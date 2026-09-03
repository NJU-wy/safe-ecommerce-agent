"""会话级退款状态机：把跨轮授权从自然语言历史提升为显式状态。"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from app.agent.refund_safety import is_explicit_refund_confirmation

_AUDIT_LOCK = threading.Lock()
_ORDER_RE = re.compile(r"ORD-[\w-]+", re.IGNORECASE)


class RefundState(str, Enum):
    IDLE = "idle"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class RefundWorkflow:
    user_id: str
    audit_path: str
    state: RefundState = RefundState.IDLE
    order_id: str = ""
    reason: str = ""
    idempotency_key: str = ""

    def observe_user(self, text: str) -> None:
        orders = [item.upper() for item in _ORDER_RE.findall(text)]
        if re.search(r"取消退款|不退了|不要退", text):
            self._transition(RefundState.CANCELLED, "user_cancelled")
            return
        mentions_refund = bool(re.search(r"退款|退掉|取消订单", text))
        if mentions_refund:
            order_id = orders[-1] if orders else self.order_id
            if order_id != self.order_id or self.state in {
                RefundState.IDLE, RefundState.EXECUTED, RefundState.REJECTED,
                RefundState.CANCELLED,
            }:
                self.order_id = order_id
                self._transition(RefundState.AWAITING_CONFIRMATION, "refund_requested")
        elif orders and self.state is RefundState.AWAITING_CONFIRMATION and not self.order_id:
            self.order_id = orders[-1]
            self.audit("order_bound", detail="order_supplied_in_followup")
        if is_explicit_refund_confirmation(text):
            if self.state is RefundState.AWAITING_CONFIRMATION:
                if orders and orders[-1] != self.order_id:
                    self.audit("confirmation_blocked", detail="order_mismatch")
                elif not self.order_id:
                    self.audit("confirmation_blocked", detail="missing_order")
                else:
                    self._transition(RefundState.CONFIRMED, "user_confirmed")
            else:
                self.audit("confirmation_blocked", detail="no_pending_refund")

    def authorize(self, order_id: str, latest_user_text: str, idempotency_key: str = "",
                  reason: str = "") -> bool:
        normalized = order_id.strip().upper()
        allowed = (
            self.state is RefundState.CONFIRMED
            and normalized == self.order_id
            and is_explicit_refund_confirmation(latest_user_text)
        )
        self.idempotency_key = idempotency_key
        self.reason = reason.strip()
        self.audit(
            "authorization_allowed" if allowed else "authorization_blocked",
            order_id=normalized,
            detail="state_and_order_bound_check",
        )
        return allowed

    def record_result(self, result: dict) -> None:
        target = RefundState.EXECUTED if result.get("success") else RefundState.REJECTED
        self._transition(target, str(result.get("refund_status") or result.get("error") or ""))

    def _transition(self, target: RefundState, detail: str) -> None:
        previous = self.state
        self.state = target
        self.audit("state_transition", detail=detail, previous_state=previous.value)

    def audit(self, event: str, detail: str = "", order_id: str = "", previous_state: str = "") -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "user_id": self.user_id,
            "order_id": order_id or self.order_id,
            "previous_state": previous_state,
            "state": self.state.value,
            "detail": detail,
            "idempotency_key": self.idempotency_key,
            "reason": self.reason,
        }
        path = Path(self.audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict, user_id: str, audit_path: str) -> "RefundWorkflow":
        return cls(user_id=user_id, audit_path=audit_path,
                   state=RefundState(data.get("state", RefundState.IDLE.value)),
                   order_id=str(data.get("order_id", "")), reason=str(data.get("reason", "")),
                   idempotency_key=str(data.get("idempotency_key", "")))
