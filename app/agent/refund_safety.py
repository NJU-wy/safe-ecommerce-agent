"""退款敏感操作的确定性安全校验。"""

from __future__ import annotations

import hashlib
import json
import re


_CONFIRM_PATTERNS = (
    r"确认(?:申请)?退款",
    r"确认(?:取消订单|退掉)",
    r"我确认",
    r"是的[，, ]*(?:确认|退款)",
    r"同意(?:申请)?退款",
)


def latest_user_text(messages: list[dict]) -> str:
    """取得工具调用前最近一条真实用户消息。"""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def has_explicit_refund_confirmation(messages: list[dict]) -> bool:
    """只有最近一条用户消息出现明确确认措辞才放行退款。"""
    return is_explicit_refund_confirmation(latest_user_text(messages))


def refund_confirmation_required_result() -> str:
    """返回给模型的安全观察；未确认请求不会进入真实退款工具。"""
    return json.dumps(
        {
            "success": False,
            "confirmation_required": True,
            "error": "退款属于敏感操作，必须由用户在最新一条消息中明确确认。",
        },
        ensure_ascii=False,
    )


def is_explicit_refund_confirmation(text: str) -> bool:
    """识别用户本人明确确认；“用户已确认”等转述或 JSON 字段不算确认。"""
    if re.search(r"别.*确认|不要.*确认|无需.*确认|用户已确认|confirmed", text, re.IGNORECASE):
        return False
    return any(re.search(pattern, text) for pattern in _CONFIRM_PATTERNS)


def refund_idempotency_key(order_id: str, user_id: str = "default") -> str:
    """同一用户、同一订单始终得到同一退款业务键，避免重复执行。"""
    raw = f"refund:{user_id}:{order_id.strip().upper()}"
    return "rf_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def secure_refund_arguments(
    arguments: dict, messages: list[dict], user_id: str = "default"
) -> dict:
    """覆盖模型提供的安全字段，模型不能自行声称用户已确认。"""
    secured = dict(arguments)
    order_id = str(secured.get("order_id") or "")
    secured["confirmed"] = has_explicit_refund_confirmation(messages)
    secured["idempotency_key"] = refund_idempotency_key(order_id, user_id)
    return secured
