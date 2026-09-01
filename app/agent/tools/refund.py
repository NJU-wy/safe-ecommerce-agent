import json
import os
import threading
from pathlib import Path

from app.agent.tools.mock_data import ORDERS
from app.config.settings import settings


_LEDGER_LOCK = threading.Lock()


def _load_ledger(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp_path, path)


def apply_refund(
    order_id: str,
    reason: str,
    confirmed: bool = False,
    idempotency_key: str = "",
) -> dict:
    """经明确确认后幂等地申请退款。"""
    order_id = order_id.strip().upper()
    reason = reason.strip()
    if not confirmed:
        return {
            "success": False,
            "confirmation_required": True,
            "error": "退款尚未执行：请用户明确回复“确认退款”后再提交",
        }
    if not reason:
        return {"success": False, "error": "退款原因不能为空"}
    if not idempotency_key:
        return {"success": False, "error": "缺少幂等键，退款未执行"}

    order = ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"未找到订单 {order_id}，请核实订单号"}

    if order["status"] == "refund_processing":
        return {"success": False, "error": "该订单已有退款申请正在处理中，请耐心等待"}

    if order["status"] == "shipped":
        return {
            "success": False,
            "error": "订单已经发货，不能直接退款；请先拒收或签收后按退货流程申请",
        }

    payload = {"order_id": order_id, "reason": reason}
    ledger_path = Path(settings.refund_ledger_path)
    with _LEDGER_LOCK:
        ledger = _load_ledger(ledger_path)
        previous = ledger.get(idempotency_key)
        if previous:
            if previous.get("request") != payload:
                return {
                    "success": False,
                    "idempotency_conflict": True,
                    "error": "该幂等键已用于另一笔退款请求，退款未执行",
                }
            replay = dict(previous["result"])
            replay["idempotent_replay"] = True
            return replay

        if order["status"] == "pending":
            result = {
                "success": True,
                "refund_status": "submitted",
                "idempotent_replay": False,
                "message": (
                    f"订单 {order_id} 尚未发货，已取消并发起退款。"
                    f"退款原因：{reason}。退款将在 1-3 个工作日内原路退回。"
                ),
            }
        else:
            result = {
                "success": True,
                "refund_status": "submitted",
                "idempotent_replay": False,
                "message": (
                    f"退款申请已提交。订单 {order_id}，退款原因：{reason}。"
                    f"预计 1-3 个工作日内审核完成，届时会通知您退货地址。"
                ),
            }
        ledger[idempotency_key] = {"request": payload, "result": result}
        _save_ledger(ledger_path, ledger)
        return result
