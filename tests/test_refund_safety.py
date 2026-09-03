import json

from app.agent.refund_safety import (
    has_explicit_refund_confirmation,
    refund_idempotency_key,
    secure_refund_arguments,
)
from app.agent.tools.refund import apply_refund
from app.agent.refund_workflow import RefundState, RefundWorkflow
from app.config.settings import settings


def test_confirmation_must_come_from_latest_user_message():
    messages = [{"role": "user", "content": "订单有问题，我想退款"}]
    assert has_explicit_refund_confirmation(messages) is False
    assert secure_refund_arguments(
        {"order_id": "ORD-20240120-002", "reason": "不想要了", "confirmed": True},
        messages,
    )["confirmed"] is False

    messages.append({"role": "assistant", "content": "请确认"})
    messages.append({"role": "user", "content": "确认退款"})
    assert has_explicit_refund_confirmation(messages) is True


def test_refund_requires_confirmation_and_is_idempotent(tmp_path, monkeypatch):
    ledger = tmp_path / "refund-ledger.json"
    monkeypatch.setattr(settings, "refund_ledger_path", str(ledger))
    key = refund_idempotency_key("ORD-20240120-002", "test-user")

    rejected = apply_refund("ORD-20240120-002", "不想要了", False, key)
    assert rejected["success"] is False
    assert rejected["confirmation_required"] is True
    assert not ledger.exists()

    first = apply_refund("ORD-20240120-002", "不想要了", True, key)
    replay = apply_refund("ORD-20240120-002", "不想要了", True, key)
    assert first["success"] is True
    assert first["idempotent_replay"] is False
    assert replay["success"] is True
    assert replay["idempotent_replay"] is True
    assert len(json.loads(ledger.read_text(encoding="utf-8"))) == 1


def test_refund_validates_order_state_and_key_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "refund_ledger_path", str(tmp_path / "ledger.json"))
    key = "rf_test_conflict"

    shipped = apply_refund("ORD-20240115-001", "不想要了", True, key)
    assert shipped["success"] is False
    assert "已经发货" in shipped["error"]

    assert apply_refund("ORD-20240122-005", "买错了", True, key)["success"] is True
    conflict = apply_refund("ORD-20240120-002", "重复请求", True, key)
    assert conflict["success"] is False
    assert conflict["idempotency_conflict"] is True


def test_refund_workflow_binds_confirmation_to_pending_order(tmp_path):
    workflow = RefundWorkflow("u1", str(tmp_path / "audit.jsonl"))
    workflow.observe_user("订单 ORD-20240122-005 买错了，我想退款")
    assert workflow.state is RefundState.AWAITING_CONFIRMATION
    assert not workflow.authorize("ORD-20240122-005", "好吧")
    workflow.observe_user("确认退款")
    assert workflow.state is RefundState.CONFIRMED
    assert workflow.authorize("ORD-20240122-005", "确认退款")
    assert not workflow.authorize("ORD-20240120-002", "确认退款")


def test_old_confirmation_cannot_authorize_new_order(tmp_path):
    workflow = RefundWorkflow("u1", str(tmp_path / "audit.jsonl"))
    workflow.observe_user("订单 ORD-20240122-005 申请退款")
    workflow.observe_user("确认退款")
    workflow.observe_user("改成订单 ORD-20240120-002 退款")
    assert workflow.state is RefundState.AWAITING_CONFIRMATION
    assert workflow.order_id == "ORD-20240120-002"
    assert not workflow.authorize("ORD-20240120-002", "改成订单 ORD-20240120-002 退款")


def test_order_can_be_bound_in_followup_before_confirmation(tmp_path):
    workflow = RefundWorkflow("u1", str(tmp_path / "audit.jsonl"))
    workflow.observe_user("我想退款")
    workflow.observe_user("是订单 ORD-20240122-005")
    workflow.observe_user("确认退款")
    assert workflow.authorize("ORD-20240122-005", "确认退款")


def test_workflow_audit_and_terminal_state(tmp_path):
    audit = tmp_path / "audit.jsonl"
    workflow = RefundWorkflow("u1", str(audit))
    workflow.observe_user("订单 ORD-20240122-005 申请退款")
    workflow.observe_user("确认退款")
    workflow.record_result({"success": True, "refund_status": "submitted"})
    assert workflow.state is RefundState.EXECUTED
    records = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
    states = [r["state"] for r in records if r["event"] == "state_transition"]
    assert states == ["awaiting_confirmation", "confirmed", "executed"]
