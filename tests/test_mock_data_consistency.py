from app.agent.tools.logistics import query_logistics
from app.agent.tools.mock_data import LOGISTICS, ORDERS, PRODUCTS
from app.agent.tools.refund import apply_refund


def test_order_skus_and_tracking_references_exist():
    for order in ORDERS.values():
        for item in order["items"]:
            assert item["sku"] in PRODUCTS, (order["order_id"], item["sku"])
        if order.get("tracking_number"):
            assert order["tracking_number"] in LOGISTICS


def test_catalog_and_orders_cover_supported_categories():
    assert len(ORDERS) >= 13
    assert len(PRODUCTS) >= 7
    assert any(order.get("split_shipment") for order in ORDERS.values())
    assert any(
        item.get("activation_status") == "activated"
        for order in ORDERS.values() for item in order["items"]
    )


def test_refunded_order_cannot_be_refunded_again(tmp_path, monkeypatch):
    from app.agent.tools import refund
    monkeypatch.setattr(refund.settings, "refund_ledger_path", str(tmp_path / "ledger.json"))
    result = apply_refund("ORD-20240218-013", "重复购买", True, "completed-order")
    assert result["success"] is False
    assert "已退款完成" in result["error"]


def test_new_order_logistics_is_queryable():
    result = query_logistics("ORD-20240208-009")
    assert result["success"] is True
    assert result["logistics"]["status"] == "refused"
