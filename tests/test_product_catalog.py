from app.agent.tools.product import query_product


def test_unknown_product_is_not_fabricated():
    result = query_product("生鲜三文鱼")
    assert result["success"] is False
    assert result["products"] == []
    assert "未找到" in result["error"]


def test_existing_catalog_product_is_returned():
    result = query_product("运动鞋")
    assert result["success"] is True
    assert result["products"]
    assert all(not item["product_id"].startswith("MOCK-") for item in result["products"])
