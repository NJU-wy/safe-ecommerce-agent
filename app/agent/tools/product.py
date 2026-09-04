import re

from app.agent.tools.mock_data import PRODUCTS


def _matches_token(searchable: str, token: str) -> bool:
    """中文按连续词匹配；英文/数字按完整词匹配，避免 ``VI`` 命中 Levi's。"""
    if re.search(r"[\u4e00-\u9fff]", token):
        return len(token) >= 2 and token in searchable
    if len(token) < 3:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", searchable))


def _match_score(product: dict, keywords: list[str]) -> int:
    """计算商品与关键词列表的匹配度（命中关键词数量）。"""
    searchable = " ".join([
        product["name"],
        product["category"],
        product.get("description", ""),
        " ".join(str(v) for v in product.get("specs", {}).values()),
    ]).lower()
    return sum(1 for kw in keywords if _matches_token(searchable, kw))


def query_product(keyword: str) -> dict:
    """根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。"""
    if keyword in PRODUCTS:
        return {"success": True, "products": [PRODUCTS[keyword]]}

    keywords = [kw.lower() for kw in keyword.split() if kw.strip()]
    if not keywords:
        keywords = [keyword.lower()]

    scored = [(p, _match_score(p, keywords)) for p in PRODUCTS.values()]
    results = [p for p, score in scored if score > 0]

    if not results:
        # 商品目录是结构化事实源；未命中时禁止随机生成价格、库存或类目。
        return {
            "success": False,
            "error": f"未找到与“{keyword}”匹配的商品",
            "products": [],
        }
    return {"success": True, "products": results}
