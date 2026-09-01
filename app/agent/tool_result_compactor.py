"""压缩写入模型上下文的工具结果，完整业务工具返回值保持不变。"""

from __future__ import annotations

import json


def _pick(source: dict, keys: tuple[str, ...]) -> dict:
    return {key: source[key] for key in keys if key in source}


def compact_tool_result(name: str, result_json: str) -> str:
    """保留回答所需字段，减少后续 ReAct 与多轮对话的 Prompt Token。"""
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return result_json[:2000]

    if not data.get("success"):
        return json.dumps(data, ensure_ascii=False)

    if name == "query_order" and isinstance(data.get("order"), dict):
        order = data["order"]
        compact = _pick(order, (
            "order_id", "status", "total", "created_at", "shipped_at",
            "carrier", "tracking_number", "estimated_delivery",
            "delivered_at", "refund_status", "refund_reason",
        ))
        compact["items"] = [
            _pick(item, ("name", "quantity", "price"))
            for item in order.get("items", [])
        ]
        data["order"] = compact

    elif name == "query_logistics" and isinstance(data.get("logistics"), dict):
        logistics = data["logistics"]
        compact = _pick(logistics, ("tracking_number", "carrier", "status"))
        compact["events"] = logistics.get("events", [])[-3:]
        data["logistics"] = compact

    elif name == "query_product":
        data["products"] = [
            _pick(product, (
                "product_id", "name", "price", "stock", "description", "specs"
            ))
            for product in data.get("products", [])[:5]
        ]

    elif name == "search_knowledge":
        data["results"] = [
            {
                **_pick(hit, ("doc", "section", "score")),
                "text": str(hit.get("text", ""))[:360],
            }
            for hit in data.get("results", [])[:3]
        ]

    elif name == "load_skill" and "instructions" in data:
        data["instructions"] = str(data["instructions"])[:800]

    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
