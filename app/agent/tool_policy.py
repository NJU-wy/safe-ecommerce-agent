"""按当前用户意图裁剪 Function Calling 工具集。

完整暴露 8 个工具会让模型在简单问题中额外调用 load_skill、memory 或无关查询。
该策略只裁剪“可见工具”，不改变工具实现，是低风险且可测试的成本优化。
"""

from __future__ import annotations

import re

from app.agent.response_policy import classify_intent, requires_human_service
from app.agent.refund_safety import is_explicit_refund_confirmation
from app.schemas.response import IntentType


def allowed_tool_names(user_input: str) -> set[str]:
    text = user_input.strip()
    intent = classify_intent(text)

    if requires_human_service(text) and not re.search(r"ORD-[\w-]+", text, re.IGNORECASE):
        return set()
    # 无结构化实体的规则/政策问题统一走知识库，不凭模型常识回答。
    if re.search(r"政策|规则|发票|权益|七天无理由|怎么办|多久到账|运费谁", text):
        return {"search_knowledge"}
    if intent is IntentType.RETURN_REQUEST:
        has_order = bool(re.search(r"ORD-[\w-]+", text, re.IGNORECASE))
        if re.search(r"政策|规则|多久|到账|运费|流程|哪里", text) and not has_order:
            return {"search_knowledge"}
        tools = {"query_order", "list_user_orders"}
        # 具体订单提供事实，RAG提供类目/退换边界；模型需要同时看到两者。
        if has_order:
            tools.add("search_knowledge")
        if is_explicit_refund_confirmation(text.splitlines()[-1]):
            tools.add("apply_refund")
        return tools
    if intent is IntentType.ORDER_QUERY:
        if re.search(r"物流|快递|到哪|运单|派送|送达", text):
            return {"query_logistics", "list_user_orders"}
        if re.search(r"一般|政策|规则|怎么办|多久", text) and not re.search(r"ORD-", text, re.IGNORECASE):
            return {"search_knowledge"}
        return {"query_order", "list_user_orders"}
    if intent is IntentType.PRODUCT_CONSULT:
        return {"query_product"}
    if intent is IntentType.AFTER_SALE:
        tools = {"search_knowledge"}
        if re.search(r"ORD-[\w-]+", text, re.IGNORECASE):
            tools.add("query_order")
        return tools
    if intent in {IntentType.PROMOTION, IntentType.ACCOUNT}:
        return {"search_knowledge"}
    if intent is IntentType.COMPLAINT:
        return {"query_order"} if re.search(r"ORD-[\w-]+", text, re.IGNORECASE) else set()

    allowed: set[str] = set()
    if re.search(r"记得|之前|偏好|上次", text):
        allowed.add("recall_user_memory")
    if re.search(r"标准流程|完整流程|复杂流程", text):
        allowed.add("load_skill")
    return allowed


def select_tool_definitions(user_input: str, definitions: list[dict]) -> list[dict]:
    allowed = allowed_tool_names(user_input)
    return [item for item in definitions if item["function"]["name"] in allowed]


def should_include_skill_catalog(user_input: str) -> bool:
    """常规客服不注入 Skill 目录，只为明确的复杂流程请求提供发现信息。"""
    return bool(re.search(r"标准流程|完整流程|复杂流程|按步骤处理", user_input))
