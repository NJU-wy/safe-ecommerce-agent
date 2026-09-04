"""按当前用户意图裁剪 Function Calling 工具集。

完整暴露 8 个工具会让模型在简单问题中额外调用 load_skill、memory 或无关查询。
该策略只裁剪“可见工具”，不改变工具实现，是低风险且可测试的成本优化。
"""

from __future__ import annotations

import re

from app.agent.response_policy import classify_intent, requires_human_service
from app.agent.refund_safety import is_explicit_refund_confirmation
from app.schemas.response import IntentType


_ORDER_ID = r"ORD-[\w-]+"
_PRIVACY_REQUEST = r"(?:同事|朋友|别人|他人|非本人).{0,8}订单|替我查.{0,8}(?:同事|朋友|别人|他人)"
_POLICY_TERMS = r"政策|规则|七天无理由|能否|还能|可以退|是否支持|运费|包装|激活|拆封|洗过|吊牌|试穿|入耳"


def allowed_tool_names(user_input: str) -> set[str]:
    text = user_input.strip()
    intent = classify_intent(text)

    # 身份校验不能依赖模型：即使消息含合法格式订单号，也绝不向模型暴露查询工具。
    if re.search(_PRIVACY_REQUEST, text, re.IGNORECASE):
        return {"escalate_to_human"}
    if requires_human_service(text) and not re.search(_ORDER_ID, text, re.IGNORECASE):
        return {"escalate_to_human"}
    # 无结构化实体的规则/政策问题统一走知识库，不凭模型常识回答。
    if re.search(r"政策|规则|发票|权益|七天无理由|怎么办|多久到账|运费谁", text):
        return {"search_knowledge"}
    if intent is IntentType.RETURN_REQUEST:
        has_order = bool(re.search(_ORDER_ID, text, re.IGNORECASE))
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
        tools = {"escalate_to_human"}
        if re.search(_ORDER_ID, text, re.IGNORECASE):
            tools.add("query_logistics" if re.search(r"物流|快递|配送|派送", text) else "query_order")
        return tools

    allowed: set[str] = set()
    if re.search(r"记得|之前|偏好|上次", text):
        allowed.add("recall_user_memory")
    if re.search(r"标准流程|完整流程|复杂流程", text):
        allowed.add("load_skill")
    return allowed


def required_tool_names(user_input: str) -> set[str]:
    """代码级定义本轮必须取得的证据/动作，防止模型只调用一半就作答。"""
    text = user_input.strip()
    visible = allowed_tool_names(text)
    required: set[str] = set()
    if "escalate_to_human" in visible:
        required.add("escalate_to_human")
    if re.search(_ORDER_ID, text, re.IGNORECASE):
        if "query_logistics" in visible:
            required.add("query_logistics")
        elif "query_order" in visible:
            required.add("query_order")
    if (
        re.search(_ORDER_ID, text, re.IGNORECASE)
        and re.search(_POLICY_TERMS, text, re.IGNORECASE)
        and "search_knowledge" in visible
    ):
        required.add("search_knowledge")
    return required


def select_tool_definitions(user_input: str, definitions: list[dict]) -> list[dict]:
    allowed = allowed_tool_names(user_input)
    return [item for item in definitions if item["function"]["name"] in allowed]


def is_tool_call_allowed(user_input: str, tool_name: str) -> bool:
    """执行前再次校验工具权限，防止模型或上游路由绕过可见工具集。"""
    return tool_name in allowed_tool_names(user_input)


def should_include_skill_catalog(user_input: str) -> bool:
    """常规客服不注入 Skill 目录，只为明确的复杂流程请求提供发现信息。"""
    return bool(re.search(r"标准流程|完整流程|复杂流程|按步骤处理", user_input))
