"""客服响应元数据的确定性策略。

原实现为了生成 intent/requires_human，会在每轮 ReAct 结束后再次调用大模型。
100 条评估显示该步骤显著增加 Token，且退款常被误分为 after_sale。本模块在正文
生成后使用可审计规则产生主意图、次要意图与转人工元数据，不增加模型调用。
"""

from __future__ import annotations

import re

from app.schemas.response import CustomerServiceResponse, IntentType


_HUMAN_PATTERNS = (
    r"转人工|人工客服|真人客服",
    r"12315|消协|监管|起诉|律师|报警",
    r"曝光|媒体|赔偿",
    r"账号被盗|支付密码.*泄露|隐私|同事.*订单|朋友.*订单|别人.*订单|他人.*订单",
    r"安全隐患|伤到人|人身安全|食品安全",
)

_COMPLAINT_PATTERNS = (
    r"投诉|破服务|态度.*差|体验.*不好|太差|不满",
    r"12315|消协|起诉|曝光|赔偿|安全隐患|伤到人|转人工|人工客服",
)


def recent_user_context(messages: list[dict], limit: int = 2) -> str:
    """合并最近两条用户消息，让“就是那笔订单”等追问继承上一轮意图。"""
    user_turns = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    ]
    return "\n".join(user_turns[-limit:])


def requires_human_service(user_input: str) -> bool:
    """对高风险和明确人工诉求做代码级兜底，避免模型漏转。"""
    return any(re.search(pattern, user_input, re.IGNORECASE) for pattern in _HUMAN_PATTERNS)


def classify_intent(user_input: str) -> IntentType:
    """按互斥优先级识别意图，优先处理投诉和退款等高价值业务。"""
    text = user_input.strip()
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in _COMPLAINT_PATTERNS):
        return IntentType.COMPLAINT
    if re.search(r"退款|退货|换货|退掉|不想要|拒收|七天无理由|apply_refund", text, re.IGNORECASE):
        return IntentType.RETURN_REQUEST
    if re.search(r"账号|账户|密码|登录|绑定手机|手机号|隐私|被盗|同事.*订单|朋友.*订单|别人.*订单|他人.*订单", text):
        return IntentType.ACCOUNT
    if re.search(r"优惠券|活动|促销|折扣|满减|包邮|会员权益", text):
        return IntentType.PROMOTION
    if re.search(r"维修|保修|售后", text):
        return IntentType.AFTER_SALE
    if re.search(r"订单|ORD-|物流|快递|发货|配送|签收|运单|下单.*送到|预计.*到", text, re.IGNORECASE):
        return IntentType.ORDER_QUERY
    if re.search(r"商品|库存|价格|多少钱|规格|材质|续航|推荐|耳机|手机|鞋|吸尘器|牛仔裤|AirPods|Nike|小米|戴森|Levi|保护壳|相机", text, re.IGNORECASE):
        return IntentType.PRODUCT_CONSULT
    if re.fullmatch(r"[\s。！!,.，…]*", text):
        return IntentType.OTHER
    if re.search(r"你好|您好|嗨|谢谢|再见|👋", text):
        return IntentType.GREETING
    return IntentType.OTHER


def classify_intents(user_input: str) -> tuple[IntentType, list[IntentType]]:
    """识别主、次意图；用于响应元数据和评估，不直接触发多 Agent 并行调度。"""
    primary = classify_intent(user_input)
    rules = (
        (IntentType.COMPLAINT, _COMPLAINT_PATTERNS),
        (IntentType.RETURN_REQUEST, (r"退款|退货|换货|退掉|不想要|拒收|七天无理由|apply_refund",)),
        (IntentType.ACCOUNT, (r"账号|账户|密码|登录|绑定手机|手机号|隐私|被盗|同事.*订单|朋友.*订单|别人.*订单|他人.*订单",)),
        (IntentType.PROMOTION, (r"优惠券|活动|促销|折扣|满减|包邮|会员权益",)),
        (IntentType.AFTER_SALE, (r"维修|保修|售后",)),
        (IntentType.ORDER_QUERY, (r"订单|ORD-|物流|快递|发货|配送|签收|运单|下单.*送到|预计.*到",)),
        (IntentType.PRODUCT_CONSULT, (r"商品|库存|价格|多少钱|规格|材质|续航|推荐|耳机|手机|鞋|吸尘器|牛仔裤|AirPods|Nike|小米|戴森|Levi|保护壳|相机",)),
        (IntentType.GREETING, (r"你好|您好|嗨|谢谢|再见|👋",)),
    )
    matched = [
        intent for intent, patterns in rules
        if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in patterns)
    ]
    secondary = [intent for intent in matched if intent != primary]
    return primary, secondary


def build_customer_response(user_input: str, reply: str) -> CustomerServiceResponse:
    """不增加模型调用地构建对外响应结构。"""
    intent, secondary_intents = classify_intents(user_input)
    follow_up = None
    if reply.rstrip().endswith(("？", "?")):
        follow_up = reply.rstrip().splitlines()[-1]
    return CustomerServiceResponse(
        intent=intent,
        primary_intent=intent,
        secondary_intents=secondary_intents,
        confidence=0.95 if intent is not IntentType.OTHER else 0.75,
        reply=reply,
        requires_human=requires_human_service(user_input),
        follow_up_question=follow_up,
    )
