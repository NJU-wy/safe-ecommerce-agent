"""独立于向量相似度的知识库可回答性判断。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerabilityDecision:
    answerable: bool
    reason: str
    confidence: float


_UNSUPPORTED_PATTERNS = (
    (r"关税|寄(?:到|往)?法国|跨境", "当前知识库不覆盖跨境与关税"),
    (r"主播|开播|直播时间", "静态知识库不包含直播排期"),
    (r"线下门店|门店地址|门店.*几点", "当前知识库不包含线下门店实时信息"),
    (r"天气", "超出电商知识库范围"),
    (r"身份证号码|(?:支付密码|短信验证码).*(?:是什么|多少|告诉|给我|查询)", "敏感信息不得检索或披露"),
    (r"快递员.*(?:电话|手机号码|身份证)", "不提供配送人员隐私信息"),
    (r"(?:下月|下个月|下季度|明年|下一版|下一次|下次|重新补货).*(?:出|推出|补货|发布|更新|会不会|多少|哪天|什么|价格)", "静态知识库不能预测未来信息"),
    (r"(?:今晚|今天|现在).*(?:几点|排队|多少台|便宜多少)", "需要实时系统而非静态知识库"),
    (r"(?:京东|官方店|其他平台).*(?:同款|便宜|价格)", "知识库不含外部平台实时价格"),
    (r"具体激活日期|最终会判定|一定出库", "需要订单、品牌或履约实时系统"),
    (r"用户.+(?:用了|使用了|一共).+多少积分", "需要用户账户实时数据"),
)


def classify_answerability(query: str) -> AnswerabilityDecision:
    if re.search(r"知识库.*(?:能|可以).*(?:预测|回答|查询)", query):
        return AnswerabilityDecision(True, "这是关于知识库能力边界的政策问题", 0.95)
    for pattern, reason in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return AnswerabilityDecision(False, reason, 0.98)
    return AnswerabilityDecision(True, "未命中确定性越界规则，继续检查检索证据", 0.70)
