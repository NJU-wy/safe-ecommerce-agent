"""确定性中文复合查询拆解。

不额外调用LLM，避免每次RAG检索增加生成成本。拆解只在存在明显并列、转折或
分别询问标记时触发；原始问题始终保留，防止子句失去商品/订单上下文。
"""

from __future__ import annotations

import re

_BOUNDARY_RE = re.compile(r"[；;。！？!?，,]|(?:但|但是|另外|并且|而且|以及)")


def decompose_query(query: str, max_subqueries: int = 4) -> list[str]:
    normalized = " ".join((query or "").strip().split())
    if not normalized:
        return []
    raw_parts = [part.strip(" ，,、：:") for part in _BOUNDARY_RE.split(normalized)]
    parts: list[str] = []
    for part in raw_parts:
        if len(part) < 6 and parts:
            parts[-1] += "，" + part
        elif len(part) >= 4:
            parts.append(part)
    unique: list[str] = []
    for part in parts:
        if part != normalized and part not in unique:
            unique.append(part)
    # 单一短句无需拆解；原查询由检索器另行保留。
    return unique[:max_subqueries] if len(unique) >= 2 else []


def looks_composite(query: str) -> bool:
    """只让明显复合问题进入较昂贵的LLM拆解。"""
    return bool(re.search(r"[；;]|同时|分别|以及|而且|并且|但|还是|又|只.*(?:退|换|取消)", query))


def needs_extended_context(query: str) -> bool:
    """复杂问题返回Top-5，普通问题保持Top-3以控制上下文长度。"""
    if looks_composite(query):
        return True
    aspects = (
        r"订单|发货|物流|签收|拒收|分包",
        r"退货|退款|换货|取消",
        r"质量|故障|开胶|重启|损坏",
        r"证据|材料|照片|视频|质检|检测",
        r"激活|拆封|使用|试穿|洗过",
        r"政策|规则|新版|旧版|生效",
        r"运费|赔偿|扣减|金额",
    )
    return sum(bool(re.search(pattern, query)) for pattern in aspects) >= 2
