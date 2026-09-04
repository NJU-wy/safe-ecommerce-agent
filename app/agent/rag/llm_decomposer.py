"""使用聊天模型把复杂检索问题改写为自包含的原子子查询。"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from app.agent.rag.query_decomposer import looks_composite


class LLMQueryDecomposer:
    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def decompose(self, query: str, max_subqueries: int = 4) -> list[str]:
        if not looks_composite(query):
            return []
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是电商知识检索查询拆解器。把复合问题拆成2到4个可独立检索的中文问题。"
                        "每个子问题必须继承原问题中的商品、订单和条件，不能使用‘它/这件/那条’等失去上下文的指代。"
                        "不要回答问题，不要增加原文没有的事实。只输出JSON：{\"queries\":[\"...\"]}。"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0)).get("queries", [])
        except (json.JSONDecodeError, AttributeError):
            return []
        output: list[str] = []
        for item in items:
            text = str(item).strip()
            if len(text) >= 4 and text != query and text not in output:
                output.append(text)
        return output[:max_subqueries] if len(output) >= 2 else []
