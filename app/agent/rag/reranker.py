"""阿里云百炼文本重排客户端。"""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.agent.rag.backends.base import RetrievedChunk


def derive_rerank_url(openai_base_url: str) -> str:
    base = openai_base_url.rstrip("/")
    for suffix in ("/compatible-mode/v1", "/compatible-api/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + "/api/v1/services/rerank/text-rerank/text-rerank"


class BailianReranker:
    def __init__(self, api_key: str, base_url: str, model: str = "qwen3.7-text-rerank"):
        self._api_key = api_key
        self._url = derive_rerank_url(base_url)
        self._model = model
        self.total_tokens = 0
        self.calls = 0

    def rerank(
        self, query: str, hits: list[RetrievedChunk], top_n: int = 3
    ) -> list[RetrievedChunk]:
        if len(hits) <= 1:
            return hits[:top_n]
        payload = {
            "model": self._model,
            "input": {"query": query, "documents": [hit.chunk.text for hit in hits]},
            "parameters": {
                "top_n": min(top_n, len(hits)),
                "instruct": "Given an ecommerce support query, retrieve passages that directly provide the rules needed to answer it.",
            },
        }
        request = Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"百炼Rerank调用失败({exc.code}): {detail}") from exc
        self.calls += 1
        self.total_tokens += int(data.get("usage", {}).get("total_tokens", 0))
        results = data.get("output", {}).get("results", [])
        return [
            RetrievedChunk(hits[int(item["index"])].chunk, float(item["relevance_score"]))
            for item in results
        ]
