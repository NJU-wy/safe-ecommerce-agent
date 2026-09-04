"""知识库检索器：query → 向量化 → 委托后端检索。

设计上 retriever 只负责"问句怎么变向量""结果怎么聚合"，
存储和打分都交给 VectorBackend 实现，对应两套：
- NumpyBackend：手写余弦 + JSON 持久化（教学透明，零依赖）
- ChromaBackend：嵌入式向量数据库 + HNSW（生产代表性）

校验逻辑：加载后比对 backend 持久化的 embedding_model 与当前 Embedder.model，
不一致直接报错——避免"换了 embedding 但还在用老索引"这种隐蔽问题。
"""

from __future__ import annotations

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.embedder import Embedder
from app.agent.rag.bm25 import BM25Retriever
from app.agent.rag.query_decomposer import decompose_query

__all__ = ["KnowledgeRetriever", "RetrievedChunk"]


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给具体 backend。"""

    def __init__(self, embedder: Embedder, backend: VectorBackend, mode: str = "semantic"):
        self._embedder = embedder
        self._backend = backend
        self._mode = mode.lower()
        if self._mode not in {"semantic", "bm25", "hybrid"}:
            raise ValueError("检索模式必须是 semantic / bm25 / hybrid")
        self._bm25: BM25Retriever | None = None
        self.last_subqueries: list[str] = []
        self._loaded = False

    @property
    def backend(self) -> VectorBackend:
        return self._backend

    @property
    def size(self) -> int:
        return self._backend.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()

        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        if self._mode in {"bm25", "hybrid"}:
            self._bm25 = BM25Retriever(self._backend.all_chunks())
        self._loaded = True

    def search(
        self, query: str, top_k: int = 3, *, decompose: bool = False,
        candidate_k: int | None = None, subqueries: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._loaded:
            self.load()
        candidate_k = max(top_k, candidate_k or top_k)
        subqueries = subqueries if subqueries is not None else (
            decompose_query(query) if decompose else []
        )
        self.last_subqueries = subqueries
        if subqueries:
            rankings = [self._search_one(query, candidate_k)]
            rankings.extend(self._search_one(part, candidate_k) for part in subqueries)
            return _coverage_fusion(rankings, top_k)
        return self._search_one(query, top_k)

    def _search_one(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._mode == "bm25":
            return self._bm25.search(query, top_k)  # type: ignore[union-attr]
        q_vec = self._embedder.encode_one(query)
        semantic = self._backend.search(q_vec, top_k=top_k if self._mode == "semantic" else min(self.size, max(20, top_k * 5)))
        if self._mode == "semantic":
            return semantic
        lexical = self._bm25.search(query, min(self.size, max(20, top_k * 5)))  # type: ignore[union-attr]
        return _reciprocal_rank_fusion(semantic, lexical, top_k)


def _coverage_fusion(
    rankings: list[list[RetrievedChunk]], top_k: int, k: int = 60
) -> list[RetrievedChunk]:
    """融合原查询和子查询排名，并优先覆盖不同子查询的首选证据。"""
    chunks = {hit.chunk.chunk_id: hit.chunk for hits in rankings for hit in hits}
    scores = {cid: 0.0 for cid in chunks}
    for hits in rankings:
        for rank, hit in enumerate(hits, 1):
            scores[hit.chunk.chunk_id] += 1.0 / (k + rank)

    selected: list[str] = []
    # 原查询负责整体语义；每个子查询最多贡献一个尚未选择的首选片段。
    for hits in rankings:
        first_new = next((h.chunk.chunk_id for h in hits if h.chunk.chunk_id not in selected), None)
        if first_new:
            selected.append(first_new)
        if len(selected) >= top_k:
            break
    for cid in sorted(scores, key=lambda item: (-scores[item], item)):
        if cid not in selected:
            selected.append(cid)
        if len(selected) >= top_k:
            break
    max_score = len(rankings) / (k + 1)
    return [RetrievedChunk(chunks[cid], scores[cid] / max_score) for cid in selected[:top_k]]


def _reciprocal_rank_fusion(
    semantic: list[RetrievedChunk], lexical: list[RetrievedChunk], top_k: int, k: int = 60
) -> list[RetrievedChunk]:
    """用 RRF 融合异质量纲分数，避免直接相加余弦分与 BM25 分。"""
    chunks = {hit.chunk.chunk_id: hit.chunk for hit in semantic + lexical}
    scores: dict[str, float] = {cid: 0.0 for cid in chunks}
    for hits in (semantic, lexical):
        for rank, hit in enumerate(hits, 1):
            scores[hit.chunk.chunk_id] += 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
    max_score = 2.0 / (k + 1)
    return [RetrievedChunk(chunks[cid], scores[cid] / max_score) for cid in ranked]
