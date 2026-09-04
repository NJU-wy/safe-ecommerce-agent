"""轻量中文 BM25 检索器，无需分词服务或额外运行时依赖。

中文文本采用汉字 unigram + bigram；英文、数字和订单式标识按连续词切分。
unigram 保证短查询有召回，bigram 提供词序区分。BM25 负责精确词面匹配，
与语义/哈希向量召回互补，不承担答案生成。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_-]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in _TOKEN_RE.findall((text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            chars = list(part)
            tokens.extend(chars)
            tokens.extend(a + b for a, b in zip(chars, chars[1:]))
        else:
            tokens.append(part)
    return tokens


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._term_freqs = [Counter(tokenize(c.text)) for c in chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avgdl = sum(self._lengths) / len(chunks) if chunks else 0.0
        doc_freq: dict[str, int] = defaultdict(int)
        for tf in self._term_freqs:
            for term in tf:
                doc_freq[term] += 1
        n = len(chunks)
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_terms = Counter(tokenize(query))
        scored: list[tuple[int, float]] = []
        for idx, tf in enumerate(self._term_freqs):
            dl = self._lengths[idx]
            score = 0.0
            for term, query_freq in query_terms.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = freq + self._k1 * (
                    1.0 - self._b + self._b * dl / (self._avgdl or 1.0)
                )
                term_score = self._idf.get(term, 0.0) * freq * (self._k1 + 1.0) / norm
                score += term_score * (1.0 + math.log(query_freq))
            scored.append((idx, score))
        scored.sort(key=lambda item: (-item[1], self._chunks[item[0]].chunk_id))
        return [
            RetrievedChunk(self._chunks[idx], score)
            for idx, score in scored[:top_k]
        ]
