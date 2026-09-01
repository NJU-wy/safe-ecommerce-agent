"""向量化封装：调用 OpenAI Embeddings 接口。

- 支持批量编码（list[str] → list[list[float]]）。
- 可配置 model 与 base_url（与 chat 模型共用一套 OpenAI 客户端配置）。
- 返回原始 list[float]，由调用方决定如何持久化（这里用 json，不引入 numpy 依赖）。
"""

from typing import Iterable
import hashlib
import math

from openai import OpenAI


class Embedder:
    """OpenAI Embeddings 同步封装。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """批量编码，自动按 batch_size 分批请求。"""
        texts = list(texts)
        if not texts:
            return []
        if self._model == "local-hash-embedding-v1":
            return [self._encode_local(text) for text in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    @staticmethod
    def _encode_local(text: str, dimensions: int = 1024) -> list[float]:
        """Deterministic character n-gram hashing for the bundled Chinese KB.

        This zero-download fallback is intentionally small and transparent. It
        is suitable for local demos/tests, not a replacement for a production
        semantic embedding model.
        """
        normalized = "".join(text.lower().split())
        features = list(normalized)
        features.extend(normalized[i:i + 2] for i in range(len(normalized) - 1))
        features.extend(normalized[i:i + 3] for i in range(len(normalized) - 2))

        vector = [0.0] * dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % dimensions
            sign = 1.0 if (value >> 63) == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector
