# services/embedding.py
"""Async client for an OpenAI-compatible embedding server (e.g. llama-server).
Vectors are L2-normalized after truncation so that cosine == 1 - d²/2."""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


class EmbeddingError(Exception):
    """Any failure to produce an embedding vector."""


def _normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class EmbeddingService:
    def __init__(self, *, api_url: str | None = None,
                 api_key: str | None = None, model: str | None = None,
                 dim: int | None = None):
        self._url = (api_url or settings.embedding_api_url).rstrip("/") + "/v1/embeddings"
        self._key = api_key or settings.embedding_api_key
        self._model = model or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.embedding_timeout,
                                  connect=settings.embedding_connect_timeout),
            headers=({"Authorization": f"Bearer {self._key}"} if self._key else {}),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed_document(self, text: str) -> list[float]:
        return await self._call(text, prefix=settings.embedding_document_prefix)

    async def embed_query(self, text: str) -> list[float]:
        return await self._call(text, prefix=settings.embedding_query_prefix)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            out.extend(await self._call_batch(batch,
                                              prefix=settings.embedding_document_prefix))
        return out

    async def _call_batch(self, texts: list[str], *, prefix: str) -> list[list[float]]:
        payload = {"model": self._model,
                   "input": [prefix + t for t in texts]}
        try:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise EmbeddingError(f"embedding request failed: {e}") from e

        try:
            raw = [item["embedding"] for item in
                   sorted(data["data"], key=lambda d: d["index"])]
        except (KeyError, IndexError, TypeError) as e:
            raise EmbeddingError(f"malformed embedding response: {e}") from e

        if len(raw) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} vectors, got {len(raw)}")

        out = []
        for vec in raw:
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError("empty embedding vector")
            if settings.embedding_truncate_matryoshka and len(vec) > self._dim:
                vec = vec[: self._dim]
            if len(vec) != self._dim:
                raise EmbeddingError(
                    f"embedding dim {len(vec)} != expected {self._dim}")
            out.append(_normalize([float(x) for x in vec]))
        return out

    async def _call(self, text: str, *, prefix: str = "") -> list[float]:
        return (await self._call_batch([text], prefix=prefix))[0]
