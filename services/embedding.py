# services/embedding.py
"""Embedding client: llama.cpp /v1/embeddings, task prefixes, matryoshka truncation."""
from __future__ import annotations

import logging
from typing import Sequence

import httpx
import numpy as np
import sqlite_vec

from config.settings import settings

logger = logging.getLogger(__name__)

CHUNK = 32


class EmbeddingError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self) -> None:
        headers = {}
        if settings.embedding_api_key:
            headers["Authorization"] = f"Bearer {settings.embedding_api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.embedding_api_url,
            timeout=httpx.Timeout(settings.embedding_timeout,
                                  connect=settings.embedding_connect_timeout),
            headers=headers,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def serialize(emb: Sequence[float]) -> bytes:
        """sqlite-vec float32 blob for vec0 inserts."""
        return sqlite_vec.serialize_float32([float(x) for x in emb])

    def _postprocess(self, emb: Sequence[float]) -> list[float]:
        arr = np.asarray(emb, dtype=np.float32)
        if settings.embedding_truncate_matryoshka and arr.size > settings.embedding_dim:
            arr = arr[: settings.embedding_dim]
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return [float(x) for x in arr]

    # ------------------------------------------------------------ API
    async def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        prefix = (settings.embedding_query_prefix if kind == "query"
                  else settings.embedding_document_prefix)
        payload = [prefix + t for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(payload), CHUNK):
            chunk = payload[i:i + CHUNK]
            try:
                resp = await self._client.post(
                    "/v1/embeddings",
                    json={"input": chunk, "model": settings.embedding_model})
                resp.raise_for_status()
                data = resp.json()["data"]
            except httpx.HTTPError as e:
                raise EmbeddingError(f"embedding request failed: {e}") from e
            except (KeyError, TypeError, ValueError) as e:
                raise EmbeddingError(f"malformed embedding response: {e}") from e
            data.sort(key=lambda d: d.get("index", 0))
            out.extend(self._postprocess(d["embedding"]) for d in data)
        if len(out) != len(texts):
            raise EmbeddingError(f"embedding count mismatch: {len(out)} != {len(texts)}")
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text], kind="query"))[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed(list(texts), kind="document")
