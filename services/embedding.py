# services/embedding.py
"""Async client for an OpenAI-compatible embedding server (e.g. llama-server)."""
from __future__ import annotations

import logging
from typing import Sequence

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Any failure to produce an embedding vector."""


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

    # -------------------------------------------------------------- single
    async def embed_document(self, text: str) -> list[float]:
        """Embed a stored fact / episode (no prefix)."""
        return await self._call(text, prefix=settings.embedding_document_prefix)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (query prefix so the model treats it as a query)."""
        return await self._call(text, prefix=settings.embedding_query_prefix)

    # -------------------------------------------------------------- batch
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out = []
        for t in texts:
            out.append(await self.embed_document(t))
        return out

    # -------------------------------------------------------------- internals
    async def _call(self, text: str, *, prefix: str = "") -> list[float]:
        payload = {"model": self._model, "input": prefix + text}
        try:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise EmbeddingError(f"embedding request failed: {e}") from e

        try:
            vec = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as e:
            raise EmbeddingError(f"malformed embedding response: {e}") from e

        if not isinstance(vec, list) or not vec:
            raise EmbeddingError("empty embedding vector")

        if settings.embedding_truncate_matryoshka and len(vec) > self._dim:
            vec = vec[: self._dim]
        return [float(x) for x in vec]
