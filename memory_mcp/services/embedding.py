# services/embedding.py
"""Async client for an OpenAI-compatible embedding server (e.g. llama-server).
Vectors are L2-normalized after truncation so that cosine == 1 - d²/2."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import httpx

from memory_mcp.config.settings import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


class EmbeddingError(Exception):
    """Any failure to produce an embedding vector."""


def _normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]

class OnnxEmbedder:
    """Experimental in-process bge-m3 (ONNX, CLS pooling). Lazy-loaded.

    Not used by default; selected via MEMORY_EMBEDDING_BACKEND=onnx.
    Requires onnxruntime + transformers + huggingface_hub.
    """

    def __init__(self, model_ref: str):
        self._ref = model_ref
        self._sess = None
        self._tok = None

    def _ensure(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        path = hf_hub_download(repo_id=self._ref, filename=settings.onnx_embedding_file)
        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._tok = AutoTokenizer.from_pretrained(self._ref)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=512, return_tensors="np")
        feeds = {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}
        names = {i.name for i in self._sess.get_inputs()}
        if "token_type_ids" in enc and "token_type_ids" in names:
            feeds["token_type_ids"] = enc["token_type_ids"]
        out = self._sess.run(None, feeds)[0]
        import numpy as np
        arr = np.asarray(out)
        if arr.ndim == 3:            # last_hidden_state -> CLS pooling
            arr = arr[:, 0, :]
        return arr.tolist()          # 2D = already pooled sentence embedding


_ONNX_SINGLETON: OnnxEmbedder | None = None

def get_onnx_embedder() -> OnnxEmbedder:
    global _ONNX_SINGLETON
    if _ONNX_SINGLETON is None:
        _ONNX_SINGLETON = OnnxEmbedder(settings.onnx_embedding_model)
    return _ONNX_SINGLETON

class EmbeddingService:
    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ):
        self._url = (api_url or settings.embedding_api_url).rstrip("/") + "/v1/embeddings"
        self._key = api_key or settings.embedding_api_key
        self._model = model or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.embedding_timeout, connect=settings.embedding_connect_timeout
            ),
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
            batch = texts[i : i + BATCH_SIZE]
            out.extend(await self._call_batch(batch, prefix=settings.embedding_document_prefix))
        return out

    async def _call_batch(self, texts: list[str], *, prefix: str) -> list[list[float]]:
        if settings.embedding_backend == "onnx":
            raw = await self._onnx_raw(texts, prefix)
        else:
            raw = await self._http_raw(texts, prefix)
        if len(raw) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} vectors, got {len(raw)}")
        out = []
        for vec in raw:
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError("empty embedding vector")
            if settings.embedding_truncate_matryoshka and len(vec) > self._dim:
                vec = vec[: self._dim]
            if len(vec) != self._dim:
                raise EmbeddingError(f"embedding dim {len(vec)} != expected {self._dim}")
            out.append(_normalize([float(x) for x in vec]))
        return out

    async def _http_raw(self, texts: list[str], prefix: str) -> list[list[float]]:
        payload = {"model": self._model, "input": [prefix + t for t in texts]}
        try:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise EmbeddingError(f"embedding request failed: {e}") from e
        try:
            return [item["embedding"]
                    for item in sorted(data["data"], key=lambda d: d["index"])]
        except (KeyError, IndexError, TypeError) as e:
            raise EmbeddingError(f"malformed embedding response: {e}") from e

    async def _onnx_raw(self, texts: list[str], prefix: str) -> list[list[float]]:
        try:
            return get_onnx_embedder().embed([prefix + t for t in texts])
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"onnx embedding failed: {e}") from e

    async def _call(self, text: str, *, prefix: str = "") -> list[float]:
        return (await self._call_batch([text], prefix=prefix))[0]
