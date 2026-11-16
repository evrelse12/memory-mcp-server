# tests/test_phase_5.py
"""Phase 5: pluggable embedding backend (http default, onnx experimental)."""
import pytest

from memory_mcp.config.settings import settings
from memory_mcp.services import embedding as emb_mod


def test_default_backend_is_http():
    assert settings.embedding_backend == "http"


@pytest.mark.asyncio
async def test_onnx_backend_delegates(monkeypatch):
    monkeypatch.setattr(settings, "embedding_backend", "onnx")

    class FakeOnnx:
        def embed(self, texts):
            return [[1.0] * settings.embedding_dim for _ in texts]

    monkeypatch.setattr(emb_mod, "get_onnx_embedder", lambda: FakeOnnx())
    svc = emb_mod.EmbeddingService()
    vec = await svc.embed_query("hello")
    assert len(vec) == settings.embedding_dim
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-6  # normalized
    await svc.close()
