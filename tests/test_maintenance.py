# tests/test_maintenance.py
"""Maintenance unit tests: fake LLM + fake embeddings, throwaway DB per test."""

import pytest

from mcp_server import server
from services import embedding as emb_mod


class FakeLLM:
    """Deterministic LLM: extraction JSON, 'no' for causal/conflict questions."""

    async def chat(self, messages, **kw):
        return "no"

    async def chat_json(self, messages, **kw):
        return {"facts": [{
            "content": "The user lives in Moscow",
            "type": "semantic",
            "confidence": 0.9,
            "sources": [0],
            "entities": [{"name": "Moscow", "type": "place"}],
        }]}

    async def close(self):
        pass

async def _fake_call_batch(self, texts, *, prefix=""):
    """Fake batch embedding: return deterministic vectors for all texts."""
    return [[0.1] * 256 for _ in texts]

async def _fake_call(self, text, *, prefix=""):
    """Fake single embedding."""
    return [0.1] * 256

@pytest.fixture
def patched_services(monkeypatch):
    """Swap LLM client and embedding network calls for offline determinism."""
    import agents.consolidator as cons_mod
    monkeypatch.setattr(cons_mod, "LLMClient", lambda: FakeLLM())
    # Patch BOTH the batch and single methods
    monkeypatch.setattr(emb_mod.EmbeddingService, "_call_batch", _fake_call_batch)
    monkeypatch.setattr(emb_mod.EmbeddingService, "_call", _fake_call)

@pytest.mark.asyncio
async def test_consolidation_extracts_and_marks(fresh_db, patched_services):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await ctx.hot_path.record_message("sX", "I live in Moscow now")
        eps = [n for n in await ctx.storage.fetch_nodes_by_session("sX", None)
               if n["type"] == "episodic"]
        ok = await ctx.consolidator.consolidate_session("sX", episodes=eps)
        assert ok is True
        marked = await ctx.storage.session_consolidated_at("sX")
        assert marked is not None

        facts = await ctx.tools.search_memory("Moscow", limit=5)
        assert any("Moscow" in f["content"] for f in facts)


@pytest.mark.asyncio
async def test_consolidation_retries_on_embedding_failure(fresh_db, monkeypatch):
    """EmbeddingError must NOT mark the session consolidated (retry next run)."""
    import agents.consolidator as cons_mod
    monkeypatch.setattr(cons_mod, "LLMClient", lambda: FakeLLM())
    # leave the real (unreachable) embedding endpoint -> EmbeddingError on demand

    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await ctx.hot_path.record_message("sY", "some episode text")
        eps = [n for n in await ctx.storage.fetch_nodes_by_session("sY", None)
               if n["type"] == "episodic"]
        ok = await ctx.consolidator.consolidate_session("sY", episodes=eps)
        assert ok is False
        assert await ctx.storage.session_consolidated_at("sY") is None


@pytest.mark.asyncio
async def test_pending_sessions_only_new_episodes(fresh_db, patched_services):
    """_consolidate_pending must skip sessions without new episodes."""
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await ctx.hot_path.record_message("sZ", "old episode")
        await ctx.storage.execute(
            "UPDATE nodes SET event_time = event_time - 7200", (), wait=True)

        report = await ctx.maintenance._consolidate_pending(
            int(__import__("time").time()))
        assert report["sessions"] == 1

        report2 = await ctx.maintenance._consolidate_pending(
            int(__import__("time").time()))
        assert report2["sessions"] == 0


@pytest.mark.asyncio
async def test_purge_respects_retention(fresh_db):
    """Soft-deleted nodes survive within the retention grace period."""
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r = await ctx.tools.remember_fact("temp fact", node_type="semantic")
        await ctx.tools.forget_fact(r["node_id"])
        purged = await ctx.decay.purge_expired(int(__import__("time").time()))
        assert purged == 0
        node = await ctx.storage.get_node(r["node_id"], include_deleted=True)
        assert node is not None

@pytest.mark.asyncio
async def test_remember_duplicate_and_pin(fresh_db, patched_services):
    """Same content twice -> duplicate; pin on a duplicate pins the original."""
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r1 = await ctx.tools.remember_fact("The user works at Google")
        assert r1["status"] == "created"
        first_id = r1["node_id"]

        r2 = await ctx.tools.remember_fact("The user works at Google")
        assert r2["status"] == "duplicate"
        assert r2["existing_id"] == first_id
        assert r2["pinned"] is False

        r3 = await ctx.tools.remember_fact("The user works at Google", pin=True)
        assert r3["status"] == "duplicate"
        assert r3["pinned"] is True
        node = await ctx.storage.get_node(first_id)
        assert node["decay_rate"] == 0.0
        assert node["confidence"] == 1.0
