# tests/test_phase_1.py
"""Phase 1: consolidation writes hypotheses; read path hides them."""
import pytest

from memory_mcp.mcp_server import server


class FakeLLM:
    async def chat(self, messages, **kw):
        return "different"

    async def chat_json(self, messages, **kw):
        return {"facts": [{"content": "User drinks oat milk latte",
                           "type": "semantic", "confidence": 0.8,
                           "sources": [0], "entities": []}]}

    async def close(self):
        pass


@pytest.fixture
def const_emb(monkeypatch):
    from memory_mcp.services import embedding as emb_mod

    async def _call(self, text, *, prefix=""):
        return [0.1] * 256

    async def _call_batch(self, texts, *, prefix=""):
        return [[0.1] * 256 for _ in texts]

    monkeypatch.setattr(emb_mod.EmbeddingService, "_call", _call)
    monkeypatch.setattr(emb_mod.EmbeddingService, "_call_batch", _call_batch)


@pytest.mark.asyncio
async def test_consolidation_marks_hypothesis(fresh_db, const_emb):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        ctx.consolidator.llm = FakeLLM()
        await ctx.hot_path.record_message("s1", "I switched to oat milk lattes")
        assert await ctx.consolidator.consolidate_session("s1")
        async with ctx.storage.db.get_read_connection().execute(
                "SELECT status, generator_model FROM nodes WHERE type='semantic'") as cur:
            rows = await cur.fetchall()
        assert rows and rows[0]["status"] == "hypothesis"
        assert rows[0]["generator_model"]


@pytest.mark.asyncio
async def test_search_hides_hypotheses(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r = await ctx.tools.remember_fact("User drinks oat milk latte")
        await ctx.storage.execute(
            "UPDATE nodes SET status='hypothesis' WHERE id=?",
            (r["node_id"],), wait=True)
        hidden = await ctx.tools.search_memory("oat milk")
        assert all(h["id"] != r["node_id"] for h in hidden)
        shown = await ctx.tools.search_memory("oat milk", include_hypotheses=True)
        assert any(h["id"] == r["node_id"] for h in shown)


@pytest.mark.asyncio
async def test_turn_context_hides_hypotheses(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r = await ctx.tools.remember_fact("User drinks oat milk latte")
        await ctx.storage.execute(
            "UPDATE nodes SET status='hypothesis' WHERE id=?",
            (r["node_id"],), wait=True)
        tc = await ctx.tools.get_turn_context("oat milk")
        assert all(f["id"] != r["node_id"] for f in tc["facts"])
