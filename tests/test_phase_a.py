# tests/test_phase_a.py
"""Phase A: turn context, directives, structured conflicts, protocol prompt."""

import pytest
from fastmcp import Client

from memory_mcp.mcp_server import server


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
async def test_constraint_created_and_listed(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r = await ctx.tools.remember_fact("Never use Java in this project", constraint=True)
        assert r["status"] == "created" and r["constraint"] is True
        node = await ctx.storage.get_node(r["node_id"])
        assert node["decay_rate"] == 0.0 and node["confidence"] == 1.0
        d = await ctx.tools.get_active_directives()
        assert any(x["id"] == r["node_id"] and x["kind"] == "constraint" for x in d)


@pytest.mark.asyncio
async def test_pinned_opinion_listed(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r = await ctx.tools.remember_fact("User dislikes verbose answers", node_type="opinion")
        await ctx.tools.confirm_fact(r["node_id"])
        d = await ctx.tools.get_active_directives()
        assert any(x["id"] == r["node_id"] and x["kind"] == "preference" for x in d)


@pytest.mark.asyncio
async def test_constraint_upgrades_duplicate(fresh_db, const_emb):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r1 = await ctx.tools.remember_fact("User prefers dark theme", node_type="opinion")
        r2 = await ctx.tools.remember_fact("User prefers dark theme", constraint=True)
        assert r2["status"] == "duplicate"
        node = await ctx.storage.get_node(r1["node_id"])
        assert node["decay_rate"] == 0.0
        d = await ctx.tools.get_active_directives()
        assert any(x["id"] == r1["node_id"] and x["kind"] == "constraint" for x in d)


@pytest.mark.asyncio
async def test_similar_requires_clarification(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx

        async def fake_dedupe(content, **kw):
            return "conflict", {
                "id": "x",
                "content": "User loves Python",
                "type": "opinion",
                "decay_rate": 0.1,
            }

        ctx.search.dedupe_candidate = fake_dedupe

        r = await ctx.tools.remember_fact("User wants Rust for this task")
        assert r["status"] == "similar"
        assert r["requires_user_clarification"] is True
        assert r["clarification_prompt"]


@pytest.mark.asyncio
async def test_turn_context_block(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await ctx.tools.remember_fact("Never use Java in this project", constraint=True)
        p = await ctx.tools.remember_fact("User works at Yandex")
        await ctx.tools.confirm_fact(p["node_id"])
        await ctx.tools.remember_fact("User lives in Moscow")

        r = await ctx.tools.get_turn_context("Moscow")
        block = r["block"]
        assert block.startswith("<user_memory>") and block.endswith("</user_memory>")
        assert "Directives" in block and "Never use Java" in block
        assert "User profile" in block and "Yandex" in block
        assert "Moscow" in block
        seen = {d["id"] for d in r["directives"]} | {x["id"] for x in r["profile"]}
        assert all(f["id"] not in seen for f in r["facts"])


@pytest.mark.asyncio
async def test_turn_context_empty(fresh_db):
    async with server.lifespan(server.mcp):
        r = await server.ctx.tools.get_turn_context("anything")
        assert "(no stored memory yet)" in r["block"]
        assert r["directives"] == [] and r["facts"] == []


@pytest.mark.asyncio
async def test_memory_protocol_prompt(fresh_db):
    from memory_mcp.prompts import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_RU

    assert "get_turn_context" in SYSTEM_PROMPT_EN
    assert "get_turn_context" in SYSTEM_PROMPT_RU
    async with server.lifespan(server.mcp):
        async with Client(server.mcp) as c:
            prompts = await c.list_prompts()
            assert any(p.name == "memory_protocol" for p in prompts)
