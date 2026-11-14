# tests/test_scaling.py
import time

import pytest

from memory_mcp.mcp_server import server


@pytest.mark.asyncio
async def test_entity_index_combined_regex(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        eid = await ctx.entity_index.ensure("New York", "place")
        await ctx.storage.merge_alias(eid, "NYC")
        ctx.entity_index._by_norm.setdefault("nyc", (eid, "New York"))
        ctx.entity_index._add_pattern("NYC")
        hits = ctx.entity_index.find_in_text("Flew to NYC via New York, NYC again")
        assert hits == ["New York"]          # alias -> canonical, deduped


@pytest.mark.asyncio
async def test_weak_facts_sql_path(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r = await ctx.tools.remember_fact("An old fading fact", confidence=0.9)
        await ctx.storage.execute(
            "UPDATE nodes SET created_at=?, last_accessed=NULL WHERE id=?",
            (int(time.time()) - 100 * 86400, r["node_id"]))
        weak = await ctx.tools.weak_facts()
        assert any(w["id"] == r["node_id"] for w in weak)
