# tests/test_memory.py
import json
import pytest
from fastmcp import Client

from mcp_server import server


def _data(res):
    text = "".join(getattr(c, "text", "") or "" for c in res.content)
    return json.loads(text) if text else None


@pytest.mark.asyncio
async def test_remember_search_correct_forget():
    async with server.lifespan(server.mcp):
        async with Client(server.mcp) as c:
            # 1. Remember
            r = _data(await c.call_tool("remember_fact", {
                "content": "The user works at Yandex", "node_type": "semantic"}))
            assert r["status"] == "created"
            old_id = r["node_id"]

            # 2. Search
            r = _data(await c.call_tool("search_memory", {"query": "Yandex"}))
            assert any("Yandex" in x["content"] for x in r)

            # 3. Correct
            r = _data(await c.call_tool("correct_fact", {
                "old_node_id": old_id, "new_content": "The user works at Google"}))
            assert r["status"] == "corrected"

            r = _data(await c.call_tool("search_memory", {"query": "Google"}))
            assert any("Google" in x["content"] for x in r)
            assert not any("Yandex" in x["content"] for x in r)

            # 4. Forget
            res = await c.call_tool("forget_fact", {"node_id": r[0]["id"]})
            assert not res.is_error

            r = _data(await c.call_tool("search_memory", {"query": "Google"}))
            assert r in ([], None)

            # 5. Confirm (should fail on missing ID)
            res = await c.call_tool("confirm_fact", {"node_id": "missing-id"}, raise_on_error=False)
            assert res.is_error


@pytest.mark.asyncio
async def test_timeline_is_chronological():
    async with server.lifespan(server.mcp):
        await server.ctx.hot_path.record_message("s1", "first message")
        await server.ctx.hot_path.record_message("s1", "second message")
        async with Client(server.mcp) as c:
            r = _data(await c.call_tool("get_timeline", {"limit": 5}))
        contents = [x["content"] for x in r]
        assert contents.index("first message") < contents.index("second message")


@pytest.mark.asyncio
async def test_stats_keys():
    async with server.lifespan(server.mcp):
        async with Client(server.mcp) as c:
            r = _data(await c.call_tool("memory_stats", {}))
    assert {"nodes_by_type", "edges_by_type", "entities_by_type",
            "orphans", "db_size_mb", "vec_available"} <= set(r)


def test_decay_curve_matches_spec():
    from services.decay import DecayService
    eff = DecayService.effective_confidence
    assert 0.45 < eff(0.9, 0.1, 9 * 86400) < 0.55
    assert 0.04 < eff(0.9, 0.1, 67 * 86400) < 0.06
    assert eff(0.9, 0.0, 365 * 86400) == pytest.approx(0.9)
