# tests/test_phase_b.py
"""Phase B: MCP resources — static endpoints and URI templates."""

import pytest
from fastmcp import Client

from memory_mcp.mcp_server import server


def _text(res) -> str:
    return "".join(getattr(b, "text", "") or "" for b in res)


@pytest.mark.asyncio
async def test_resources_and_templates_listed(fresh_db):
    async with server.lifespan(server.mcp):
        async with Client(server.mcp) as c:
            uris = " ".join(str(r.uri) for r in await c.list_resources())
            for expected in (
                "memory://user/profile",
                "memory://user/constraints",
                "memory://user/preferences",
                "memory://episodes/recent",
            ):
                assert expected in uris
            templates = " ".join(
                getattr(t, "uri_template", None) or t.uriTemplate
                for t in await c.list_resource_templates()
            )
            assert "memory://entities/{entity_id}" in templates
            assert "memory://session/{session_id}/summary" in templates


@pytest.mark.asyncio
async def test_profile_and_directive_resources(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        p = await ctx.tools.remember_fact("User works at Yandex")
        await ctx.tools.confirm_fact(p["node_id"])
        await ctx.tools.remember_fact("Never use Java in this project", constraint=True)
        o = await ctx.tools.remember_fact("User dislikes verbose answers", node_type="opinion")
        await ctx.tools.confirm_fact(o["node_id"])
        async with Client(server.mcp) as c:
            assert "Yandex" in _text(await c.read_resource("memory://user/profile"))
            assert "Never use Java" in _text(await c.read_resource("memory://user/constraints"))
            assert "verbose answers" in _text(await c.read_resource("memory://user/preferences"))


@pytest.mark.asyncio
async def test_recent_episodes_resource(fresh_db):
    async with server.lifespan(server.mcp):
        await server.ctx.hot_path.record_message("sB", "first line")
        await server.ctx.hot_path.record_message("sB", "second line")
        async with Client(server.mcp) as c:
            txt = _text(await c.read_resource("memory://episodes/recent"))
            assert "first line" in txt and "second line" in txt


@pytest.mark.asyncio
async def test_entity_resource_template(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        eid = await ctx.entity_index.ensure("Yandex", "organization")
        await ctx.tools.remember_fact("User works at Yandex")
        async with Client(server.mcp) as c:
            txt = _text(await c.read_resource(f"memory://entities/{eid}"))
            assert "Yandex" in txt


@pytest.mark.asyncio
async def test_session_summary_template(fresh_db):
    async with server.lifespan(server.mcp):
        await server.ctx.hot_path.record_message("sC", "talking about Moscow")
        async with Client(server.mcp) as c:
            txt = _text(await c.read_resource("memory://session/sC/summary"))
            assert "sC" in txt and "talking about Moscow" in txt
