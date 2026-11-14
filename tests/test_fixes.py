# tests/test_fixes.py
"""Regression tests for the bug-fix batch (offline, deterministic)."""
import asyncio
import json

import pytest

from config.settings import settings
from core.tasks import drain
from mcp_server import server
from services.search import build_fts_query


class FakeLLM:
    async def chat(self, messages, **kw):
        return "contradicts"

    async def chat_json(self, messages, **kw):
        return {"facts": [{"content": "The user lives in Moscow",
                           "type": "semantic", "confidence": 0.9, "sources": [0],
                           "entities": [{"name": "Moscow", "type": "place"}]}]}

    async def close(self):
        pass


def _vec_for(text):
    t = text.casefold()
    if "google" in t or "гугл" in t:
        return [1.0] + [0.0] * 255
    if "moscow" in t or "москв" in t:
        return [0.0, 1.0] + [0.0] * 254
    return [0.0, 0.0, 1.0] + [0.0] * 253


@pytest.fixture
def patched(monkeypatch):
    import agents.consolidator as cons_mod
    from services import embedding as emb_mod
    monkeypatch.setattr(cons_mod, "LLMClient", lambda: FakeLLM())

    async def _call(self, text, *, prefix=""):
        return _vec_for(text)

    async def _call_batch(self, texts, *, prefix=""):
        return [_vec_for(t) for t in texts]

    monkeypatch.setattr(emb_mod.EmbeddingService, "_call", _call)
    monkeypatch.setattr(emb_mod.EmbeddingService, "_call_batch", _call_batch)


def test_fts_query_builder():
    q = build_fts_query("Где работает пользователь?")
    assert '"работа"*' in q and "где" not in q
    assert build_fts_query("работу") == '"рабо"*'
    assert build_fts_query("Where does the user work?") == '"user"* OR "work"*'


@pytest.mark.asyncio
async def test_russian_fts_offline(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await drain(timeout=2)   # let startup catch-up finish before the test acts
        await ctx.tools.remember_fact("Пользователь работал в Яндексе")
        r = await ctx.tools.search_memory("работу", limit=5)
        assert any("Яндексе" in x["content"] for x in r)


@pytest.mark.asyncio
async def test_merge_alias_atomic(fresh_db):
    async with server.lifespan(server.mcp):
        s = server.ctx.storage
        eid = await s.upsert_entity(name_norm="google", canonical="Google")
        await asyncio.gather(s.merge_alias(eid, "Гугл"),
                             s.merge_alias(eid, "Alphabet"),
                             s.merge_alias(eid, "Гугл"))
        ent = await s.get_entity(eid)
        aliases = json.loads(ent["aliases"])
        assert len(aliases) == 2
        assert set(aliases) == {"Гугл", "Alphabet"}


@pytest.mark.asyncio
async def test_session_rotation_and_explicit(fresh_db):
    async with server.lifespan(server.mcp):
        hp = server.ctx.hot_path
        s1 = hp.current_session()
        assert hp.current_session() == s1
        hp._auto_last -= settings.session_timeout_minutes * 60 + 1
        assert hp.current_session() != s1
        assert hp.current_session("chat-42") == "chat-42"


@pytest.mark.asyncio
async def test_assistant_episodes_excluded(fresh_db, patched):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await drain(timeout=2)
        await ctx.hot_path.record_message("sA", "I live in Moscow", role="assistant")
        eps = [n for n in await ctx.storage.fetch_nodes_by_session("sA", None)
               if n["type"] == "episodic"]
        assert await ctx.consolidator.consolidate_session("sA", episodes=eps) is True
        assert await ctx.tools.search_memory("Moscow") == []


@pytest.mark.asyncio
async def test_retro_dedup_sweep(fresh_db, patched):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await drain(timeout=2)   # startup catch-up must not race the sweep
        a = await ctx.tools.remember_fact("The user lives in Moscow")
        b = await ctx.tools.remember_fact("The user lives in Moscow", force=True)
        rep = await ctx.search.sweep_duplicates()
        assert rep["merged"] == 1
        assert await ctx.storage.get_node(b["node_id"]) is None
        assert await ctx.storage.get_node(a["node_id"]) is not None


@pytest.mark.asyncio
async def test_entity_vector_alias(fresh_db, patched):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        eid1 = await ctx.entity_index.ensure("Google", "organization")
        await drain(timeout=2)   # background entity embedding must land in vec_entities
        eid2 = await ctx.entity_index.ensure("Гугл")
        assert eid2 == eid1
        ent = await ctx.storage.get_entity(eid1)
        assert "Гугл" in json.loads(ent["aliases"])
        assert ctx.entity_index.resolve("Гугл") == eid1


@pytest.mark.asyncio
async def test_supersede_preserves_type(fresh_db, patched):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await drain(timeout=2)
        old = await ctx.tools.remember_fact("The user dislikes Java", node_type="opinion")
        old_id = old["node_id"]

        async def fake_dedupe(content, **kw):
            return "conflict", {"id": old_id, "content": "The user dislikes Java",
                                "type": "opinion", "decay_rate": 0.1}
        ctx.consolidator._search.dedupe_candidate = fake_dedupe

        await ctx.hot_path.record_message("sT", "Actually I love Java now")
        eps = [n for n in await ctx.storage.fetch_nodes_by_session("sT", None)
               if n["type"] == "episodic"]
        assert await ctx.consolidator.consolidate_session("sT", episodes=eps)
        edges = await ctx.storage.fetch_edges_for_nodes([old_id])
        sup = [e for e in edges if e["relation_type"] == "supersedes"]
        assert sup
        new = await ctx.storage.get_node(sup[0]["source_id"])
        assert new["type"] == "opinion"   # was: semantic (the bug)
