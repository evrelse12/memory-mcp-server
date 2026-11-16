# tests/test_phase_3.py
"""Phase 3: adversarial LLM critic with grounding and approval_rate."""
import time

import pytest

from memory_mcp.mcp_server import server


class FakeCritic:
    def __init__(self, verdict):
        self._verdict = verdict

    async def chat_json(self, messages, **kw):
        return self._verdict

    async def chat(self, messages, **kw):
        return ""

    async def close(self):
        pass


async def _make_hypothesis(ctx, content="User drinks oat milk latte"):
    ep = await ctx.hot_path.record_message("s1", "I drink oat milk lattes every morning")
    r = await ctx.tools.remember_fact(content, source_episode_ids=[ep])
    await ctx.storage.execute(
        "UPDATE nodes SET status='hypothesis' WHERE id=?", (r["node_id"],), wait=True)
    return ep, r["node_id"]


async def _status(ctx, nid):
    async with ctx.storage.db.get_read_connection().execute(
            "SELECT status, critic_model FROM nodes WHERE id=?", (nid,)) as cur:
        return await cur.fetchone()


@pytest.mark.asyncio
async def test_critic_approves_with_evidence(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        ep, nid = await _make_hypothesis(ctx)
        ctx.consolidator.llm = FakeCritic(
            {"approve": True, "reason": "supported", "evidence_ids": [ep]})
        rep = await ctx.maintenance._run_critic(int(time.time()))
        assert rep["approved"] == 1 and rep["approval_rate"] == 1.0
        row = await _status(ctx, nid)
        assert row["status"] == "canonical" and row["critic_model"]


@pytest.mark.asyncio
async def test_critic_rejects(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        ep, nid = await _make_hypothesis(ctx)
        ctx.consolidator.llm = FakeCritic({"approve": False, "reason": "unsupported"})
        rep = await ctx.maintenance._run_critic(int(time.time()))
        assert rep["rejected"] == 1
        row = await _status(ctx, nid)
        assert row["status"] == "rejected"
        # rejected facts are hidden from reads
        hits = await ctx.tools.search_memory("oat milk")
        assert all(h["id"] != nid for h in hits)


@pytest.mark.asyncio
async def test_critic_approve_without_evidence_is_inconclusive(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        ep, nid = await _make_hypothesis(ctx)
        ctx.consolidator.llm = FakeCritic({"approve": True, "evidence_ids": []})
        rep = await ctx.maintenance._run_critic(int(time.time()))
        assert rep["pending"] == 1 and rep["approved"] == 0
        row = await _status(ctx, nid)
        assert row["status"] == "hypothesis"  # left for next run


@pytest.mark.asyncio
async def test_critic_approval_rate(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        ep1, nid1 = await _make_hypothesis(ctx, "User drinks oat milk latte")
        ep2, nid2 = await _make_hypothesis(ctx, "User prefers window seats")

        class TwoVerdicts:
            def __init__(self):
                self.calls = 0

            async def chat_json(self, messages, **kw):
                self.calls += 1
                if self.calls == 1:
                    return {"approve": True, "evidence_ids": [ep1]}
                return {"approve": False}

            async def chat(self, messages, **kw):
                return ""

            async def close(self):
                pass

        ctx.consolidator.llm = TwoVerdicts()
        rep = await ctx.maintenance._run_critic(int(time.time()))
        assert rep["approved"] == 1 and rep["rejected"] == 1
        assert rep["approval_rate"] == 0.5
