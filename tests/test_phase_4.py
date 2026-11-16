# tests/test_phase_4.py
"""Phase 4: mathematical regulator (NLI gate, TTL archival, confidence calibration)."""
import time

import pytest

from memory_mcp.config.settings import settings
from memory_mcp.mcp_server import server


async def _hypothesis(ctx, nli=None, age_days=0):
    ep = await ctx.hot_path.record_message("s1", "I drink oat milk lattes every morning")
    r = await ctx.tools.remember_fact("User drinks oat milk latte", source_episode_ids=[ep])
    nid = r["node_id"]
    now = int(time.time())
    await ctx.storage.execute(
        "UPDATE nodes SET status='hypothesis', nli_score=?, created_at=? WHERE id=?",
        (nli, now - age_days * 86400, nid), wait=True)
    return nid


async def _row(ctx, nid):
    async with ctx.storage.db.get_read_connection().execute(
            "SELECT status, confidence, valid_until FROM nodes WHERE id=?", (nid,)) as cur:
        return await cur.fetchone()


@pytest.mark.asyncio
async def test_gate_rejects_strong_contradiction(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        nid = await _hypothesis(ctx, nli=-0.9)
        rep = await ctx.maintenance._regulate(int(time.time()))
        assert rep["gated_reject"] == 1
        assert (await _row(ctx, nid))["status"] == "rejected"


@pytest.mark.asyncio
async def test_auto_promote_off_by_default(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        nid = await _hypothesis(ctx, nli=0.9)
        await ctx.maintenance._regulate(int(time.time()))
        assert (await _row(ctx, nid))["status"] == "hypothesis"


@pytest.mark.asyncio
async def test_auto_promote_when_enabled(fresh_db, monkeypatch):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        monkeypatch.setattr(settings, "nli_auto_promote", True)
        nid = await _hypothesis(ctx, nli=0.9)
        rep = await ctx.maintenance._regulate(int(time.time()))
        assert rep["gated_promote"] == 1
        row = await _row(ctx, nid)
        assert row["status"] == "canonical" and row["confidence"] > 0.6


@pytest.mark.asyncio
async def test_stale_hypothesis_archived(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        nid = await _hypothesis(ctx, nli=None, age_days=30)
        rep = await ctx.maintenance._regulate(int(time.time()))
        assert rep["archived"] == 1
        assert (await _row(ctx, nid))["valid_until"] is not None

@pytest.mark.asyncio
async def test_confidence_calibration_on_approve(fresh_db):
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        ep = await ctx.hot_path.record_message("s1", "I drink oat milk lattes")
        r = await ctx.tools.remember_fact("User drinks oat milk latte",
                                          source_episode_ids=[ep])
        nid = r["node_id"]
        await ctx.storage.execute(
            "UPDATE nodes SET status='hypothesis', nli_score=0.8 WHERE id=?",
            (nid,), wait=True)

        class Approve:
            async def chat_json(self, messages, **kw):
                return {"approve": True, "evidence_ids": [ep]}

            async def chat(self, messages, **kw):
                return ""

            async def close(self):
                pass

        ctx.consolidator.llm = Approve()
        await ctx.maintenance._run_critic(int(time.time()))
        row = await _row(ctx, nid)
        assert row["status"] == "canonical"
        # Calibration floor is 0.92; search reinforcement may raise it further.
        assert row["confidence"] >= 0.92 - 1e-6
        assert row["confidence"] <= 1.0


def test_calibrated_confidence_formula():
    from memory_mcp.agents.maintenance import MaintenanceService
    f = MaintenanceService._calibrated_confidence
    assert f(0.8) == 0.92
    assert f(None) == 0.8
    assert f(-0.5) == 0.6
    assert f(1.0) == 1.0
