# tests/test_phase_2.py
"""Phase 2: NLI shadow scoring (logs only, no decisions)."""
import time

import pytest

from memory_mcp.mcp_server import server


class FakeNLI:
    """Deterministic NLI for testing: always returns entailment with high confidence."""

    async def classify(self, premise: str, hypothesis: str):
        if "contradict" in premise.lower():
            return "contradiction", 0.9
        return "entailment", 0.8

    async def classify_batch(self, pairs):
        return [await self.classify(p, h) for p, h in pairs]

    def unload(self):
        pass

@pytest.mark.asyncio
async def test_nli_shadow_scores_hypotheses(fresh_db, monkeypatch):
    """NLI shadow mode scores hypotheses against their source episodes."""
    async with server.lifespan(server.mcp):
        ctx = server.ctx

        # Episode first; pass its id so the fact gets a derived_from source
        ep = await ctx.hot_path.record_message("s1", "I drink oat milk lattes every morning")
        r = await ctx.tools.remember_fact("User drinks oat milk latte",
                                          source_episode_ids=[ep])
        await ctx.storage.execute(
            "UPDATE nodes SET status='hypothesis', generator_model='test' WHERE id=?",
            (r["node_id"],), wait=True)

        from memory_mcp.services import nli as nli_mod
        monkeypatch.setattr(nli_mod, "NLIClassifier", lambda path: FakeNLI())

        await ctx.maintenance._run_nli_shadow(int(time.time()))

        async with ctx.storage.db.get_read_connection().execute(
                "SELECT nli_score FROM nodes WHERE id=?", (r["node_id"],)
        ) as cur:
            row = await cur.fetchone()
        assert row["nli_score"] is not None
        assert row["nli_score"] > 0.0  # entailment > contradiction

@pytest.mark.asyncio
async def test_nli_shadow_skips_canonical(fresh_db):
    """NLI shadow mode skips canonical facts (already verified)."""
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        r = await ctx.tools.remember_fact("User lives in Moscow")
        # status='canonical' by default

        await ctx.maintenance._run_nli_shadow(int(time.time()))

        async with ctx.storage.db.get_read_connection().execute(
                "SELECT nli_score FROM nodes WHERE id=?", (r["node_id"],)
        ) as cur:
            row = await cur.fetchone()
        assert row["nli_score"] is None  # not scored


@pytest.mark.asyncio
async def test_nli_shadow_handles_missing_episodes(fresh_db, monkeypatch):
    """NLI shadow mode gracefully handles hypotheses without episodes."""
    async with server.lifespan(server.mcp):
        ctx = server.ctx

        # Insert hypothesis with no session_id (orphan)
        await ctx.storage.execute(
            "INSERT INTO nodes (id, type, content, event_time, created_at, "
            "status, generator_model, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("orphan1", "semantic", "Test", 1700000000, 1700000000,
             "hypothesis", "test", "{}"),
            wait=True)

        from memory_mcp.services import nli as nli_mod
        monkeypatch.setattr(nli_mod, "NLIClassifier", lambda path: FakeNLI())

        # Should not crash
        await ctx.maintenance._run_nli_shadow(int(time.time()))
