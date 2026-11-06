# agents/maintenance.py
"""Nightly maintenance: consolidation, decay, enrichment, causal, housekeeping."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from agents.llm_client import LLMError
from config.settings import settings
from core.storage import Storage
from core.tasks import spawn
from services.decay import DecayService
from services.embedding import EmbeddingError, EmbeddingService
from services.entity_index import EntityIndex

logger = logging.getLogger(__name__)

OVERDUE_S = 20 * 3600


class MaintenanceService:
    def __init__(self, storage: Storage, consolidator, decay: DecayService,
                 embedding: EmbeddingService, entity_index: EntityIndex):
        self._s = storage
        self._c = consolidator
        self._decay = decay
        self._emb = embedding
        self._ei = entity_index
        self._lock = asyncio.Lock()

    async def run_if_overdue(self) -> None:
        last = await self._s.meta_get("last_maintenance_ts")
        now = int(time.time())
        if last is None or now - int(last) > OVERDUE_S:
            spawn(self.run_nightly(), name="catch-up-maintenance")

    async def run_nightly(self) -> dict:
        if self._lock.locked():
            logger.info("maintenance already running, skip")
            return {}
        async with self._lock:
            now = int(time.time())
            report = {}
            steps = (
                ("consolidate", self._consolidate_pending),
                ("decay", self._decay_step),
                ("enrich", self._enrich_entities),
                ("housekeep", self._housekeep),
            )
            for name, step in steps:
                try:
                    report[name] = await step(now)
                except Exception:
                    logger.exception("maintenance step %s failed", name)
                    report[name] = "error"
            await self._s.meta_set("last_maintenance_ts", str(int(time.time())))
            logger.info("nightly maintenance report: %s", report)
            return report

    async def _consolidate_pending(self, now: int) -> dict:
        """Only sessions with NEW episodes since last consolidation (no N writes)."""
        cutoff = now - settings.session_timeout_minutes * 60
        sql = """
            SELECT n.session_id FROM nodes n
            LEFT JOIN sessions s ON s.session_id = n.session_id
            WHERE n.type='episodic' AND n.session_id IS NOT NULL
              AND n.valid_until IS NULL
            GROUP BY n.session_id
            HAVING MAX(n.event_time) < ?
               AND MAX(n.event_time) > COALESCE(s.consolidated_at, 0)"""
        async with self._s.db.get_read_connection().execute(sql, (cutoff,)) as cur:
            sids = [r["session_id"] for r in await cur.fetchall()]
        consolidated = 0
        for sid in sids:
            since = await self._s.session_consolidated_at(sid)
            eps = [n for n in await self._s.fetch_nodes_by_session(sid, since)
                   if n["type"] == "episodic"]
            if not eps:
                continue
            if await self._c.consolidate_session(sid, episodes=eps):
                consolidated += 1
        return {"sessions": len(sids), "consolidated": consolidated}

    async def _decay_step(self, now: int) -> dict:
        return await self._decay.run_night_job(now)

    async def _enrich_entities(self, now: int) -> dict:
        async with self._s.db.get_read_connection().execute("""
                SELECT * FROM entities WHERE mention_count >= 3
                ORDER BY mention_count DESC LIMIT 20""") as cur:
            rows = await cur.fetchall()
        updated = 0
        for e in rows:
            attrs = json.loads(e["attributes"] or "{}")
            summary_at = attrs.get("summary_mentions", 0)
            if e["summary"] and e["mention_count"] < 2 * max(summary_at, 1):
                continue
            nodes = await self._s.fetch_nodes_for_entity(e["id"])
            if not nodes:
                continue
            ctx = "\n".join(f"- {n['content'][:200]}" for n in nodes[:10])
            prompt = (f"Summarize in 2-3 sentences what is known about "
                      f"{e['canonical_name']}:\n{ctx}")
            try:
                summary = (await self._c.llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=200, temperature=0.3)).strip()[:600]
            except LLMError:
                continue
            if not summary:
                continue
            attrs["summary_mentions"] = e["mention_count"]
            await self._s.execute(
                "UPDATE entities SET summary=?, attributes=?, last_updated=? WHERE id=?",
                (summary, json.dumps(attrs, ensure_ascii=False), now, e["id"]))
            try:
                await self._ei.embed_entity(e["id"], f"{e['canonical_name']}. {summary}")
            except EmbeddingError:
                logger.warning("entity re-embed deferred: %s", e["id"])
            updated += 1
        return {"updated": updated}

    async def _housekeep(self, now: int) -> dict:
        """No purge here (decay night job owns it). Backfill + optimize only."""
        backfilled = 0
        ids = await self._s.nodes_without_vectors(100)
        if ids:
            nodes = await self._s.fetch_nodes(ids)
            try:
                embs = await self._emb.embed_documents([n["content"] for n in nodes])
                for n, emb in zip(nodes, embs, strict=False):
                    await self._s.insert_embedding_and_clear_flag(n["id"], emb)
                    backfilled += 1
            except EmbeddingError:
                logger.warning("embedding backfill skipped: service unavailable")
        await self._s.raw("INSERT INTO nodes_fts(nodes_fts) VALUES('optimize')")
        await self._s.raw("PRAGMA wal_checkpoint(TRUNCATE)")
        await self._s.raw("PRAGMA optimize")
        return {"backfilled": backfilled}
