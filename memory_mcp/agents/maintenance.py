# agents/maintenance.py
"""Nightly maintenance: consolidation, dedup, decay, enrichment, housekeeping."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from memory_mcp.agents.llm_client import LLMError
from memory_mcp.config.settings import settings
from memory_mcp.core.storage import Storage
from memory_mcp.core.tasks import spawn
from memory_mcp.services.decay import DecayService
from memory_mcp.services.embedding import EmbeddingError, EmbeddingService
from memory_mcp.services.entity_index import EntityIndex

logger = logging.getLogger(__name__)

OVERDUE_S = 20 * 3600

CRITIC_PROMPT = """You are an adversarial memory verifier. Presume the hypothesis is WRONG until proven.
Find contradictions, logical gaps and unsupported generalizations.
Hypothesis: {hypothesis}
NLI score (entailment minus contradiction, -1..1): {nli}
Source episodes (the ONLY admissible evidence):
{sources}
Related existing facts (context, not evidence):
{neighbors}
Return ONLY valid JSON: {{"approve": bool, "reason": string, "evidence_ids": [ids]}}
Approve ONLY if the hypothesis is directly supported by the source episodes and you cite
their ids in evidence_ids. If unsupported, contradicted or an over-generalization, set
approve=false. Never approve without evidence_ids drawn from the source episodes."""

class MaintenanceService:
    def __init__(
        self,
        storage: Storage,
        consolidator,
        decay: DecayService,
        embedding: EmbeddingService,
        entity_index: EntityIndex,
        search,
    ):
        self._s = storage
        self._c = consolidator
        self._decay = decay
        self._emb = embedding
        self._ei = entity_index
        self._search = search
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
                ("dedup", self._dedup_step),
                ("decay", self._decay_step),
                ("enrich", self._enrich_entities),
                ("housekeep", self._housekeep),
                ("nli_shadow", self._run_nli_shadow),
                ("regulate", self._regulate),
                ("critic", self._run_critic),
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

    async def run_consolidation_tick(self) -> None:
        """Consolidate idlsteps = (
                ("consolidate"e sessions periodically; full maintenance stays nightly."""
        if self._lock.locked():
            return
        async with self._lock:
            try:
                report = await self._consolidate_pending(int(time.time()))
                if report["sessions"]:
                    logger.info("consolidation tick: %s", report)
            except Exception:
                logger.exception("consolidation tick failed")

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
            eps = [
                n
                for n in await self._s.fetch_nodes_by_session(sid, since)
                if n["type"] == "episodic"
            ]
            if not eps:
                continue
            if await self._c.consolidate_session(sid, episodes=eps):
                consolidated += 1
        return {"sessions": len(sids), "consolidated": consolidated}

    async def _run_nli_shadow(self, now: int) -> dict:
        """Score unverified hypotheses against their source episodes (shadow mode).

        Grounding = the episodes the hypothesis was derived from (derived_from
        edges), not "same session". Logs scores for threshold calibration;
        does not promote/reject anything yet (Phase 4).
        """
        from memory_mcp.services.nli import NLIClassifier, NLIError

        async with self._s.db.get_read_connection().execute(
                "SELECT id, content, generator_model FROM nodes "
                "WHERE status='hypothesis' AND nli_score IS NULL"
        ) as cur:
            hyps = await cur.fetchall()
        if not hyps:
            logger.info("NLI shadow: no new hypotheses to score")
            return {"scored": 0, "total": 0}

        nli = NLIClassifier(settings.nli_models_path)
        scored = 0
        try:
            for h in hyps:
                try:
                    sources, _, _ = await self._s.provenance(h["id"])
                    eps = sources[: settings.nli_top_k]
                    if not eps:
                        logger.debug("NLI shadow: no sources for %s", h["id"])
                        continue

                    max_entailment = 0.0
                    max_contradiction = 0.0
                    for ep in eps:
                        label, conf = await nli.classify(ep["content"], h["content"])
                        if label == "entailment":
                            max_entailment = max(max_entailment, conf)
                        elif label == "contradiction":
                            max_contradiction = max(max_contradiction, conf)

                    final_score = max_entailment - max_contradiction
                    await self._s.execute(
                        "UPDATE nodes SET nli_score=? WHERE id=?",
                        (final_score, h["id"]), wait=True)
                    scored += 1
                except NLIError as e:
                    logger.warning("NLI failed for hypothesis %s: %s", h["id"], e)
                    break
            logger.info("NLI shadow: scored %d/%d hypotheses", scored, len(hyps))
        finally:
            nli.unload()
        return {"scored": scored, "total": len(hyps)}
    @staticmethod
    def _validate_verdict(v, source_ids: set) -> str | None:
        """approve requires cited evidence from the grounding set; else reject/inconclusive."""
        if not isinstance(v, dict) or not isinstance(v.get("approve"), bool):
            return None
        if not v["approve"]:
            return "reject"
        ev = v.get("evidence_ids") or []
        if not ev or not set(ev) <= source_ids:
            return None
        return "approve"

    async def _run_critic(self, now: int) -> dict:
        """Adversarial LLM verification of hypotheses; promote/reject with provenance."""
        async with self._s.db.get_read_connection().execute(
                "SELECT id, content, nli_score FROM nodes WHERE status='hypothesis'"
        ) as cur:
            hyps = await cur.fetchall()
        if not hyps:
            return {"approved": 0, "rejected": 0, "pending": 0, "approval_rate": None}

        approved = rejected = pending = 0
        for h in hyps:
            sources, _, _ = await self._s.provenance(h["id"])
            if not sources:
                pending += 1
                continue
            eps = sources[: settings.nli_top_k]
            neighbors = [n for n in await self._search.search(h["content"], limit=3)
                         if n["id"] != h["id"]][:3]
            prompt = CRITIC_PROMPT.format(
                hypothesis=h["content"],
                nli=h["nli_score"] if h["nli_score"] is not None else "n/a",
                sources="\n".join(f"- [{e['id']}] {e['content'][:300]}" for e in eps),
                neighbors="\n".join(f"- {n['content'][:200]}" for n in neighbors) or "(none)",
            )
            try:
                verdict = await self._c.llm.chat_json(
                    [{"role": "user", "content": prompt}],
                    max_tokens=256, temperature=0.0)
            except LLMError:
                pending += 1
                continue
            decision = self._validate_verdict(verdict, {e["id"] for e in eps})
            if decision == "approve":
                await self._s.execute(
                    "UPDATE nodes SET status='canonical', critic_model=?, confidence=? "
                    "WHERE id=?",
                    (settings.llm_model, self._calibrated_confidence(h["nli_score"]),
                     h["id"]), wait=True)
                approved += 1
            elif decision == "reject":
                await self._s.execute(
                    "UPDATE nodes SET status='rejected', critic_model=? WHERE id=?",
                    (settings.llm_model, h["id"]), wait=True)
                rejected += 1
            else:
                pending += 1

        judged = approved + rejected
        rate = round(approved / judged, 4) if judged else None
        logger.info("critic: approved=%d rejected=%d pending=%d approval_rate=%s",
                    approved, rejected, pending, rate)
        return {"approved": approved, "rejected": rejected,
                "pending": pending, "approval_rate": rate}

    @staticmethod
    def _calibrated_confidence(nli_score: float | None) -> float:
        """Map NLI entailment strength to a stored confidence in [0.6, 1.0]."""
        base = nli_score if nli_score is not None else 0.5
        return round(min(1.0, max(0.6, 0.6 + 0.4 * max(0.0, base))), 4)

    async def _regulate(self, now: int) -> dict:
        """Pure-code regulator: NLI gate + stale-hypothesis archival (no LLM)."""
        gated_reject = gated_promote = archived = 0
        async with self._s.db.get_read_connection().execute(
                "SELECT id, nli_score, created_at FROM nodes WHERE status='hypothesis'"
        ) as cur:
            hyps = await cur.fetchall()
        for h in hyps:
            score = h["nli_score"]
            if score is not None and score <= settings.nli_reject_threshold:
                await self._s.execute(
                    "UPDATE nodes SET status='rejected', critic_model='nli-gate' "
                    "WHERE id=?", (h["id"],), wait=True)
                gated_reject += 1
                continue
            if (settings.nli_auto_promote and score is not None
                    and score >= settings.nli_promote_threshold):
                await self._s.execute(
                    "UPDATE nodes SET status='canonical', critic_model='nli-gate', "
                    "confidence=? WHERE id=?",
                    (self._calibrated_confidence(score), h["id"]), wait=True)
                gated_promote += 1
                continue
            if now - h["created_at"] > settings.hypothesis_ttl_days * 86400:
                await self._s.soft_delete_nodes([h["id"]])
                archived += 1
        logger.info("regulator: gated_reject=%d gated_promote=%d archived=%d",
                    gated_reject, gated_promote, archived)
        return {"gated_reject": gated_reject, "gated_promote": gated_promote,
                "archived": archived}

    async def _dedup_step(self, now: int) -> dict:
        return await self._search.sweep_duplicates()

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
            prompt = f"Summarize in 2-3 sentences what is known about {e['canonical_name']}:\n{ctx}"
            try:
                summary = (
                    await self._c.llm.chat(
                        [{"role": "user", "content": prompt}], max_tokens=200, temperature=0.3
                    )
                ).strip()[:600]
            except LLMError:
                continue
            if not summary:
                continue
            attrs["summary_mentions"] = e["mention_count"]
            await self._s.execute(
                "UPDATE entities SET summary=?, attributes=?, last_updated=? WHERE id=?",
                (summary, json.dumps(attrs, ensure_ascii=False), now, e["id"]),
            )
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
