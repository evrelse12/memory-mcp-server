# services/decay.py
"""Ebbinghaus decay: effective confidence on the fly; night job expires/purges.

effective = confidence * exp(-decay_rate * idle_days ** 0.8)
Idle is measured from LAST ACCESS (falls back to created_at), so a recent
lookup resets the forgetting curve, as the formula intends.
"""
from __future__ import annotations

import logging
import math
import time

from config.settings import settings
from core.storage import Storage

logger = logging.getLogger(__name__)

DAY = 86400.0


class DecayService:
    def __init__(self, storage: Storage):
        self._s = storage

    @staticmethod
    def effective_confidence(confidence: float, decay_rate: float,
                              idle_s: float) -> float:
        days = max(0.0, idle_s / DAY)
        return confidence * math.exp(-decay_rate * (days ** settings.decay_exponent))

    async def expire_weak(self, now_ts: int | None = None) -> int:
        now = now_ts or int(time.time())
        thr = settings.decay_threshold
        if self._s.db.has_math_functions:
            return await self._s.execute("""
                UPDATE nodes SET valid_until = ?
                WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL
                  AND confidence * EXP(-decay_rate *
                      POWER((? - COALESCE(last_accessed, created_at)) / ?, ?)) < ?""",
                (now, now, DAY, settings.decay_exponent, thr)) or 0
        async with self._s.db.get_read_connection().execute("""
                SELECT id, confidence, decay_rate, created_at, last_accessed
                FROM nodes
                WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL""") as cur:
            rows = await cur.fetchall()
        ids = [r["id"] for r in rows
               if self.effective_confidence(
                   r["confidence"], r["decay_rate"],
                   now - (r["last_accessed"] or r["created_at"])) < thr]
        if not ids:
            return 0
        ph = ",".join("?" for _ in ids)
        return await self._s.execute(
            f"UPDATE nodes SET valid_until=? WHERE id IN ({ph})", (now, *ids)) or 0

    async def purge_expired(self, now_ts: int | None = None) -> int:
        """Physical delete only after a retention grace period (audit trail)."""
        now = now_ts or int(time.time())
        cutoff = now - int(settings.soft_delete_retention_days * DAY)
        n1 = await self._s.execute(
            "DELETE FROM nodes WHERE valid_until IS NOT NULL AND valid_until < ?",
            (cutoff,)) or 0
        n2 = 0
        if settings.episode_retention_days > 0:
            n2 = await self._s.execute(
                "DELETE FROM nodes WHERE type='episodic' AND event_time < ?",
                (now - int(settings.episode_retention_days * DAY),)) or 0
        return n1 + n2

    async def run_night_job(self, now_ts: int | None = None) -> dict:
        now = now_ts or int(time.time())
        expired = await self.expire_weak(now)
        purged = await self.purge_expired(now)
        logger.info("decay night job: expired=%d purged=%d", expired, purged)
        return {"expired": expired, "purged": purged}
