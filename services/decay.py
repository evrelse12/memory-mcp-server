# services/decay.py
"""Ebbinghaus decay: effective confidence on the fly; night job only expires/purges.

effective(t) = confidence * exp(-decay_rate * age_days ** 0.8)
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

from config.settings import settings
from core.storage import Storage

logger = logging.getLogger(__name__)

DAY = 86400.0


class DecayService:
    def __init__(self, storage: Storage):
        self._s = storage

    @staticmethod
    def effective_confidence(confidence: float, decay_rate: float,
                              age_s: float) -> float:
        days = max(0.0, age_s / DAY)
        return confidence * math.exp(-decay_rate * (days ** settings.decay_exponent))

    async def expire_weak(self, now_ts: Optional[int] = None) -> int:
        """Single UPDATE: effective confidence below threshold -> valid_until=now."""
        now = now_ts or int(time.time())
        thr = settings.decay_threshold
        if self._s.db.has_math_functions:
            return await self._s.execute("""
                UPDATE nodes SET valid_until = ?
                WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL
                  AND confidence * EXP(-decay_rate *
                      POWER((? - created_at) / ?, ?)) < ?""",
                (now, now, DAY, settings.decay_exponent, thr)) or 0
        async with self._s.db.get_read_connection().execute("""
                SELECT id, confidence, decay_rate, created_at FROM nodes
                WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL""") as cur:
            rows = await cur.fetchall()
        ids = [r["id"] for r in rows
               if self.effective_confidence(r["confidence"], r["decay_rate"],
                                            now - r["created_at"]) < thr]
        if not ids:
            return 0
        ph = ",".join("?" for _ in ids)
        return await self._s.execute(
            f"UPDATE nodes SET valid_until=? WHERE id IN ({ph})", (now, *ids)) or 0

    async def purge_expired(self, now_ts: Optional[int] = None,
                            retention_days: Optional[int] = None) -> int:
        """Physical delete of long-expired rows; triggers clean FTS/vec/cascades."""
        now = now_ts or int(time.time())
        ret = settings.episode_retention_days if retention_days is None else retention_days
        n1 = await self._s.execute(
            "DELETE FROM nodes WHERE valid_until IS NOT NULL AND valid_until < ?",
            (now,)) or 0
        n2 = 0
        if ret and ret > 0:
            n2 = await self._s.execute(
                "DELETE FROM nodes WHERE type='episodic' AND event_time < ?",
                (now - int(ret * DAY),)) or 0
        return n1 + n2

    async def run_night_job(self, now_ts: Optional[int] = None) -> dict:
        now = now_ts or int(time.time())
        expired = await self.expire_weak(now)
        purged = await self.purge_expired(now)
        logger.info("decay night job: expired=%d purged=%d", expired, purged)
        return {"expired": expired, "purged": purged}
