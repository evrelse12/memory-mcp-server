# services/hot_path.py
"""Hot path: record episodes, link entities, defer embeddings to background."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from memory_mcp.config.settings import settings
from memory_mcp.core.storage import Storage
from memory_mcp.core.tasks import spawn
from memory_mcp.services.embedding import EmbeddingError, EmbeddingService
from memory_mcp.services.entity_index import EntityIndex

logger = logging.getLogger(__name__)


class HotPath:
    def __init__(self, storage: Storage, embedding: EmbeddingService, entity_index: EntityIndex):
        self._s = storage
        self._emb = embedding
        self._ei = entity_index
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._auto_sid: str | None = None
        self._auto_last: int = 0

    def current_session(self, explicit: str | None = None) -> str:
        """Host-provided chat id wins; otherwise rotate the auto session on idle.

        Rotation makes the previous conversation consolidatable while the
        process is still alive (no 03:00 wait, no restart needed).
        """
        if explicit:
            return explicit
        now = int(time.time())
        if self._auto_sid is None or now - self._auto_last > settings.session_timeout_minutes * 60:
            self._auto_sid = uuid.uuid4().hex[:12]
        self._auto_last = now
        return self._auto_sid

    def _lock(self, session_id: str) -> asyncio.Lock:
        # per-session lock: two parallel messages must not fork the temporal chain
        return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def record_message(self, session_id: str, message: str, *, role: str = "user") -> str:
        text = message[: settings.max_episode_chars]
        async with self._lock(session_id):
            now = int(time.time())
            stmt, nid = self._s.node_stmt(
                {
                    "type": "episodic",
                    "content": text,
                    "session_id": session_id,
                    "event_time": now,
                    "confidence": 1.0,
                    "decay_rate": 0.0,
                    "needs_embedding": True,
                    "metadata": {"role": role},
                }
            )
            stmts = [stmt]
            # NOTE: temporal edges are not written; ordering is recovered
            # from (session_id, event_time, rid).
            for canon in self._ei.find_in_text(text):
                eid = await self._ei.ensure(canon)
                stmts += self._s.link_entities_stmts(nid, [eid])
            await self._s.transaction(stmts, wait=True)
        spawn(self._embed_episode(nid, text), name=f"embed-{nid[:8]}")
        return nid

    async def _embed_episode(self, nid: str, text: str) -> None:
        try:
            emb = await self._emb.embed_document(text)
            await self._s.insert_embedding_and_clear_flag(nid, emb)
        except EmbeddingError:
            logger.warning("episode embedding deferred: %s", nid)
