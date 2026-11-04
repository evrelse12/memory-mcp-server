# services/hot_path.py
"""Hot path: record episodes, link entities, defer embeddings to background."""
from __future__ import annotations

import asyncio
import logging
import time

from config.settings import settings
from core.storage import Storage
from core.tasks import spawn
from services.embedding import EmbeddingError, EmbeddingService
from services.entity_index import EntityIndex

logger = logging.getLogger(__name__)


class HotPath:
    def __init__(self, storage: Storage, embedding: EmbeddingService,
                 entity_index: EntityIndex):
        self._s = storage
        self._emb = embedding
        self._ei = entity_index
        self._session_locks: dict[str, asyncio.Lock] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        # per-session lock: two parallel messages must not fork the temporal chain
        return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def record_message(self, session_id: str, message: str, *,
                             role: str = "user") -> str:
        text = message[: settings.max_episode_chars]   # enforce configured limit
        async with self._lock(session_id):
            prev = await self._s.last_episode_id(session_id)
            now = int(time.time())
            stmt, nid = self._s.node_stmt({
                "type": "episodic", "content": text, "session_id": session_id,
                "event_time": now, "confidence": 1.0, "decay_rate": 0.0,
                "needs_embedding": True, "metadata": {"role": role},
            })
            stmts = [stmt]
            if prev:
                stmts.append(self._s.edge_stmt(nid, prev, "temporal",
                                               predicate="follows"))
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

    def last_episode(self, session_id: str) -> str | None:
        return None  # superseded by storage.last_episode_id; kept for compatibility
