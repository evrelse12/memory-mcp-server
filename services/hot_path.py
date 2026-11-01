# services/hot_path.py
"""Live dialogue write path: episodic node + temporal chain + entity links, atomic."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from core.storage import Storage
from services.embedding import EmbeddingError, EmbeddingService
from services.entity_index import EntityIndex

logger = logging.getLogger(__name__)


class HotPath:
    def __init__(self, storage: Storage, embedding: EmbeddingService,
                 entity_index: EntityIndex):
        self._s = storage
        self._emb = embedding
        self._ei = entity_index
        self._last: dict[str, str] = {}   # session_id -> last episode id (cache)

    async def record_message(self, session_id: str, content: str, *,
                             role: str = "user") -> str:
        """One transaction: episode + temporal edge + entity links. Embedding async."""
        stmt, node_id = self._s.node_stmt({
            "type": "episodic", "content": content, "session_id": session_id,
            "decay_rate": 0.0, "needs_embedding": True,
            "metadata": {"role": role},
        })
        stmts = [stmt]

        prev = self._last.get(session_id) or await self._fetch_last(session_id)
        if prev:
            stmts.append(self._s.edge_stmt(node_id, prev, "temporal",
                                           predicate="follows"))
        for _name, entity_id in self._ei.find_in_text(content):
            stmts.append(("INSERT OR IGNORE INTO node_entities(node_id, entity_id) "
                          "VALUES (?,?)", (node_id, entity_id)))

        await self._s.transaction(stmts, wait=True)
        self._last[session_id] = node_id

        asyncio.get_running_loop().create_task(self._embed_later(node_id, content))
        return node_id

    def last_episode(self, session_id: str) -> Optional[str]:
        """For remember_fact provenance (derived_from to the live dialogue)."""
        return self._last.get(session_id)

    # ------------------------------------------------------------ internals
    async def _fetch_last(self, session_id: str) -> Optional[str]:
        async with self._s.db.get_read_connection().execute(
                """SELECT id FROM nodes
                   WHERE session_id=? AND type='episodic' AND valid_until IS NULL
                   ORDER BY event_time DESC, rid DESC LIMIT 1""",
                (session_id,)) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None

    async def _embed_later(self, node_id: str, content: str) -> None:
        """Best effort: on failure needs_embedding stays 1 and backfill fixes it."""
        try:
            emb = (await self._emb.embed_documents([content]))[0]
        except EmbeddingError as e:
            logger.warning("episode embedding deferred: %s", e)
            return
        try:
            await self._s.insert_embedding_and_clear_flag(node_id, emb)
        except Exception:
            logger.exception("episode embedding insert failed: %s", node_id)
