# tools/memory_tools.py
"""Tool-layer orchestration over storage/search; auto entity extraction internal."""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from core.storage import Storage
from services.decay import DecayService
from services.embedding import EmbeddingError, EmbeddingService
from services.entity_index import EntityIndex
from services.search import SearchService

logger = logging.getLogger(__name__)

FACT_TYPES = {"semantic", "procedural", "opinion"}


class MemoryToolError(Exception):
    """User-facing tool error (mapped to MCP ToolError by the server)."""


class MemoryTools:
    def __init__(self, storage: Storage, search: SearchService,
                 embedding: EmbeddingService, entity_index: EntityIndex,
                 decay: DecayService):
        self._s = storage
        self._search = search
        self._emb = embedding
        self._ei = entity_index
        self._decay = decay

    async def search_memory(self, query: str, *, node_types=None,
                            entity_ids=None, time_from=None, time_to=None,
                            limit: int = 10,
                            include_episodes: bool = False) -> list[dict]:
        # entity_ids kept in signature for tool compatibility, but ignored
        return await self._search.search(
            query, limit=limit, node_types=node_types,
            time_from=time_from, time_to=time_to,
            include_episodes=include_episodes)

    async def remember_fact(self, content: str, *, node_type: str = "semantic",
                            confidence: float = 0.9,
                            entities: Sequence[str] | None = None,
                            source_episode_ids: Sequence[str] = (),
                            pin: bool = False,
                            force: bool = False) -> dict:
        if node_type not in FACT_TYPES:
            raise MemoryToolError(f"bad node_type: {node_type!r}")
        confidence = min(1.0, max(0.0, float(confidence)))

        # Embed ONCE; reuse the vector for dedupe and for storage.
        emb = None
        try:
            emb = await self._emb.embed_document(content)
        except EmbeddingError as e:
            logger.warning("embedding unavailable, storing without vector: %s", e)

        action, existing = await self._search.dedupe_candidate(content, emb=emb)

        if action == "skip" and existing and not force:
            # A repeated mention confirms the fact: reinforce it.
            await self._s.access_nodes([existing["id"]])
            pinned = False
            if pin:
                await self.confirm_fact(existing["id"])
                pinned = True
            return {"status": "duplicate",
                    "existing_id": existing["id"],
                    "existing_content": existing["content"],
                    "pinned": pinned,
                    "hint": "call correct_fact if the new text contradicts it"}

        if action == "conflict" and existing and not force:
            # Conflict band: let the model choose replace vs keep-both.
            return {"status": "similar",
                    "existing_id": existing["id"],
                    "existing_content": existing["content"],
                    "hint": "call correct_fact to replace it, or "
                            "remember_fact(force=true) to keep both"}

        names = list(entities or []) or self._ei.find_in_text(content)
        eids = [await self._ei.ensure(n) for n in names]
        nid = await self._s.create_fact(
            content=content, node_type=node_type,
            confidence=1.0 if pin else confidence,
            decay_rate=0.0 if pin else None,
            embedding=emb, entity_ids=eids,
            source_episode_ids=list(source_episode_ids))
        return {"status": "created", "node_id": nid,
                "embedded": emb is not None, "pinned": pin}

    async def correct_fact(self, old_node_id: str, new_content: str, *,
                          new_confidence: float = 0.9,
                          source_episode_ids: Sequence[str] = ()) -> dict:
        old = await self._s.get_node(old_node_id)
        if not old:
            raise MemoryToolError(f"node not found: {old_node_id}")
        emb = None
        try:
            emb = await self._emb.embed_document(new_content)
        except EmbeddingError as e:
            logger.warning("embedding unavailable for correction: %s", e)
        names = self._ei.find_in_text(new_content)
        eids = [await self._ei.ensure(n) for n in names]
        nid = await self._s.supersede_fact(
            old_node_id, node_type=old["type"], content=new_content,
            confidence=new_confidence, embedding=emb,
            source_episode_ids=list(source_episode_ids), entity_ids=eids)
        return {"status": "corrected", "old_node_id": old_node_id,
                "new_node_id": nid, "preserved_type": old["type"]}

    async def confirm_fact(self, node_id: str) -> dict:
        n = await self._s.execute(
            "UPDATE nodes SET confidence=1.0, decay_rate=0.0, last_accessed=? "
            "WHERE id=? AND valid_until IS NULL",
            (int(time.time()), node_id), wait=True)
        if not n:
            raise MemoryToolError(f"node not found or deleted: {node_id}")
        return {"status": "confirmed", "node_id": node_id}

    async def forget_fact(self, node_id: str, *, hard: bool = False) -> dict:
        node = await self._s.get_node(node_id, include_deleted=True)
        if not node:
            raise MemoryToolError(f"node not found: {node_id}")
        if hard:
            await self._s.hard_delete_nodes([node_id])
            mode = "hard"
        else:
            await self._s.soft_delete_nodes([node_id])
            mode = "soft"
        return {"status": "forgotten", "node_id": node_id, "mode": mode}

    async def get_entity_info(self, entity_name: str) -> dict:
        eid = self._ei.resolve(entity_name)
        if not eid:
            raise MemoryToolError(f"unknown entity: {entity_name!r}")
        entity = await self._s.get_entity(eid)
        nodes = await self._s.fetch_nodes_for_entity(eid)
        return {**entity, "linked_facts": [
            {"id": n["id"], "content": n["content"], "type": n["type"]}
            for n in nodes[:20]]}

    async def get_timeline(self, *, entity_name: str | None = None,
                           time_from=None, time_to=None,
                           limit: int = 20) -> list[dict]:
        eid = self._ei.resolve(entity_name) if entity_name else None
        rows = await self._s.fetch_timeline(
            entity_id=eid, time_from=time_from, time_to=time_to, limit=limit)
        return rows

    async def memory_stats(self) -> dict:
        s = await self._s.stats()
        # vec_available = is the embedding service answering RIGHT NOW?
        try:
            await self._emb.embed_query("ping")
            vec_available = True
        except Exception:
            vec_available = False
        return {**s, "vec_available": vec_available}

    async def explain_fact(self, node_id: str) -> dict:
        node = await self._s.get_node(node_id, include_deleted=True)
        if not node:
            raise MemoryToolError(f"node not found: {node_id}")
        sources, superseded, ents = await self._s.provenance(node_id)
        return {"node": node, "source_episodes": sources,
                "superseded": superseded, "entities": ents}

    async def weak_facts(self, limit: int = 20) -> list[dict]:
        """Facts whose EFFECTIVE confidence (from last access) is below 0.3."""
        now = int(time.time())
        async with self._s.db.get_read_connection().execute("""
                SELECT id, type, content, confidence, decay_rate,
                       created_at, last_accessed FROM nodes
                WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL""") as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            idle = now - (r["last_accessed"] or r["created_at"])
            eff = self._decay.effective_confidence(
                r["confidence"], r["decay_rate"], idle)
            if eff < 0.3:
                out.append({"id": r["id"], "type": r["type"],
                            "content": r["content"],
                            "effective_confidence": round(eff, 4)})
        out.sort(key=lambda d: d["effective_confidence"])
        return out[:limit]
