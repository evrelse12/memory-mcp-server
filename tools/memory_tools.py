# tools/memory_tools.py
"""Business logic for the memory tools: typed, transactional, race-safe."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional, Sequence

import sqlite_vec

from config.settings import settings
from core.storage import Storage
from services.decay import DecayService
from services.embedding import EmbeddingError, EmbeddingService
from services.entity_index import EntityIndex
from services.search import SearchService

logger = logging.getLogger(__name__)

FACT_TYPES = ("semantic", "procedural", "opinion")


class MemoryToolError(RuntimeError):
    """User-facing failure; the MCP layer maps it to ToolError(isError=true)."""


def _project(node: dict, entities: Optional[list[dict]] = None) -> dict:
    """Minimal payload for the model: no metadata/decay/valid_until noise."""
    return {
        "id": node["id"],
        "type": node["type"],
        "content": node["content"],
        "confidence": round(float(node["confidence"]), 3),
        "event_time": node["event_time"],
        "entities": [{"id": e["id"], "name": e["canonical_name"], "type": e["type"]}
                     for e in (entities or [])],
    }


class MemoryTools:
    def __init__(self, storage: Storage, search: SearchService,
                 embedding: EmbeddingService, entity_index: EntityIndex,
                 decay: DecayService):
        self._s = storage
        self._search = search
        self._emb = embedding
        self._ei = entity_index
        self._decay = decay
        self._lock = asyncio.Lock()   # dedup check-then-insert is not atomic otherwise

    # ------------------------------------------------------------ helpers
    async def _embed_or_none(self, text: str) -> Optional[list[float]]:
        try:
            return (await self._emb.embed_documents([text]))[0]
        except EmbeddingError as e:
            logger.warning("embedding unavailable, storing without vector: %s", e)
            return None

    async def _find_duplicate(self, emb: Sequence[float],
                              exclude_id: Optional[str] = None) -> Optional[dict]:
        """L2 distance on unit vectors -> cosine: sim = 1 - d^2 / 2."""
        sql = """
            SELECT n.id, n.content, v.distance FROM (
                SELECT node_id, distance FROM vec_nodes
                WHERE embedding MATCH ? AND k = 8
            ) v JOIN nodes n ON n.id = v.node_id
            WHERE n.valid_until IS NULL AND n.type != 'episodic' AND n.id IS NOT ?
            ORDER BY v.distance LIMIT 1"""
        async with self._s.db.get_read_connection().execute(
                sql, (sqlite_vec.serialize_float32([float(x) for x in emb]),
                      exclude_id or "")) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        sim = 1.0 - row["distance"] ** 2 / 2.0
        if sim < settings.dedup_similarity_threshold:
            return None
        return {"id": row["id"], "content": row["content"], "similarity": round(sim, 3)}

    # ------------------------------------------------------------ tools
    async def search_memory(self, query: str, *, node_types=None, entity_ids=None,
                            time_from=None, time_to=None, limit: int = 10,
                            include_episodes: bool = False) -> list[dict]:
        rows = await self._search.search(
            query, limit=limit, node_types=node_types, entity_ids=entity_ids,
            time_from=time_from, time_to=time_to, include_episodes=include_episodes)
        if not rows:
            return []
        ent = await self._s.get_entities_for_nodes([r["id"] for r in rows])  # one query
        return [_project(r, ent.get(r["id"])) for r in rows]

    async def remember_fact(self, content: str, *, node_type: str = "semantic",
                            confidence: float = 0.9,
                            entities: Optional[Sequence[str]] = None,
                            source_episode_ids: Sequence[str] = (),
                            pin: bool = False) -> dict:
        if node_type not in FACT_TYPES:
            raise MemoryToolError(f"bad node_type: {node_type!r}")
        confidence = min(1.0, max(0.0, float(confidence)))
        emb = await self._embed_or_none(content)
        async with self._lock:
            if emb is not None:
                dup = await self._find_duplicate(emb)
                if dup:
                    return {"status": "duplicate", **dup}
            eids = [await self._ei.ensure(name) for name in (entities or [])]
            nid = await self._s.create_fact(
                content=content, node_type=node_type,
                confidence=1.0 if pin else confidence,
                decay_rate=0.0 if pin else None,
                embedding=emb, entity_ids=eids,
                source_episode_ids=list(source_episode_ids))
        return {"status": "created", "node_id": nid, "embedded": emb is not None,
                "pinned": pin}

    async def correct_fact(self, old_node_id: str, new_content: str, *,
                          new_confidence: float = 0.9,
                          source_episode_ids: Sequence[str] = ()) -> dict:
        """No dedup here: the new text is intentionally close to the old one."""
        emb = await self._embed_or_none(new_content)
        async with self._lock:
            try:
                nid = await self._s.supersede_fact(
                    old_node_id, content=new_content, confidence=new_confidence,
                    embedding=emb, source_episode_ids=list(source_episode_ids))
            except KeyError as e:
                raise MemoryToolError(str(e)) from e
        return {"status": "corrected", "old_node_id": old_node_id, "new_node_id": nid}

    async def confirm_fact(self, node_id: str) -> dict:
        n = await self._s.execute(
            "UPDATE nodes SET confidence=1.0, decay_rate=0.0 "
            "WHERE id=? AND valid_until IS NULL", (node_id,))
        if not n:
            raise MemoryToolError(f"node not found or deleted: {node_id}")
        return {"status": "confirmed", "node_id": node_id}

    async def forget_fact(self, node_id: str, *, hard: bool = False) -> dict:
        n = (await self._s.hard_delete_nodes([node_id]) if hard
             else await self._s.soft_delete_nodes([node_id]))
        if not n:
            raise MemoryToolError(f"node not found or already deleted: {node_id}")
        return {"status": "forgotten", "node_id": node_id,
                "mode": "hard" if hard else "soft"}

    async def get_entity_info(self, entity_name: str, *,
                              facts_per_type: int = 10) -> dict:
        eid = self._ei.resolve(entity_name)
        entity = await self._s.get_entity(eid) if eid else None
        if entity is None:
            raise MemoryToolError(f"entity not found: {entity_name!r}")
        nodes = await self._s.fetch_nodes_for_entity(eid)
        grouped, truncated = {}, False
        for ntype in ("semantic", "procedural", "opinion", "episodic"):
            subset = sorted((n for n in nodes if n["type"] == ntype),
                            key=lambda n: n["confidence"], reverse=True)
            if len(subset) > facts_per_type:
                truncated = True
                subset = subset[:facts_per_type]
            grouped[ntype] = [_project(n) for n in subset]
        return {
            "id": entity["id"], "name": entity["canonical_name"],
            "type": entity["type"], "summary": entity["summary"],
            "aliases": json.loads(entity["aliases"] or "[]"),
            "mention_count": entity["mention_count"],
            "first_seen": entity["first_seen"], "last_updated": entity["last_updated"],
            "facts": grouped, "total_facts": len(nodes), "truncated": truncated,
        }

    async def get_timeline(self, *, entity_name: Optional[str] = None,
                           time_from=None, time_to=None, limit: int = 20) -> list[dict]:
        eid = self._ei.resolve(entity_name) if entity_name else None
        if entity_name and eid is None:
            return []
        rows = await self._s.fetch_episodes(entity_id=eid, time_from=time_from,
                                            time_to=time_to, limit=limit)
        return [_project(r) for r in reversed(rows)]   # chronological order

    async def memory_stats(self) -> dict:
        read = self._s.db.get_read_connection()
        async with read.execute("SELECT type, COUNT(*) c FROM nodes "
                                "WHERE valid_until IS NULL GROUP BY type") as cur:
            nodes_by_type = {r["type"]: r["c"] for r in await cur.fetchall()}
        async with read.execute("SELECT relation_type t, COUNT(*) c FROM edges "
                                "WHERE valid_until IS NULL GROUP BY relation_type") as cur:
            edges_by_type = {r["t"]: r["c"] for r in await cur.fetchall()}
        async with read.execute("SELECT type, COUNT(*) c FROM entities "
                                "GROUP BY type") as cur:
            entities_by_type = {r["type"]: r["c"] for r in await cur.fetchall()}
        async with read.execute("""
                SELECT COUNT(*) c FROM nodes n WHERE n.valid_until IS NULL
                  AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = n.id)
                  AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.target_id = n.id)
                  AND NOT EXISTS (SELECT 1 FROM node_entities ne
                                  WHERE ne.node_id = n.id)""") as cur:
            orphans = (await cur.fetchone())["c"]
        return {
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "entities_by_type": entities_by_type,
            "orphans": orphans,
            "db_size_mb": round(os.path.getsize(self._s.db.db_path) / 1048576, 2),
            "vec_available": True,
        }

    async def explain_fact(self, node_id: str) -> dict:
        node = await self._s.get_node(node_id, include_deleted=True)
        if node is None:
            raise MemoryToolError(f"fact not found: {node_id}")
        edges = await self._s.fetch_edges_for_nodes([node_id])
        derived = [e for e in edges
                   if e["relation_type"] == "derived_from" and e["source_id"] == node_id]
        sup_out = [e for e in edges
                   if e["relation_type"] == "supersedes" and e["source_id"] == node_id]
        sup_in = [e for e in edges
                  if e["relation_type"] == "supersedes" and e["target_id"] == node_id]
        eps = await self._s.fetch_nodes([e["target_id"] for e in derived])
        olds = await self._s.fetch_nodes([e["target_id"] for e in sup_out])
        news = await self._s.fetch_nodes([e["source_id"] for e in sup_in])
        ents = (await self._s.get_entities_for_nodes([node_id])).get(node_id, [])
        return {
            "fact": _project(node, ents),
            "source_episodes": [_project(e) for e in eps],
            "supersedes": [_project(o) for o in olds],
            "superseded_by": [_project(n) for n in news],
            "decay_rate": node["decay_rate"],
            "access_count": node["access_count"],
            "last_accessed": node["last_accessed"],
        }

    async def weak_facts(self, limit: int = 20) -> list[dict]:
        """Effective confidence (Ebbinghaus) below 0.3, most at-risk first."""
        now = int(time.time())
        read = self._s.db.get_read_connection()
        async with read.execute("""
                SELECT id, type, content, confidence, decay_rate, created_at, event_time
                FROM nodes WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL""") as cur:
            rows = await cur.fetchall()
        scored = []
        for r in rows:
            eff = self._decay.effective_confidence(
                r["confidence"], r["decay_rate"], now - r["created_at"])
            if eff < 0.3:
                scored.append((eff, dict(r)))
        scored.sort(key=lambda p: p[0])
        out = []
        for eff, r in scored[:limit]:
            p = _project(r)
            p["effective_confidence"] = round(eff, 3)
            out.append(p)
        return out
