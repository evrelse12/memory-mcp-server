# services/search.py
"""Hybrid retrieval: FTS5 (OR) + sqlite-vec KNN + anchored graph + entities, RRF.
Final ranking multiplies RRF score by effective confidence measured from the
LAST ACCESS (falls back to created_at), so used facts stay on top."""
from __future__ import annotations

import logging
import re
import time
from typing import Optional, Sequence

import sqlite_vec

from config.settings import settings
from core.storage import Storage
from services.decay import DecayService
from services.embedding import EmbeddingError, EmbeddingService
from services.entity_index import EntityIndex

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)
STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
             "is", "are", "was", "were", "what", "where", "when", "who",
             "how", "why", "do", "does", "did", "i", "you", "my", "me"}


def build_fts_query(text: str) -> str:
    toks = [t for t in TOKEN_RE.findall(text.casefold()) if t not in STOPWORDS]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"*' for t in toks[:12])


class SearchService:
    def __init__(self, storage: Storage, embedding: EmbeddingService,
                 entity_index: EntityIndex):
        self._s = storage
        self._emb = embedding
        self._ei = entity_index
        self._decay = DecayService(storage)

    # ---------------------------------------------------------- public
    async def search(self, query: str, *, limit: int = 10,
                     node_types: Optional[Sequence[str]] = None,
                     entity_ids: Optional[Sequence[str]] = None,
                     time_from: Optional[int] = None,
                     time_to: Optional[int] = None,
                     include_episodes: bool = False) -> list[dict]:
        limit = min(limit, settings.max_search_limit)
        now = int(time.time())

        fts = await self._fts(query, node_types, time_from, time_to, include_episodes)
        vec = await self._vec(query, node_types, time_from, time_to, include_episodes)
        seeds = [i for i, _ in (vec[:3] or fts[:3])]
        graph = await self._graph(seeds, node_types)
        entity = await self._entity(query, node_types)

        fused = self._rrf({"fts": fts, "vec": vec, "graph": graph,
                           "entity": entity}, limit)
        rows = await self._s.fetch_nodes([i for i, _ in fused])
        by_id = {r["id"]: r for r in rows}

        out = []
        for nid, score in fused:
            r = by_id.get(nid)
            if not r:
                continue
            idle = now - (r["last_accessed"] or r["created_at"])
            eff = self._decay.effective_confidence(
                r["confidence"], r["decay_rate"], idle)
            ents = await self._s.fetch_entities_for_node(nid)
            out.append({
                "id": r["id"], "type": r["type"], "content": r["content"],
                "confidence": round(eff, 4), "event_time": r["event_time"],
                "entities": ents, "score": round(score * eff, 6),
            })
        out.sort(key=lambda d: d["score"], reverse=True)
        out = out[:limit]
        # reinforcement capped to top-3 inside storage.access_nodes
        await self._s.access_nodes([d["id"] for d in out])
        return out

    # ---------------------------------------------------------- channels
    def _filters(self, node_types, time_from, time_to, include_episodes):
        clauses, params = [], []
        types = list(node_types) if node_types else (
            ["semantic", "procedural", "opinion"] + (["episodic"] if include_episodes else []))
        clauses.append(f"type IN ({','.join('?' for _ in types)})")
        params += types
        clauses.append("valid_until IS NULL")
        if time_from:
            clauses.append("event_time >= ?"); params.append(time_from)
        if time_to:
            clauses.append("event_time <= ?"); params.append(time_to)
        return " AND ".join(clauses), params

    async def _fts(self, query, node_types, t_from, t_to, include_episodes):
        fts_q = build_fts_query(query)
        if not fts_q:
            return []
        where, params = self._filters(node_types, t_from, t_to, include_episodes)
        sql = f"""
            SELECT n.id, bm25(nodes_fts) AS rank
            FROM nodes_fts f JOIN nodes n ON n.rowid = f.rowid
            WHERE nodes_fts MATCH ? AND {where}
            ORDER BY rank LIMIT 40"""
        async with self._s.db.get_read_connection().execute(
                sql, (fts_q, *params)) as cur:
            rows = await cur.fetchall()
        return [(r["id"], -r["rank"]) for r in rows]

    async def _vec(self, query, node_types, t_from, t_to, include_episodes):
        try:
            emb = await self._emb.embed_query(query)
        except EmbeddingError:
            logger.warning("embedding unavailable, FTS-only mode")
            return []
        where, params = self._filters(node_types, t_from, t_to, include_episodes)
        sql = f"""
            SELECT n.id, v.distance FROM (
                SELECT node_id, distance FROM vec_nodes
                WHERE embedding MATCH ? AND k = ?
            ) v JOIN nodes n ON n.id = v.node_id
            WHERE {where} ORDER BY v.distance LIMIT 40"""
        async with self._s.db.get_read_connection().execute(
                sql, (sqlite_vec.serialize_float32([float(x) for x in emb]),
                      settings.rrf_k, *params)) as cur:
            rows = await cur.fetchall()
        return [(r["id"], 1.0 - r["distance"] ** 2 / 2.0) for r in rows]

    async def _graph(self, seeds: list[str], node_types) -> list[tuple[str, float]]:
        if not seeds:
            return []
        ph = ",".join("?" for _ in seeds)
        sql = f"""
            SELECT n.id, MAX(e.weight) AS w FROM edges e
            JOIN nodes n ON n.id = CASE WHEN e.source_id IN ({ph})
                                        THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id IN ({ph}) OR e.target_id IN ({ph}))
              AND e.valid_until IS NULL AND n.valid_until IS NULL
              AND n.type != 'episodic'
            GROUP BY n.id ORDER BY w DESC LIMIT 20"""
        async with self._s.db.get_read_connection().execute(
                sql, (*seeds, *seeds, *seeds)) as cur:
            rows = await cur.fetchall()
        return [(r["id"], r["w"]) for r in rows]

    async def _entity(self, query: str, node_types) -> list[tuple[str, float]]:
        hits = []
        for canon in self._ei.find_in_text(query):
            eid = self._ei.resolve(canon)
            if not eid:
                continue
            nodes = await self._s.fetch_nodes_for_entity(eid)
            hits += [(n["id"], 0.5) for n in nodes[:10]]
        return hits

    # ---------------------------------------------------------- fusion
    @staticmethod
    def _rrf(channels: dict, limit: int) -> list[tuple[str, float]]:
        weights = {"fts": settings.rrf_weight_fts, "vec": settings.rrf_weight_vec,
                   "graph": settings.rrf_weight_graph, "entity": 0.8}
        scores: dict[str, float] = {}
        for name, rows in channels.items():
            ordered = sorted(rows, key=lambda t: t[1], reverse=True)
            for rank, (nid, _) in enumerate(ordered):
                scores[nid] = scores.get(nid, 0.0) + \
                    weights[name] / (settings.rrf_k + rank + 1)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
