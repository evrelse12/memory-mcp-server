# services/search.py
"""Hybrid retrieval: FTS5 (OR) + sqlite-vec KNN + anchored graph walk + entity lookup, RRF."""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

from config.settings import settings
from core.storage import Storage
from services.embedding import EmbeddingError, EmbeddingService
from services.entity_index import EntityIndex
from services.intent_classifier import classify_intent

logger = logging.getLogger(__name__)

MAX_LIMIT = 50

_STOP = {
    # EN
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "what", "who",
    "where", "when", "why", "how", "you", "your", "my", "me", "i", "we", "us", "it",
    "its", "of", "in", "on", "at", "to", "for", "and", "or", "about", "tell", "know",
    # RU
    "и", "в", "на", "с", "по", "о", "об", "от", "к", "у", "за", "из", "что", "кто",
    "где", "когда", "почему", "как", "я", "мы", "ты", "вы", "он", "она", "оно", "они",
    "меня", "мне", "тебя", "тебе", "его", "её", "их", "это", "этот", "такой", "есть",
    "был", "была", "было", "были", "будет", "мне", "вам", "нам", "них", "него", "нее",
}


def build_fts_query(query: str) -> Optional[str]:
    """OR-joined prefix tokens; stopwords dropped; quotes impossible via \\w+."""
    toks = [t for t in re.findall(r"\w+", query.casefold())
            if t not in _STOP and len(t) > 1]
    if not toks:
        return None
    return " OR ".join(f'"{t}" *' for t in toks[:12])


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SearchService:
    def __init__(self, storage: Storage, embedding: EmbeddingService,
                 entity_index: EntityIndex):
        self._s = storage
        self._emb = embedding
        self._ei = entity_index

    # ------------------------------------------------------------ public
    async def search(self, query: str, *, limit: int = 10,
                     node_types: Optional[Sequence[str]] = None,
                     entity_ids: Optional[Sequence[str]] = None,
                     time_from: Optional[int] = None,
                     time_to: Optional[int] = None,
                     include_episodes: bool = False) -> list[dict]:
        limit = max(1, min(int(limit), MAX_LIMIT))
        k = limit * 3  # headroom: soft-deleted filtered after KNN
        intent = classify_intent(query)

        try:
            emb = await self._emb.embed_query(query)
        except EmbeddingError as e:
            logger.warning("embedding unavailable, FTS-only mode: %s", e)
            emb = None

        filters = self._filter_clauses(node_types, entity_ids, time_from,
                                       time_to, include_episodes)

        fts = await self._fts(query, k, filters)
        vec = await self._vec(emb, k, filters) if emb is not None else []
        if not fts and not vec:
            fts = await self._like(query, k, filters)

        anchors = self._anchors(fts, vec)
        graph = await self._graph(anchors, intent, k, filters) if anchors else []
        entity = await self._entity(query, k, filters) if intent in ("who", "what") else []

        results = self._rrf({"fts": fts, "vec": vec, "graph": graph,
                             "entity": entity}, limit)
        if results:
            await self._s.access_nodes([r["id"] for r in results], wait=False)
        logger.info("search %r intent=%s hits=%d", query[:60], intent, len(results))
        return results

    # ------------------------------------------------------------ filters
    @staticmethod
    def _filter_clauses(node_types, entity_ids, time_from, time_to,
                        include_episodes) -> tuple[str, list[Any]]:
        cl, params = [], []
        if node_types:
            cl.append(f"n.type IN ({','.join('?' for _ in node_types)})")
            params.extend(node_types)
        elif not include_episodes:
            cl.append("n.type != 'episodic'")  # facts outrank raw dialogue by default
        if time_from is not None:
            cl.append("n.event_time >= ?")
            params.append(time_from)
        if time_to is not None:
            cl.append("n.event_time <= ?")
            params.append(time_to)
        if entity_ids:
            ph = ",".join("?" for _ in entity_ids)
            cl.append(f"EXISTS(SELECT 1 FROM node_entities ne "
                      f"WHERE ne.node_id = n.id AND ne.entity_id IN ({ph}))")
            params.extend(entity_ids)
        clause = (" AND " + " AND ".join(cl)) if cl else ""
        return clause, params

    # ------------------------------------------------------------ strategies
    async def _fts(self, query: str, k: int, filters) -> list[dict]:
        fts_q = build_fts_query(query)
        if not fts_q:
            return []
        clause, params = filters
        sql = f"""
            SELECT n.*, bm25(nodes_fts) AS rank
            FROM nodes_fts JOIN nodes n ON n.rid = nodes_fts.rowid
            WHERE nodes_fts MATCH ? AND n.valid_until IS NULL{clause}
            ORDER BY rank LIMIT ?"""
        async with self._s.db.get_read_connection().execute(
                sql, (fts_q, *params, k)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def _vec(self, emb: Sequence[float], k: int, filters) -> list[dict]:
        """KNN inside subquery so sqlite-vec sees k; filters applied after join."""
        import sqlite_vec
        clause, params = filters
        sql = f"""
            SELECT n.*, v.distance FROM (
                SELECT node_id, distance FROM vec_nodes
                WHERE embedding MATCH ? AND k = ?
            ) v JOIN nodes n ON n.id = v.node_id
            WHERE n.valid_until IS NULL{clause}
            ORDER BY v.distance LIMIT ?"""
        try:
            async with self._s.db.get_read_connection().execute(
                    sql, (sqlite_vec.serialize_float32([float(x) for x in emb]),
                          k, *params, k)) as cur:
                rows = await cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("vector search unavailable: %s", e)
            return []

    async def _like(self, query: str, k: int, filters) -> list[dict]:
        words = [w for w in re.findall(r"\w+", query.casefold())
                 if w not in _STOP and len(w) > 1][:6]
        if not words:
            return []
        clause, params = filters
        likes, lparams = [], []
        for w in words:
            likes.append("n.content LIKE ? ESCAPE '\\'")
            lparams.append(f"%{_like_escape(w)}%")
        sql = f"""
            SELECT n.* FROM nodes n
            WHERE ({' OR '.join(likes)}) AND n.valid_until IS NULL{clause}
            ORDER BY n.confidence DESC, n.event_time DESC LIMIT ?"""
        async with self._s.db.get_read_connection().execute(
                sql, (*lparams, *params, k)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def _graph(self, anchors: Sequence[str], intent: str, k: int,
                     filters) -> list[dict]:
        if intent not in ("why", "when"):
            return []
        rel = "causal" if intent == "why" else "temporal"
        depth = (settings.max_graph_depth_complex
                 if len(anchors) > 3 else settings.max_graph_depth_simple)
        clause, params = filters
        ph = ",".join("?" for _ in anchors)
        sql = f"""
            WITH RECURSIVE chain(node_id, depth) AS (
                SELECT id, 0 FROM nodes WHERE id IN ({ph}) AND valid_until IS NULL
                UNION
                SELECT CASE WHEN e.source_id = c.node_id THEN e.target_id
                            ELSE e.source_id END, c.depth + 1
                FROM edges e JOIN chain c
                  ON (e.source_id = c.node_id OR e.target_id = c.node_id)
                WHERE e.relation_type = ? AND e.valid_until IS NULL AND c.depth < ?
            )
            SELECT DISTINCT n.* FROM nodes n JOIN chain c ON n.id = c.node_id
            WHERE n.valid_until IS NULL AND c.depth > 0{clause}
            ORDER BY n.event_time DESC LIMIT ?"""
        async with self._s.db.get_read_connection().execute(
                sql, (*anchors, rel, depth, *params, k)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def _entity(self, query: str, k: int, filters) -> list[dict]:
        hits = self._ei.find_in_text(query)
        if not hits:
            return []
        ids = [eid for _, eid in hits]
        clause, params = filters
        ph = ",".join("?" for _ in ids)
        sql = f"""
            SELECT n.* FROM node_entities ne JOIN nodes n ON n.id = ne.node_id
            WHERE ne.entity_id IN ({ph}) AND n.valid_until IS NULL{clause}
            ORDER BY n.confidence DESC, n.event_time DESC LIMIT ?"""
        async with self._s.db.get_read_connection().execute(
                sql, (*ids, *params, k)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ fusion
    @staticmethod
    def _anchors(fts: list[dict], vec: list[dict], n: int = 5) -> list[str]:
        seen: list[str] = []
        for row in fts + vec:
            if row["id"] not in seen:
                seen.append(row["id"])
            if len(seen) >= n:
                break
        return seen

    @staticmethod
    def _rrf(lists: dict[str, list[dict]], limit: int) -> list[dict]:
        weights = {"fts": settings.rrf_weight_fts, "vec": settings.rrf_weight_vec,
                   "graph": settings.rrf_weight_graph,
                   "entity": settings.rrf_weight_graph}
        scores: dict[str, float] = {}
        items: dict[str, dict] = {}
        for name, rows in lists.items():
            w = weights[name]
            for rank, item in enumerate(rows, 1):
                nid = item["id"]
                conf = 0.5 + 0.5 * float(item.get("confidence", 1.0))
                scores[nid] = scores.get(nid, 0.0) + (w / (settings.rrf_k + rank)) * conf
                items.setdefault(nid, item)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
        return [items[nid] for nid in ranked]
