# services/search.py
"""Hybrid retrieval: FTS5 (OR) + sqlite-vec KNN + anchored graph + entities, RRF.

Final ranking multiplies the RRF score by effective confidence measured from
LAST ACCESS (falls back to created_at), so recently used facts stay on top.
"""
from __future__ import annotations

import asyncio
import logging
import re
import struct
import time
from collections.abc import Sequence

import sqlite_vec

from memory_mcp.config.settings import settings
from memory_mcp.core.storage import Storage
from memory_mcp.core.tasks import spawn
from memory_mcp.services.decay import DecayService
from memory_mcp.services.embedding import EmbeddingError, EmbeddingService
from memory_mcp.services.entity_index import EntityIndex

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)
CYR_RE = re.compile(r"[а-яё]")
STOPWORDS_EN = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
                "is", "are", "was", "were", "what", "where", "when", "who",
                "how", "why", "do", "does", "did", "i", "you", "my", "me"}
STOPWORDS_RU = {"что", "как", "где", "когда", "почему", "зачем", "кто", "какой",
    "какая", "какие", "это", "этот", "эта", "эти", "тот", "там", "тут", "здесь",
    "все", "весь", "вся", "еще", "уже", "или", "если", "чтобы", "тоже", "также",
    "потому", "поэтому", "очень", "можно", "нужно", "надо", "был", "была",
    "было", "были", "будет", "будут", "есть", "нет", "меня", "мне", "мой",
    "моя", "мое", "мои", "твой", "твоя", "свой", "своя", "свое", "свои",
    "него", "нее", "них", "ему", "ей", "ими", "вас", "вам", "вами", "нас",
    "нам", "нами", "тебя", "тебе", "тобой", "при", "про", "под", "над",
    "без", "для", "через", "после", "перед", "между", "только", "даже",
    "вот", "ну", "да", "не", "ни"}

def build_fts_query(text: str) -> str:
    seen, toks = set(), []
    for t in TOKEN_RE.findall(text.casefold()):
        t = t.replace("ё", "е")
        if t in STOPWORDS_EN or t in STOPWORDS_RU or t in seen:
            continue
        if CYR_RE.search(t) and len(t) >= 5:
            t = t[: max(4, len(t) - 2)]   # crude RU stem: prefix covers word forms
        seen.add(t)
        toks.append(t)
    return " OR ".join(f'"{t}"*' for t in toks[:12])

class SearchService:
    def __init__(self, storage: Storage, embedding: EmbeddingService,
                 entity_index: EntityIndex):
        self._s = storage
        self._emb = embedding
        self._ei = entity_index
        self._decay = DecayService(storage)

    # ---------------------------------------------------------- public API
    async def search(self, query: str, *, limit: int = 10,
                     node_types: Sequence[str] | None = None,
                     time_from: int | None = None,
                     time_to: int | None = None,
                     include_episodes: bool = False) -> list[dict]:
        limit = min(limit, settings.max_search_limit)
        now = int(time.time())

        fts, vec, entity = await asyncio.gather(
            self._fts(query, node_types, time_from, time_to, include_episodes),
            self._vec(query, node_types, time_from, time_to, include_episodes),
            self._entity(query, node_types))
        seeds = [i for i, _ in (vec[:3] or fts[:3])]
        graph = await self._graph(seeds, node_types)

        fused = self._rrf({"fts": fts, "vec": vec, "graph": graph,
                           "entity": entity}, limit)
        node_ids = [i for i, _ in fused]
        rows = await self._s.fetch_nodes(node_ids)
        by_id = {r["id"]: r for r in rows}

        # ONE batched call for entities of ALL result nodes (fixes N+1)
        entities_map = await self._s.get_entities_for_nodes(node_ids)

        out = []
        for nid, score in fused:
            r = by_id.get(nid)
            if not r:
                continue
            idle = now - (r["last_accessed"] or r["created_at"])
            eff = self._decay.effective_confidence(
                r["confidence"], r["decay_rate"], idle)
            # Only canonical names go to the LLM, not full entity rows
            ents = [e["canonical_name"] for e in entities_map.get(nid, [])]
            out.append({
                "id": r["id"], "type": r["type"], "content": r["content"],
                "confidence": round(eff, 4), "event_time": r["event_time"],
                "entities": ents, "score": round(score * eff, 6),
            })
        out.sort(key=lambda d: d["score"], reverse=True)
        out = out[:limit]

        # Reinforcement is a WRITE: run it in the background so a failed or
        # slow write (e.g. SQLite without LN()) can never kill the search.
        spawn(self._safe_access([d["id"] for d in out]), name="access-nodes")
        return out

    async def _safe_access(self, ids: Sequence[str]) -> None:
        try:
            await self._s.access_nodes(ids)
        except Exception as e:
            logger.debug("access_nodes failed (non-fatal): %s", e)

    async def dedupe_candidate(self, content: str, *,
                               emb: Sequence[float] | None = None,
                               exclude_id: str | None = None,
                               node_types: Sequence[str] = (
                                   "semantic", "procedural", "opinion"),
                               threshold: float | None = None
                               ) -> tuple[str, dict | None]:
        """Pure vector-similarity duplicate check (no LLM arbitration).

        Returns ('skip' | 'conflict' | 'insert', existing_node_or_None).
        Pass emb= when the caller already embedded the text to avoid a
        second HTTP round-trip per fact.

        KNN overfetches (dedup_knn_k): episodes share vec_nodes, so a small k
        could be entirely filled with episodes and miss the duplicate fact.
        sqlite-vec is brute-force, so a large k is essentially free.
        """
        thr = (threshold if threshold is not None
               else settings.dedup_similarity_threshold)
        if emb is None:
            try:
                emb = await self._emb.embed_document(content)
            except EmbeddingError:
                return "insert", None

        where, params = self._filters(node_types, None, None, False)
        if exclude_id:
            where += " AND n.id != ?"
            params.append(exclude_id)
        sql = f"""
            SELECT n.id, n.content, n.type, n.decay_rate, v.distance FROM (
                SELECT node_id, distance FROM vec_nodes
                WHERE embedding MATCH ? AND k = ?
            ) v JOIN nodes n ON n.id = v.node_id
            WHERE {where} ORDER BY v.distance LIMIT 1"""
        async with self._s.db.get_read_connection().execute(
                sql, (sqlite_vec.serialize_float32([float(x) for x in emb]),
                      settings.dedup_knn_k, *params)) as cur:
            row = await cur.fetchone()
        if not row:
            return "insert", None
        # embeddings are L2-normalized, so cosine == 1 - d^2 / 2
        sim = 1.0 - row["distance"] ** 2 / 2.0
        info = {"id": row["id"], "content": row["content"],
                "type": row["type"], "decay_rate": row["decay_rate"]}
        if sim >= thr:
            return "skip", info
        if sim >= settings.conflict_band_min:
            return "conflict", info
        return "insert", None

    async def sweep_duplicates(self, limit: int | None = None) -> dict:
        """Facts stored while the embedder was down never got deduped — fix that.

        Vectors are read back from vec_nodes via point lookups (vec0 does not
        support full scans without MATCH), so no re-embedding happens here.
        Keeper = pinned fact if any, else the older one. Same-type only.
        """
        limit = settings.dedup_sweep_limit if limit is None else limit
        if limit <= 0:
            return {"checked": 0, "merged": 0}
        async with self._s.db.get_read_connection().execute("""
                SELECT id, content, type, decay_rate FROM nodes
                WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL
                ORDER BY rid DESC LIMIT ?""", (limit,)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        merged = 0
        for r in rows:
            if await self._s.get_node(r["id"]) is None:
                continue                       # already merged earlier in this sweep
            async with self._s.db.get_read_connection().execute(
                    "SELECT embedding FROM vec_nodes WHERE node_id = ?",
                    (r["id"],)) as cur:
                vrow = await cur.fetchone()
            if vrow is None:
                continue                       # not embedded yet
            emb = struct.unpack(f"<{settings.embedding_dim}f", vrow["embedding"])
            action, existing = await self.dedupe_candidate(
                r["content"], emb=emb, exclude_id=r["id"], node_types=(r["type"],))
            if action != "skip" or not existing:
                continue
            keeper, loser = existing["id"], r["id"]
            if r["decay_rate"] == 0.0 and existing["decay_rate"] != 0.0:
                keeper, loser = loser, keeper  # never drop a pinned fact
            await self._s.merge_duplicate_facts(keeper, loser)
            await self._s.access_nodes([keeper])
            merged += 1
        return {"checked": len(rows), "merged": merged}

    # ---------------------------------------------------------- channels
    def _filters(self, node_types, time_from, time_to, include_episodes):
        clauses, params = [], []
        types = list(node_types) if node_types else (
            ["semantic", "procedural", "opinion"]
            + (["episodic"] if include_episodes else []))
        clauses.append(f"type IN ({','.join('?' for _ in types)})")
        params += types
        clauses.append("valid_until IS NULL")
        if time_from:
            clauses.append("event_time >= ?")
            params.append(time_from)
        if time_to:
            clauses.append("event_time <= ?")
            params.append(time_to)
        return " AND ".join(clauses), params

    async def _fts(self, query, node_types, t_from, t_to, include_episodes):
        fts_q = build_fts_query(query)
        if not fts_q:
            return []
        where, params = self._filters(node_types, t_from, t_to, include_episodes)
        # rowid works with or without the explicit rid column
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
        # Overfetch: the type/valid_until filter runs AFTER KNN, and episodes
        # share vec_nodes — a small k could be all episodes. Brute-force scan
        # makes a larger k cheap.
        k = min(settings.knn_k * settings.vec_overfetch, 10000)
        sql = f"""
            SELECT n.id, v.distance FROM (
                SELECT node_id, distance FROM vec_nodes
                WHERE embedding MATCH ? AND k = ?
            ) v JOIN nodes n ON n.id = v.node_id
            WHERE {where} ORDER BY v.distance LIMIT 40"""
        async with self._s.db.get_read_connection().execute(
                sql, (sqlite_vec.serialize_float32([float(x) for x in emb]),
                      k, *params)) as cur:
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
