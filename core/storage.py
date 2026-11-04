# core/storage.py
"""Typed CRUD over nodes/edges/entities; every write goes through WriteQueue."""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any, Optional, Sequence

import sqlite_vec

from config.settings import settings
from core.database import Database
from core.write_queue import WriteQueue

logger = logging.getLogger(__name__)

NODE_COLS = ("id", "type", "content", "session_id", "event_time", "created_at",
             "last_accessed", "access_count", "confidence", "decay_rate",
             "needs_embedding", "valid_until", "metadata")
NODE_UPDATE_ALLOW = {"content", "confidence", "decay_rate", "valid_until",
                     "metadata", "session_id"}
EDGE_COLS = ("id", "source_id", "target_id", "relation_type", "predicate",
             "weight", "confidence", "valid_from", "valid_until", "evidence",
             "created_at")
ENTITY_COLS = ("id", "name_norm", "canonical_name", "type", "aliases", "summary",
               "first_seen", "last_updated", "mention_count", "attributes")


def _now() -> int:
    return int(time.time())


def _ph(ids: Sequence[Any]) -> str:
    return ",".join("?" for _ in ids)


class Storage:
    def __init__(self, db: Database, queue: WriteQueue):
        self._db = db
        self._q = queue
        self._read = db.get_read_connection()

    @property
    def db(self) -> Database:
        return self._db

    # ------------------------------------------------------------ passthrough
    async def execute(self, sql: str, params: Any = (), *, wait: bool = True,
                      many: bool = False) -> Optional[int]:
        return await self._q.execute(sql, params, wait=wait, many=many)

    async def transaction(self, stmts: Sequence[tuple[str, Any]], *,
                          wait: bool = True, many: bool = False) -> Optional[int]:
        return await self._q.transaction(stmts, wait=wait, many=many)

    async def raw(self, sql: str, params: Any = (), *, wait: bool = True) -> Optional[int]:
        return await self._q.raw(sql, params, wait=wait)

    # ------------------------------------------------------------ schema extras
    async def ensure_vec_triggers(self) -> None:
        """vec0 virtual tables do not cascade; clean them on row delete."""
        await self._q.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_vec_ad AFTER DELETE ON nodes BEGIN
              DELETE FROM vec_nodes WHERE node_id = old.id;
            END""")
        await self._q.execute("""
            CREATE TRIGGER IF NOT EXISTS entities_vec_ad AFTER DELETE ON entities BEGIN
              DELETE FROM vec_entities WHERE entity_id = old.id;
            END""")

    # ------------------------------------------------------------ stmt builders
    def node_stmt(self, data: dict) -> tuple[tuple[str, tuple], str]:
        """(INSERT stmt, node_id) for composite transactions."""
        now = _now()
        decay = data.get("decay_rate")
        event_time = data.get("event_time")
        if event_time is None:
            event_time = now
        created_at = data.get("created_at")
        if created_at is None:
            created_at = now
        row = {
            "id": data.get("id") or uuid.uuid4().hex,
            "type": data.get("type", "episodic"),
            "content": data["content"],
            "session_id": data.get("session_id"),
            "event_time": event_time,
            "created_at": created_at,
            "last_accessed": data.get("last_accessed"),
            "access_count": data.get("access_count", 0),
            "confidence": data.get("confidence", 1.0),
            "decay_rate": settings.default_decay_rate if decay is None else decay,
            "needs_embedding": 1 if data.get("needs_embedding", True) else 0,
            "valid_until": data.get("valid_until"),
            "metadata": json.dumps(data.get("metadata") or {}, ensure_ascii=False),
        }
        cols, ph = ", ".join(NODE_COLS), _ph(NODE_COLS)
        return (f"INSERT INTO nodes ({cols}) VALUES ({ph})",
                tuple(row[c] for c in NODE_COLS)), row["id"]

    def edge_stmt(self, source_id: str, target_id: str, relation_type: str, *,
                  confidence: float = 1.0, weight: float = 1.0,
                  predicate: Optional[str] = None,
                  evidence: Optional[list] = None,
                  now: Optional[int] = None) -> tuple[str, tuple]:
        now = now or _now()
        row = {
            "id": uuid.uuid4().hex,
            "source_id": source_id, "target_id": target_id,
            "relation_type": relation_type, "predicate": predicate,
            "weight": weight, "confidence": confidence,
            "valid_from": now, "valid_until": None,
            "evidence": json.dumps(evidence or [], ensure_ascii=False),
            "created_at": now,
        }
        cols, ph = ", ".join(EDGE_COLS), _ph(EDGE_COLS)
        return (f"INSERT INTO edges ({cols}) VALUES ({ph}) ON CONFLICT DO NOTHING",
                tuple(row[c] for c in EDGE_COLS))

    @staticmethod
    def _vec_stmt(node_id: str, embedding: Sequence[float]) -> tuple[str, tuple]:
        import sqlite_vec
        # vec0 не поддерживает ON CONFLICT DO UPDATE, используем INSERT OR REPLACE
        return ("""INSERT OR REPLACE INTO vec_nodes(node_id, embedding) VALUES (?,?)""",
                (node_id, sqlite_vec.serialize_float32([float(x) for x in embedding])))

    # ------------------------------------------------------------ nodes
    async def insert_node(self, data: dict, *, wait: bool = True) -> str:
        stmt, nid = self.node_stmt(data)
        await self._q.execute(stmt[0], stmt[1], wait=wait)
        return nid

    async def get_node(self, node_id: str, *,
                       include_deleted: bool = False) -> Optional[dict]:
        if include_deleted:
            sql, params = "SELECT * FROM nodes WHERE id=?", (node_id,)
        else:
            sql = ("SELECT * FROM nodes WHERE id=? "
                   "AND (valid_until IS NULL OR valid_until > ?)")
            params = (node_id, _now())
        async with self._read.execute(sql, params) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def fetch_nodes(self, ids: Sequence[str]) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        async with self._read.execute(
                f"SELECT * FROM nodes WHERE id IN ({_ph(ids)})", ids) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def fetch_nodes_by_session(self, session_id: str,
                                     since_ts: Optional[int] = None) -> list[dict]:
        sql = "SELECT * FROM nodes WHERE session_id=?"
        params: list[Any] = [session_id]
        if since_ts is not None:
            sql += " AND event_time >= ?"
            params.append(since_ts)
        sql += " ORDER BY event_time, rowid"
        async with self._read.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def update_node_fields(self, node_id: str, fields: dict, *,
                                 wait: bool = True) -> Optional[int]:
        bad = set(fields) - NODE_UPDATE_ALLOW
        if bad:
            raise ValueError(f"fields not updatable: {sorted(bad)}")
        sets, params = [], []
        for key, val in fields.items():
            if key == "metadata":
                val = json.dumps(val, ensure_ascii=False)
            sets.append(f"{key}=?")
            params.append(val)
        params.append(node_id)
        return await self._q.execute(
            f"UPDATE nodes SET {', '.join(sets)} WHERE id=?", tuple(params), wait=wait)

    async def access_nodes(self, ids, now=None):
        """Reinforce ONLY the top-3 ranked hits, not the whole result page."""
        ids = list(ids)[:3]
        if not ids:
            return 0
        now = now or _now()
        ph = ",".join("?" for _ in ids)
        return await self._q.execute(
            f"UPDATE nodes SET last_accessed=?, access_count=access_count+1, "
            f"confidence=MIN(1.0, confidence * (1 + LN(access_count + 2) / 10)) "
            f"WHERE id IN ({ph}) AND valid_until IS NULL", (now, *ids), wait=True)

    async def soft_delete_nodes(self, ids: Sequence[str], *,
                                wait: bool = True) -> Optional[int]:
        ids = list(ids)
        if not ids:
            return 0
        return await self._q.execute(
            f"UPDATE nodes SET valid_until=? WHERE id IN ({_ph(ids)}) "
            "AND valid_until IS NULL", (_now(), *ids), wait=wait)

    async def hard_delete_nodes(self, ids: Sequence[str], *,
                                wait: bool = True) -> Optional[int]:
        ids = list(ids)
        if not ids:
            return 0
        ph = _ph(ids)
        return await self._q.transaction([
            (f"DELETE FROM vec_nodes WHERE node_id IN ({ph})", ids),
            (f"DELETE FROM node_entities WHERE node_id IN ({ph})", ids),
            (f"DELETE FROM edges WHERE source_id IN ({ph}) OR target_id IN ({ph})", ids),
            (f"DELETE FROM nodes WHERE id IN ({ph})", ids),
        ], wait=wait)

    async def nodes_without_vectors(self, limit: int = 200) -> list[str]:
        async with self._read.execute(
                "SELECT id FROM nodes WHERE needs_embedding=1 "
                "ORDER BY rowid LIMIT ?", (limit,)) as cur:
            rows = await cur.fetchall()
        return [r["id"] for r in rows]

    async def insert_embedding_and_clear_flag(self, node_id: str,
                                              emb: Sequence[float], *,
                                              wait: bool = True) -> Optional[int]:
        """One transaction: vector upsert + flag clear (no orphan vectors)."""
        return await self._q.transaction([
            self._vec_stmt(node_id, emb),
            ("UPDATE nodes SET needs_embedding=0 WHERE id=?", (node_id,)),
        ], wait=wait)

    # ------------------------------------------------------------ edges
    async def insert_edge(self, data: dict, *, wait: bool = True) -> Optional[str]:
        stmt = self.edge_stmt(
            data["source_id"], data["target_id"], data["relation_type"],
            confidence=data.get("confidence", 1.0),
            weight=data.get("weight", 1.0),
            predicate=data.get("predicate"),
            evidence=data.get("evidence"))
        n = await self._q.execute(stmt[0], stmt[1], wait=wait)
        return stmt[1][0] if (n or 0) > 0 else None

    async def fetch_edges_for_nodes(self, ids: Sequence[str]) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        ph = _ph(ids)
        sql = f"""SELECT * FROM edges
                  WHERE (source_id IN ({ph}) OR target_id IN ({ph}))
                    AND (valid_until IS NULL OR valid_until > ?)"""
        async with self._read.execute(sql, (*ids, *ids, _now())) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ entities
    async def upsert_entity(self, *, name_norm: str, canonical: str,
                            etype: str = "concept", summary=None,
                            wait: bool = True) -> str:
        """Never clobber aliases/attributes/canonical; upgrade concept->specific."""
        sql = """
            INSERT INTO entities(id, name_norm, canonical_name, type, aliases,
                                 summary, first_seen, last_updated,
                                 mention_count, attributes)
            VALUES (?,?,?,?,'[]',?, ?,?,1,'{}')
            ON CONFLICT(name_norm) DO UPDATE SET
                mention_count = mention_count + 1,
                last_updated  = excluded.last_updated,
                type = CASE WHEN entities.type='concept'
                            THEN excluded.type ELSE entities.type END,
                summary = COALESCE(excluded.summary, entities.summary)"""
        now = _now()
        await self._q.execute(sql, (uuid.uuid4().hex, name_norm, canonical,
                                    etype, summary, now, now), wait=wait)
        async with self._db.get_read_connection().execute(
                "SELECT id FROM entities WHERE name_norm=?", (name_norm,)) as cur:
            row = await cur.fetchone()
        return row["id"]

    async def merge_alias(self, entity_id: str, alias: str,
                          wait: bool = True) -> None:
        """Read-merge-write aliases in Python; never overwrite the list."""
        async with self._db.get_read_connection().execute(
                "SELECT aliases, canonical_name FROM entities WHERE id=?",
                (entity_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return
        aliases = json.loads(row["aliases"] or "[]")
        if alias == row["canonical_name"] or alias in aliases:
            return
        aliases.append(alias)
        await self._q.execute("UPDATE entities SET aliases=? WHERE id=?",
                              (json.dumps(aliases, ensure_ascii=False), entity_id),
                              wait=wait)

    def link_entities_stmts(self, node_id: str, entity_ids) -> list[tuple[str, tuple]]:
        return [("INSERT OR IGNORE INTO node_entities(node_id, entity_id) "
                 "VALUES (?,?)", (node_id, eid)) for eid in entity_ids]

    async def last_episode_id(self, session_id: str):
        async with self._db.get_read_connection().execute(
                "SELECT id FROM nodes WHERE type='episodic' AND session_id=? "
                "ORDER BY event_time DESC, rowid DESC LIMIT 1",
                (session_id,)) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None

    async def link_node_entity(self, node_id: str, entity_id: str, *,
                               wait: bool = True) -> Optional[int]:
        return await self._q.execute(
            "INSERT OR IGNORE INTO node_entities(node_id, entity_id) VALUES (?,?)",
            (node_id, entity_id), wait=wait)

    async def link_nodes_entities(self, pairs: Sequence[tuple[str, str]], *,
                                  wait: bool = True) -> Optional[int]:
        return await self._q.execute(
            "INSERT OR IGNORE INTO node_entities(node_id, entity_id) VALUES (?,?)",
            [list(p) for p in pairs], wait=wait, many=True)

    async def get_entities_for_nodes(self, node_ids: Sequence[str]) -> dict[str, list[dict]]:
        node_ids = list(node_ids)
        out: dict[str, list[dict]] = {nid: [] for nid in node_ids}
        if not node_ids:
            return out
        sql = f"""SELECT ne.node_id AS node_id, e.* FROM node_entities ne
                  JOIN entities e ON e.id = ne.entity_id
                  WHERE ne.node_id IN ({_ph(node_ids)})"""
        async with self._read.execute(sql, node_ids) as cur:
            rows = await cur.fetchall()
        for r in rows:
            d = dict(r)
            out.setdefault(d.pop("node_id"), []).append(d)
        return out

    async def get_entity(self, entity_id: str) -> Optional[dict]:
        async with self._read.execute(
                "SELECT * FROM entities WHERE id=?", (entity_id,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def get_entity_by_norm(self, norm: str) -> Optional[dict]:
        async with self._read.execute(
                "SELECT * FROM entities WHERE name_norm=?", (norm,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def fetch_entities(self, ids: Sequence[str]) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        async with self._read.execute(
                f"SELECT * FROM entities WHERE id IN ({_ph(ids)})", ids) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def fetch_nodes_for_entity(self, entity_id: str) -> list[dict]:
        async with self._read.execute("""
                SELECT n.* FROM nodes n JOIN node_entities ne ON ne.node_id = n.id
                WHERE ne.entity_id = ? AND n.valid_until IS NULL""",
                (entity_id,)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def fetch_episodes(self, *, entity_id: Optional[str] = None,
                             time_from: Optional[int] = None,
                             time_to: Optional[int] = None,
                             limit: int = 20) -> list[dict]:
        sql = "SELECT n.* FROM nodes n WHERE n.type='episodic' AND n.valid_until IS NULL"
        params: list[Any] = []
        if entity_id:
            sql += (" AND EXISTS(SELECT 1 FROM node_entities ne "
                    "WHERE ne.node_id=n.id AND ne.entity_id=?)")
            params.append(entity_id)
        if time_from is not None:
            sql += " AND n.event_time >= ?"
            params.append(time_from)
        if time_to is not None:
            sql += " AND n.event_time <= ?"
            params.append(time_to)
        sql += " ORDER BY n.event_time DESC, n.rowid DESC LIMIT ?"
        params.append(limit)
        async with self._read.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ composite ops
    async def create_fact(self, *, content: str, node_type: str,
                          confidence: float = 0.9,
                          decay_rate: Optional[float] = None,
                          embedding: Optional[Sequence[float]] = None,
                          session_id: Optional[str] = None,
                          event_time: Optional[int] = None,
                          entity_ids: Sequence[str] = (),
                          source_episode_ids: Sequence[str] = (),
                          metadata: Optional[dict] = None,
                          wait: bool = True) -> str:
        """Node + vector + entity links + derived_from edges in ONE transaction."""
        stmt, nid = self.node_stmt({
            "type": node_type, "content": content, "confidence": confidence,
            "decay_rate": decay_rate, "session_id": session_id,
            "event_time": event_time, "needs_embedding": embedding is None,
            "metadata": metadata,
        })
        stmts = [stmt]
        if embedding is not None:
            stmts.append(self._vec_stmt(nid, embedding))
        stmts += [("INSERT OR IGNORE INTO node_entities(node_id, entity_id) VALUES (?,?)",
                   (nid, eid)) for eid in entity_ids]
        stmts += [self.edge_stmt(nid, ep, "derived_from", confidence=confidence)
                  for ep in source_episode_ids]
        await self._q.transaction(stmts, wait=wait)
        return nid

    async def supersede_fact(self, old_id, *, content, confidence=0.9,
                             embedding=None, source_episode_ids=(),
                             entity_ids=(), wait: bool = True) -> str:
        now = _now()
        stmt, nid = self.node_stmt({
            "type": "semantic", "content": content, "confidence": confidence,
            "event_time": now, "needs_embedding": embedding is None,
        })
        stmts = [
            ("UPDATE nodes SET valid_until=? WHERE id=? AND valid_until IS NULL",
             (now, old_id)),
            stmt,
            self.edge_stmt(nid, old_id, "supersedes", predicate="replaces"),
        ]
        stmts += self.link_entities_stmts(nid, entity_ids)   # entities on NEW node
        for ep in source_episode_ids:
            stmts.append(self.edge_stmt(nid, ep, "derived_from", predicate="from"))
        if embedding is not None:
            stmts.append(self._vec_stmt(nid, embedding))
            stmts.append(("UPDATE nodes SET needs_embedding=0 WHERE id=?", (nid,)))
        await self._q.transaction(stmts, wait=wait)
        return nid

    # ------------------------------------------------------------ sessions
    async def mark_session_consolidated(self, session_id: str,
                                        ts: Optional[int] = None, *,
                                        wait: bool = True) -> Optional[int]:
        return await self._q.execute(
            """INSERT INTO sessions(session_id, consolidated_at) VALUES (?,?)
               ON CONFLICT(session_id) DO UPDATE SET consolidated_at=excluded.consolidated_at""",
            (session_id, ts or _now()), wait=wait)

    async def session_consolidated_at(self, session_id: str) -> Optional[int]:
        async with self._read.execute(
                "SELECT consolidated_at FROM sessions WHERE session_id=?",
                (session_id,)) as cur:
            row = await cur.fetchone()
        return row["consolidated_at"] if row else None

    # ------------------------------------------------------------ causal checks
    async def mark_causal_checked(self, cause_id: str, effect_id: str,
                                  ts: Optional[int] = None, *,
                                  wait: bool = True) -> Optional[int]:
        return await self._q.execute(
            "INSERT OR IGNORE INTO causal_checked(cause_id, effect_id, checked_at) "
            "VALUES (?,?,?)", (cause_id, effect_id, ts or _now()), wait=wait)

    async def is_causal_checked(self, cause_id: str, effect_id: str) -> bool:
        async with self._read.execute(
                "SELECT 1 FROM causal_checked WHERE cause_id=? AND effect_id=?",
                (cause_id, effect_id)) as cur:
            row = await cur.fetchone()
        return row is not None

    # ------------------------------------------------------------ meta
    async def meta_get(self, key: str) -> Optional[str]:
        async with self._read.execute(
                "SELECT value FROM meta WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
        return row["value"] if row else None

    async def meta_set(self, key: str, value: str, *,
                       wait: bool = True) -> Optional[int]:
        return await self._q.execute(
            "INSERT INTO meta(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value), wait=wait)

    # ------------------------------------------------------------ add methods
    async def fetch_entities_for_node(self, node_id: str) -> list[dict]:
        """Single-node variant of get_entities_for_nodes."""
        result = await self.get_entities_for_nodes([node_id])
        return result.get(node_id, [])

    async def fetch_timeline(self, *, entity_id: Optional[str] = None,
                             time_from: Optional[int] = None,
                             time_to: Optional[int] = None,
                             limit: int = 20) -> list[dict]:
        """Alias for fetch_episodes with chronological order (ASC)."""
        sql = "SELECT n.* FROM nodes n WHERE n.type='episodic' AND n.valid_until IS NULL"
        params: list[Any] = []
        if entity_id:
            sql += (" AND EXISTS(SELECT 1 FROM node_entities ne "
                    "WHERE ne.node_id=n.id AND ne.entity_id=?)")
            params.append(entity_id)
        if time_from is not None:
            sql += " AND n.event_time >= ?"
            params.append(time_from)
        if time_to is not None:
            sql += " AND n.event_time <= ?"
            params.append(time_to)
        sql += " ORDER BY n.event_time ASC, n.rowid ASC LIMIT ?"  # ← ASC вместо DESC
        params.append(limit)
        async with self._read.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def stats(self) -> dict:
        """Graph metrics: node/edge/entity counts, DB size."""
        async with self._read.execute("""
                SELECT type, COUNT(*) as cnt FROM nodes
                WHERE valid_until IS NULL GROUP BY type""") as cur:
            nodes_rows = await cur.fetchall()
        nodes_by_type = {r["type"]: r["cnt"] for r in nodes_rows}

        async with self._read.execute("""
                SELECT relation_type, COUNT(*) as cnt FROM edges
                WHERE valid_until IS NULL GROUP BY relation_type""") as cur:
            edges_rows = await cur.fetchall()
        edges_by_type = {r["relation_type"]: r["cnt"] for r in edges_rows}

        async with self._read.execute("""
                SELECT type, COUNT(*) as cnt FROM entities
                GROUP BY type""") as cur:
            ents_rows = await cur.fetchall()
        entities_by_type = {r["type"]: r["cnt"] for r in ents_rows}

        async with self._read.execute("""
                SELECT COUNT(*) FROM nodes n
                WHERE n.valid_until IS NULL AND n.type != 'episodic'
                  AND NOT EXISTS (SELECT 1 FROM edges e
                                  WHERE e.source_id=n.id OR e.target_id=n.id)
                  AND NOT EXISTS (SELECT 1 FROM node_entities ne WHERE ne.node_id=n.id)
            """) as cur:
            orphans_row = await cur.fetchone()
        orphans = orphans_row[0] if orphans_row else 0

        async with self._read.execute("""
                SELECT COUNT(*) FROM nodes WHERE needs_embedding=1
                  AND valid_until IS NULL""") as cur:
            vec_row = await cur.fetchone()
        vec_available = vec_row[0] if vec_row else 0

        db_size_mb = 0.0
        if self._db.db_path.exists():
            db_size_mb = round(self._db.db_path.stat().st_size / (1024 * 1024), 2)

        return {
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "entities_by_type": entities_by_type,
            "orphans": orphans,
            "vec_available": vec_available,
            "db_size_mb": db_size_mb,
        }

    async def provenance(self, node_id: str) -> tuple[list[dict], list[dict], list[dict]]:
        """Source episodes, supersedes chain, linked entities."""
        # Soures (derived_from edges)
        async with self._read.execute("""
                SELECT n.* FROM nodes n
                JOIN edges e ON e.target_id=n.id
                WHERE e.source_id=? AND e.relation_type='derived_from'
                  AND n.type='episodic'
                ORDER BY n.event_time""", (node_id,)) as cur:
            sources = [dict(r) for r in await cur.fetchall()]

        # Supersedes chain (walk backwards)
        superseded = []
        current = node_id
        visited = set()
        while current and current not in visited:
            visited.add(current)
            async with self._read.execute("""
                    SELECT target_id FROM edges
                    WHERE source_id=? AND relation_type='supersedes'
                    LIMIT 1""", (current,)) as cur:
                row = await cur.fetchone()
            if row:
                old_id = row["target_id"]
                old_node = await self.get_node(old_id, include_deleted=True)
                if old_node:
                    superseded.append(old_node)
                current = old_id
            else:
                break

        # Entities
        entities = await self.fetch_entities_for_node(node_id)
        return sources, superseded, entities

    async def insert_entity_embedding(self, entity_id: str, emb: Sequence[float], *,
                                      wait: bool = True) -> Optional[int]:
        """Store embedding vector for entity."""
        import sqlite_vec
        vec_data = sqlite_vec.serialize_float32([float(x) for x in emb])
        return await self._q.execute(
            "INSERT OR REPLACE INTO vec_entities(entity_id, embedding) VALUES (?,?)",
            (entity_id, vec_data), wait=wait)
