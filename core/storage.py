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
        sql += " ORDER BY event_time, rid"
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

    async def access_nodes(self, node_ids: Sequence[str], *,
                           wait: bool = True) -> Optional[int]:
        """Single UPDATE with ln() bonus; Python fallback without math functions."""
        ids = list(node_ids)
        if not ids:
            return 0
        now = _now()
        if self._db.has_math_functions:
            return await self._q.execute(f"""
                UPDATE nodes SET access_count = access_count + 1, last_accessed = ?,
                  confidence = MIN(1.0, confidence * (1 + LN(access_count + 2) / 10))
                WHERE id IN ({_ph(ids)})""", (now, *ids), wait=wait)
        rows = await self.fetch_nodes(ids)
        stmts = [(
            "UPDATE nodes SET access_count=?, last_accessed=?, confidence=? WHERE id=?",
            (r["access_count"] + 1, now,
             min(1.0, r["confidence"] * (1 + math.log(r["access_count"] + 2) / 10)),
             r["id"])) for r in rows]
        return await self._q.transaction(stmts, wait=wait)

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
                "ORDER BY rid LIMIT ?", (limit,)) as cur:
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
    async def upsert_entity(self, data: dict, *, wait: bool = True) -> str:
        """Deterministic id from name_norm; atomic mention_count bump."""
        now = _now()
        norm = data["name_norm"]
        row = {
            "id": data.get("id") or ("e_" + uuid.uuid5(uuid.NAMESPACE_URL, norm).hex[:16]),
            "name_norm": norm,
            "canonical_name": data.get("canonical_name") or norm,
            "type": data["type"],
            "aliases": json.dumps(data.get("aliases") or [], ensure_ascii=False),
            "summary": data.get("summary"),
            "first_seen": data.get("first_seen", now),
            "last_updated": now,
            "mention_count": 1,
            "attributes": json.dumps(data.get("attributes") or {}, ensure_ascii=False),
        }
        cols, ph = ", ".join(ENTITY_COLS), _ph(ENTITY_COLS)
        await self._q.execute(f"""
            INSERT INTO entities ({cols}) VALUES ({ph})
            ON CONFLICT(name_norm) DO UPDATE SET
              mention_count  = mention_count + 1,
              last_updated   = excluded.last_updated,
              canonical_name = COALESCE(excluded.canonical_name, entities.canonical_name),
              summary        = COALESCE(excluded.summary, entities.summary),
              aliases        = excluded.aliases,
              attributes     = excluded.attributes""",
            tuple(row[c] for c in ENTITY_COLS), wait=wait)
        return row["id"]

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
        sql += " ORDER BY n.event_time DESC, n.rid DESC LIMIT ?"
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

    async def supersede_fact(self, old_id: str, *, content: str,
                             confidence: float = 0.9,
                             embedding: Optional[Sequence[float]] = None,
                             source_episode_ids: Sequence[str] = (),
                             wait: bool = True) -> str:
        """New node + copied entity links + supersedes edge + old invalidated."""
        old = await self.get_node(old_id)
        if old is None:
            raise KeyError(f"node not found: {old_id}")
        now = _now()
        stmt, nid = self.node_stmt({
            "type": old["type"], "content": content, "confidence": confidence,
            "session_id": old["session_id"], "event_time": now,
            "needs_embedding": embedding is None,
        })
        stmts = [stmt]
        if embedding is not None:
            stmts.append(self._vec_stmt(nid, embedding))
        stmts.append(("INSERT INTO node_entities(node_id, entity_id) "
                      "SELECT ?, entity_id FROM node_entities WHERE node_id=?",
                      (nid, old_id)))
        stmts.append(self.edge_stmt(nid, old_id, "supersedes", confidence=0.95))
        stmts.append(("UPDATE nodes SET valid_until=?, confidence=0.3 WHERE id=?",
                      (now, old_id)))
        stmts += [self.edge_stmt(nid, ep, "derived_from", confidence=confidence)
                  for ep in source_episode_ids]
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
