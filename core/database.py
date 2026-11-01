# core/database.py
"""SQLite schema v2: stable rowid for FTS, sqlite-vec from pip, meta/version checks."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import aiosqlite

from config.settings import settings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


class Database:
    """Owns the two connections. Write connection is handed to WriteQueue only."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.db_file)
        self._write: Optional[aiosqlite.Connection] = None
        self._read: Optional[aiosqlite.Connection] = None
        self.has_math_functions = False

    # ------------------------------------------------------------ lifecycle
    async def initialize(self) -> None:
        parent = self.db_path.parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)

        self._write = await aiosqlite.connect(self.db_path, isolation_level=None)
        self._write.row_factory = aiosqlite.Row
        for stmt in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA foreign_keys=ON",
            "PRAGMA busy_timeout=5000",
            "PRAGMA wal_autocheckpoint=1000",
            "PRAGMA cache_size=-64000",
        ):
            await self._write.execute(stmt)

        self._read = await aiosqlite.connect(self.db_path)
        self._read.row_factory = aiosqlite.Row
        await self._read.execute("PRAGMA query_only=ON")
        await self._read.execute("PRAGMA busy_timeout=5000")

        self.has_math_functions = await self._probe(self._write, "SELECT ln(2.0)")
        if not self.has_math_functions:
            logger.warning("SQLite built without math functions; access bonus computed in Python")

        await self._load_sqlite_vec()
        await self._create_schema()
        await self._check_meta()
        logger.info("Database initialized: %s (schema v%d)", self.db_path, SCHEMA_VERSION)

    async def close(self) -> None:
        if self._write:
            try:
                await self._write.execute("PRAGMA optimize")
            except Exception:
                pass
            await self._write.close()
            self._write = None
        if self._read:
            await self._read.close()
            self._read = None

    # ------------------------------------------------------------ helpers
    @staticmethod
    async def _probe(conn: aiosqlite.Connection, sql: str) -> bool:
        try:
            await conn.execute(sql)
            return True
        except Exception:
            return False

    async def _load_sqlite_vec(self) -> None:
        """sqlite-vec from pip, required in BOTH connections or fail loudly."""
        try:
            import sqlite_vec
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip package 'sqlite-vec' is required: pip install sqlite-vec") from e
        path = sqlite_vec.loadable_path()
        for name, conn in (("write", self._write), ("read", self._read)):
            await conn.enable_load_extension(True)
            try:
                await conn.load_extension(path)
            except Exception as e:
                raise RuntimeError(f"sqlite-vec failed to load into {name} connection: {e}") from e
            await conn.enable_load_extension(False)
        logger.info("sqlite-vec loaded into both connections: %s", path)

    async def _create_schema(self) -> None:
        dim = settings.embedding_dim
        w = self._write
        async with w.cursor() as cur:
            await cur.execute(f"""
                CREATE TABLE IF NOT EXISTS nodes (
                    rid INTEGER PRIMARY KEY,
                    id TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL CHECK(type IN
                        ('episodic','semantic','procedural','opinion')),
                    content TEXT NOT NULL,
                    session_id TEXT,
                    event_time INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_accessed INTEGER,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    decay_rate REAL NOT NULL DEFAULT 0.1,
                    needs_embedding INTEGER NOT NULL DEFAULT 0,
                    valid_until INTEGER,
                    metadata TEXT NOT NULL DEFAULT '{{}}'
                )""")
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL CHECK(relation_type IN
                        ('temporal','causal','entity','derived_from','supersedes')),
                    predicate TEXT,
                    weight REAL NOT NULL DEFAULT 1.0,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    valid_from INTEGER NOT NULL,
                    valid_until INTEGER,
                    evidence TEXT NOT NULL DEFAULT '[]',
                    created_at INTEGER NOT NULL
                )""")
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name_norm TEXT UNIQUE NOT NULL,
                    canonical_name TEXT NOT NULL,
                    type TEXT NOT NULL CHECK(type IN
                        ('person','project','organization','place','concept','tool')),
                    aliases TEXT NOT NULL DEFAULT '[]',
                    summary TEXT,
                    first_seen INTEGER NOT NULL,
                    last_updated INTEGER NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    attributes TEXT NOT NULL DEFAULT '{}'
                )""")
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS node_entities (
                    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    PRIMARY KEY (node_id, entity_id)
                )""")
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    consolidated_at INTEGER NOT NULL
                )""")
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS causal_checked (
                    cause_id TEXT NOT NULL,
                    effect_id TEXT NOT NULL,
                    checked_at INTEGER NOT NULL,
                    PRIMARY KEY (cause_id, effect_id)
                )""")
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )""")
            await cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    content, content=nodes, content_rowid=rid, tokenize='unicode61'
                )""")
            await cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(
                    node_id TEXT PRIMARY KEY, embedding float[{dim}]
                )""")
            await cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_entities USING vec0(
                    entity_id TEXT PRIMARY KEY, embedding float[{dim}]
                )""")

            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_nodes_session_time ON nodes(session_id, event_time)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_valid_type ON nodes(valid_until, type)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_needs_emb ON nodes(needs_embedding) WHERE needs_embedding=1",
                "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(source_id, relation_type, valid_until)",
                "CREATE INDEX IF NOT EXISTS idx_edges_tgt ON edges(target_id, relation_type, valid_until)",
                "CREATE INDEX IF NOT EXISTS idx_ne_entity ON node_entities(entity_id)",
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_active_pair
                   ON edges(source_id, target_id, relation_type) WHERE valid_until IS NULL""",
                # FTS sync triggers; update trigger fires ONLY on content change
                """CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                     INSERT INTO nodes_fts(rowid, content) VALUES (new.rid, new.content);
                   END""",
                """CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                     INSERT INTO nodes_fts(nodes_fts, rowid, content)
                     VALUES ('delete', old.rid, old.content);
                   END""",
                """CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE OF content ON nodes BEGIN
                     INSERT INTO nodes_fts(nodes_fts, rowid, content)
                     VALUES ('delete', old.rid, old.content);
                     INSERT INTO nodes_fts(rowid, content) VALUES (new.rid, new.content);
                   END""",
            ):
                await cur.execute(stmt)

            await cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await w.execute("COMMIT") if False else None  # isolation_level=None: autocommit
        logger.info("Schema v%d created/verified", SCHEMA_VERSION)

    async def _check_meta(self) -> None:
        w = self._write
        async with w.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
            version = row[0]
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version {version} != {SCHEMA_VERSION}. "
                "Run scripts/migrate_v1_to_v2.py or start with a fresh database."
            )
        for key, value in (
            ("embedding_dim", str(settings.embedding_dim)),
            ("embedding_model", settings.embedding_model),
        ):
            async with w.execute("SELECT value FROM meta WHERE key=?", (key,)) as cur:
                row = await cur.fetchone()
            if row is None:
                await w.execute(
                    "INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO NOTHING",
                    (key, value),
                )
            elif row["value"] != value:
                raise RuntimeError(
                    f"Stored {key}={row['value']} but settings say {value}. "
                    "Re-embed or change settings; mismatched vectors corrupt search."
                )

    # ------------------------------------------------------------ accessors
    def get_read_connection(self) -> aiosqlite.Connection:
        if self._read is None:
            raise RuntimeError("Database not initialized")
        return self._read

    def take_write_connection(self) -> aiosqlite.Connection:
        """Handed to WriteQueue exactly once; not part of the public API."""
        if self._write is None:
            raise RuntimeError("Database not initialized")
        return self._write
