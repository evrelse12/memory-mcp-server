# core/database.py
"""Dual-connection SQLite DB (WAL). Write connection is owned by WriteQueue."""
from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

import aiosqlite
import sqlite_vec

from config.settings import settings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


class Database:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.db_file
        self._write: aiosqlite.Connection | None = None
        self._read: aiosqlite.Connection | None = None
        self.has_math_functions: bool = False

    def get_read_connection(self) -> aiosqlite.Connection:
        assert self._read is not None, "Database not initialized"
        return self._read

    def take_write_connection(self) -> aiosqlite.Connection:
        assert self._write is not None, "Database not initialized"
        return self._write

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write = await aiosqlite.connect(self.db_path, isolation_level=None)
        self._write.row_factory = aiosqlite.Row
        self._read = await aiosqlite.connect(
            f"file:{self.db_path}?mode=ro", uri=True)
        self._read.row_factory = aiosqlite.Row
        await self._load_vec()
        await self._guard_schema_version()
        await self._create_schema()
        await self._apply_pragmas()
        logger.info("Database initialized: %s (schema v%d)",
                    self.db_path, SCHEMA_VERSION)

    async def _load_vec(self) -> None:
        """Load sqlite-vec extension into BOTH connections."""
        ext_dir = Path(sqlite_vec.__file__).parent
        ext_name = "vec0" if platform.system() != "Windows" else "vec0.dll"
        ext_path = ext_dir / ext_name
        await self._write.enable_load_extension(True)
        await self._read.enable_load_extension(True)
        await self._write.load_extension(str(ext_path))
        await self._read.load_extension(str(ext_path))
        logger.info("sqlite-vec loaded into both connections: %s", ext_path)

    async def _guard_schema_version(self) -> None:
        """Fail fast on an old DB instead of silently opening incompatible tables."""
        async with self._write.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes'") as cur:
            has_nodes = await cur.fetchone() is not None
        async with self._write.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
        version = row[0] if row else 0
        if has_nodes and version != SCHEMA_VERSION:
            await self._write.close()
            await self._read.close()
            raise RuntimeError(
                f"incompatible schema v{version} (expected v{SCHEMA_VERSION}); "
                f"run scripts/migrate_v1_to_v2.py before starting")

    async def _create_schema(self) -> None:
        w = self._write
        async with w.execute(
                "SELECT name FROM sqlite_master WHERE type='table'") as cur:
            existing = {r[0] for r in await cur.fetchall()}

        if "nodes" not in existing:
            async with w.execute("""
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
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
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """):
                pass

            async with w.execute("""
                CREATE TABLE edges (
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
                )
            """):
                pass

            async with w.execute("""
                CREATE TABLE entities (
                    id TEXT PRIMARY KEY,
                    name_norm TEXT NOT NULL UNIQUE,
                    canonical_name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    aliases TEXT NOT NULL DEFAULT '[]',
                    summary TEXT,
                    first_seen INTEGER NOT NULL,
                    last_updated INTEGER NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    attributes TEXT NOT NULL DEFAULT '{}'
                )
            """):
                pass

            async with w.execute("""
                CREATE TABLE node_entities (
                    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    PRIMARY KEY (node_id, entity_id)
                )
            """):
                pass

            async with w.execute("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    consolidated_at INTEGER
                )
            """):
                pass

            async with w.execute("""
                CREATE TABLE causal_checked (
                    cause_id TEXT NOT NULL,
                    effect_id TEXT NOT NULL,
                    checked_at INTEGER NOT NULL,
                    PRIMARY KEY (cause_id, effect_id)
                )
            """):
                pass

            async with w.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """):
                pass

        # FTS5 — virtual table, recreated if schema changes; stable rowid for joins
        async with w.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
            USING fts5(content, content='nodes', content_rowid='rowid',
                       tokenize='porter unicode61')
        """):
            pass

        # Triggers to keep FTS5 in sync with nodes
        async with w.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
            END
        """):
            pass
        async with w.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content)
                VALUES ('delete', OLD.rowid, OLD.content);
            END
        """):
            pass
        async with w.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content)
                VALUES ('delete', OLD.rowid, OLD.content);
                INSERT INTO nodes_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
            END
        """):
            pass

        # vec0 virtual table (does NOT support UPSERT — use INSERT OR REPLACE)
        async with w.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes
            USING vec0(node_id TEXT PRIMARY KEY, embedding float[256])
        """):
            pass

        # vec0 virtual table for entities
        async with w.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_entities
            USING vec0(entity_id TEXT PRIMARY KEY, embedding float[256])
        """):
            pass

        # Indexes (idempotent via IF NOT EXISTS)
        for idx in (
            "CREATE INDEX IF NOT EXISTS idx_nodes_type_valid ON nodes(type, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id, event_time)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_event_time ON nodes(event_time)",
            "CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation_type, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
        ):
            async with w.execute(idx):
                pass

        async with w.execute(f"PRAGMA user_version = {SCHEMA_VERSION}"):
            pass

    async def _apply_pragmas(self) -> None:
        w = self._write
        # Check whether sqlite was built with math functions (needed for decay SQL)
        async with w.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                "AND name='nodes'") as cur:
            await cur.fetchone()
        try:
            async with w.execute("SELECT EXP(1.0) AS v") as cur:
                await cur.fetchone()
            self.has_math_functions = True
        except aiosqlite.OperationalError:
            self.has_math_functions = False

        for p in ("PRAGMA journal_mode=WAL",
                  "PRAGMA synchronous=NORMAL",
                  "PRAGMA foreign_keys=ON",
                  "PRAGMA temp_store=MEMORY",
                  "PRAGMA cache_size=-20000",
                  "PRAGMA busy_timeout=5000"):
            async with w.execute(p) as cur:
                await cur.fetchone()

        logger.info("Pragmas applied (math_functions=%s)", self.has_math_functions)

    async def close(self) -> None:
        if self._write:
            try:
                await self._write.close()
            except Exception:
                pass
        if self._read:
            try:
                await self._read.close()
            except Exception:
                pass
