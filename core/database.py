# core/database.py
"""Dual-connection SQLite DB (WAL). Write connection is owned by WriteQueue."""
from __future__ import annotations

import logging
import platform
from pathlib import Path

import aiosqlite
import sqlite_vec

from config.settings import settings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3


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
        await self._check_embedding_meta()   # AFTER _apply_pragmas, OUTSIDE it
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
            # rid is an explicit autoincrement PK; FTS5 rowid joins against it
            # and VACUUM is guaranteed to preserve it because it is the PK.
            async with w.execute("""
                CREATE TABLE nodes (
                    rid INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
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

        # FTS5 virtual table: rowid joins against nodes.rid
        async with w.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
            USING fts5(content, content='nodes', content_rowid='rid',
                       tokenize='porter unicode61')
        """):
            pass

        async with w.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, content) VALUES (NEW.rid, NEW.content);
            END
        """):
            pass
        async with w.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content)
                VALUES ('delete', OLD.rid, OLD.content);
            END
        """):
            pass
        # Fire ONLY on content change — other UPDATEs must not churn FTS.
        async with w.execute("""
            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE OF content ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content)
                VALUES ('delete', OLD.rid, OLD.content);
                INSERT INTO nodes_fts(rowid, content) VALUES (NEW.rid, NEW.content);
            END
        """):
            pass

        # vec0 virtual tables; dimension comes from settings, not hardcoded
        dim = settings.embedding_dim
        async with w.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes
            USING vec0(node_id TEXT PRIMARY KEY, embedding float[{dim}])
        """):
            pass
        async with w.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_entities
            USING vec0(entity_id TEXT PRIMARY KEY, embedding float[{dim}])
        """):
            pass

        # Indexes (idempotent)
        for idx in (
            "CREATE INDEX IF NOT EXISTS idx_nodes_type_valid "
            "ON nodes(type, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_session "
            "ON nodes(session_id, event_time)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_event_time "
            "ON nodes(event_time)",
            "CREATE INDEX IF NOT EXISTS idx_edges_relation "
            "ON edges(relation_type, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_edges_source "
            "ON edges(source_id, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_edges_target "
            "ON edges(target_id, valid_until)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type "
            "ON entities(type)",
            "CREATE INDEX IF NOT EXISTS idx_node_entities_entity "
            "ON node_entities(entity_id)",
            # unique triple for ON CONFLICT DO NOTHING to actually dedupe edges
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_triple "
            "ON edges(source_id, target_id, relation_type)",
        ):
            async with w.execute(idx):
                pass

        async with w.execute(f"PRAGMA user_version = {SCHEMA_VERSION}"):
            pass

    async def _apply_pragmas(self) -> None:
        w = self._write
        # Probe math functions (for decay SQL).
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

    async def _check_embedding_meta(self) -> None:
        """Fail fast when embedding model or dim changed since DB creation."""
        async with self._write.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='meta'") as cur:
            if await cur.fetchone() is None:
                return
        async with self._write.execute(
                "SELECT key, value FROM meta "
                "WHERE key IN ('embedding_dim','embedding_model')") as cur:
            stored = {r["key"]: r["value"] for r in await cur.fetchall()}

        dim = stored.get("embedding_dim")
        if dim is not None and int(dim) != settings.embedding_dim:
            await self._close_both()
            raise RuntimeError(
                f"embedding_dim mismatch: DB has {dim}, config wants "
                f"{settings.embedding_dim}. Set MEMORY_EMBEDDING_DIM={dim} "
                f"or migrate.")
        model = stored.get("embedding_model")
        if model is not None and model != settings.embedding_model:
            await self._close_both()
            raise RuntimeError(
                f"embedding_model mismatch: DB has {model!r}, config wants "
                f"{settings.embedding_model!r}. Vectors are not comparable "
                f"across models; migrate or set MEMORY_EMBEDDING_MODEL={model}.")

        # isolation_level=None -> autocommit; no WriteQueue needed here
        await self._write.execute(
            "INSERT INTO meta(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("embedding_dim", str(settings.embedding_dim)))
        await self._write.execute(
            "INSERT INTO meta(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("embedding_model", settings.embedding_model))

    async def _close_both(self) -> None:
        for conn in (self._write, self._read):
            try:
                await conn.close()
            except Exception:
                pass

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
