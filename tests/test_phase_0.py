# tests/test_phase_0.py
"""Phase 0: schema v4 migration (status, nli_score, generator/critic_model)."""
import aiosqlite
import pytest

from memory_mcp.core.database import Database


async def _create_v3_schema(db_file, with_data=False):
    """Create a complete v3 schema (all tables) for migration tests."""
    async with aiosqlite.connect(db_file) as conn:
        await conn.execute("""
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
        """)
        await conn.execute("""
            CREATE TABLE edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                predicate TEXT,
                weight REAL NOT NULL DEFAULT 1.0,
                confidence REAL NOT NULL DEFAULT 1.0,
                valid_from INTEGER NOT NULL,
                valid_until INTEGER,
                evidence TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER NOT NULL
            )
        """)
        await conn.execute("""
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
        """)
        await conn.execute("""
            CREATE TABLE node_entities (
                node_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (node_id, entity_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                consolidated_at INTEGER
            )
        """)
        await conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        await conn.execute("PRAGMA user_version = 3")
        if with_data:
            await conn.execute(
                "INSERT INTO nodes (id, type, content, event_time, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("n1", "semantic", "Test fact", 1700000000, 1700000000, "{}")
            )
        await conn.commit()


async def _user_version(db) -> int:
    async with db.get_read_connection().execute("PRAGMA user_version") as cur:
        row = await cur.fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_migration_v3_to_v4(tmp_path):
    """Migration adds columns and sets version; existing data preserved."""
    db_file = tmp_path / "test.db"
    await _create_v3_schema(db_file)

    db = Database(db_file)
    await db.initialize()

    cols = await db._column_names("nodes")
    assert {"status", "nli_score", "generator_model", "critic_model"} <= set(cols)
    assert await _user_version(db) == 4
    await db.close()


@pytest.mark.asyncio
async def test_new_db_starts_at_v4(tmp_path):
    """Fresh DB is created at schema v4 immediately."""
    db_file = tmp_path / "fresh.db"
    db = Database(db_file)
    await db.initialize()

    assert await _user_version(db) == 4
    cols = await db._column_names("nodes")
    assert {"status", "nli_score", "generator_model", "critic_model"} <= set(cols)
    await db.close()


@pytest.mark.asyncio
async def test_migration_idempotent(tmp_path):
    """Running migration twice doesn't fail or duplicate columns."""
    db_file = tmp_path / "test.db"
    await _create_v3_schema(db_file)

    db2 = Database(db_file)
    await db2.initialize()
    await db2.close()

    db3 = Database(db_file)
    await db3.initialize()
    assert await _user_version(db3) == 4
    await db3.close()


@pytest.mark.asyncio
async def test_column_names_helper(tmp_path):
    """_column_names returns all column names for a table."""
    db = Database(tmp_path / "test.db")
    await db.initialize()

    cols = await db._column_names("nodes")
    assert isinstance(cols, list)
    assert len(cols) > 10
    assert "id" in cols and "content" in cols
    await db.close()


@pytest.mark.asyncio
async def test_existing_data_preserved(tmp_path):
    """Migration doesn't lose existing facts."""
    db_file = tmp_path / "test.db"
    await _create_v3_schema(db_file, with_data=True)

    db = Database(db_file)
    await db.initialize()

    async with db.get_read_connection().execute(
        "SELECT content, status FROM nodes WHERE id='n1'"
    ) as cur:
        r = await cur.fetchone()
    assert r["content"] == "Test fact"
    assert r["status"] == "canonical"
    await db.close()
