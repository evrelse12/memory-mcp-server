#!/usr/bin/env python3
"""One-shot v1 -> v2 migration via ATTACH. Target DB must not exist."""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.database import Database
from services.entity_index import normalize_name

ENTITY_TYPES = {"person", "project", "organization", "place", "concept", "tool"}


async def migrate(source: Path) -> None:
    db = Database()
    if db.db_path.exists():
        raise SystemExit(f"target exists: {db.db_path} - move it aside first")
    await db.initialize()
    conn = db.take_write_connection()
    await conn.execute("ATTACH ? AS old", (str(source),))

    await conn.execute("""
        INSERT INTO main.nodes (id, type, content, session_id, event_time, created_at,
            last_accessed, access_count, confidence, decay_rate, needs_embedding,
            valid_until, metadata)
        SELECT id, type, content, session_id, event_time, created_at,
            last_accessed, access_count, confidence, decay_rate, 1,
            valid_until, metadata
        FROM old.nodes""")
    await conn.execute("""
        INSERT INTO main.vec_nodes(node_id, embedding)
        SELECT node_id, embedding FROM old.vec_nodes""")
    await conn.execute("""
        UPDATE main.nodes SET needs_embedding = 0
        WHERE id IN (SELECT node_id FROM main.vec_nodes)""")
    await conn.execute("""
        INSERT INTO main.edges (id, source_id, target_id, relation_type, predicate,
            weight, confidence, valid_from, valid_until, evidence, created_at)
        SELECT id, source_id, target_id, relation_type, predicate, weight,
            confidence, valid_from, valid_until, evidence, created_at
        FROM old.edges""")

    async with conn.execute("SELECT * FROM old.entities") as cur:
        entities = await cur.fetchall()
    skipped = 0
    for e in entities:
        etype = e["type"] if e["type"] in ENTITY_TYPES else "concept"
        try:
            await conn.execute("""
                INSERT INTO main.entities (id, name_norm, canonical_name, type,
                    aliases, summary, first_seen, last_updated, mention_count,
                    attributes)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (e["id"], normalize_name(e["canonical_name"]), e["canonical_name"],
                 etype, e["aliases"], e["summary"], e["first_seen"], e["last_updated"],
                 e["mention_count"], e["attributes"]))
        except Exception:
            skipped += 1

    await conn.execute("""
        INSERT OR IGNORE INTO main.node_entities(node_id, entity_id)
        SELECT node_id, entity_id FROM old.node_entities""")
    # Old sessions are treated as consolidated so nightly job does not redo them.
    await conn.execute("""
        INSERT INTO main.sessions(session_id, consolidated_at)
        SELECT session_id, MAX(event_time) FROM old.nodes
        WHERE type='episodic' AND session_id IS NOT NULL
        GROUP BY session_id""")
    await conn.execute("DETACH old")
    await db.close()
    print(f"migration complete ({len(entities)} entities, {skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="path to the v1 memory.db")
    asyncio.run(migrate(parser.parse_args().source))
