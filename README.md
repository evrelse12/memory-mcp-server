# memory-mcp

Long-term memory for local LLMs: a SQLite knowledge graph (episodes, facts,
entities) with hybrid search, Ebbinghaus decay and MCP tools.

## Architecture

- **Nodes**: episodic (immutable dialogue), semantic / procedural / opinion (decay).
- **Edges**: temporal, causal, entity, derived_from, supersedes (soft-delete via valid_until).
- **Search**: FTS5 (BM25, OR) + sqlite-vec KNN + anchored graph walk + entity lookup,
  fused with RRF and confidence weighting.
- **Decay**: effective confidence = confidence * exp(-rate * age_days^0.8), computed on
  the fly; nightly job only expires and purges.
- **Consolidation**: LLM extracts per-fact sources and entities from finished sessions;
  conflicts resolved via similarity band + LLM arbitration (same/contradicts/different).
- **Single writer**: one WriteQueue (batched commits, savepoints) owns the SQLite
  connection; reads use a separate query-only connection (WAL).

## Install

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # adjust URLs/models

## Run (stdio, llama-server)

    "memory": {
      "command": "<project>/venv/bin/python3",
      "args": ["<project>/main.py"],
      "env": {"PYTHONPATH": "<project>"}
    }

Optional HTTP daemon mode: MEMORY_MCP_TRANSPORT=http (port 9932, routes /mcp, /ingest,
/healthz). stdio->http proxy for legacy clients: mcp_server/bridge.py.

Maintenance (consolidation, decay, enrichment, causal checks, housekeeping) runs inside
the server process: cron at 03:00 plus catch-up on start if it was down. No systemd unit
or crontab required.

## Tools

search_memory, remember_fact(pin), correct_fact, confirm_fact, forget_fact(hard),
get_entity_info, get_timeline, memory_stats, explain_fact, weak_facts.

## Tests

    pytest -q

## Migration from v1

    python scripts/migrate_v1_to_v2.py /path/to/old/memory.db
