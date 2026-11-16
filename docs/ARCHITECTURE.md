# Architecture

## Storage
- One SQLite file, WAL mode. **Two connections**: a dedicated writer owned by a
  single-writer `WriteQueue` (serialises all mutations) and a read-only connection
  for queries. No cross-process write contention.
- `sqlite-vec` loaded into both connections; `vec0` virtual tables for KNN.
- FTS5 external-content table joined on `nodes.rid` (stable autoincrement PK),
  triggers keep it in sync on insert/delete/content-update only.

## Schema (v4)
- `nodes`: episodic / semantic / procedural / opinion. Fields for decay
  (`confidence`, `decay_rate`, `last_accessed`, `access_count`), lifecycle
  (`valid_until` soft-delete), and Reflection (`status`, `nli_score`,
  `generator_model`, `critic_model`).
- `edges`: temporal / causal / entity / derived_from / supersedes, with weight,
  confidence and validity window. Unique (source,target,relation) dedupes edges.
- `entities` + `node_entities`: canonical names, aliases, mention counts.
- `sessions`: consolidation watermark per session.
- `meta`: embedding identity + maintenance timestamp. `PRAGMA user_version` = schema
  version; older DBs are migrated in-place at startup, newer DBs fail fast.

## Retrieval (hybrid)
Query → parallel channels → **Reciprocal Rank Fusion**:
1. FTS5 lexical (porter + unicode61, RU/EN),
2. dense KNN over `vec_nodes`,
3. graph expansion over edges,
4. entity-channel boost via the in-memory entity index.
Results carry effective confidence (decay applied) so stale facts rank lower.

## Memory lifecycle
- **Write path**: `record_message` stores the raw episode; `remember_fact` dedups
  against existing vectors (skip / conflict / insert) and links provenance.
- **Decay**: Ebbinghaus `confidence * exp(-rate * idle^exponent)`; access reinforces;
  nightly job soft-deletes below threshold. Pinned facts (`decay_rate=0`) never decay.
- **Directives**: pinned constraints/preferences are always injected into context;
  conflicts return `requires_user_clarification` instead of overwriting.

## MCP surface
- Tools: record_message, get_turn_context, get_active_directives, search_memory,
  remember_fact, correct_fact, confirm_fact, forget_fact, get_entity_info,
  get_timeline, memory_stats, explain_fact, weak_facts.
- Resources (application-driven hints): user/profile, user/constraints,
  user/preferences, episodes/recent, entities/{id}, session/{id}/summary.
- Prompt: `memory_protocol` (EN/RU) — see [SYSTEM_PROMPT.md](SYSTEM_PROMPT.md).

## Nightly pipeline
See [CONSOLIDATION_REFLECTION.md](CONSOLIDATION_REFLECTION.md).
