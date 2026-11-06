# memory-mcp

Long-term memory for local LLMs: a SQLite-backed knowledge graph of episodes,
facts and entities, with hybrid search, Ebbinghaus-style forgetting and
MCP tools. Runs as a single stdio process spawned by `llama-server`.

## Principles

Memory is a **directed, typed graph** persisted in SQLite, where every piece of
knowledge is a node and every relationship between pieces is an edge. The graph
is built and maintained through four mechanisms working together:

1. **Hot path** — the model records every dialogue turn as an immutable episode.
2. **Online fact extraction** — the model stores durable facts immediately,
   deduplicated against the existing graph.
3. **Nightly consolidation** — a background agent re-reads finished sessions
   and extracts facts that were missed online.
4. **Decay and maintenance** — rarely accessed facts fade and eventually expire;
   the database is kept healthy by a nightly housekeeping job.

### Nodes

Every piece of knowledge is a `nodes` row with a stable `id` (UUID) and a
surrogate `rid` (autoincrement PK, preserved by `VACUUM`). Node types:

| Type | Purpose | Decay |
|------|---------|--------|
| `episodic` | Immutable dialogue turns. Never decay. | No |
| `semantic` | Facts about the world ("User works at Google"). | Yes |
| `procedural` | Skills and habits ("User prefers dark theme in editors"). | Yes |
| `opinion` | Subjective preferences ("User dislikes Java"). | Yes |

Each node carries:

- `content` — self-contained sentence, explicit names, no pronouns.
- `confidence` ∈ [0, 1] — how sure we are the fact is still true.
- `decay_rate` — how fast the fact should be forgotten (0 for pinned facts).
- `last_accessed` — timestamp of the most recent successful search hit.
- `needs_embedding` — flag used by the backfill job.
- `valid_until` — soft-delete marker; set by decay, used as the "forgotten" flag.
- `session_id`, `event_time`, `metadata` — context for episodes and facts.

### Edges

Edges are first-class typed rows (not just node pairs):

| Type | Direction | Meaning |
|------|-----------|---------|
| `temporal` | episode → previous episode | "follows" in a dialogue |
| `derived_from` | fact → episode | provenance of the fact |
| `entity` | node → entity (via `node_entities`) | mentions |
| `supersedes` | new fact → old fact | correction chain |
| `causal` | cause episode → effect episode | (reserved, currently unused) |

All edges have a `valid_until` column so that when a node is soft-deleted,
downstream references (e.g. `derived_from`) can be cleaned up consistently.
A unique index on `(source_id, target_id, relation_type)` prevents duplicate
edges when `ON CONFLICT DO NOTHING` is used.

### Entities

Named concepts (people, projects, organisations, places, tools) live in the
`entities` table with:

- `canonical_name` — the form the model should use in output.
- `aliases` — JSON array of all surface forms ever seen (never overwritten).
- `summary` — LLM-generated paragraph, regenerated when mentions double.
- `mention_count` — incremented by the upsert, drives enrichment.

The in-memory `EntityIndex` holds a compiled regex with word boundaries
(`(?<!\w)name(?!\w)`, min length 3) so short names like `Go` or `AI` don't
match inside other words.

## Forgetting: the Ebbinghaus formula

Effective confidence is computed on the fly at read time:
eff = confidence · exp(-decay_rate · idle_days^0.8)

where `idle_days` is measured from **`last_accessed`** (falls back to
`created_at` if the node has never been hit). This is the key property:
**looking a fact up resets the forgetting curve**, as the Ebbinghaus formula
intends. A 70-day-old fact accessed yesterday stays near full confidence; the
same fact untouched for 70 days is practically forgotten.

The nightly `DecayService` job does the complementary work:

- `expire_weak` — set `valid_until` on nodes whose `eff` has dropped below
  `settings.decay_threshold` (default 0.05).
- `purge_expired` — physically delete soft-deleted nodes older than
  `settings.soft_delete_retention_days` (default 30 days), giving an audit
  grace period for `explain_fact`. Episodes have their own
  `settings.episode_retention_days` (default 0 = keep forever).

Pinned facts (`confirm_fact`) set `decay_rate=0` and `confidence=1`, so the
exponential term is 1 regardless of age.

## Building the graph

### Online path

Every dialogue turn, the model (instructed by `SERVER_INSTRUCTIONS`) first calls
`record_message` with the exact user text. `HotPath.record_message`:

1. Grabs a per-session lock so two parallel messages can't fork the temporal chain.
2. Truncates the text to `settings.max_episode_chars` (default 4000).
3. Finds the previous episode in the session and links them with a `temporal` edge.
4. Runs `EntityIndex.find_in_text` (bounded regex) to detect mentioned entities.
5. Inserts the episode node, the temporal edge and the entity links in one transaction.
6. Spawns a background task to embed the episode (best-effort; deferred on failure).

**Limitation**: this path only works if the model reliably calls
`record_message` every turn and passes the exact text. Small models sometimes
skip the call or paraphrase. Deterministic ingestion would require an external
proxy in front of the MCP client.

**Secrets**: the current implementation does not filter tokens or passwords.
Add a filter in `record_message` or scrub the database before sharing.

When the user states a durable fact, the model calls `remember_fact`. Before
any write happens, `SearchService.dedupe_candidate` runs a pure vector lookup:

- `sim ≥ dedup_similarity_threshold` (default 0.92) → `duplicate`, return
  `existing_id` and `existing_content` so the model can call `correct_fact`
  if the new text contradicts the stored one.
- `sim ≥ conflict_band_min` (default 0.80) → `conflict` band, the consolidator
  uses LLM arbitration here but `remember_fact` still inserts.
- otherwise → `insert`.

This prevents "User works at Google" from piling up as dozens of near-duplicates
when the user repeats themselves.

### Nightly consolidation

At `settings.consolidation_cron_hour:minute` (default 03:00) plus a catch-up on
process start, the `MaintenanceService` picks sessions whose **latest episode is
older than the timeout AND newer than the session's last `consolidated_at`**.
This guarantees:

- A session is consolidated exactly once per "wave" of new episodes (no N writes).
- A failed run leaves `consolidated_at` untouched, so the session is retried
  next run.

For each pending session the `Consolidator`:

1. Splits episodes into chunks of `settings.consolidation_chunk_chars`.
2. Asks the LLM for a JSON array of facts with per-fact sources (episode indexes)
   and entities.
3. For each fact, vector-dedupes against the existing graph; in the narrow
   conflict band, asks the LLM `same | contradicts | different` and either skips,
   supersedes (preserving the old node's type), or inserts.
4. Links the new facts to their source episodes via `derived_from` edges and to
   the mentioned entities via `node_entities`.

Embedding failures during consolidation propagate to the caller, which returns
`False` and leaves the session un-marked — the next maintenance run will retry.

## Searching the graph

`search_memory` runs four channels in parallel and fuses them with RRF:

- **FTS5** — BM25 over a stemmed FTS index. Stopwords stripped, token prefixes
  (`"word"*`) for partial matches, OR-combined. Triggers fire only on actual
  `content` changes, so unrelated updates don't churn the index.
- **vec** — KNN over `vec_nodes` (sqlite-vec). `k = settings.knn_k`
  (default 20). Distance is cosine because vectors are L2-normalised after
  Matryoshka truncation, so `sim = 1 - d²/2` is exact.
- **graph** — one-hop walk from the top 3 vec seeds, weighted by edge weight.
- **entity** — nodes linked to entities mentioned in the query.

RRF: `score(n) = Σ_c weight_c / (rrf_k + rank_c(n) + 1)`. The final score is
multiplied by `effective_confidence`, so frequently-used facts stay on top.

**Reinforcement**: after ranking, the top 3 result nodes have their
`last_accessed` bumped and `access_count` incremented via a background task.
If SQLite was built with math functions, `confidence` also grows
(`MIN(1.0, confidence · (1 + LN(access_count + 2) / 10))`); otherwise the
fallback just updates the access metadata. Either way, the search call never
blocks on a write failure.

**Entities in the response** are returned as a list of canonical names only,
not full rows — keeps the LLM context small.

## Tools

| Tool | Purpose |
|------|---------|
| `record_message` | Write the user's exact text as an episode. **First call every turn.** |
| `search_memory` | Hybrid search over facts and episodes. |
| `remember_fact` | Store a durable fact. Returns `duplicate` if a similar one exists. |
| `correct_fact` | Replace an outdated fact (preserves the old node's type via `supersedes`). |
| `confirm_fact` | Pin a fact permanently (`decay_rate=0`, `confidence=1`). |
| `forget_fact` | Soft- or hard-delete a fact. |
| `get_entity_info` | Entity profile: summary, aliases, linked facts. |
| `get_timeline` | Last N episodes in chronological order. |
| `memory_stats` | Graph metrics, `pending_embeddings`, `vec_available` (live ping). |
| `explain_fact` | Provenance: source episodes, supersedes chain, linked entities. |
| `weak_facts` | Facts whose effective confidence is below 0.3. |

## Maintenance

A single `AsyncIOScheduler` runs the nightly job, plus a catch-up on process
start when `last_maintenance_ts` is older than 20 hours. Steps, in order:

1. **consolidate** — pick sessions with new episodes, extract facts.
2. **decay** — expire weak facts, purge soft-deleted nodes past the grace period.
3. **enrich** — regenerate entity summaries when `mention_count` has doubled.
4. **housekeep** — backfill embeddings for nodes with `needs_embedding=1`,
   `INSERT INTO nodes_fts('optimize')`, `PRAGMA wal_checkpoint(TRUNCATE)`,
   `PRAGMA optimize`.

Each step runs inside the same process; a global `asyncio.Lock` prevents two
maintenance runs from overlapping.

## Install

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # adjust URLs and models

## Run (stdio, llama-server)

In `~/.config/llama-server/mcp_config.json`:

    "memory": {
      "command": "<project>/venv/bin/python3",
      "args": ["<project>/main.py"],
      "env": {
        "PYTHONPATH": "<project>",
        "FASTMCP_CHECK_FOR_UPDATES": "false"
      }
    }

Maintenance runs inside the same process (APScheduler cron + catch-up on start).
No systemd unit, crontab or HTTP daemon is required.

## Migration from v1

    python scripts/migrate_v1_to_v2.py /path/to/old/memory.db

The script backs up the old file and migrates data into the new schema, then
sets `PRAGMA user_version = 2`.

## Tests

    pytest -q

## Schema invariants worth knowing

- `nodes.rid` is the explicit autoincrement PK; FTS5 rowid joins against it.
  Stable across `VACUUM` because it is the PK.
- `nodes_fts` triggers fire only on actual `content` changes
  (`AFTER UPDATE OF content`).
- `vec_nodes` and `vec_entities` are sqlite-vec virtual tables with hardcoded
  `float[256]`; on startup the process verifies `meta.embedding_dim` matches
  the configured value and fails fast on mismatch.
- `edges(source_id, target_id, relation_type)` has a unique index so
  `ON CONFLICT DO NOTHING` dedupes properly.
- `node_entities(entity_id)` is indexed — lookups by entity are hot.
