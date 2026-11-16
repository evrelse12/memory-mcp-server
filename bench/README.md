# Benchmarks

Internal evaluation of memory-mcp on LongMemEval and LoCoMo. **Not run in CI**
(needs the local LLM, embeddings and a dataset file).

## Get the data

- **LongMemEval** — HuggingFace dataset `xiaowu0162/longmemeval`
  (repo: `github.com/xiaowu0162/LongMemEval`). Start with the `_s` haystack;
  `_oracle` is useful as an upper bound.
- **LoCoMo** — `github.com/snap-research/locomo` (the `locomo10` data file).

Download manually and pass the file with `--data`. No auto-download: URLs and
file layouts change between revisions.

## Step 1 — inspect (always do this first)

```bash
./venv/bin/python -m bench.cli --data /path/longmemeval_s.json --inspect
```

It prints record count, top-level keys, detected format, date/session-id samples
and QA categories. If the mapping looks wrong, adjust `bench/datasets.py`.

## Step 2 — run

```bash
# cheap first pass: 20 items, episodic retrieval, with a no-memory baseline
./venv/bin/python -m bench.cli --data /path/longmemeval_s.json --limit 20 --baseline

# FTS-only ablation (embeddings disabled)
./venv/bin/python -m bench.cli --data /path/longmemeval_s.json --limit 20 --fts-only

# full pipeline on a small subset (LLM consolidation — slow)
./venv/bin/python -m bench.cli --data /path/longmemeval_s.json \
    --limit 5 --mode hybrid --consolidate --max-sessions 10
```

Output: `bench/results/<timestamp>/{results.json,report.md}` (gitignored).

## What is measured

- **accuracy** overall and per ability/category (LLM-as-judge, abstention handled separately)
- **recall@k** — session-level: did retrieval hit an evidence session
- **retrieval latency** p50/p95
- `judge_errors`, `pending_embeddings`, `unparsed_dates` for transparency

## Caveats (read before quoting numbers)

- The judge is the **local model**, not the official GPT-4o protocol. Scores are
  for comparing configurations internally, **not** comparable to published ones.
- Default mode is `episodes` (no consolidation): it measures episodic retrieval.
  `facts`/`hybrid` require `--consolidate`.
- With `--consolidate`, `consolidate_assistant` is forced on (LongMemEval asks
  about assistant-side content); this differs from the production default.
- Session timestamps are patched after ingest; unparsed dates fall back to
  ingest time and weaken temporal reasoning.
- LoCoMo evidence is turn-level, so session recall is reported as n/a.
- Each item gets its own throwaway DB; the production database is never opened.
