# Benchmarks

Internal evaluation on LongMemEval / LoCoMo via `bench/`. **Not run in CI** (needs
live LLM + embeddings + a dataset file).

## Data
- LongMemEval: HF `xiaowu0162/longmemeval` (start with `_s`).
- LoCoMo: `github.com/snap-research/locomo`.
Download manually; pass with `--data`. Datasets are gitignored.

## Always inspect first
```bash
./venv/bin/python -m bench.cli --data <file>.json --inspect
```
Prints record count, keys, detected format, date/session-id samples, QA categories.

## Run
```bash
# baseline: 20 stratified items, episodic retrieval, no-memory control
./venv/bin/python -m bench.cli --data <file>.json --limit 20 --stratify --baseline \
    --out bench/results/main20
# FTS-only ablation
... --fts-only --out bench/results/fts20
# full pipeline on a small subset (LLM consolidation; slow)
... --limit 5 --mode hybrid --consolidate --max-sessions 10
```
Output: `bench/results/<run>/{results.json,report.md}` (gitignored).

## Metrics
accuracy (LLM-as-judge, abstention handled separately), session-level recall@k,
retrieval latency p50/p95, judge_errors / pending_embeddings / unparsed_dates.

## Caveats (read before quoting)
- Judge is the **local** model, not the official GPT-4o protocol → internal
  comparison only, not comparable to published scores.
- Session-level recall is coarse (hitting the evidence session ≠ the evidence turn).
- `--max-sessions/--max-turns` truncate haystacks → inflated accuracy; smoke only.
- Reasoning models need a sufficient answer token budget or `content` stays empty.

## Results

### Memory vs no-memory (LongMemEval_S, 20 stratified, nomic/http, local judge)
Same 20 questions, same judge, single run (`main20`):

| arm | judged | accuracy | recall@8 |
|---|---|---|---|
| memory | 19/20 | **0.53** | 1.00 |
| no-memory | 20/20 | **0.20** | — |

Per-kind (memory): abstention 1.00 · information-extraction 0.75 ·
knowledge-updates 0.67 · multi-session 0.25 · temporal 0.00.
The no-memory control scores 0 on every non-abstention kind.

### Embedding A/B (preliminary, INCONCLUSIVE)
| run | backend | judged | accuracy | recall@8 |
|---|---|---|---|---|
| main20 | http/nomic | 19/20 | 0.53 | 1.00 |
| bge20 | onnx/bge-m3 | 15/20 | 0.73 | 1.00 |

**Do not read this as "bge-m3 wins".** The judged subsets differ (15 vs 19) and the
per-kind counts differ, so the accuracies are not comparable. Temporal reasoning is
0.00 in both. At n=20 with a local judge this A/B is under-powered.

**Decision: keep nomic (http) as default.** A swap requires a matched, larger A/B
(same items, same judged set, baseline in both arms) plus threshold recalibration,
since bge-m3 is not Matryoshka and invalidates the calibrated cosine thresholds.

### Known weakness
Temporal-reasoning accuracy is 0.00 even with perfect session recall: session-level
recall does not guarantee the evidence *turn* is in top-8, and cross-session date
reasoning is hard for the small local model. Targeted next step: turn-level evidence
metrics + date-aware retrieval, not an embedding swap.
