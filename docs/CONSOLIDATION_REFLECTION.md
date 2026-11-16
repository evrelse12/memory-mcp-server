# Consolidation & Reflection

Goal: turn raw dialogue into durable knowledge **without** letting confident-but-
wrong LLM inferences pollute answers.

## Provenance rule
- Stated by the user (`remember_fact`, episodes) → `status='canonical'`.
- Inferred by the LLM (consolidation) → `status='hypothesis'`, hidden from reads
  until verified. Rejected → `status='rejected'`, also hidden.

## Pipeline (nightly, sequential)
```
consolidate → dedup → decay → enrich → housekeep → nli_shadow → regulate → critic
```
1. **consolidate** — LLM extracts facts per idle session, each with source episode
   indexes and entities; written as `hypothesis` with `derived_from` edges.
2. **dedup / decay / enrich / housekeep** — vector sweep, Ebbinghaus purge, entity
   summaries, embedding backfill + FTS optimize.
3. **nli_shadow** — multilingual NLI (mDeBERTa-v3-XNLI, ONNX) scores each hypothesis
   against its **source episodes** (grounding = `derived_from`, not "same session").
   Writes `nli_score = max_entailment − max_contradiction`.
4. **regulate** (pure code) — NLI gate: strong contradiction → `rejected` without an
   LLM call; strong entailment → auto-promote only if `NLI_AUTO_PROMOTE` (off by
   default). Unresolved hypotheses older than TTL → soft-deleted (archived).
5. **critic** — adversarial LLM verdict over a grounding packet
   `[hypothesis, source episodes, neighbouring facts, nli_score]`.
   Schema `{approve, reason, evidence_ids}`. **Approve requires cited evidence ids
   from the grounding set**; otherwise inconclusive (left for next run).
   Approved → `canonical` with confidence calibrated from `nli_score`.

## Anti echo-chamber
- Critic prompt presumes the hypothesis wrong; reward is finding contradictions.
- Grounding forces citation of real episodes, not vibes.
- `critic_approval_rate` is logged each run: ≈1.0 over several nights means a
  rubber-stamp critic (add a second model family / stricter prompt); ≈0 means too
  harsh (recalibrate thresholds).

## Deferred (measured, not assumed)
- HDBSCAN clustering (we group by session + entities instead).
- A second large model family for the critic (same model + adversarial prompt first).
- Small ONNX judge models and agentic critic with live tools.
- Sparse vectors / Qdrant (FTS5 already covers the lexical channel).
- Embedding swap nomic→bge-m3: decide via A/B on the bench, not by guesswork.
