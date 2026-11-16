# Changelog

## v0.8.0 — Verified memory

### Added
- **Turn protocol:** `get_turn_context`, `get_active_directives`, `constraint=true`,
  structured conflict responses (`requires_user_clarification`), reference system
  prompt + MCP prompt `memory_protocol`.
- **MCP Resources:** user/profile, user/constraints, user/preferences,
  episodes/recent, entities/{id}, session/{id}/summary (priority/audience hints).
- **HTTP transport** with bearer auth (StaticTokenVerifier), loopback guard,
  multi-client sessions.
- **Consolidation → Reflection:** hypotheses quarantined; multilingual NLI shadow
  scoring grounded on source episodes; pure-code regulator (NLI gate, TTL archival,
  confidence calibration); adversarial LLM critic requiring cited evidence;
  `critic_approval_rate` observability.
- **Schema v4** with in-place live migration (`status`, `nli_score`,
  `generator_model`, `critic_model`); fail-fast only on newer DBs.
- **Benchmark harness** (LongMemEval / LoCoMo): inspect, stratify, evidence-preserving
  trimming, memory vs no-memory control, FTS-only ablation, latency percentiles.
- **Pluggable embedding backend** (`http` default, experimental `onnx` bge-m3).
- **Docs overhaul:** README as showcase; INSTALL / CONFIGURATION / ARCHITECTURE /
  CONSOLIDATION_REFLECTION / BENCHMARKS / TROUBLESHOOTING / CONTRIBUTING.

### Fixed
- Reasoning-model empty `content` fallback to `reasoning_content` (completed runs only).
- Consolidation conflict arbitration (regex verdict, adequate token budget).
- FastMCP `Literal` env pitfalls (`check_for_updates=off`, telemetry off by default).
- mcp-config `MEMORY_DATA_DIR` nesting and embedding-identity mismatch guidance.
- Test isolation: no background catch-up maintenance / network NLI downloads in tests.
- Bench honesty: answer token budget, embedding metadata reflects active backend.

### Notes
- Benchmarks use a local judge (internal comparison only). Embedding A/B
  (nomic vs bge-m3) inconclusive at n=20 → nomic remains default.
- macOS excluded from CI (python.org builds lack `enable_load_extension`).
