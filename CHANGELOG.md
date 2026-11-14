# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/); versioning: SemVer.

## [0.7.0] - 2026-11-14
### Added
- Nightly retro-dedup sweep with provenance-preserving merge.
- Vector-based entity disambiguation (`vec_entities` now used).
- Per-chat sessions: explicit `session_id` + idle auto-rotation + consolidation tick.
- Russian FTS: stop-words + crude prefix stemming.
- Combined-regex entity matching; SQL `weak_facts`; opt-in rotating file log.
- MIT LICENSE, CHANGELOG, CONTRIBUTING, CI matrix, `memory-mcp` console script.
### Fixed
- KNN overfetch for dedupe/search; supersede preserves old type; atomic alias
  merge; assistant episodes excluded from fact extraction; hermetic tests;
  dead code removed.
### Changed
- Single namespace package `memory_mcp`; search channels truly parallel;
  README (EN/RU) synced with code.

## [0.6.0] - 2026-11-06
- Initial public version: graph schema v3, WriteQueue, hybrid RRF search,
  Ebbinghaus decay, nightly maintenance, 11 MCP tools.
