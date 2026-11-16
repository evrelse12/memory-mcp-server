# Configuration

Settings come from `memory_mcp/config/settings.py` (pydantic-settings).
Precedence: **process env > `.env` > defaults**. Env vars use the `MEMORY_` prefix.

Dump the authoritative full schema (names + current values):
```bash
./venv/bin/python -c "from memory_mcp.config.settings import settings; print(settings.model_dump_json(indent=2))"
```

## Storage & logging
| Var | Meaning |
|---|---|
| `MEMORY_DATA_DIR` | project data root; DB lives at `<data_dir>/db/memory.db` |
| `MEMORY_DB_PATH` | override DB file path explicitly |
| `MEMORY_LOG_LEVEL` | DEBUG/INFO/WARNING/ERROR |
| `MEMORY_LOG_TO_FILE` | also write rotating log file |

> `MEMORY_DATA_DIR` must be the **project root**, not the `db/` folder — the `db/`
> segment is appended automatically.

## LLM (chat / consolidation / critic)
| Var | Meaning |
|---|---|
| `MEMORY_LLM_API_URL` | OpenAI-compatible base URL |
| `MEMORY_LLM_MODEL` | model alias sent in requests |
| `MEMORY_LLM_TIMEOUT` | request timeout (s) |
| `MEMORY_LLM_MAX_TOKENS` | extraction budget |

## Embeddings
| Var | Meaning |
|---|---|
| `MEMORY_EMBEDDING_API_URL` | embedding server base URL |
| `MEMORY_EMBEDDING_MODEL` | identity string; **must match DB meta** (fail-fast) |
| `MEMORY_EMBEDDING_DIM` | vector dim; **must match DB meta** (fail-fast) |
| `MEMORY_EMBEDDING_BACKEND` | `http` (default) or `onnx` (experimental bge-m3) |
| `MEMORY_ONNX_EMBEDDING_MODEL` | HF repo for onnx backend |
| `MEMORY_ONNX_EMBEDDING_FILE` | onnx filename in that repo |
| `MEMORY_EMBEDDING_TRUNCATE_MATRYOSHKA` | truncate to `DIM` (nomic only) |

Changing model/dim on an existing DB raises at startup by design (vectors are not
comparable across models). Migrate deliberately, don't edit the var to silence it.

## Sessions, consolidation, decay
| Var | Meaning |
|---|---|
| `MEMORY_SESSION_TIMEOUT_MINUTES` | idle gap before a session consolidates |
| `MEMORY_CONSOLIDATION_CRON_HOUR/MINUTE` | nightly maintenance time |
| `MEMORY_CONSOLIDATE_ASSISTANT` | also extract from assistant turns |
| `MEMORY_DECAY_EXPONENT` | Ebbinghaus curve shape |

Dedup / conflict-band / entity-match thresholds are tunable in `settings.py`
(see dump above); they are calibrated to the default embedding model.

## Reflection (NLI + critic + regulator)
| Var | Meaning |
|---|---|
| `MEMORY_NLI_MODEL_DIR` | where the NLI ONNX model is cached |
| `MEMORY_NLI_TOP_K` | source episodes used as grounding |
| `MEMORY_NLI_REJECT_THRESHOLD` | auto-reject below this NLI score |
| `MEMORY_NLI_PROMOTE_THRESHOLD` | auto-promote above this (if enabled) |
| `MEMORY_NLI_AUTO_PROMOTE` | allow NLI-only promotion (default off) |
| `MEMORY_HYPOTHESIS_TTL_DAYS` | archive unresolved hypotheses after N days |

## Transport
| Var | Meaning |
|---|---|
| `MEMORY_TRANSPORT` | `stdio` (default) or `http` |
| `MEMORY_HTTP_HOST/PORT/PATH` | HTTP bind |
| `MEMORY_HTTP_TOKEN` | bearer token; required for non-loopback binds |

Details & security: [HTTP.md](HTTP.md).

## FastMCP runtime (host-side, `FASTMCP_` prefix)
These are **FastMCP's own** settings, not ours. Non-boolean fields are `Literal`:
a wrong value kills the server at import (host shows "0 tools").
| Var | Valid | Recommended |
|---|---|---|
| `FASTMCP_CHECK_FOR_UPDATES` | stable/prerelease/off | `off` |
| `FASTMCP_TELEMETRY_MODE` | native/propagation_only/off | `off` |
| `FASTMCP_SHOW_SERVER_BANNER` | bool | `false` |
| `FASTMCP_LOG_LEVEL` | DEBUG..CRITICAL | `WARNING` |

Check allowed values before setting:
```bash
./venv/bin/python -c "from fastmcp import settings; print(type(settings).model_fields['telemetry_mode'].annotation)"
```
