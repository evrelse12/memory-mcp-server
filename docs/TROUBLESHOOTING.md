# Troubleshooting

## Host reports "0 tools"
The client hides the child's stderr. Reproduce the spawn and read the real error:
```bash
./venv/bin/python main.py < /dev/null
```
Common causes:
- **Bad `FASTMCP_*` value.** Non-boolean FastMCP fields are `Literal`; e.g.
  `FASTMCP_CHECK_FOR_UPDATES=false` is invalid (use `off`) and kills the process at
  import. Check annotations before setting (see CONFIGURATION.md).
- **Embedding identity mismatch.** `embedding_model`/`embedding_dim` in DB meta differ
  from config → deliberate fail-fast. Align config to the DB, or migrate; don't silence.
- **`MEMORY_DATA_DIR` pointed at `db/`.** The `db/` segment is appended automatically,
  so you'd open/create `db/db/memory.db`. Point it at the project root.

## LLM returns empty content
Reasoning models may put short answers in `reasoning_content` with empty `content`.
The client falls back to the reasoning channel only when generation completed
(`finish_reason != length`). If answers are empty with `finish_reason=length`, raise
the token budget (bench uses 512 for answers).

## Tests: "Event loop is closed" warnings
The threaded-uvicorn HTTP test emits benign aiosqlite thread warnings on teardown.
The test passes; safe to ignore.

## macOS: sqlite-vec won't load
python.org builds disable `enable_load_extension`. Use Homebrew/conda Python or run
on Linux/Windows (CI covers ubuntu + windows).

## Downloaded models in git
NLI/ONNX weights land under `models/` and the HF cache. `.gitignore` excludes
`models/`, `*.onnx`, `*.gguf`, etc. If already staged:
```bash
git rm -r --cached --ignore-unmatch models
```
