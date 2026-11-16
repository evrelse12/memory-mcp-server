# Contributing

Thanks for helping make local memory better!

## Setup
```bash
git clone https://github.com/evrelse12/memory-mcp-server.git
cd memory-mcp-server
python3 -m venv venv
./venv/bin/pip install -e ".[dev]"
```

## Before you push
```bash
./venv/bin/ruff check . --fix     # lint + autofix
./venv/bin/pytest -q              # offline suite must stay green
```
CI runs Ubuntu + Windows on Python 3.10–3.14. macOS is excluded (python.org builds
lack `enable_load_extension`, so `sqlite-vec` cannot load).

## Style
- Async-first, type hints, small focused functions.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Never commit secrets or model weights (`models/`, `*.onnx`, `*.gguf` are gitignored).
- Keep logs minimal; user-facing behaviour changes need a test.

## Proposing a change / new phase
1. Open an issue describing the problem and the proposed design **before** coding.
2. Agree on the design, then implement in small staged steps.
3. Each step ships with tests; run the suite after every step.
4. Prefer full-file listings or precise anchored edits over vague diffs.
5. Benchmark-affecting changes must include a before/after run (see
   [BENCHMARKS.md](BENCHMARKS.md)); don't claim wins from unmatched samples.

## Reporting bugs
Include: OS + Python version, the command, and the real stderr
(`./venv/bin/python main.py < /dev/null` for MCP spawn issues).
