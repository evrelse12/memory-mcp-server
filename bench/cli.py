# bench/cli.py
"""Benchmark CLI.

python -m bench.cli --data longmemeval_s.json --inspect
python -m bench.cli --data longmemeval_s.json --limit 20 --baseline
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(prog="bench", description="memory-mcp benchmarks")
    p.add_argument("--data", required=True, help="path to dataset JSON")
    p.add_argument("--format", default="auto", choices=["auto", "longmemeval", "locomo"])
    p.add_argument("--inspect", action="store_true", help="print structure and exit")
    p.add_argument("--limit", type=int, default=20, help="items to run (0 = all)")
    p.add_argument(
        "--stratify",
        action="store_true",
        help="round-robin items across abilities instead of taking the first N",
    )
    p.add_argument("--max-sessions", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=0)
    p.add_argument("--mode", default="episodes", choices=["episodes", "facts", "hybrid"])
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--consolidate", action="store_true", help="run LLM consolidation after ingest (slow)"
    )
    p.add_argument(
        "--fts-only", action="store_true", help="disable embeddings (ablation: FTS recall only)"
    )
    p.add_argument("--baseline", action="store_true", help="also run no-memory arm")
    p.add_argument("--keep-db", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from bench import datasets

    if args.inspect:
        print(datasets.inspect(args.data))
        return 0

    items = datasets.load(args.data, args.format)
    if args.stratify:
        buckets = defaultdict(list)
        for it in items:
            buckets[it.questions[0].kind if it.questions else "?"].append(it)
        picked = []
        for row in zip_longest(*[buckets[k] for k in sorted(buckets)]):
            picked += [x for x in row if x is not None]
        items = picked[: args.limit] if args.limit else picked
    elif args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out) if args.out else Path("bench/results") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="membench-"))

    # Isolation must be set BEFORE memory_mcp is imported.
    os.environ["MEMORY_DATA_DIR"] = str(tmp)
    os.environ["MEMORY_DB_PATH"] = str(tmp / "memory.db")
    os.environ["MEMORY_LOG_LEVEL"] = "WARNING"
    os.environ["MEMORY_LOG_TO_FILE"] = "false"
    if args.fts_only:
        os.environ["MEMORY_EMBEDDING_API_URL"] = "http://127.0.0.1:9"

    from bench import harness
    from memory_mcp.config.settings import settings

    cfg = harness.Config(
        mode=args.mode,
        top_k=args.top_k,
        consolidate=args.consolidate,
        baseline=args.baseline,
        max_sessions=args.max_sessions,
        max_turns=args.max_turns,
        keep_db=args.keep_db,
    )
    meta = {
        "dataset": args.data,
        "format": args.format,
        "items": len(items),
        "mode": cfg.mode,
        "top_k": cfg.top_k,
        "consolidate": cfg.consolidate,
        "fts_only": args.fts_only,
        "baseline": cfg.baseline,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "embedding_backend": settings.embedding_backend,
        "max_sessions": cfg.max_sessions,
        "max_turns": cfg.max_turns,
    }
    try:
        results = asyncio.run(harness.run_all(items, cfg, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    report = harness.aggregate(results)
    (out_dir / "results.json").write_text(harness.results_json(results), encoding="utf-8")
    (out_dir / "report.md").write_text(
        harness.render_report(results, report, meta), encoding="utf-8"
    )
    print("\n" + harness.render_report(results, report, meta))
    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
