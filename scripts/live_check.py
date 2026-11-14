#!/usr/bin/env python3
"""Live integration check: real LLM + real embeddings, throwaway DB.

Production DB is NEVER opened. A temp data dir is forced before import and
removed in finally, even on failure. Run: ./venv/bin/python scripts/live_check.py
Requires real services up (embeddings :8081, chat LLM :9931).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = tempfile.mkdtemp(prefix="memory-mcp-live-")
os.environ["MEMORY_DATA_DIR"] = TMP
os.environ["MEMORY_DB_PATH"] = str(Path(TMP) / "db" / "memory.db")   # bulletproof isolation
os.environ["MEMORY_LOG_PATH"] = str(Path(TMP) / "logs" / "memory.log")
os.environ["MEMORY_LOG_LEVEL"] = "WARNING"
# NOTE: offline endpoints are NOT set — we want the real .env services.

from fastmcp import Client  # noqa: E402

from memory_mcp.config.settings import settings  # noqa: E402
from memory_mcp.core.tasks import drain  # noqa: E402
from memory_mcp.mcp_server import server  # noqa: E402

RESULTS = []

def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

def info(name, detail=""):
    print(f"[INFO] {name}" + (f" — {detail}" if detail else ""))

def _data(res):
    text = "".join(getattr(c, "text", "") or "" for c in res.content)
    return json.loads(text) if text else None

async def run():
    async with server.lifespan(server.mcp):
        ctx = server.ctx
        await drain(timeout=15)   # let startup catch-up finish before checks
        check("isolated DB (prod untouched)", str(TMP) in str(settings.db_file), str(settings.db_file))

        # --- pre-flight: real services must be up ---
        try:
            await ctx.embedding.embed_query("ping")
            check("embedding service up", True, settings.embedding_api_url)
        except Exception as e:
            check("embedding service up", False, str(e))
            raise SystemExit(2) from e
        try:
            await ctx.consolidator.llm.chat([{"role": "user", "content": "ping"}], max_tokens=4)
            check("LLM service up", True, settings.llm_api_url)
        except Exception as e:
            check("LLM service up", False, str(e))
            raise SystemExit(2) from e

        # --- dedup on real nomic vectors ---
        r = await ctx.tools.remember_fact("Пользователь работает в Яндексе")
        check("remember_fact created", r.get("status") == "created", str(r))
        r2 = await ctx.tools.remember_fact("Пользователь работает в Яндексе")
        check("duplicate detected", r2.get("status") == "duplicate", str(r2))

        # --- correct_fact: type preserved + supersedes chain ---
        op = await ctx.tools.remember_fact("Пользователь не любит Java", node_type="opinion")
        corr = await ctx.tools.correct_fact(op["node_id"], "Пользователь любит Java")
        check("correct_fact", corr.get("status") == "corrected")
        expl = await ctx.tools.explain_fact(corr["new_node_id"])
        check("supersede preserves type", expl["node"]["type"] == "opinion", expl["node"]["type"])
        check("supersedes chain present", len(expl.get("superseded", [])) >= 1)

        # --- Russian FTS recall ---
        await ctx.tools.remember_fact("Пользователь работал в Яндексе")
        sr = await ctx.tools.search_memory("работу", limit=5)
        check("RU FTS search", any("Яндексе" in x["content"] for x in sr), f"{len(sr)} hits")

        # --- MCP surface (real tool wrappers) ---
        async with Client(server.mcp) as c:
            await c.call_tool("record_message", {"message": "Привет, это тест", "session_id": "live-s1"})
            tl = _data(await c.call_tool("get_timeline", {"limit": 5}))
            check("record_message + timeline", any("это тест" in x["content"] for x in (tl or [])))
            st = _data(await c.call_tool("memory_stats", {}))
            check("vec_available live", st.get("vec_available") is True, str(st.get("nodes_by_type")))

        # --- entity disambiguation (threshold-dependent, measured) ---
        eid1 = await ctx.entity_index.ensure("Google", "organization")
        await drain(timeout=10)
        va = await ctx.embedding.embed_document("Google")
        vb = await ctx.embedding.embed_document("Гугл")
        sim = sum(a * b for a, b in zip(va, vb, strict=False))
        eid2 = await ctx.entity_index.ensure("Гугл")
        info("entity cosine Google/Гугл", f"{sim:.3f} (threshold {settings.entity_match_threshold})")
        check("entity alias merge", eid2 == eid1,
              "merged" if eid2 == eid1 else f"not merged — lower threshold to ≤ {sim:.2f}")

        # --- atomic alias merge ---
        eid = await ctx.storage.upsert_entity(name_norm="acme", canonical="Acme")
        await asyncio.gather(ctx.storage.merge_alias(eid, "ACME Corp"),
                             ctx.storage.merge_alias(eid, "Акме"),
                             ctx.storage.merge_alias(eid, "ACME Corp"))
        al = json.loads((await ctx.storage.get_entity(eid))["aliases"])
        check("atomic alias merge", len(al) == 2 and set(al) == {"ACME Corp", "Акме"}, str(al))

        # --- retro-dedup sweep on real vectors ---
        a = await ctx.tools.remember_fact("Пользователь любит кофе")
        b = await ctx.tools.remember_fact("Пользователь любит кофе", force=True)
        rep = await ctx.search.sweep_duplicates()
        check("retro-dedup sweep",
              rep["merged"] >= 1 and await ctx.storage.get_node(b["node_id"]) is None
              and await ctx.storage.get_node(a["node_id"]) is not None, str(rep))

        # --- session rotation ---
        hp = ctx.hot_path
        s1 = hp.current_session()
        s2 = hp.current_session()
        hp._auto_last -= settings.session_timeout_minutes * 60 + 1
        s3 = hp.current_session()
        check("session rotation", s1 == s2 and s3 != s1 and hp.current_session("chat-42") == "chat-42")

        # --- weak_facts SQL path ---
        wf = await ctx.tools.remember_fact("Старый факт для проверки затухания", confidence=0.9)
        await ctx.storage.execute("UPDATE nodes SET created_at=?, last_accessed=NULL WHERE id=?",
                                  (int(time.time()) - 100 * 86400, wf["node_id"]))
        weak = await ctx.tools.weak_facts()
        check("weak_facts", any(w["id"] == wf["node_id"] for w in weak))

        # --- real-LLM consolidation + assistant exclusion ---
        sid = "live-consol"
        await ctx.hot_path.record_message(sid, "Меня зовут Алексей. Я живу в Москве и работаю в Яндексе.", role="user")
        await ctx.hot_path.record_message(sid, "Пользователь живёт на Марсе и работает в NASA.", role="assistant")
        ok = await ctx.consolidator.consolidate_session(sid)
        check("consolidation ran (real LLM)", ok is True)
        check("session marked consolidated", await ctx.storage.session_consolidated_at(sid) is not None)
        mars = await ctx.tools.search_memory("Марсе NASA", limit=10)
        check("assistant fact excluded", not any("Марс" in x["content"] or "NASA" in x["content"] for x in mars), f"{len(mars)} hits")
        msk = await ctx.tools.search_memory("Москве Алексей", limit=10)
        check("user facts extracted", any("Москв" in x["content"] or "Алексей" in x["content"] or "Яндекс" in x["content"] for x in msk), f"{len(msk)} hits")

async def main():
    try:
        await run()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        print(f"\ncleaned temp DB: {TMP} (removed={not Path(TMP).exists()})")
    passed, total = sum(RESULTS), len(RESULTS)
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
