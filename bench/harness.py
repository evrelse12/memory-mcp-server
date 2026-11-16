# bench/harness.py
"""Benchmark pipeline: ingest -> retrieve -> answer -> judge -> metrics.

Uses the REAL local LLM + embeddings on a throwaway DB per item; the production
database is never opened. Not run in CI (needs services + dataset); the pure
helpers below are unit-tested offline.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from memory_mcp.agents.llm_client import LLMClient, LLMError
from memory_mcp.config.settings import settings
from memory_mcp.core.tasks import drain
from memory_mcp.mcp_server import server
from memory_mcp.services.embedding import EmbeddingError, EmbeddingService

ANSWER_PROMPT = """Answer the question using ONLY the conversation excerpts below.
If the answer is not present there, say that you do not know.
Answer in one short sentence.

Conversation excerpts:
{context}

Question: {question}
Answer:"""

NO_MEMORY_PROMPT = """Answer the question in one short sentence.
If you do not know the answer, say that you do not know.

Question: {question}
Answer:"""

JUDGE_PROMPT = """Grade a model answer against a reference answer.
Question: {q}
Reference: {gold}
Model answer: {pred}
Is the model answer semantically equivalent to the reference? Extra wording is
fine; missing or contradicting key facts is not.
Answer with exactly one word: yes or no."""

ABSTAIN_JUDGE_PROMPT = """This question cannot be answered from the conversation.
Question: {q}
Model answer: {pred}
Did the model correctly refuse or say it does not know, instead of inventing an answer?
Answer with exactly one word: yes or no."""


@dataclass
class Config:
    mode: str = "episodes"  # episodes | facts | hybrid
    top_k: int = 8
    consolidate: bool = False
    baseline: bool = False
    max_sessions: int = 0  # 0 = all
    max_turns: int = 0  # 0 = all
    context_chars: int = 6000
    embed_timeout: float = 1800.0
    keep_db: bool = False
    judge_max_tokens: int = 512


@dataclass
class Result:
    item: str
    qid: str
    kind: str
    abstention: bool
    arm: str  # memory | no-memory
    correct: bool | None
    recall: bool | None
    retrieved: int
    retrieval_ms: float
    answer_ms: float
    pending_embeddings: int = 0
    unparsed_dates: int = 0
    pred: str = ""
    gold: str = ""
    judge_raw: str = ""
    error: str | None = None


def _yesno(text: str) -> bool | None:
    """Tolerant: accepts 'Yes.', '**Yes**', 'Answer: no', etc."""
    m = re.search(r"\b(yes|no)\b", (text or "").lower())
    return None if m is None else m.group(1) == "yes"


def aggregate(results: list[Result]) -> dict:
    out: dict = {}
    for arm in sorted({r.arm for r in results}):
        rs = [r for r in results if r.arm == arm]
        graded = [r for r in rs if r.correct is not None]
        by_kind = {}
        for kind in sorted({r.kind for r in rs}):
            ks = [r for r in graded if r.kind == kind]
            by_kind[kind] = {
                "n": len(ks),
                "accuracy": round(sum(r.correct for r in ks) / len(ks), 4) if ks else None,
            }
        rec = [r.recall for r in rs if r.recall is not None]
        lat = sorted(r.retrieval_ms for r in rs)
        errs: dict[str, int] = {}
        for r in rs:
            if r.error:
                key = r.error.split(":")[0][:60]
                errs[key] = errs.get(key, 0) + 1
        out[arm] = {
            "n": len(rs),
            "judged": len(graded),
            "judge_errors": len(rs) - len(graded),
            "errors": errs,
            "accuracy": (
                round(sum(r.correct for r in graded) / len(graded), 4) if graded else None
            ),
            "by_kind": by_kind,
            "recall_n": len(rec),
            "recall_at_k": round(sum(rec) / len(rec), 4) if rec else None,
            "retrieval_ms_p50": round(statistics.median(lat), 2) if lat else None,
            "retrieval_ms_p95": (
                round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2) if lat else None
            ),
        }
    return out


def results_json(results: list[Result]) -> str:
    return json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2)


def render_report(results: list[Result], report: dict, meta: dict) -> str:
    lines = [
        "# memory-mcp benchmark report",
        "",
        "## Config",
        "```json",
        json.dumps(meta, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    for arm, m in report.items():
        lines += [
            f"## arm: {arm}",
            f"- questions: {m['n']} (judged {m['judged']}, ungraded {m['judge_errors']})",
            f"- accuracy: {m['accuracy']}",
            f"- recall@k session-level: {m['recall_at_k']} (n={m['recall_n']})",
            f"- retrieval ms p50/p95: {m['retrieval_ms_p50']} / {m['retrieval_ms_p95']}",
        ]
        if m["errors"]:
            lines.append(f"- errors: {m['errors']}")
        lines += ["", "| kind | n | accuracy |", "|---|---|---|"]
        for kind, k in m["by_kind"].items():
            lines.append(f"| {kind} | {k['n']} | {k['accuracy']} |")
        lines.append("")
    lines += [
        "## Caveats",
        "- Judge is the local model, not the official GPT-4o protocol; "
        "numbers are for internal comparison only.",
        "- Session-level recall is coarse: hitting the evidence session "
        "does not guarantee the evidence turn was retrieved.",
        "- Truncated haystacks (--max-sessions/--max-turns) inflate accuracy; "
        "use them for smoke tests only.",
        "- Unparsed session dates fall back to ingest time.",
        "",
    ]
    return "\n".join(lines)


def _select_sessions(item, cfg: Config):
    """Trim the haystack but never drop evidence sessions (keeps items answerable)."""
    sessions = item.sessions
    if not cfg.max_sessions or len(sessions) <= cfg.max_sessions:
        return sessions
    evidence = {e for q in item.questions for e in q.evidence}
    keep = [s for s in sessions if s.sid in evidence]
    rest = [s for s in sessions if s.sid not in evidence]
    return keep + rest[: max(0, cfg.max_sessions - len(keep))]


async def _retrieve(ctx, question: str, cfg: Config) -> tuple[str, list[str]]:
    ids: list[str] = []
    parts: list[str] = []
    if cfg.mode in ("facts", "hybrid"):
        tc = await ctx.tools.get_turn_context(question, limit=cfg.top_k)
        parts.append(tc["block"])
        ids += [f["id"] for f in tc["facts"]]
    if cfg.mode in ("episodes", "hybrid"):
        types = ["episodic"] if cfg.mode == "episodes" else None
        hits = await ctx.search.search(
            question, limit=cfg.top_k, node_types=types, include_episodes=True
        )
        ids += [h["id"] for h in hits]
        parts.append(
            "## Excerpts\n"
            + "\n".join(
                f"- [{time.strftime('%Y-%m-%d', time.gmtime(h['event_time']))}] "
                f"{h['content'][:500]}"
                for h in hits
            )
        )
    return "\n\n".join(parts)[: cfg.context_chars], ids


async def _answer(llm, question: str, context: str, arm: str) -> tuple[str, float, str | None]:
    prompt = (
        ANSWER_PROMPT.format(context=context, question=question)
        if arm == "memory"
        else NO_MEMORY_PROMPT.format(question=question)
    )
    t0 = time.perf_counter()
    try:
        text = await llm.chat([{"role": "user", "content": prompt}],
                              max_tokens=512, temperature=0.0)
    except LLMError as e:
        return "", (time.perf_counter() - t0) * 1000, f"answer: {e}"
    return text, (time.perf_counter() - t0) * 1000, None


async def _judge(llm, q, pred: str, max_tokens: int = 512) -> tuple[bool | None, str, str | None]:
    prompt = (
        ABSTAIN_JUDGE_PROMPT.format(q=q.question, pred=pred)
        if q.abstention
        else JUDGE_PROMPT.format(q=q.question, gold=q.answer, pred=pred)
    )
    try:
        raw = await llm.chat(
            [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0
        )
    except LLMError as e:
        return None, "", f"judge: {e}"
    finish = getattr(llm, "last_finish_reason", None)
    verdict = _yesno(raw)
    if verdict is None and not raw.strip():
        # Reasoning models can spend the whole budget before emitting content.
        try:
            raw = await llm.chat(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens * 4, temperature=0.0
            )
            finish = getattr(llm, "last_finish_reason", None)
            verdict = _yesno(raw)
        except LLMError as e:
            return None, "", f"judge retry: {e}"
    if verdict is None:
        return None, raw, f"judge unparsable (finish={finish}): {raw[:80]!r}"
    return verdict, raw, None


async def run_item(item, cfg: Config, run_dir: Path, llm) -> list[Result]:
    db_dir = run_dir / item.iid
    settings.db_path = db_dir / "memory.db"
    if cfg.consolidate:
        # LongMemEval asks about assistant-side content too.
        settings.consolidate_assistant = True
    sessions = _select_sessions(item, cfg)
    sids = {s.sid for s in sessions}
    results: list[Result] = []
    try:
        async with server.lifespan(server.mcp):
            ctx = server.ctx
            await drain(timeout=30)  # let startup catch-up settle
            unparsed = 0
            for s in sessions:
                turns = s.turns[: cfg.max_turns] if cfg.max_turns else s.turns
                for t in turns:
                    await ctx.hot_path.record_message(s.sid, t.text, role=t.role)
                if s.ts:
                    await ctx.storage.execute(
                        "UPDATE nodes SET event_time=?, created_at=? WHERE session_id=?",
                        (s.ts, s.ts, s.sid),
                    )
                else:
                    unparsed += 1
            await drain(timeout=cfg.embed_timeout)
            if cfg.consolidate:
                for s in sessions:
                    await ctx.consolidator.consolidate_session(s.sid)
            pending = (await ctx.storage.stats()).get("pending_embeddings", 0)

            for q in item.questions:
                arms = ["memory", "no-memory"] if cfg.baseline else ["memory"]
                for arm in arms:
                    t0 = time.perf_counter()
                    context, ids = ("", [])
                    if arm == "memory":
                        context, ids = await _retrieve(ctx, q.question, cfg)
                    rms = (time.perf_counter() - t0) * 1000
                    recall = None
                    if arm == "memory" and ids and (set(q.evidence) & sids):
                        rows = await ctx.storage.fetch_nodes(ids)
                        recall = bool({r["session_id"] for r in rows} & set(q.evidence))
                    pred, ams, aerr = await _answer(llm, q.question, context, arm)
                    verdict, jraw, jerr = (None, "", None)
                    if pred.strip():
                        verdict, jraw, jerr = await _judge(llm, q, pred, cfg.judge_max_tokens)
                    results.append(
                        Result(
                            item=item.iid,
                            qid=q.qid,
                            kind=q.kind,
                            abstention=q.abstention,
                            arm=arm,
                            correct=verdict,
                            recall=recall,
                            retrieved=len(ids),
                            retrieval_ms=rms,
                            answer_ms=ams,
                            pending_embeddings=pending,
                            unparsed_dates=unparsed,
                            pred=pred,
                            gold=q.answer,
                            judge_raw=jraw,
                            error=aerr or jerr,
                        )
                    )
    finally:
        settings.db_path = None
        if not cfg.keep_db:
            shutil.rmtree(db_dir, ignore_errors=True)
    return results


async def run_all(items, cfg: Config, run_dir: Path) -> list[Result]:
    llm = LLMClient()
    out: list[Result] = []
    try:
        # Fail fast BEFORE the expensive ingest if the chat model is unreachable.
        try:
            await llm.chat(
                [{"role": "user", "content": "Reply with one word: ok"}],
                max_tokens=8,
                temperature=0.0,
            )
            print(f"chat LLM: OK ({settings.llm_api_url})", flush=True)
            probe = await llm.chat(
                [
                    {
                        "role": "user",
                        "content": "Is 2+2 equal to 4? Answer with exactly one word: yes or no.",
                    }
                ],
                max_tokens=cfg.judge_max_tokens,
                temperature=0.0,
            )
            print(
                f"judge probe: {probe[:40]!r} "
                f"(finish={llm.last_finish_reason}, verdict={_yesno(probe)})",
                flush=True,
            )
        except LLMError as e:
            raise SystemExit(
                f"chat LLM unreachable at {settings.llm_api_url}: {e}\n"
                "Start llama-server or fix MEMORY_LLM_API_URL before benchmarking."
            ) from e
        emb = EmbeddingService()
        try:
            await emb.embed_query("ping")
            print(f"embeddings: OK ({settings.embedding_api_url})", flush=True)
        except EmbeddingError:
            print("embeddings: UNAVAILABLE — running FTS-only", flush=True)
        finally:
            await emb.close()

        for i, item in enumerate(items, 1):
            rs = await run_item(item, cfg, run_dir, llm)
            out += rs
            mem = [r for r in rs if r.arm == "memory"]
            ok = sum(r.correct is True for r in mem)
            graded = sum(r.correct is not None for r in mem)
            print(
                f"[{i}/{len(items)}] {item.iid}: {ok}/{graded} graded"
                + (f" ERR {mem[0].error}" if mem and mem[0].error else ""),
                flush=True,
            )
    finally:
        await llm.close()
    return out
