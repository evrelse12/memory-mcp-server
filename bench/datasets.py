# bench/datasets.py
"""Loaders/normalizers for LongMemEval and LoCoMo into one common schema.

Field names vary between dataset revisions. Use `--inspect` to print the real
structure and adjust the mapping instead of trusting a silent guess.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DATE_FORMATS = (
    "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d",
    "%I:%M %p on %d %B, %Y", "%H:%M %d %B %Y", "%d %B %Y", "%B %d, %Y",
)

LM_ABILITY = {
    "single-session-user": "information-extraction",
    "single-session-assistant": "information-extraction",
    "single-session-preference": "information-extraction",
    "multi-session": "multi-session-reasoning",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-updates",
}

# Best-effort labels; verify against the dataset docs via --inspect.
LOCOMO_CATEGORY = {
    1: "multi-hop", 2: "temporal", 3: "open-domain",
    4: "single-hop", 5: "adversarial",
}


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class Session:
    sid: str
    ts: int
    turns: list[Turn]


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    kind: str
    abstention: bool
    ts: int | None
    evidence: list[str]


@dataclass
class Item:
    """One haystack + the questions asked over it."""
    iid: str
    sessions: list[Session]
    questions: list[Question]


def parse_ts(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", str(value))).strip()
    for fmt in DATE_FORMATS:
        try:
            return int(datetime.strptime(s, fmt).timestamp())
        except ValueError:
            continue
    return None


def _turns(raw, human: bool = False) -> list[Turn]:
    out = []
    for t in raw or []:
        if not isinstance(t, dict):
            continue
        text = t.get("content") or t.get("text") or t.get("utterance") or ""
        role = str(t.get("role") or t.get("speaker") or "user").lower()
        if human:
            # LoCoMo: both speakers are human participants of the source dialogue
            role = "user"
        else:
            role = "assistant" if role in ("assistant", "bot", "agent") else "user"
        if str(text).strip():
            out.append(Turn(role=role, text=str(text)))
    return out


def load_longmemeval(raw) -> list[Item]:
    items = []
    for i, r in enumerate(raw):
        hs = r.get("haystack_sessions") or []
        dates = r.get("haystack_dates") or []
        sids = r.get("haystack_session_ids") or []
        sessions = []
        for j, sess in enumerate(hs):
            ts = parse_ts(dates[j]) if j < len(dates) else None
            sid = str(sids[j]) if j < len(sids) else f"s{j}"
            sessions.append(Session(sid=sid, ts=ts or 0, turns=_turns(sess)))
        qtype = str(r.get("question_type") or "unknown")
        qid = str(r.get("question_id") or f"q{i}")
        abst = "abs" in qid.lower() or "abstention" in qtype.lower()
        items.append(Item(iid=qid, sessions=sessions, questions=[Question(
            qid=qid, question=str(r.get("question") or ""),
            answer=str(r.get("answer") or ""),
            kind="abstention" if abst else LM_ABILITY.get(qtype, qtype),
            abstention=abst, ts=parse_ts(r.get("question_date")),
            evidence=[str(x) for x in (r.get("answer_session_ids") or [])])]))
    return items


def load_locomo(raw) -> list[Item]:
    items = []
    for i, r in enumerate(raw):
        conv = r.get("conversation") or {}
        keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                      key=lambda k: int(k.split("_")[1]))
        sessions = [Session(sid=k, ts=parse_ts(conv.get(f"{k}_date_time")) or 0,
                            turns=_turns(conv[k], human=True)) for k in keys]
        questions = []
        for j, qa in enumerate(r.get("qa") or []):
            kind = LOCOMO_CATEGORY.get(qa.get("category"), str(qa.get("category")))
            ans = qa.get("answer")
            if ans is None:
                ans = qa.get("adversarial_answer", "")
            questions.append(Question(
                qid=f"{r.get('sample_id', i)}-{j}",
                question=str(qa.get("question") or ""), answer=str(ans or ""),
                kind=kind, abstention=(kind == "adversarial"), ts=None,
                evidence=[str(x) for x in (qa.get("evidence") or [])]))
        items.append(Item(iid=str(r.get("sample_id") or f"conv{i}"),
                          sessions=sessions, questions=questions))
    return items


def detect_format(raw) -> str:
    first = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(first, dict):
        raise ValueError("dataset must be a JSON object or a list of objects")
    if "haystack_sessions" in first:
        return "longmemeval"
    if "conversation" in first and "qa" in first:
        return "locomo"
    raise ValueError(f"unrecognized dataset; top-level keys: {sorted(first)}")


def load(path: str, fmt: str = "auto") -> list[Item]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    fmt = detect_format(raw) if fmt == "auto" else fmt
    if fmt == "longmemeval":
        return load_longmemeval(raw)
    if fmt == "locomo":
        return load_locomo(raw)
    raise ValueError(f"unknown format: {fmt}")


def inspect(path: str) -> str:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    first = raw[0] if isinstance(raw, list) and raw else raw
    lines = [f"records: {len(raw) if isinstance(raw, list) else 1}",
             f"top-level keys: {sorted(first) if isinstance(first, dict) else type(first)}"]
    try:
        lines.append(f"detected format: {detect_format(raw)}")
    except ValueError as e:
        lines.append(f"detect failed: {e}")
    if isinstance(first, dict):
        for key in ("question_type", "question_id", "question_date",
                    "haystack_dates", "haystack_session_ids", "answer_session_ids"):
            if key in first:
                lines.append(f"{key}: {json.dumps(first[key], ensure_ascii=False)[:200]}")
        if "haystack_sessions" in first and first["haystack_sessions"]:
            lines.append("turn sample: " + json.dumps(
                first["haystack_sessions"][0][:1], ensure_ascii=False)[:300])
        if "conversation" in first:
            ck = sorted(first["conversation"])[:6]
            lines.append(f"conversation keys (sample): {ck}")
            for k in ck:
                if re.fullmatch(r"session_\d+", k) and first["conversation"][k]:
                    lines.append("turn sample: " + json.dumps(
                        first["conversation"][k][:1], ensure_ascii=False)[:300])
                    break
        if "qa" in first and first["qa"]:
            lines.append("qa sample: " + json.dumps(first["qa"][0], ensure_ascii=False)[:300])
            lines.append("qa categories: " + str(sorted(
                {q.get("category") for q in first["qa"]}, key=str)))
    return "\n".join(lines)
