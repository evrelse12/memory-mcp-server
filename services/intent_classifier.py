# services/intent_classifier.py
"""Regex-only intent routing. Sync, stateless, no embedding cache to invalidate."""
from __future__ import annotations

import re

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("why", re.compile(
        r"\b(почему|почем\w*|причин\w*|из-за\s+чего|why|because|caused\s+by|reason)\b",
        re.I)),
    ("when", re.compile(
        r"\b(когда|хронолог\w*|таймлайн|последовательност\w*|when|timeline|chronology)\b",
        re.I)),
    ("who", re.compile(
        r"\b(кто|кого|кем|чей|чья|чьё|who|whom|whose)\b", re.I)),
    ("what", re.compile(
        r"\b(что\s+такое|расскажи\s+про|опиши|what\s+is|tell\s+me\s+about|describe)\b",
        re.I)),
]


def classify_intent(query: str) -> str:
    """Return one of: why | when | who | what | general."""
    for intent, rx in _RULES:
        if rx.search(query):
            return intent
    return "general"
