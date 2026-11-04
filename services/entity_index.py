# services/entity_index.py
"""In-memory entity index: word-boundary matching, canonical names, alias merge."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Optional

from core.storage import Storage
from services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

ENTITY_TYPES = {"person", "project", "organization", "place", "concept", "tool"}
MIN_NAME_LEN = 3


def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold().strip()


class EntityIndex:
    def __init__(self, storage: Storage, embedding: EmbeddingService):
        self._s = storage
        self._emb = embedding
        self._by_norm: dict[str, tuple[str, str]] = {}   # norm -> (id, canonical)
        self._patterns: list[tuple[re.Pattern, str]] = []  # (boundary regex, surface)

    async def load(self) -> None:
        self._by_norm.clear()
        self._patterns.clear()
        async with self._s.db.get_read_connection().execute(
                "SELECT id, name_norm, canonical_name, aliases FROM entities") as cur:
            rows = await cur.fetchall()
        for r in rows:
            self._by_norm[r["name_norm"]] = (r["id"], r["canonical_name"])
            self._add_pattern(r["canonical_name"])
            for a in json.loads(r["aliases"] or "[]"):
                self._add_pattern(a)
        logger.info("EntityIndex loaded: %d names", len(self._by_norm))

    def _add_pattern(self, surface: str) -> None:
        # word boundaries + min length: "Go"/"AI" must not match inside words
        if len(surface) < MIN_NAME_LEN:
            return
        pat = re.compile(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", re.IGNORECASE)
        self._patterns.append((pat, surface))

    def find_in_text(self, text: str) -> list[str]:
        """Canonical names of known entities mentioned in text (bounded match)."""
        hits, seen = [], set()
        for pat, surface in self._patterns:
            if pat.search(text):
                entry = self._by_norm.get(normalize_name(surface))
                canon = entry[1] if entry else surface
                if canon not in seen:
                    seen.add(canon)
                    hits.append(canon)
        return hits

    def resolve(self, name: str) -> Optional[str]:
        entry = self._by_norm.get(normalize_name(name))
        return entry[0] if entry else None

    def canonical(self, name: str) -> str:
        """Stored canonical name, NOT the normalized key."""
        entry = self._by_norm.get(normalize_name(name))
        return entry[1] if entry else name

    async def ensure(self, name: str, etype: str = "concept") -> str:
        norm = normalize_name(name)
        existing = self._by_norm.get(norm)
        canonical = existing[1] if existing else name
        eid = await self._s.upsert_entity(name_norm=norm, canonical=canonical,
                                          etype=etype)
        if existing is None:
            self._by_norm[norm] = (eid, canonical)
            self._add_pattern(canonical)
        elif name != canonical:
            # new surface form becomes an alias; never clobber existing aliases
            await self._s.merge_alias(eid, name)
        return eid

    async def embed_entity(self, eid: str, text: str) -> None:
        emb = await self._emb.embed_document(text)
        await self._s.insert_entity_embedding(eid, emb)
