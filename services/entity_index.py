# services/entity_index.py
"""In-memory entity index: word-boundary matching, canonical names, alias merge."""
from __future__ import annotations

import json
import logging
import re
import unicodedata

import sqlite_vec

from config.settings import settings
from core.storage import Storage
from core.tasks import spawn
from services.embedding import EmbeddingError, EmbeddingService

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

    def resolve(self, name: str) -> str | None:
        entry = self._by_norm.get(normalize_name(name))
        return entry[0] if entry else None

    def canonical(self, name: str) -> str:
        """Stored canonical name, NOT the normalized key."""
        entry = self._by_norm.get(normalize_name(name))
        return entry[1] if entry else name

    async def _find_by_vector(self, name: str) -> str | None:
        """Nearest known entity by embedding; None on miss or embedder outage."""
        try:
            emb = await self._emb.embed_document(name)
        except EmbeddingError:
            return None
        async with self._s.db.get_read_connection().execute(
                "SELECT entity_id, distance FROM vec_entities "
                "WHERE embedding MATCH ? AND k = ?",
                (sqlite_vec.serialize_float32(emb), settings.entity_knn_k)) as cur:
            rows = await cur.fetchall()
        for r in rows:
            if 1.0 - r["distance"] ** 2 / 2.0 >= settings.entity_match_threshold:
                return r["entity_id"]
        return None

    async def _embed_safe(self, eid: str, text: str) -> None:
        try:
            await self.embed_entity(eid, text)
        except EmbeddingError:
            logger.debug("entity embedding deferred: %s", eid)

    async def ensure(self, name: str, etype: str = "concept") -> str:
        norm = normalize_name(name)
        existing = self._by_norm.get(norm)
        if existing is None:
            # Vector disambiguation: "Гугл" must land on "Google", not a new entity.
            eid = await self._find_by_vector(name)
            if eid is not None:
                ent = await self._s.get_entity(eid)
                if ent:
                    existing = (eid, ent["canonical_name"])
                    self._by_norm[norm] = existing
        canonical = existing[1] if existing else name
        # Upsert by CANONICAL norm: an alias norm must never create a new row.
        eid = await self._s.upsert_entity(name_norm=normalize_name(canonical),
                                          canonical=canonical, etype=etype)
        if existing is None:
            self._by_norm[norm] = (eid, canonical)
            self._add_pattern(canonical)
            spawn(self._embed_safe(eid, canonical), name=f"ent-emb-{eid[:8]}")
        elif name != canonical:
            await self._s.merge_alias(eid, name)
            if normalize_name(name) != normalize_name(canonical):
                self._add_pattern(name)   # alias matches in-process, not only after restart
        return eid

    async def embed_entity(self, eid: str, text: str) -> None:
        emb = await self._emb.embed_document(text)
        await self._s.insert_entity_embedding(eid, emb)
