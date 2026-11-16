# memory_mcp/services/entity_index.py
"""In-memory entity index: one combined word-boundary regex, canonical names."""

from __future__ import annotations

import json
import logging
import re
import unicodedata

import sqlite_vec

from memory_mcp.config.settings import settings
from memory_mcp.core.storage import Storage
from memory_mcp.core.tasks import spawn
from memory_mcp.services.embedding import EmbeddingError, EmbeddingService

logger = logging.getLogger(__name__)

ENTITY_TYPES = {"person", "project", "organization", "place", "concept", "tool"}
MIN_NAME_LEN = 3


def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold().strip()


class EntityIndex:
    def __init__(self, storage: Storage, embedding: EmbeddingService):
        self._s = storage
        self._emb = embedding
        self._by_norm: dict[str, tuple[str, str]] = {}  # norm -> (id, canonical)
        self._surfaces: list[str] = []
        self._surface_keys: set[str] = set()
        self._combined: re.Pattern | None = None

    async def load(self) -> None:
        self._by_norm.clear()
        self._surfaces.clear()
        self._surface_keys.clear()
        self._combined = None
        async with self._s.db.get_read_connection().execute(
            "SELECT id, name_norm, canonical_name, aliases FROM entities"
        ) as cur:
            rows = await cur.fetchall()
        for r in rows:
            self._by_norm[r["name_norm"]] = (r["id"], r["canonical_name"])
            self._add_pattern(r["canonical_name"])
            for a in json.loads(r["aliases"] or "[]"):
                self._by_norm.setdefault(normalize_name(a), (r["id"], r["canonical_name"]))
                self._add_pattern(a)
        logger.info("EntityIndex loaded: %d names", len(self._by_norm))

    def _add_pattern(self, surface: str) -> None:
        if len(surface) < MIN_NAME_LEN:
            return
        key = normalize_name(surface)
        if key in self._surface_keys:
            return
        self._surface_keys.add(key)
        self._surfaces.append(surface)
        self._combined = None  # invalidate; rebuilt lazily on next find

    def _pattern(self) -> re.Pattern | None:
        if self._combined is None and self._surfaces:
            # longest-first so "New York" wins over "York" at the same position
            alts = "|".join(re.escape(s) for s in sorted(self._surfaces, key=len, reverse=True))
            self._combined = re.compile(r"(?<!\w)(?:" + alts + r")(?!\w)", re.IGNORECASE)
        return self._combined

    def find_in_text(self, text: str) -> list[str]:
        pat = self._pattern()
        if pat is None:
            return []
        hits, seen = [], set()
        for m in pat.finditer(text):
            entry = self._by_norm.get(normalize_name(m.group(0)))
            canon = entry[1] if entry else m.group(0)
            if canon not in seen:
                seen.add(canon)
                hits.append(canon)
        return hits

    def resolve(self, name: str) -> str | None:
        entry = self._by_norm.get(normalize_name(name))
        return entry[0] if entry else None

    def canonical(self, name: str) -> str:
        entry = self._by_norm.get(normalize_name(name))
        return entry[1] if entry else name

    async def _find_by_vector(self, name: str) -> str | None:
        try:
            emb = await self._emb.embed_document(name)
        except EmbeddingError:
            return None
        async with self._s.db.get_read_connection().execute(
            "SELECT entity_id, distance FROM vec_entities WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(emb), settings.entity_knn_k),
        ) as cur:
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
            eid = await self._find_by_vector(name)
            if eid is not None:
                ent = await self._s.get_entity(eid)
                if ent:
                    existing = (eid, ent["canonical_name"])
                    self._by_norm[norm] = existing
        canonical = existing[1] if existing else name
        eid = await self._s.upsert_entity(
            name_norm=normalize_name(canonical), canonical=canonical, etype=etype
        )
        if existing is None:
            self._by_norm[norm] = (eid, canonical)
            self._add_pattern(canonical)
            spawn(self._embed_safe(eid, canonical), name=f"ent-emb-{eid[:8]}")
        elif name != canonical:
            await self._s.merge_alias(eid, name)
            if normalize_name(name) != normalize_name(canonical):
                self._by_norm.setdefault(normalize_name(name), (eid, canonical))
                self._add_pattern(name)
        return eid

    async def embed_entity(self, eid: str, text: str) -> None:
        emb = await self._emb.embed_document(text)
        await self._s.insert_entity_embedding(eid, emb)
