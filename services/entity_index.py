# services/entity_index.py
"""Normalized entity registry: NFKC+casefold names, alias map, in-text matching."""
from __future__ import annotations

import json
import logging
import unicodedata
from typing import Optional, Sequence

import sqlite_vec

from core.storage import Storage
from services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

ENTITY_TYPES = {"person", "project", "organization", "place", "concept", "tool"}


def normalize_name(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold().strip()


class EntityIndex:
    def __init__(self, storage: Storage, embedding: EmbeddingService):
        self._s = storage
        self._emb = embedding
        self._map: dict[str, str] = {}     # name_norm | alias_norm -> entity_id
        self._names: list[str] = []        # sorted by length desc for scanning

    # ------------------------------------------------------------ registry
    async def load(self) -> None:
        read = self._s.db.get_read_connection()
        async with read.execute(
                "SELECT id, name_norm, aliases FROM entities") as cur:
            rows = await cur.fetchall()
        self._map.clear()
        for r in rows:
            self._map[r["name_norm"]] = r["id"]
            for alias in json.loads(r["aliases"] or "[]"):
                self._map.setdefault(normalize_name(alias), r["id"])
        self._rebuild_names()
        logger.info("EntityIndex loaded: %d names", len(self._map))

    def _rebuild_names(self) -> None:
        self._names = sorted(self._map, key=len, reverse=True)

    def resolve(self, name: str) -> Optional[str]:
        return self._map.get(normalize_name(name))

    def find_in_text(self, text: str) -> list[tuple[str, str]]:
        """Longest-name-first substring scan over the normalized text."""
        norm = normalize_name(text)
        best: dict[str, tuple[str, str]] = {}
        for name in self._names:
            if name and name in norm:
                eid = self._map[name]
                if eid not in best or len(name) > len(best[eid][0]):
                    best[eid] = (name, eid)
        return list(best.values())

    # ------------------------------------------------------------ writes
    async def ensure(self, name: str, etype: str = "concept", *,
                     summary: Optional[str] = None,
                     aliases: Optional[Sequence[str]] = None) -> str:
        if etype not in ENTITY_TYPES:
            etype = "concept"
        norm = normalize_name(name)
        alias_norms = [normalize_name(a) for a in (aliases or []) if normalize_name(a)]
        eid = await self._s.upsert_entity({
            "name_norm": norm,
            "canonical_name": name.strip(),
            "type": etype,
            "summary": summary,
            "aliases": alias_norms,
        })
        if norm not in self._map:
            self._map[norm] = eid
            for a in alias_norms:
                self._map.setdefault(a, eid)
            self._rebuild_names()
        return eid

    async def embed_entity(self, entity_id: str, text: str) -> None:
        """DELETE+INSERT in one transaction: vec0 PK conflict-safe refresh."""
        emb = (await self._emb.embed_documents([text]))[0]
        blob = sqlite_vec.serialize_float32(emb)
        await self._s.transaction([
            ("DELETE FROM vec_entities WHERE entity_id=?", (entity_id,)),
            ("INSERT INTO vec_entities(entity_id, embedding) VALUES (?,?)",
             (entity_id, blob)),
        ])
