# memory_mcp/tools/memory_tools.py
"""Tool-layer orchestration over storage/search; auto entity extraction internal."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence

from memory_mcp.core.storage import Storage
from memory_mcp.services.decay import DAY, DecayService
from memory_mcp.services.embedding import EmbeddingError, EmbeddingService
from memory_mcp.services.entity_index import EntityIndex
from memory_mcp.services.search import SearchService

logger = logging.getLogger(__name__)

FACT_TYPES = {"semantic", "procedural", "opinion"}
DIRECTIVES_LIMIT = 20
PROFILE_LIMIT = 10
WEAK_THRESHOLD = 0.3


class MemoryToolError(Exception):
    """User-facing tool error (mapped to MCP ToolError by the server)."""


class MemoryTools:
    def __init__(
        self,
        storage: Storage,
        search: SearchService,
        embedding: EmbeddingService,
        entity_index: EntityIndex,
        decay: DecayService,
    ):
        self._s = storage
        self._search = search
        self._emb = embedding
        self._ei = entity_index
        self._decay = decay

    async def search_memory(
        self,
        query: str,
        *,
        node_types=None,
        time_from=None,
        time_to=None,
        limit: int = 10,
        include_episodes: bool = False,
    ) -> list[dict]:
        return await self._search.search(
            query,
            limit=limit,
            node_types=node_types,
            time_from=time_from,
            time_to=time_to,
            include_episodes=include_episodes,
        )

    async def get_turn_context(self, user_message: str, *, limit: int = 8) -> dict:
        """One-call turn context: directives + profile + relevant facts.

        Returns a ready-to-read <user_memory> block plus structured parts.
        Weak facts (effective confidence < 0.3) are dropped; facts already
        present in directives/profile are not repeated.
        """
        directives = await self._s.fetch_directives(DIRECTIVES_LIMIT)
        profile = await self._s.fetch_profile(PROFILE_LIMIT)
        hits = await self._search.search(user_message, limit=limit)
        seen = {d["id"] for d in directives} | {p["id"] for p in profile}
        facts = [h for h in hits if h["id"] not in seen and h["confidence"] >= WEAK_THRESHOLD]

        lines = ["<user_memory>"]
        if directives:
            lines.append("## Directives (must follow)")
            lines += [f"- [{d['kind']}] {d['content'][:300]}" for d in directives]
        if profile:
            lines.append("## User profile")
            lines += [f"- {p['content'][:300]}" for p in profile]
        if facts:
            lines.append("## Relevant facts")
            lines += [f"- ({f['type']}, {f['confidence']:.2f}) {f['content'][:300]}" for f in facts]
        if len(lines) == 1:
            lines.append("(no stored memory yet)")
        lines.append("</user_memory>")
        return {
            "block": "\n".join(lines),
            "directives": directives,
            "profile": profile,
            "facts": facts,
        }

    async def get_active_directives(self, limit: int = 50) -> list[dict]:
        """Hard constraints and pinned preferences, absolute priority."""
        return await self._s.fetch_directives(limit)

    # ------------------------------------------------------------ resources
    @staticmethod
    def _bullets(items: Sequence[str]) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "(empty)"

    async def render_profile(self) -> str:
        rows = await self._s.fetch_profile(20)
        return "# User profile\n\n" + self._bullets([r["content"] for r in rows])

    async def render_constraints(self) -> str:
        rows = [d for d in await self._s.fetch_directives(100) if d["kind"] == "constraint"]
        return "# Constraints (must follow)\n\n" + self._bullets([d["content"] for d in rows])

    async def render_preferences(self) -> str:
        rows = [d for d in await self._s.fetch_directives(100) if d["kind"] == "preference"]
        return "# Preferences\n\n" + self._bullets([d["content"] for d in rows])

    async def render_entity(self, entity_id: str) -> str:
        ent = await self._s.get_entity(entity_id)
        if not ent:
            raise MemoryToolError(f"unknown entity id: {entity_id}")
        nodes = await self._s.fetch_nodes_for_entity(entity_id)
        aliases = json.loads(ent["aliases"] or "[]")
        lines = [f"# {ent['canonical_name']}", ""]
        if ent.get("summary"):
            lines += [ent["summary"], ""]
        if aliases:
            lines += [f"Aliases: {', '.join(aliases)}", ""]
        lines += ["## Linked facts", self._bullets([n["content"] for n in nodes[:20]])]
        return "\n".join(lines)

    async def render_recent_episodes(self, limit: int = 20) -> str:
        rows = await self._s.fetch_timeline(limit=limit)
        return "# Recent episodes\n\n" + self._bullets([r["content"][:300] for r in rows])

    async def render_session_digest(self, session_id: str) -> str:
        """Extractive digest (no LLM call): recent turns + facts of this session."""
        rows = [
            r for r in await self._s.fetch_nodes_by_session(session_id) if r["valid_until"] is None
        ]
        eps = [r for r in rows if r["type"] == "episodic"][-10:]
        facts = [r for r in rows if r["type"] != "episodic"][:20]
        return "\n".join(
            [
                f"# Session {session_id}",
                "",
                "## Recent turns",
                self._bullets([e["content"][:300] for e in eps]),
                "",
                "## Facts from this session",
                self._bullets([f["content"] for f in facts]),
            ]
        )

    # ------------------------------------------------------------ end resources

    async def remember_fact(
        self,
        content: str,
        *,
        node_type: str = "semantic",
        confidence: float = 0.9,
        entities: Sequence[str] | None = None,
        source_episode_ids: Sequence[str] = (),
        pin: bool = False,
        force: bool = False,
        constraint: bool = False,
    ) -> dict:
        if node_type not in FACT_TYPES:
            raise MemoryToolError(f"bad node_type: {node_type!r}")
        confidence = min(1.0, max(0.0, float(confidence)))

        emb = None
        try:
            emb = await self._emb.embed_document(content)
        except EmbeddingError as e:
            logger.warning("embedding unavailable, storing without vector: %s", e)

        action, existing = await self._search.dedupe_candidate(content, emb=emb)

        if action == "skip" and existing and not force:
            await self._s.access_nodes([existing["id"]])
            pinned = False
            if pin or constraint:
                await self.confirm_fact(existing["id"])
                pinned = True
            if constraint:
                # Upgrade the existing fact to a hard directive.
                node = await self._s.get_node(existing["id"])
                meta = json.loads(node["metadata"] or "{}")
                meta["kind"] = "constraint"
                await self._s.update_node_fields(existing["id"], {"metadata": meta})
            return {
                "status": "duplicate",
                "existing_id": existing["id"],
                "existing_content": existing["content"],
                "pinned": pinned,
                "constraint": constraint,
                "hint": "call correct_fact if the new text contradicts it",
            }

        if action == "conflict" and existing and not force:
            return {
                "status": "similar",
                "existing_id": existing["id"],
                "existing_content": existing["content"],
                "new_content": content,
                "requires_user_clarification": True,
                "clarification_prompt": (
                    f'Stored fact: "{existing["content"]}". '
                    f'New statement: "{content}". '
                    "Ask the user: is this a permanent change or a one-off "
                    "exception? Permanent -> correct_fact(old_node_id, ...). "
                    "One-off -> remember_fact(..., force=true)."
                ),
                "hint": "call correct_fact to replace it, or "
                "remember_fact(force=true) to keep both",
            }

        names = list(entities or []) or self._ei.find_in_text(content)
        eids = [await self._ei.ensure(n) for n in names]
        nid = await self._s.create_fact(
            content=content,
            node_type=node_type,
            confidence=1.0 if (pin or constraint) else confidence,
            decay_rate=0.0 if (pin or constraint) else None,
            embedding=emb,
            entity_ids=eids,
            source_episode_ids=list(source_episode_ids),
            metadata={"kind": "constraint"} if constraint else None,
        )
        return {
            "status": "created",
            "node_id": nid,
            "embedded": emb is not None,
            "pinned": pin or constraint,
            "constraint": constraint,
        }

    async def correct_fact(
        self,
        old_node_id: str,
        new_content: str,
        *,
        new_confidence: float = 0.9,
        source_episode_ids: Sequence[str] = (),
    ) -> dict:
        old = await self._s.get_node(old_node_id)
        if not old:
            raise MemoryToolError(f"node not found: {old_node_id}")
        emb = None
        try:
            emb = await self._emb.embed_document(new_content)
        except EmbeddingError as e:
            logger.warning("embedding unavailable for correction: %s", e)
        names = self._ei.find_in_text(new_content)
        eids = [await self._ei.ensure(n) for n in names]
        nid = await self._s.supersede_fact(
            old_node_id,
            node_type=old["type"],
            content=new_content,
            confidence=new_confidence,
            embedding=emb,
            source_episode_ids=list(source_episode_ids),
            entity_ids=eids,
        )
        return {
            "status": "corrected",
            "old_node_id": old_node_id,
            "new_node_id": nid,
            "preserved_type": old["type"],
        }

    async def confirm_fact(self, node_id: str) -> dict:
        n = await self._s.execute(
            "UPDATE nodes SET confidence=1.0, decay_rate=0.0, last_accessed=? "
            "WHERE id=? AND valid_until IS NULL",
            (int(time.time()), node_id),
            wait=True,
        )
        if not n:
            raise MemoryToolError(f"node not found or deleted: {node_id}")
        return {"status": "confirmed", "node_id": node_id}

    async def forget_fact(self, node_id: str, *, hard: bool = False) -> dict:
        node = await self._s.get_node(node_id, include_deleted=True)
        if not node:
            raise MemoryToolError(f"node not found: {node_id}")
        if hard:
            await self._s.hard_delete_nodes([node_id])
            mode = "hard"
        else:
            await self._s.soft_delete_nodes([node_id])
            mode = "soft"
        return {"status": "forgotten", "node_id": node_id, "mode": mode}

    async def get_entity_info(self, entity_name: str) -> dict:
        eid = self._ei.resolve(entity_name)
        if not eid:
            raise MemoryToolError(f"unknown entity: {entity_name!r}")
        entity = await self._s.get_entity(eid)
        nodes = await self._s.fetch_nodes_for_entity(eid)
        return {
            **entity,
            "linked_facts": [
                {"id": n["id"], "content": n["content"], "type": n["type"]} for n in nodes[:20]
            ],
        }

    async def get_timeline(
        self, *, entity_name: str | None = None, time_from=None, time_to=None, limit: int = 20
    ) -> list[dict]:
        eid = self._ei.resolve(entity_name) if entity_name else None
        return await self._s.fetch_timeline(
            entity_id=eid, time_from=time_from, time_to=time_to, limit=limit
        )

    async def memory_stats(self) -> dict:
        s = await self._s.stats()
        try:
            await self._emb.embed_query("ping")
            vec_available = True
        except Exception:
            vec_available = False
        return {**s, "vec_available": vec_available}

    async def explain_fact(self, node_id: str) -> dict:
        node = await self._s.get_node(node_id, include_deleted=True)
        if not node:
            raise MemoryToolError(f"node not found: {node_id}")
        sources, superseded, ents = await self._s.provenance(node_id)
        return {
            "node": node,
            "source_episodes": sources,
            "superseded": superseded,
            "entities": ents,
        }

    async def weak_facts(self, limit: int = 20) -> list[dict]:
        """Facts whose EFFECTIVE confidence (from last access) is below 0.3.

        SQL path when SQLite has math functions; Python fallback otherwise.
        """
        now = int(time.time())
        thr = WEAK_THRESHOLD
        if self._s.db.has_math_functions:
            async with self._s.db.get_read_connection().execute(
                """
                    SELECT id, type, content, eff FROM (
                        SELECT id, type, content,
                               confidence * EXP(-decay_rate *
                                   POWER((? - COALESCE(last_accessed, created_at)) / ?, ?)) AS eff
                        FROM nodes
                        WHERE type IN ('semantic','procedural','opinion')
                          AND valid_until IS NULL
                    ) WHERE eff < ? ORDER BY eff LIMIT ?""",
                (now, DAY, self._decay_exponent(), thr, limit),
            ) as cur:
                return [
                    {
                        "id": r["id"],
                        "type": r["type"],
                        "content": r["content"],
                        "effective_confidence": round(r["eff"], 4),
                    }
                    for r in await cur.fetchall()
                ]
        async with self._s.db.get_read_connection().execute("""
                SELECT id, type, content, confidence, decay_rate,
                       created_at, last_accessed FROM nodes
                WHERE type IN ('semantic','procedural','opinion')
                  AND valid_until IS NULL""") as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            idle = now - (r["last_accessed"] or r["created_at"])
            eff = self._decay.effective_confidence(r["confidence"], r["decay_rate"], idle)
            if eff < thr:
                out.append(
                    {
                        "id": r["id"],
                        "type": r["type"],
                        "content": r["content"],
                        "effective_confidence": round(eff, 4),
                    }
                )
        out.sort(key=lambda d: d["effective_confidence"])
        return out[:limit]

    @staticmethod
    def _decay_exponent() -> float:
        from memory_mcp.config.settings import settings

        return settings.decay_exponent
