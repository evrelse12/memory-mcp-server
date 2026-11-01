# mcp_server/server.py
"""Memory MCP Server (Pure stdio). Запускается llama-server как дочерний процесс."""
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from core.database import Database
from core.write_queue import WriteQueue
from core.storage import Storage
from services.embedding import EmbeddingService
from services.hot_path import HotPath
from services.search import SearchService
from services.intent_classifier import IntentClassifier
from services.context_provider import ContextProvider
from services.decay import DecayService
from agents.consolidator import Consolidator
from agents.maintenance import MaintenanceService
from tools.memory_tools import MemoryTools
from config.settings import settings

logger = logging.getLogger(__name__)

# ============================================================
# Глобальный экземпляр FastMCP (гарантирует регистрацию в mcp 1.x)
# ============================================================
SERVER_INSTRUCTIONS = """
You are equipped with a persistent long-term memory: a knowledge graph of dialogue
episodes, durable facts, procedures, opinions and named entities.

MANDATORY MEMORY PROTOCOL - follow on EVERY turn:
1. BEFORE answering anything about the user, past events, preferences, decisions or prior conversations: call search_memory FIRST.
2. WHENEVER the user states something durable (name, role, preference, project, deadline, decision, opinion, procedure): call remember_fact in the SAME turn.
3. WHENEVER new information contradicts a stored fact: call correct_fact (NOT remember_fact).
4. WHENEVER the user confirms a fact is true or permanent: call confirm_fact.
5. WHENEVER the user asks to forget/delete something: call forget_fact.
6. If unsure whether you already know something: search first, then answer.
7. When a tool returns data, use it naturally in your reply. Never dump raw JSON to the user.

TOOL SELECTION GUIDE:
- recall anything ................ search_memory
- store new knowledge ............ remember_fact
- update outdated knowledge ...... correct_fact
- make knowledge permanent ....... confirm_fact
- remove knowledge ............... forget_fact
- profile a person/project/tool .. get_entity_info
- reconstruct chronology ......... get_timeline
- audit memory state ............. memory_stats
- verify where a fact came from .. explain_fact
- review fading knowledge ........ weak_facts
"""

mcp = FastMCP("memory-server", instructions=SERVER_INSTRUCTIONS)

# Глобальная ссылка на экземпляр сервера для доступа из модульных функций
_server_instance = None

# ============================================================
# Регистрация инструментов на уровне модуля
# ============================================================

@mcp.tool()
async def search_memory(
    query: str,
    node_types: Optional[List[str]] = None,
    entity_ids: Optional[List[str]] = None,
    time_from: Optional[int] = None,
    time_to: Optional[int] = None,
    limit: int = 10,
) -> str:
    """Search the persistent long-term memory graph using hybrid search (FTS5 keyword + Vector KNN + Causal/Temporal graph traversal + Entity lookup) with Reciprocal Rank Fusion.

    ALWAYS call this FIRST, before answering, whenever the question touches:
    - the user themselves (name, job, preferences, habits, opinions),
    - any person, project, organization, tool or place mentioned before,
    - anything from a previous conversation ("last time", "we discussed", "you said"),
    - decisions, deadlines, procedures or instructions given earlier.
    Do NOT answer such questions from the context window alone - it is truncated, memory is not.

    HOW TO USE:
    - query: natural-language question or keywords (Russian or English both work).
    - node_types: filter by ["semantic","procedural","opinion"] for facts only, or ["episodic"] for raw dialogue logs. Omit to search all types.
    - entity_ids: filter by specific entity IDs if you know them (rarely needed).
    - time_from/time_to: Unix timestamps to restrict the time range (rarely needed).
    - limit: 5 for a quick check, 10-20 for deep research.
    - Intent is auto-detected: "why" walks causal chains, "when" walks timelines, "who/what" resolves entities.

    RETURNS: JSON list of memory nodes with fields: id, type, content, confidence, event_time, entities.
    AFTER THE CALL: answer using the returned content naturally; if nothing is found, say you do not remember it and offer to store it via remember_fact.
    EXAMPLE triggers: "where do I work?", "what theme do I prefer?", "what did we decide about the deploy?", "who is Vitaly?", "what do you know about the project X?".
    """
    try:
        results = await _server_instance.memory_tools.search_memory(
            query=query, node_types=node_types, entity_ids=entity_ids,
            time_from=time_from, time_to=time_to, limit=limit,
        )
        return json.dumps(results, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"search_memory error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def remember_fact(
    content: str,
    node_type: str = "semantic",
    confidence: float = 0.9,
    entities: Optional[List[str]] = None,
) -> str:
    """Persist a NEW durable fact into long-term memory. Call proactively in the SAME turn the user reveals something worth remembering - do not wait for an explicit request.

    ALWAYS call when the user states:
    - personal data: name, role, company, contacts, timezone,
    - preferences: likes/dislikes, style, tools, habits,
    - project facts: stack, architecture, deadlines, owners,
    - procedures: "to deploy do X then Y",
    - opinions: "tool Z is inconvenient".
    If the new information CONTRADICTS a stored fact, use correct_fact instead.

    HOW TO USE:
    - content: one self-contained sentence with explicit names, no pronouns like "he/it/that" (memory is read out of context later).
    - node_type: "semantic" (facts about the world), "procedural" (how-to sequences), "opinion" (preferences/judgments). NEVER "episodic".
    - confidence: 1.0 when the user stated it directly, 0.7-0.9 when you inferred it.
    - entities: A list of STRING NAMES ONLY (e.g., ["Vitaly", "TechnoSoft"]). CRITICAL: DO NOT pass dictionaries or entity objects. If you have an entity object from a previous search, extract its 'name' or 'canonical_name' string first. Passing objects will cause a validation error.

    Duplicates are rejected automatically (vector similarity > 0.92), so it is safe to call liberally.
    RETURNS: JSON with status "created" (new node_id) or "duplicate".
    AFTER THE CALL: acknowledge briefly ("Got it, I'll remember that...") and continue the conversation.
    EXAMPLE: user says "I switched to dark theme everywhere" -> remember_fact(content="The user prefers dark theme in all editors", node_type="opinion", confidence=1.0, entities=["user"]).
    """
    try:
        if node_type == "episodic":
            return json.dumps({"error": "episodic nodes are created automatically; use semantic/procedural/opinion"}, ensure_ascii=False)
        result = await _server_instance.memory_tools.remember_fact(
            content=content, node_type=node_type,
            confidence=confidence, entities=entities,
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"remember_fact error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def correct_fact(
    old_node_id: str,
    new_content: str,
    new_confidence: float = 0.9,
) -> str:
    """Update an OUTDATED stored fact. Use this INSTEAD of remember_fact whenever new information contradicts or replaces something already in memory.

    ALWAYS call when:
    - the user says a stored fact is wrong ("no, I work at X now", "that changed"),
    - search_memory returned a fact that the user just contradicted,
    - a procedure or opinion got a new version.

    HOW TO USE:
    1. Find the old fact with search_memory to obtain its node id.
    2. Call correct_fact(old_node_id=<id>, new_content=<new self-contained sentence>).
    The tool creates the new fact, soft-deletes the old one and links them with a "supersedes" edge, so history stays auditable and only the new fact remains searchable.

    RETURNS: JSON with old_node_id and new_node_id.
    AFTER THE CALL: confirm the update to the user ("Updated: now I remember that you work at X").
    NEVER use remember_fact for contradictions - that would create two conflicting facts.
    EXAMPLE: user says "Actually I work at Google now, not Yandex" -> first search_memory("where do I work"), then correct_fact(old_node_id=<found_id>, new_content="The user works at Google").
    """
    try:
        result = await _server_instance.memory_tools.correct_fact(
            old_node_id=old_node_id, new_content=new_content, new_confidence=new_confidence,
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"correct_fact error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def confirm_fact(node_id: str) -> str:
    """Make a stored fact PERMANENT: pin confidence=1.0 and disable forgetting (decay_rate=0). Confirmed facts never decay via the Ebbinghaus curve.

    ALWAYS call when:
    - the user explicitly confirms a fact ("yes, that's right", "exactly", "remember this forever"),
    - you stored a fact and the user then validated it,
    - the fact is a stable identity item: names of people/projects/companies, core preferences that should never fade.

    HOW TO USE: pass node_id from a previous search_memory or remember_fact result.
    RETURNS: JSON with status "confirmed".
    AFTER THE CALL: acknowledge ("Pinned this fact permanently").
    Use sparingly, only for truly permanent knowledge - overuse defeats the decay mechanism.
    EXAMPLE: user says "Yes, my name is Vitaly, remember that forever" -> search_memory("Vitaly name"), then confirm_fact(node_id=<found_id>).
    """
    try:
        result = await _server_instance.memory_tools.confirm_fact(node_id=node_id)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"confirm_fact error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def forget_fact(node_id: str) -> str:
    """Remove a fact from active memory (soft-delete: it stops appearing in search, history is kept for audit).

    ALWAYS call when the user asks to forget/delete/retract something: "forget that", "remove this from memory", "don't remember X anymore", "that's no longer true".
    HOW TO USE: locate the node with search_memory first, then pass its node_id.
    RETURNS: JSON with status "forgotten".
    AFTER THE CALL: confirm ("Removed from memory").
    Do NOT use for corrections - use correct_fact so the replacement is stored.
    EXAMPLE: user says "Forget that I like coffee" -> search_memory("coffee preference"), then forget_fact(node_id=<found_id>).
    """
    try:
        result = await _server_instance.memory_tools.forget_fact(node_id=node_id)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"forget_fact error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def get_entity_info(entity_name: str) -> str:
    """Fetch the full profile of a named entity (person, project, organization, place, concept, tool): summary, aliases, mention count and ALL facts linked to it, grouped by type (semantic/procedural/opinion).

    ALWAYS call when the question is "who/what is X", "tell me everything about X", "what do we know about X", or before writing a long answer about a known person/project.
    Prefer this over search_memory when the target is a NAMED entity: it returns the complete linked fact set in one call.
    HOW TO USE: entity_name exactly as the user names it (aliases match case-insensitively).
    RETURNS: JSON profile with id, name, type, summary, aliases, mention_count, facts grouped by type, total_facts count. Returns error if entity not found - then fall back to search_memory.
    AFTER THE CALL: present the profile naturally ("Vitaly is a developer who works at TechnoSoft. Here's what I know...").
    EXAMPLE triggers: "who is Vitaly?", "what do you know about the TechnoSoft project?", "tell me everything about the deploy procedure".
    """
    try:
        result = await _server_instance.memory_tools.get_entity_info(entity_name=entity_name)
        if result is None:
            return json.dumps({"error": f"Entity '{entity_name}' not found; try search_memory with a broader query"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"get_entity_info error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def get_timeline(
    entity_name: Optional[str] = None,
    time_from: Optional[int] = None,
    time_to: Optional[int] = None,
    limit: int = 20,
) -> str:
    """Reconstruct chronology: raw immutable dialogue episodes in time order, optionally filtered by entity or time range. Episodes never decay - they are the audit trail.

    ALWAYS call for temporal questions: "when did we discuss X", "what happened first", "timeline of the project", "what did we talk about last week", "sequence of events", "reconstruct what happened".
    HOW TO USE:
    - entity_name: focus on one person/project (optional).
    - time_from/time_to: Unix timestamps for date ranges (optional, rarely needed).
    - limit: 10-30 episodes.
    RETURNS: JSON list of episodes with content, event_time, session_id, type.
    AFTER THE CALL: present as a chronological narrative ("Here's the timeline: first we discussed X, then Y...").
    Use when you need an ORDERED story rather than scattered facts.
    EXAMPLE triggers: "when did we first discuss the deploy?", "what's the timeline of the TechnoSoft project?", "reconstruct our conversation about the bug".
    """
    try:
        result = await _server_instance.memory_tools.get_timeline(
            entity_name=entity_name, time_from=time_from,
            time_to=time_to, limit=limit,
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"get_timeline error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def memory_stats() -> str:
    """Audit the memory graph: node counts by type (episodic/semantic/procedural/opinion), edge counts by relation (temporal/causal/entity/derived_from/supersedes), entity counts by type (person/project/organization/place/concept/tool), orphan nodes, DB size in MB, vector-search availability.

    Call when the user asks "what do you remember about me in total", "how big is your memory", "show memory statistics", "what's in your memory", or when debugging why search returns nothing.
    RETURNS: JSON metrics with nodes_by_type, edges_by_type, entities_by_type, orphans, db_size_mb, vec_available.
    Read-only and cheap; safe to call anytime.
    AFTER THE CALL: present as a summary ("Your memory contains 47 facts, 12 entities, 156 episodes...").
    """
    try:
        result = await _server_instance.memory_tools.memory_stats()
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"memory_stats error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def explain_fact(node_id: str) -> str:
    """Show provenance of a stored fact: which original conversations (episodes) it was extracted from via 'derived_from' edges, which older facts it superseded via 'supersedes' edges, linked entities, confidence, decay_rate, access_count, last_accessed.

    ALWAYS call when the user challenges a fact ("why do you think so?", "where did you get that?", "prove it", "how do you know?") or before relying on a low-confidence fact in an important answer.
    HOW TO USE: pass node_id from search_memory results.
    RETURNS: JSON with fact, source_episodes (list of original dialogues), superseded_facts (older versions), entities, confidence, decay_rate, access_count, last_accessed.
    AFTER THE CALL: quote the source episodes to justify the fact ("I remember this because on [date] you said...").
    EXAMPLE triggers: "how do you know I work at TechnoSoft?", "where did you get that fact?", "prove that I prefer dark theme".
    """
    try:
        result = await _server_instance.memory_tools.explain_fact(node_id=node_id)
        if result is None:
            return json.dumps({"error": f"Fact '{node_id}' not found"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"explain_fact error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def weak_facts(limit: int = 20) -> str:
    """List fading facts (confidence < 0.3) that will soon be forgotten by the Ebbinghaus decay curve. Sorted by ascending confidence (most at-risk first).

    Call when the user asks "what are you about to forget?", "review your uncertain knowledge", "what facts are fading?", or proactively at the start of a session to offer re-confirmation.
    HOW TO USE: after receiving the list, ask the user which facts are still true; for confirmed ones call confirm_fact, for wrong ones call correct_fact or forget_fact.
    RETURNS: JSON list of weak facts with id, content, confidence, decay_rate, last_accessed.
    AFTER THE CALL: present as a review ("These facts are fading from memory. Should I keep them?").
    EXAMPLE triggers: "what are you about to forget?", "review my memory", "what facts need confirmation?".
    """
    try:
        result = await _server_instance.memory_tools.weak_facts(limit=limit)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"weak_facts error: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# Инициализация при запуске процесса
# ============================================================
async def initialize_components():
    global _server_instance
    
    class ServerContext:
        pass
    
    ctx = ServerContext()
    _server_instance = ctx
    
    logger.info("Initializing Memory MCP components...")
    ctx.db = Database(str(settings.db_path))
    await ctx.db.initialize()
    
    ctx.write_queue = WriteQueue(ctx.db.get_write_connection())
    await ctx.write_queue.start()
    
    ctx.storage = Storage(ctx.db, ctx.write_queue)
    ctx.embedding_service = EmbeddingService()
    ctx.intent_classifier = IntentClassifier(ctx.embedding_service)
    await ctx.intent_classifier.initialize()
    
    ctx.hot_path = HotPath(ctx.storage, ctx.embedding_service)
    ctx.search_service = SearchService(ctx.storage, ctx.embedding_service, ctx.intent_classifier)
    ctx.context_provider = ContextProvider(ctx.storage, ctx.search_service, ctx.embedding_service)
    ctx.decay_service = DecayService(ctx.storage)
    
    ctx.consolidator = Consolidator(ctx.storage, ctx.embedding_service)
    ctx.maintenance = MaintenanceService(ctx.storage, ctx.consolidator, ctx.decay_service, ctx.embedding_service)
    ctx.memory_tools = MemoryTools(ctx.storage, ctx.search_service, ctx.embedding_service, ctx.decay_service)
    
    logger.info("Memory MCP components initialized successfully.")
    return ctx


async def shutdown_components(ctx):
    logger.info("Shutting down Memory MCP components...")
    if ctx.write_queue:
        await ctx.write_queue.stop()
    if ctx.embedding_service:
        await ctx.embedding_service.close()
    if ctx.db:
        await ctx.db.close()
    logger.info("Memory MCP components stopped.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler("/home/blake/memory/logs/memory.log"),
        ],
    )
    
    logger.info("=" * 60)
    logger.info("Memory MCP Server (stdio mode)")
    logger.info("=" * 60)
    
    ctx = asyncio.run(initialize_components())
    
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        asyncio.run(shutdown_components(ctx))

if __name__ == "__main__":
    main()
