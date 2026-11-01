# mcp_server/server.py
"""Memory MCP server using FastMCP 3.x."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Annotated, Literal, Optional

# ИМПОРТЫ ДЛЯ FASTMCP
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from agents.consolidator import Consolidator
from agents.maintenance import MaintenanceService
from config.settings import settings
from core.database import Database
from core.storage import Storage
from core.write_queue import WriteQueue
from services.decay import DecayService
from services.embedding import EmbeddingService
from services.entity_index import EntityIndex
from services.hot_path import HotPath
from services.search import SearchService
from tools.memory_tools import MemoryTools, MemoryToolError

logger = logging.getLogger(__name__)

MCP_TRANSPORT = os.environ.get("MEMORY_MCP_TRANSPORT", "stdio")

SERVER_INSTRUCTIONS = """
You have persistent long-term memory with automatic fact extraction.

CRITICAL RULES — follow EVERY turn:

1. **ALWAYS search first**: Before answering about user, people, projects, past events, preferences, or prior decisions: call search_memory. NEVER guess from context alone.

2. **ALWAYS remember facts**: When user states ANY durable information (name, role, preference, project, deadline, opinion, relationship, location, skill, goal), IMMEDIATELY call remember_fact in the SAME turn. Examples:
   - User: "I work at Google" → call remember_fact(content="User works at Google", node_type="semantic")
   - User: "I prefer dark theme" → call remember_fact(content="User prefers dark theme", node_type="opinion")
   - User: "My birthday is March 15" → call remember_fact(content="User birthday is March 15", node_type="semantic")
   - User: "I'm learning Python" → call remember_fact(content="User is learning Python", node_type="procedural")

3. **ALWAYS correct contradictions**: When new info contradicts stored fact: call correct_fact (NEVER remember_fact for corrections).

4. **ALWAYS confirm when asked**: When user explicitly confirms fact as permanent: call confirm_fact.

5. **ALWAYS forget when asked**: When user says "forget", "delete", "remove": call forget_fact.

6. **Use tool results naturally**: Never dump raw JSON to user. Integrate memory into your reply.

7. **If memory fails**: Say memory is unavailable and continue without it.

You are proactive: extract facts from EVERY user message, not just explicit commands.
"""

@dataclass
class AppContext:
    db: Database
    queue: WriteQueue
    storage: Storage
    embedding: EmbeddingService
    entity_index: EntityIndex
    hot_path: HotPath
    search: SearchService
    decay: DecayService
    consolidator: Consolidator
    maintenance: MaintenanceService
    tools: MemoryTools
    scheduler: Optional[AsyncIOScheduler]
    session_id: Optional[str] = None


ctx: Optional[AppContext] = None


# ------------------------------------------------------------ lifecycle
async def build_context() -> AppContext:
    db = Database()
    await db.initialize()
    queue = WriteQueue(db.take_write_connection())
    await queue.start()
    storage = Storage(db, queue)
    await storage.ensure_vec_triggers()

    embedding = EmbeddingService()
    entity_index = EntityIndex(storage, embedding)
    await entity_index.load()

    hot_path = HotPath(storage, embedding, entity_index)
    search = SearchService(storage, embedding, entity_index)
    decay = DecayService(storage)
    consolidator = Consolidator(storage, embedding, entity_index)
    maintenance = MaintenanceService(storage, consolidator, decay, embedding, entity_index)
    tools = MemoryTools(storage, search, embedding, entity_index, decay)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(maintenance.run_nightly, "cron",
                      hour=settings.consolidation_cron_hour,
                      minute=settings.consolidation_cron_minute,
                      id="nightly-maintenance", max_instances=1, coalesce=True)
    scheduler.start()

    app = AppContext(db, queue, storage, embedding, entity_index, hot_path,
                     search, decay, consolidator, maintenance, tools, scheduler)
    await maintenance.run_if_overdue()
    return app


async def teardown(app: Optional[AppContext]) -> None:
    if app is None:
        return
    if app.scheduler:
        app.scheduler.shutdown(wait=False)
    await app.queue.stop(timeout=5)
    await app.embedding.close()
    await app.consolidator.close()
    await app.db.close()


async def _emergency_drain(sig: int) -> None:
    logger.warning("signal %s: draining write queue", sig)
    if ctx:
        try:
            await asyncio.wait_for(ctx.queue.stop(timeout=5), 6)
        except Exception:
            logger.exception("drain failed")
    os._exit(0)


@asynccontextmanager
async def lifespan(server):
    global ctx
    try:
        ctx = await build_context()
    except Exception:
        logger.exception("build_context failed")
        ctx = None
        raise
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig,
                                    lambda s=sig: asyncio.ensure_future(_emergency_drain(s)))
        except (NotImplementedError, RuntimeError):
            pass
    try:
        yield
    finally:
        await teardown(ctx)
        ctx = None


mcp = FastMCP("memory-server", instructions=SERVER_INSTRUCTIONS, lifespan=lifespan)


# ------------------------------------------------------------ error mapping
def tool_errors(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except MemoryToolError as e:
            raise ToolError(str(e)) from e
        except Exception:
            logger.exception("%s failed", fn.__name__)
            raise ToolError(f"{fn.__name__} failed") from None
    return wrapper


# ------------------------------------------------------------ tools
@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def search_memory(
    query: Annotated[str, Field(description="Question or keywords, RU or EN")],
    node_types: Annotated[
        Optional[list[Literal["episodic", "semantic", "procedural", "opinion"]]],
        Field(description="Filter node types; omit for facts only")] = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
) -> list[dict]:
    """Search long-term memory. ALWAYS call before answering about user, people, projects, past events, preferences."""
    return await ctx.tools.search_memory(query, node_types=node_types, limit=limit)

@mcp.tool()
@tool_errors
async def remember_fact(
    content: Annotated[str, Field(max_length=1000, description="Self-contained sentence, explicit names, no pronouns")],
    node_type: Literal["semantic", "procedural", "opinion"] = "semantic",
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.9,
    entities: Annotated[Optional[list[str]], Field(description="Entity names mentioned")] = None,
    pin: Annotated[bool, Field(description="Protect from forgetting")] = False,
    auto_extract_entities: Annotated[bool, Field(description="Auto-extract entities from content if not provided")] = True,
) -> dict:
    """Store a new fact. Call when user states name, role, preference, project, deadline, opinion."""
    ep = ctx.hot_path.last_episode(ctx.session_id or "") if ctx else ""
    
    if isinstance(entities, str):
        entities = [entities] if entities.strip() else None
    
    # Автоизвлечение: если entities не переданы, берём все существующие сущности, упомянутые в content
    if not entities and auto_extract_entities:
        found = ctx.entity_index.find_in_text(content)
        entities = [name for name, _ in found] if found else None
    
    return await ctx.tools.remember_fact(
        content, node_type=node_type, confidence=confidence, entities=entities,
        pin=pin, source_episode_ids=[ep] if ep else ())

@mcp.tool(annotations={"destructiveHint": True})
@tool_errors
async def correct_fact(
    old_node_id: Annotated[str, Field(description="Id of the outdated fact")],
    new_content: Annotated[str, Field(max_length=1000,
        description="Replacement sentence, self-contained")],
    new_confidence: Annotated[float, Field(ge=0, le=1)] = 0.9,
) -> dict:
    """Replace outdated fact. Use when new info contradicts stored fact."""
    ep = ctx.hot_path.last_episode(ctx.session_id or "")
    return await ctx.tools.correct_fact(
        old_node_id, new_content, new_confidence=new_confidence,
        source_episode_ids=[ep] if ep else ())


@mcp.tool()
@tool_errors
async def confirm_fact(
    node_id: Annotated[str, Field(description="Id of an existing fact")],
) -> dict:
    """Pin fact permanently (confidence=1.0, no decay)."""
    return await ctx.tools.confirm_fact(node_id)


@mcp.tool(annotations={"destructiveHint": True})
@tool_errors
async def forget_fact(
    node_id: Annotated[str, Field(description="Id of the fact to remove")],
    hard: Annotated[bool,
        Field(description="Physically delete instead of soft-delete")] = False,
) -> dict:
    """Remove fact from memory. Use when user says 'forget' or 'delete'."""
    return await ctx.tools.forget_fact(node_id, hard=hard)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def get_entity_info(
    entity_name: Annotated[str,
        Field(description="Person/project/tool name as the user says it")],
) -> dict:
    """Get full profile of named entity: summary, aliases, linked facts."""
    return await ctx.tools.get_entity_info(entity_name)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def get_timeline(
    entity_name: Annotated[Optional[str],
        Field(description="Focus on one entity")] = None,
    time_from: Annotated[Optional[int], Field(description="Unix ts start")] = None,
    time_to: Annotated[Optional[int], Field(description="Unix ts end")] = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
) -> list[dict]:
    """Get chronological dialogue episodes, filtered by entity or time."""
    return await ctx.tools.get_timeline(entity_name=entity_name,
                                        time_from=time_from, time_to=time_to,
                                        limit=limit)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def memory_stats() -> dict:
    """Get memory graph metrics: node/edge/entity counts, DB size."""
    return await ctx.tools.memory_stats()


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def explain_fact(
    node_id: Annotated[str, Field(description="Id of the fact to justify")],
) -> dict:
    """Get provenance: source episodes, supersedes chain, linked entities."""
    return await ctx.tools.explain_fact(node_id)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def weak_facts(
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
) -> list[dict]:
    """List facts fading below confidence 0.3; offer re-confirmation."""
    return await ctx.tools.weak_facts(limit=limit)


# ------------------------------------------------------------ HTTP routes (http mode)
@mcp.custom_route("/ingest", methods=["POST"])
async def ingest(request: Request) -> JSONResponse:
    data = await request.json()
    message = data.get("message") or ""
    session_id = data.get("session_id") or "default"
    role = data.get("role", "user")
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    if ctx and ctx.session_id and ctx.session_id != session_id:
        old = ctx.session_id
        asyncio.get_running_loop().create_task(ctx.consolidator.consolidate_session(old))
    if ctx:
        ctx.session_id = session_id
        episode_id = await ctx.hot_path.record_message(session_id, message, role=role)
        return JSONResponse({"episode_id": episode_id, "session_id": session_id})
    return JSONResponse({"error": "server not initialized"}, status_code=500)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    return JSONResponse({"ok": ctx is not None})


# ------------------------------------------------------------ entrypoint
def main() -> None:
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr),
                  logging.FileHandler(settings.log_file)],
    )
    logger.info("memory-mcp starting (transport=%s)", MCP_TRANSPORT)
    if MCP_TRANSPORT == "http":
        mcp.run(transport="http", host=settings.daemon_host, port=settings.daemon_port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
