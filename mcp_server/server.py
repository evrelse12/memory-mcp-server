# mcp_server/server.py
"""Memory MCP server (stdio only). Graceful shutdown; short model instructions."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Annotated, Literal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from agents.consolidator import Consolidator
from agents.maintenance import MaintenanceService
from config.settings import settings
from core.database import Database
from core.storage import Storage
from core.tasks import drain as drain_tasks
from core.write_queue import WriteQueue
from services.decay import DecayService
from services.embedding import EmbeddingService
from services.entity_index import EntityIndex
from services.hot_path import HotPath
from services.search import SearchService
from tools.memory_tools import MemoryToolError, MemoryTools

logger = logging.getLogger(__name__)

# Short on purpose: small models follow few crisp rules better than long essays.
SERVER_INSTRUCTIONS = (
    "You have persistent long-term memory. Every turn: first call record_message "
    "with the user's exact text. Before answering about the user, people, projects "
    "or the past, call search_memory. Store durable facts with remember_fact; "
    "replace contradictions with correct_fact; pin with confirm_fact; delete with "
    "forget_fact. Never guess from context alone.")


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
    scheduler: AsyncIOScheduler | None
    session_id: str | None = None


ctx: AppContext | None = None


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
    consolidator = Consolidator(storage, search, embedding, entity_index)
    maintenance = MaintenanceService(storage, consolidator, decay, embedding,
                                     entity_index)
    tools = MemoryTools(storage, search, embedding, entity_index, decay)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(maintenance.run_nightly, "cron",
                      hour=settings.consolidation_cron_hour,
                      minute=settings.consolidation_cron_minute,
                      id="nightly-maintenance", max_instances=1, coalesce=True)
    scheduler.start()

    app = AppContext(db, queue, storage, embedding, entity_index, hot_path,
                     search, decay, consolidator, maintenance, tools, scheduler,
                     session_id=uuid.uuid4().hex[:12])
    await maintenance.run_if_overdue()   # spawns catch-up task via registry
    return app

async def teardown(app: AppContext | None) -> None:
    if app is None:
        return
    # 1. Stop scheduler first (no wait)
    if app.scheduler:
        try:
            app.scheduler.shutdown(wait=False)
        except Exception:
            pass
    # 2. Drain background tasks
    try:
        await asyncio.wait_for(drain_tasks(timeout=3), timeout=5)
    except (asyncio.TimeoutError, Exception):
        pass
    # 3. PRAGMA optimize BEFORE stopping the queue (it needs a live connection)
    try:
        await asyncio.wait_for(app.storage.raw("PRAGMA optimize"), timeout=2)
    except (asyncio.TimeoutError, Exception):
        pass
    # 4. Flush WriteQueue
    try:
        await asyncio.wait_for(app.queue.stop(timeout=3), timeout=5)
    except (asyncio.TimeoutError, Exception):
        pass
    # 5. Close services
    for svc in (app.embedding, app.consolidator, app.db):
        try:
            await asyncio.wait_for(svc.close(), timeout=2)
        except (asyncio.TimeoutError, Exception):
            pass


# Single-flight protection for concurrent signals
_SHUTTING_DOWN = False


async def _graceful_shutdown(sig: int) -> None:
    global ctx, _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return
    _SHUTTING_DOWN = True
    logger.warning("signal %s: shutting down gracefully", sig)
    if ctx:
        try:
            await asyncio.wait_for(teardown(ctx), timeout=10)
        except Exception:
            logger.exception("teardown failed")
        ctx = None
    # Restore default handler and re-raise the signal so the process
    # exits with the correct status and the parent sees it.
    loop = asyncio.get_running_loop()
    try:
        loop.remove_signal_handler(sig)
    except (NotImplementedError, RuntimeError):
        pass
    os.kill(os.getpid(), sig)

@asynccontextmanager
async def lifespan(server):
    global ctx
    ctx = await build_context()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.ensure_future(_graceful_shutdown(s)))
        except (NotImplementedError, RuntimeError):
            pass
    try:
        yield
    finally:
        await teardown(ctx)
        ctx = None


mcp = FastMCP("memory-server", instructions=SERVER_INSTRUCTIONS, lifespan=lifespan)


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


@mcp.tool()
@tool_errors
async def record_message(
    message: Annotated[str, Field(max_length=4000,
        description="The user's exact raw message for this turn")],
    role: Literal["user", "assistant"] = "user",
) -> dict:
    """Record a dialogue message as an episode. Call FIRST every turn."""
    session = ctx.session_id or "default"
    episode_id = await ctx.hot_path.record_message(session, message, role=role)
    return {"status": "recorded", "episode_id": episode_id}


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def search_memory(
    query: Annotated[str, Field(description="Question or keywords, RU or EN")],
    node_types: Annotated[
        list[Literal["episodic", "semantic", "procedural", "opinion"]] | None,
        Field(description="Filter node types; omit for facts only")] = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
) -> list[dict]:
    """Search long-term memory before answering about user, people, projects, past."""
    return await ctx.tools.search_memory(query, node_types=node_types, limit=limit)

@mcp.tool()
@tool_errors
async def remember_fact(
    content: Annotated[str, Field(max_length=1000,
        description="Self-contained sentence, explicit names, no pronouns")],
    node_type: Literal["semantic", "procedural", "opinion"] = "semantic",
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.9,
    entities: Annotated[list[str] | None,
        Field(description="Entity names; auto-extracted when omitted")] = None,
    pin: Annotated[bool, Field(description="Protect from forgetting")] = False,
    force: Annotated[bool, Field(
        description="Store even if a similar fact already exists")] = False,
) -> dict:
    """Store a durable fact. May return duplicate/similar instead of creating."""
    ep = await ctx.storage.last_episode_id(ctx.session_id or "default")
    return await ctx.tools.remember_fact(
        content, node_type=node_type, confidence=confidence, entities=entities,
        pin=pin, force=force, source_episode_ids=[ep] if ep else ())

@mcp.tool(annotations={"destructiveHint": True})
@tool_errors
async def correct_fact(
    old_node_id: Annotated[str, Field(description="Id of the outdated fact")],
    new_content: Annotated[str, Field(max_length=1000,
        description="Replacement sentence, self-contained")],
    new_confidence: Annotated[float, Field(ge=0, le=1)] = 0.9,
) -> dict:
    """Replace an outdated fact when new info contradicts it."""
    ep = await ctx.storage.last_episode_id(ctx.session_id or "default")
    return await ctx.tools.correct_fact(
        old_node_id, new_content, new_confidence=new_confidence,
        source_episode_ids=[ep] if ep else ())


@mcp.tool()
@tool_errors
async def confirm_fact(
    node_id: Annotated[str, Field(description="Id of an existing fact")],
) -> dict:
    """Pin a fact permanently (confidence=1.0, no decay)."""
    return await ctx.tools.confirm_fact(node_id)


@mcp.tool(annotations={"destructiveHint": True})
@tool_errors
async def forget_fact(
    node_id: Annotated[str, Field(description="Id of the fact to remove")],
    hard: Annotated[bool, Field(description="Physically delete")] = False,
) -> dict:
    """Remove a fact (soft by default, hard on request)."""
    return await ctx.tools.forget_fact(node_id, hard=hard)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def get_entity_info(
    entity_name: Annotated[str, Field(description="Person/project/tool name")],
) -> dict:
    """Entity profile: summary, aliases, linked facts."""
    return await ctx.tools.get_entity_info(entity_name)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def get_timeline(
    entity_name: Annotated[str | None, Field(description="One entity")] = None,
    time_from: Annotated[int | None, Field(description="Unix ts start")] = None,
    time_to: Annotated[int | None, Field(description="Unix ts end")] = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
) -> list[dict]:
    """Chronological dialogue episodes."""
    return await ctx.tools.get_timeline(entity_name=entity_name,
                                        time_from=time_from, time_to=time_to,
                                        limit=limit)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def memory_stats() -> dict:
    """Graph metrics and DB size."""
    return await ctx.tools.memory_stats()


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def explain_fact(
    node_id: Annotated[str, Field(description="Id of the fact to justify")],
) -> dict:
    """Provenance: source episodes, supersedes chain, entities."""
    return await ctx.tools.explain_fact(node_id)


@mcp.tool(annotations={"readOnlyHint": True})
@tool_errors
async def weak_facts(
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
) -> list[dict]:
    """Facts fading below effective confidence 0.3."""
    return await ctx.tools.weak_facts(limit=limit)


def main() -> None:
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr),
                  logging.FileHandler(settings.log_file)],
    )
    logger.info("memory-mcp starting (stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
