#!/usr/bin/env python3
# mcp_stdio_bridge.py
"""stdio-мост: llama-server спавнит этот процесс по stdio, а он прозрачно
проксирует list_tools/call_tool в всегда-включённый демон памяти по Streamable HTTP.
Описания инструментов и схемы берутся из демона без дублирования."""
import asyncio
import logging
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

DAEMON_URL = os.environ.get("MEMORY_DAEMON_URL", "http://127.0.0.1:9932/mcp")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
LOG = logging.getLogger("memory-bridge")

bridge = Server("memory-bridge")

_state = {"cm_client": None, "cm_session": None, "session": None}


async def _connect():
    if _state["session"] is not None:
        return _state["session"]
    cm_client = streamablehttp_client(DAEMON_URL)
    read, write, _ = await cm_client.__aenter__()
    cm_session = ClientSession(read, write)
    session = await cm_session.__aenter__()
    await session.initialize()
    _state.update(cm_client=cm_client, cm_session=cm_session, session=session)
    LOG.info(f"Подключено к демону: {DAEMON_URL}")
    return session


async def _reset():
    for key in ("cm_session", "cm_client"):
        cm = _state.pop(key, None)
        if cm:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
    _state["session"] = None


async def _with_retry(factory):
    """Один реконнект при обрыве (например, после рестарта демона)."""
    for attempt in (1, 2):
        try:
            session = await _connect()
            return await factory(session)
        except Exception as e:
            LOG.warning(f"Вызов к демону не удался (попытка {attempt}): {e}")
            await _reset()
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


@bridge.list_tools()
async def list_tools():
    async def _f(session):
        res = await session.list_tools()
        return res.tools          # дословно из демона: name, description, inputSchema
    return await _with_retry(_f)


@bridge.call_tool()
async def call_tool(name, arguments):
    async def _f(session):
        res = await session.call_tool(name, arguments)
        return res.content        # дословно контент-блоки ответа демона
    return await _with_retry(_f)


async def _main():
    async with stdio_server() as (read_stream, write_stream):
        await bridge.run(read_stream, write_stream, bridge.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            asyncio.run(_reset())
        except Exception:
            pass
