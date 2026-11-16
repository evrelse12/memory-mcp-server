# tests/test_phase_c.py
"""Phase C: HTTP transport, bearer auth config, multi-client sessions."""
import asyncio
import json
import sys

import pytest
from fastmcp import Client

from memory_mcp.mcp_server import server


def _data(res):
    text = "".join(getattr(c, "text", "") or "" for c in res.content)
    return json.loads(text) if text else None


def test_http_requires_token_off_loopback(monkeypatch):
    from memory_mcp.config import settings as s
    monkeypatch.setattr(s.settings, "transport", "http")
    monkeypatch.setattr(s.settings, "http_host", "0.0.0.0")
    monkeypatch.setattr(s.settings, "http_token", None)
    with pytest.raises(RuntimeError):
        server._validate_http_settings()


def test_http_loopback_no_token_ok(monkeypatch):
    from memory_mcp.config import settings as s
    monkeypatch.setattr(s.settings, "transport", "http")
    monkeypatch.setattr(s.settings, "http_host", "127.0.0.1")
    monkeypatch.setattr(s.settings, "http_token", None)
    server._validate_http_settings()   # must not raise


@pytest.mark.asyncio
async def test_build_auth(monkeypatch):
    from memory_mcp.config import settings as s
    monkeypatch.setattr(s.settings, "http_token", "secret123")
    auth = server._build_auth()
    assert auth is not None
    assert await auth.verify_token("secret123") is not None
    assert await auth.verify_token("wrong") is None
    monkeypatch.setattr(s.settings, "http_token", None)
    assert server._build_auth() is None


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32",
                    reason="threaded uvicorn; verify manually on Windows")
async def test_http_transport_multi_session(fresh_db):
    import threading

    import uvicorn

    app = server.mcp.http_app(path="/mcp")
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    srv = uvicorn.Server(config)
    srv.install_signal_handlers = lambda: None   # not the main thread
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        for _ in range(250):
            if srv.started:
                break
            await asyncio.sleep(0.02)
        assert srv.started, "uvicorn did not start"
        port = srv.servers[0].sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/mcp"
        # client 1: own session, writes a fact
        async with Client(url) as c1:
            await c1.call_tool("record_message",
                               {"message": "msg from A", "session_id": "chat-A"})
            await c1.call_tool("remember_fact",
                               {"content": "User works at Yandex", "session_id": "chat-A"})
        # client 2: new MCP session, sees shared graph, own session id
        async with Client(url) as c2:
            await c2.call_tool("record_message",
                               {"message": "msg from B", "session_id": "chat-B"})
            r = _data(await c2.call_tool("search_memory", {"query": "Yandex"}))
            assert any("Yandex" in x["content"] for x in r)
            tl = _data(await c2.call_tool("get_timeline", {"limit": 10}))
            assert {"chat-A", "chat-B"} <= {e["session_id"] for e in tl}
    finally:
        srv.should_exit = True
        thread.join(timeout=10)
