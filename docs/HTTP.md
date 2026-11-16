# HTTP mode (multi-client)

One shared memory server for several clients (open-webui, IDE, llama-server).
Default transport remains `stdio`; HTTP is opt-in.

## Enable

```bash
# generate a token
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

`.env`:
```
MEMORY_TRANSPORT=http
MEMORY_HTTP_HOST=127.0.0.1
MEMORY_HTTP_PORT=8090
MEMORY_HTTP_PATH=/mcp
MEMORY_HTTP_TOKEN=<generated>
```

Run: `./venv/bin/memory-mcp` → `http://127.0.0.1:8090/mcp`.

## Auth & security

- Bearer token: clients send `Authorization: Bearer <token>`.
- `StaticTokenVerifier` is a shared static secret (FastMCP marks it dev/test).
  Fine for local/single-user; there is **no TLS**.
- Loopback bind without a token is allowed (warning logged). Non-loopback bind
  **requires** a token (fail-fast).
- For LAN/remote: put it behind TLS (reverse proxy) or an SSH tunnel; do not
  expose plain HTTP. Never commit `.env`. For production use JWT/OAuth.

## Client example (FastMCP)

```python
from fastmcp import Client

async with Client("http://127.0.0.1:8090/mcp", auth="<token>") as c:
    print(await c.call_tool("memory_stats", {}))
```

Any streamable-HTTP MCP client works with the bearer header.

## Multi-client correctness

- All writes go through a single `WriteQueue` (serialized); reads are concurrent.
- The memory graph is **shared** across clients (that is the goal).
- Each client passes its own `session_id` so episodes/consolidation are
  per-conversation. Clients that omit it fall back to the auto-managed session.
- **Do not run two server processes on the same DB** (e.g. a stdio instance for
  llama-server plus an HTTP instance): the single-writer and in-memory
  EntityIndex assumptions are per-process. Run ONE HTTP server and point every
  client at it.

## llama-server

llama-server currently spawns a stdio MCP process. To share the single HTTP
server, either use a stdio→HTTP bridge (e.g. mcp-remote / supergateway) in front
of it, or verify llama-server can act as a streamable-HTTP MCP client. This
bridge is the follow-up to validate; until then, keep llama-server on its own
stdio instance only if you accept a separate DB.

## systemd (optional)

```ini
[Unit]
Description=memory-mcp HTTP server
After=network.target

[Service]
User=blake
WorkingDirectory=/home/blake/memory
Environment=MEMORY_TRANSPORT=http
Environment=MEMORY_HTTP_HOST=127.0.0.1
Environment=MEMORY_HTTP_PORT=8090
EnvironmentFile=/home/blake/memory/.env
ExecStart=/home/blake/memory/venv/bin/memory-mcp
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Logs go to stderr → journal (`journalctl -u memory-mcp -f`); file log stays off.
