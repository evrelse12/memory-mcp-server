# mcp_server/bridge.py
"""Optional stdio->http proxy for MCP clients that cannot speak streamable HTTP."""
from fastmcp import FastMCP
from config.settings import settings

url = f"http://{settings.daemon_host}:{settings.daemon_port}/mcp"
FastMCP.as_proxy(url, name="memory-bridge").run()
