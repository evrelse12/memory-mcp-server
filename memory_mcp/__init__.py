"""memory-mcp: long-term memory MCP server."""

import os

# Runtime defaults for FastMCP, applied before fastmcp is imported.
# The host environment always wins (setdefault), so these are overridable.
os.environ.setdefault("FASTMCP_TELEMETRY_MODE", "off")
os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")
os.environ.setdefault("FASTMCP_SHOW_SERVER_BANNER", "false")
