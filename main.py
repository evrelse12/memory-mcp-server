#!/usr/bin/env python3
# main.py - single entrypoint (stdio by default, http via MEMORY_MCP_TRANSPORT)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_server.server import main

if __name__ == "__main__":
    main()
