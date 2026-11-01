#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.server import main as run_server

if __name__ == "__main__":
    run_server()
