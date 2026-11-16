# conftest.py
"""Project root on sys.path; unreachable LLM/embed endpoints; shared lifespan."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force offline degradation for the whole suite — override, don't setdefault:
# a developer's exported MEMORY_* vars must never point tests at live services.
os.environ["MEMORY_EMBEDDING_API_URL"] = "http://127.0.0.1:9"
os.environ["MEMORY_LLM_API_URL"] = "http://127.0.0.1:9"
os.environ["MEMORY_LOG_LEVEL"] = "WARNING"


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"

@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Each test gets its own database directory; no shared state between tests."""
    from memory_mcp.config import settings as s

    monkeypatch.setattr(s.settings, "data_dir", tmp_path)
    (tmp_path / "db").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path

@pytest.fixture(autouse=True)
def _no_catchup_maintenance(monkeypatch):
    """Tests must not spawn background nightly (network NLI download + races)."""
    from memory_mcp.agents.maintenance import MaintenanceService

    async def _noop(self):
        return None

    monkeypatch.setattr(MaintenanceService, "run_if_overdue", _noop)
