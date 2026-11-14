# tests/test_shutdown.py
"""SIGTERM must terminate the stdio process after a graceful teardown."""
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals / select on pipes")
def test_sigterm_terminates_process(tmp_path):
    env = dict(os.environ)
    env["MEMORY_DATA_DIR"] = str(tmp_path)
    env["MEMORY_LOG_LEVEL"] = "INFO"   # conftest forces WARNING; we need the startup line
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py")],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True, env=env, cwd=str(ROOT))
    try:
        deadline = time.monotonic() + 20
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            r, _, _ = select.select([proc.stderr], [], [], 0.5)
            if r:
                line = proc.stderr.readline()
                if "memory-mcp starting" in line:
                    ready = True
                    break
        assert ready, "server did not become ready"
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode in (-signal.SIGTERM, 0)
