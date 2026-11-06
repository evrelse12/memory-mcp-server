# tests/test_shutdown.py
"""SIGTERM must terminate the stdio process after a graceful teardown."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sigterm_terminates_process(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["MEMORY_DATA_DIR"] = str(tmp_path)   # isolate the DB for this process
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py")],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=env, cwd=str(ROOT))
    try:
        time.sleep(3.0)                      # allow startup + scheduler
        assert proc.poll() is None, "server exited before SIGTERM"
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    # Default handler re-raised the signal after teardown.
    assert proc.returncode in (-signal.SIGTERM, 0)
