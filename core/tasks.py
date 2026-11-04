# core/tasks.py
"""Registry for fire-and-forget tasks so they are never garbage-collected."""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_TASKS: set[asyncio.Task] = set()


def spawn(coro, *, name: str | None = None) -> asyncio.Task:
    """Create a background task and keep a strong reference until it finishes."""
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


async def drain(timeout: float = 5.0) -> None:
    """Wait for all background tasks to finish, with a timeout."""
    if _TASKS:
        tasks = set(_TASKS)
        try:
            await asyncio.wait(tasks, timeout=timeout)
        except Exception:
            pass
        # Cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
