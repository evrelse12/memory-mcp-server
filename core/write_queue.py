# core/write_queue.py
"""Single-writer queue: batched commits, savepoint per op, waits by default."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class WriteOp:
    stmts: list[tuple[str, tuple]]
    future: Optional[asyncio.Future]
    many: bool = False
    raw: bool = False


class WriteQueue:
    def __init__(self, conn: aiosqlite.Connection, *, batch: int = 64,
                 maxsize: int = 4096):
        self._conn = conn
        self._batch = batch
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._worker_task = asyncio.get_running_loop().create_task(self._worker())

    async def stop(self, timeout: float = 5) -> None:
        await self._queue.join()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    async def _enqueue(self, stmts, *, wait: bool, many: bool, raw: bool):
        fut = asyncio.get_running_loop().create_future() if wait else None
        # await put: apply backpressure instead of raising QueueFull at 1000 ops
        await self._queue.put(WriteOp(stmts, fut, many=many, raw=raw))
        return fut

    async def transaction(self, stmts, *, wait: bool = True,
                          many: bool = False):
        fut = await self._enqueue(list(stmts), wait=wait, many=many, raw=False)
        return await fut if fut else None

    async def execute(self, sql, params=(), *, wait: bool = True,
                      many: bool = False):
        return await self.transaction([(sql, params)], wait=wait, many=many)

    async def raw(self, sql, params=(), *, wait: bool = True):
        fut = await self._enqueue([(sql, params)], wait=wait, many=False, raw=True)
        return await fut if fut else None

    async def _worker(self) -> None:
        while True:
            first = await self._queue.get()
            ops = [first]
            while len(ops) < self._batch:
                try:
                    ops.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                results = await self._run_batch(ops)
            except asyncio.CancelledError:
                for _ in ops:
                    self._queue.task_done()
                raise
            except Exception as e:
                logger.exception("WriteQueue: batch aborted")
                try:
                    await self._conn.execute("ROLLBACK")
                except Exception:
                    pass
                results = [(op, None, e) for op in ops]
            for op, res, err in results:
                if op.future is not None and not op.future.done():
                    if err is None:
                        op.future.set_result(res)
                    else:
                        op.future.set_exception(err)
                elif op.future is None and err is not None:
                    # fire-and-forget failure must never be swallowed silently
                    logger.error("WriteQueue: fire-and-forget op failed: %s", err)
                self._queue.task_done()

    async def _run_batch(self, ops):
        out = []
        try:
            await self._conn.execute("ROLLBACK")
        except Exception:
            pass
        await self._conn.execute("BEGIN")
        for op in ops:
            if op.raw:
                await self._conn.execute("COMMIT")
                try:
                    sql, params = op.stmts[0]
                    cur = await self._conn.execute(sql, params)
                    rc = cur.rowcount
                    await cur.close()
                    out.append((op, rc, None))
                except Exception as e:
                    out.append((op, None, e))
                await self._conn.execute("BEGIN")
                continue
            try:
                await self._conn.execute("SAVEPOINT op")
                n = 0
                for sql, params in op.stmts:
                    cur = (await self._conn.executemany(sql, params)) if op.many \
                        else (await self._conn.execute(sql, params))
                    n += cur.rowcount
                    await cur.close()
                await self._conn.execute("RELEASE op")
                out.append((op, n, None))
            except Exception as e:
                try:
                    await self._conn.execute("ROLLBACK TO op")
                    await self._conn.execute("RELEASE op")
                except Exception:
                    pass
                out.append((op, None, e))
        try:
            await self._conn.execute("COMMIT")
        except Exception:
            try:
                await self._conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        return out
