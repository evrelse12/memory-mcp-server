# core/write_queue.py
"""Single-writer queue: batched commits, per-op savepoints, raw ops, drain on stop."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class WriteOp:
    stmts: list[tuple[str, Any]]          # executed atomically
    future: Optional[asyncio.Future] = None
    many: bool = False                    # params of each stmt is a sequence (executemany)
    raw: bool = False                     # run outside any transaction (VACUUM, rebuild)


class WriteQueue:
    def __init__(self, conn: aiosqlite.Connection, maxsize: int = 1000, batch: int = 64):
        self._conn = conn
        self._batch = batch
        self._queue: asyncio.Queue[WriteOp] = asyncio.Queue(maxsize)
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._worker(), name="write-queue")

    async def stop(self, timeout: float = 10.0) -> None:
        if not self._task:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout)
        except asyncio.TimeoutError:
            logger.error("WriteQueue stop timeout: %d ops dropped", self._queue.qsize())
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    # ------------------------------------------------------------ public API
    async def execute(self, sql: str, params: Any = (), *, wait: bool = True,
                      many: bool = False) -> Optional[int]:
        """Single statement. Returns rowcount when wait=True."""
        return await self.transaction([(sql, params)], wait=wait, many=many)

    async def transaction(self, stmts: Sequence[tuple[str, Any]], *,
                          wait: bool = True, many: bool = False) -> Optional[int]:
        """Multiple statements committed atomically. Returns summed rowcount."""
        fut = self._fut(wait)
        self._queue.put_nowait(WriteOp(list(stmts), future=fut, many=many))
        return await fut if fut else None

    async def raw(self, sql: str, params: Any = (), *, wait: bool = True) -> Optional[int]:
        """Statement that cannot run inside a transaction (VACUUM, fts rebuild)."""
        fut = self._fut(wait)
        self._queue.put_nowait(WriteOp([(sql, params)], future=fut, raw=True))
        return await fut if fut else None

    # ------------------------------------------------------------ internals
    @staticmethod
    def _fut(wait: bool) -> Optional[asyncio.Future]:
        return asyncio.get_running_loop().create_future() if wait else None

    async def _run_batch(self, ops: list[WriteOp]) -> list[tuple[WriteOp, Optional[int], Optional[Exception]]]:
        out: list[tuple[WriteOp, Optional[int], Optional[Exception]]] = []
        
        # 1. Гарантируем чистое начало (сбрасываем любое зависшее состояние)
        try:
            await self._conn.execute("ROLLBACK")
        except Exception:
            pass
        await self._conn.execute("BEGIN")
        
        for op in ops:
            if op.raw:
                # raw-операции (VACUUM и т.д.) не могут быть внутри транзакции
                await self._conn.execute("COMMIT")
                try:
                    sql, params = op.stmts[0]
                    cur = await self._conn.execute(sql, params)
                    rc = cur.rowcount
                    await cur.close()  # <-- КРИТИЧНО: закрываем курсор
                    out.append((op, rc, None))
                except Exception as e:
                    out.append((op, None, e))
                await self._conn.execute("BEGIN")
                continue
            
            try:
                await self._conn.execute("SAVEPOINT op")
                n = 0
                for sql, params in op.stmts:
                    if op.many:
                        cur = await self._conn.executemany(sql, params)
                    else:
                        cur = await self._conn.execute(sql, params)
                    n += cur.rowcount
                    await cur.close()  # <-- КРИТИЧНО: закрываем курсор, иначе COMMIT упадёт
                await self._conn.execute("RELEASE op")
                out.append((op, n, None))
            except Exception as e:
                try:
                    await self._conn.execute("ROLLBACK TO op")
                    await self._conn.execute("RELEASE op")
                except Exception:
                    pass
                out.append((op, None, e))
        
        # 2. Финализируем пачку
        try:
            await self._conn.execute("COMMIT")
        except Exception as e:
            try:
                await self._conn.execute("ROLLBACK")
            except Exception:
                pass
            raise e
            
        return out

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
                self._queue.task_done()

