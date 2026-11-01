# core/write_queue.py
import asyncio
import aiosqlite
from typing import Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class WriteOp:
    """Операция записи в БД"""
    sql: str
    params: tuple
    future: Optional[asyncio.Future] = None
    executemany: bool = False

class WriteQueue:
    """Single-writer queue для всех write-операций в БД."""
    
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn
        self.queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._worker())
        logger.info("WriteQueue worker запущен")
        
    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("WriteQueue worker остановлен")
        
    async def _worker(self):
        while self._running:
            try:
                op = await self.queue.get()
                
                try:
                    if op.executemany:
                        async with self.conn.executemany(op.sql, op.params) as cursor:
                            result = cursor.lastrowid
                    else:
                        async with self.conn.execute(op.sql, op.params) as cursor:
                            result = cursor.lastrowid
                            
                    await self.conn.commit()
                    
                    if op.future and not op.future.done():
                        op.future.set_result(result)
                        
                except Exception as e:
                    logger.error(f"Ошибка выполнения write-операции: {e}", exc_info=True)
                    if op.future and not op.future.done():
                        op.future.set_exception(e)
                    await self.conn.rollback()
                    
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Неожиданная ошибка в worker: {e}", exc_info=True)
                await asyncio.sleep(1)
                
    async def execute(self, sql: str, params: tuple, wait: bool = False) -> Optional[Any]:
        future = None
        if wait:
            future = asyncio.Future()
            
        op = WriteOp(sql=sql, params=params, future=future)
        await self.queue.put(op)
        
        if wait and future:
            return await future
        return None
        
    async def executemany(self, sql: str, params_list: list, wait: bool = False):
        future = None
        if wait:
            future = asyncio.Future()
            
        op = WriteOp(sql=sql, params=params_list, future=future, executemany=True)
        await self.queue.put(op)
        
        if wait and future:
            return await future
        return None
