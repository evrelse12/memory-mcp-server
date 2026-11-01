#!/usr/bin/env python3
import asyncio, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.database import Database
from core.write_queue import WriteQueue
from core.storage import Storage
from services.embedding import EmbeddingService
from agents.consolidator import Consolidator
from agents.maintenance import MaintenanceService
from services.decay import DecayService
from config.settings import settings

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting nightly maintenance...")
    
    db = Database(str(settings.db_path) + ".maintenance") # Или используйте основную, но будьте осторожны с блокировками
    # Лучше использовать основную, но с таймаутами
    db = Database(str(settings.db_path))
    await db.initialize()
    
    write_queue = WriteQueue(db.get_write_connection())
    await write_queue.start()
    
    storage = Storage(db, write_queue)
    embedding = EmbeddingService()
    consolidator = Consolidator(storage, embedding)
    decay = DecayService(storage)
    maintenance = MaintenanceService(storage, consolidator, decay, embedding)
    
    await maintenance.run_nightly_maintenance()
    
    await write_queue.stop()
    await embedding.close()
    await db.close()
    logging.info("Nightly maintenance completed.")

if __name__ == "__aio__": # Исправлено на стандартный запуск
    pass

if __name__ == "__main__":
    asyncio.run(main())
