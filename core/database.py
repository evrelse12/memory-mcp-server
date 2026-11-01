# core/database.py
import aiosqlite
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Путь к расширению sqlite-vec
VEC0_EXTENSION_PATH = "/usr/lib/vec0"


class Database:
    """
    Управление подключениями к SQLite базе данных.
    Использует WAL mode для параллельного чтения/записи.
    Поддерживает загрузку расширения sqlite-vec для векторного поиска.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._write_conn: Optional[aiosqlite.Connection] = None
        self._read_conn: Optional[aiosqlite.Connection] = None
        self._vec_loaded = False

    async def initialize(self):
        """Инициализация подключений, загрузка расширений и создание схемы БД."""
        # Создаём директорию для БД, если её нет
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Write connection (single writer)
        self._write_conn = await aiosqlite.connect(self.db_path)
        self._write_conn.row_factory = aiosqlite.Row

        # Read connection (query_only)
        self._read_conn = await aiosqlite.connect(self.db_path)
        self._read_conn.row_factory = aiosqlite.Row
        await self._read_conn.execute("PRAGMA query_only = ON")

        # Настройка WAL mode для параллельного чтения/записи
        await self._write_conn.execute("PRAGMA journal_mode = WAL")
        await self._write_conn.execute("PRAGMA synchronous = NORMAL")
        await self._write_conn.execute("PRAGMA cache_size = -64000")  # 64MB
        await self._write_conn.execute("PRAGMA foreign_keys = ON")
        await self._write_conn.execute("PRAGMA busy_timeout = 5000")
        await self._read_conn.execute("PRAGMA busy_timeout = 5000")

        # Загрузка sqlite-vec расширения в оба соединения
        self._vec_loaded = await self._load_sqlite_vec()

        # Создаём схему БД
        await self._create_schema()

        logger.info(f"База данных инициализирована: {self.db_path}")
        logger.info(f"Vector search (sqlite-vec): {'✓ доступен' if self._vec_loaded else '✗ fallback на FTS5'}")

    async def _load_sqlite_vec(self) -> bool:
        """
        Загрузка расширения sqlite-vec в оба соединения.
        Возвращает True если расширение загружено успешно.
        """
        # Пробуем несколько вариантов имени файла расширения
        extension_paths = [
            VEC0_EXTENSION_PATH,           # /usr/lib/vec0
            f"{VEC0_EXTENSION_PATH}.so",   # /usr/lib/vec0.so
            "/usr/local/lib/vec0",
            "/usr/local/lib/vec0.so",
        ]

        found_path = None
        for path in extension_paths:
            if os.path.exists(path):
                found_path = path
                break

        if not found_path:
            logger.warning(
                f"⚠ Расширение sqlite-vec не найдено по путям: {extension_paths}"
            )
            logger.warning("  Vector search будет работать через FTS5 fallback")
            return False

        # Загружаем расширение в оба соединения
        success_count = 0
        for conn_name, conn in [("write", self._write_conn), ("read", self._read_conn)]:
            try:
                # Включаем возможность загрузки расширений
                await conn.enable_load_extension(True)
                # Загружаем расширение (без .so в имени для SQLite)
                await conn.load_extension(found_path.replace(".so", ""))
                logger.info(f"✓ sqlite-vec загружен в {conn_name} соединение: {found_path}")
                success_count += 1
            except Exception as e:
                logger.warning(
                    f"⚠ Не удалось загрузить sqlite-vec в {conn_name} соединение: {e}"
                )

        # Успех, если загрузилось хотя бы в write соединение (основное)
        return success_count >= 1

    def is_vec_available(self) -> bool:
        """Проверить доступность sqlite-vec расширения."""
        return self._vec_loaded

    async def _create_schema(self):
        """Создание таблиц, индексов, триггеров FTS5 и векторных индексов."""
        async with self._write_conn.cursor() as cursor:
            # =========================================================
            # Таблица узлов (nodes)
            # =========================================================
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL CHECK(type IN 
                        ('episodic', 'semantic', 'procedural', 'opinion')),
                    content TEXT NOT NULL,
                    session_id TEXT,
                    event_time INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_accessed INTEGER,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    decay_rate REAL NOT NULL DEFAULT 0.1,
                    valid_until INTEGER,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            # =========================================================
            # Таблица рёбер (edges)
            # =========================================================
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL CHECK(relation_type IN
                        ('temporal', 'causal', 'entity', 'derived_from', 'supersedes')),
                    predicate TEXT,
                    weight REAL NOT NULL DEFAULT 1.0,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    valid_from INTEGER NOT NULL,
                    valid_until INTEGER,
                    evidence TEXT DEFAULT '[]',
                    created_at INTEGER NOT NULL
                )
            """)

            # =========================================================
            # Таблица сущностей (entities)
            # =========================================================
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    type TEXT NOT NULL CHECK(type IN
                        ('person', 'project', 'organization', 'place', 'concept', 'tool')),
                    aliases TEXT DEFAULT '[]',
                    summary TEXT,
                    embedding BLOB,
                    first_seen INTEGER NOT NULL,
                    last_updated INTEGER NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    attributes TEXT DEFAULT '{}'
                )
            """)

            # =========================================================
            # Junction table (node_entities) - связь many-to-many
            # =========================================================
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS node_entities (
                    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    PRIMARY KEY (node_id, entity_id)
                )
            """)

            # =========================================================
            # FTS5 индекс для полнотекстового поиска
            # External content table - не дублирует данные
            # =========================================================
            await cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    content,
                    content=nodes,
                    content_rowid=rowid,
                    tokenize='unicode61'
                )
            """)

            # =========================================================
            # Векторные индексы через sqlite-vec (условное создание)
            # =========================================================
            if self._vec_loaded:
                try:
                    await cursor.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(
                            node_id TEXT PRIMARY KEY,
                            embedding float[256]
                        )
                    """)

                    await cursor.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_entities USING vec0(
                            entity_id TEXT PRIMARY KEY,
                            embedding float[256]
                        )
                    """)
                    logger.info("✓ Векторные индексы sqlite-vec созданы (vec_nodes, vec_entities)")
                except Exception as e:
                    logger.error(f"Ошибка создания vec таблиц: {e}")
                    self._vec_loaded = False
            else:
                logger.warning("⚠ Пропуск создания vec_nodes и vec_entities (sqlite-vec недоступен)")

            # =========================================================
            # Индексы для быстрого поиска
            # =========================================================
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_event_time ON nodes(event_time)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_decay_rate ON nodes(decay_rate)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_valid_until ON nodes(valid_until)",
                "CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed)",
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)",
                "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)",
                "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type)",
                "CREATE INDEX IF NOT EXISTS idx_edges_valid_until ON edges(valid_until)",
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name)",
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
                "CREATE INDEX IF NOT EXISTS idx_entities_mentions ON entities(mention_count)",
                "CREATE INDEX IF NOT EXISTS idx_node_entities_node ON node_entities(node_id)",
                "CREATE INDEX IF NOT EXISTS idx_node_entities_entity ON node_entities(entity_id)",
            ]

            for idx_sql in indexes:
                await cursor.execute(idx_sql)

            # =========================================================
            # Триггеры для синхронизации FTS5 с таблицей nodes
            # Без них FTS5 индекс рассинхронизируется после UPDATE/DELETE
            # =========================================================

            # Триггер AFTER INSERT
            await cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                    INSERT INTO nodes_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)

            # Триггер AFTER DELETE
            await cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                    INSERT INTO nodes_fts(nodes_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                END
            """)

            # Триггер AFTER UPDATE
            await cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                    INSERT INTO nodes_fts(nodes_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                    INSERT INTO nodes_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)

        await self._write_conn.commit()
        logger.info("Схема базы данных создана")

    def get_write_connection(self) -> aiosqlite.Connection:
        """Получить write-соединение (single writer)."""
        if not self._write_conn:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._write_conn

    def get_read_connection(self) -> aiosqlite.Connection:
        """Получить read-соединение (query_only)."""
        if not self._read_conn:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._read_conn

    async def close(self):
        """Закрыть все подключения к базе данных."""
        if self._write_conn:
            await self._write_conn.close()
            self._write_conn = None
        if self._read_conn:
            await self._read_conn.close()
            self._read_conn = None
        logger.info("Подключения к базе данных закрыты")

    async def get_db_size_mb(self) -> float:
        """Получить размер базы данных в мегабайтах."""
        try:
            size = os.path.getsize(self.db_path)
            return round(size / (1024 * 1024), 2)
        except FileNotFoundError:
            return 0.0

    async def vacuum(self):
        """Запустить VACUUM для сжатия базы данных."""
        try:
            await self._write_conn.execute("VACUUM")
            logger.info("VACUUM выполнен успешно")
        except Exception as e:
            logger.error(f"Ошибка VACUUM: {e}", exc_info=True)

    async def get_statistics(self) -> dict:
        """Получить краткую статистику по базе данных."""
        conn = self.get_read_connection()
        stats = {}

        # Количество узлов по типам
        async with conn.execute(
            "SELECT type, COUNT(*) as count FROM nodes WHERE valid_until IS NULL GROUP BY type"
        ) as cursor:
            rows = await cursor.fetchall()
            stats["nodes_by_type"] = {row["type"]: row["count"] for row in rows}

        # Количество рёбер по типам
        async with conn.execute(
            "SELECT relation_type, COUNT(*) as count FROM edges WHERE valid_until IS NULL GROUP BY relation_type"
        ) as cursor:
            rows = await cursor.fetchall()
            stats["edges_by_type"] = {row["relation_type"]: row["count"] for row in rows}

        # Количество сущностей по типам
        async with conn.execute(
            "SELECT type, COUNT(*) as count FROM entities GROUP BY type"
        ) as cursor:
            rows = await cursor.fetchall()
            stats["entities_by_type"] = {row["type"]: row["count"] for row in rows}

        # Размер БД
        stats["db_size_mb"] = await self.get_db_size_mb()
        stats["vec_available"] = self._vec_loaded

        return stats
