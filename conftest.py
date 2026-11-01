# conftest.py
import os
import tempfile
import pytest

# Уникальная директория для тестов, чтобы не конфликтовать с реальной БД
_tmp = tempfile.mkdtemp(prefix="memory-test-")
os.environ["MEMORY_DATA_DIR"] = _tmp
os_env = {
    "MEMORY_EMBEDDING_API_URL": "http://127.0.0.1:9",  # Намеренно недоступен -> FTS-only fallback
    "MEMORY_LLM_API_URL": "http://127.0.0.1:9",
}
for k, v in os_env.items():
    os.environ[k] = v

pytest_plugins = ("pytest_asyncio",)
