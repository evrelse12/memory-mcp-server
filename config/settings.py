# config/settings.py
"""All tunables in one place; every path is derived, nothing hardcoded."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_prefix="MEMORY_", extra="ignore"
    )

    # --- Data locations (derived, overridable) ---
    data_dir: Path = Field(default=Path.home() / "memory")
    db_path: Path | None = None
    log_path: Path | None = None
    cache_dir: Path | None = None

    # --- LLM / embedding endpoints ---
    llm_api_url: str = "http://127.0.0.1:9931"
    llm_model: str = "gemma-4-E4B-it-qat-UD-Q4_K_XL"
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_tokens: int = Field(default=2048, ge=64)
    embedding_api_url: str = "http://127.0.0.1:8081"
    embedding_api_key: str | None = None
    embedding_model: str = "nomic-embed-text-v2-moe-q8_0"
    embedding_dim: int = Field(default=256, ge=64, le=4096)
    embedding_timeout: float = Field(default=10.0, gt=0)
    embedding_connect_timeout: float = Field(default=3.0, gt=0)
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    embedding_truncate_matryoshka: bool = True

    # --- Daemon ---
    daemon_host: str = "127.0.0.1"
    daemon_port: int = Field(default=9932, ge=1, le=65535)

    # --- Memory behaviour ---
    session_timeout_minutes: int = Field(default=30, ge=1)
    consolidation_cron_hour: int = Field(default=3, ge=0, le=23)
    consolidation_cron_minute: int = Field(default=0, ge=0, le=59)
    consolidation_chunk_chars: int = Field(default=6000, ge=500)
    max_facts_per_llm_call: int = Field(default=12, ge=1, le=50)
    conflict_band_min: float = Field(default=0.80, ge=0.0, lt=1.0)
    episode_retention_days: int = Field(default=0, ge=0)  # 0 = keep forever
    max_episode_chars: int = Field(default=4000, ge=100)

    default_decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    decay_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    decay_exponent: float = Field(default=0.8, gt=0.0, le=1.0)

    rrf_k: int = Field(default=60, ge=1)
    rrf_weight_fts: float = Field(default=1.0, ge=0.0)
    rrf_weight_vec: float = Field(default=1.0, ge=0.0)
    rrf_weight_graph: float = Field(default=1.0, ge=0.0)
    max_search_limit: int = Field(default=50, ge=1, le=200)
    dedup_similarity_threshold: float = Field(default=0.92, gt=0.0, lt=1.0)
    causal_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    max_causal_pairs: int = Field(default=50, ge=1, le=500)

    log_level: str = "INFO"

    # --- Derived paths ---
    @property
    def db_file(self) -> Path:
        return self.db_path or (self.data_dir / "db" / "memory.db")

    @property
    def log_file(self) -> Path:
        return self.log_path or (self.data_dir / "logs" / "memory.log")

    @property
    def intent_cache_file(self) -> Path:
        return (self.cache_dir or (self.data_dir / "cache")) / "intent_cache.json"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


settings = Settings()
