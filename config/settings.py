# config/settings.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Base paths
    base_dir: Path = Field(default=Path("/home/blake/memory"))
    db_dir: Path = Field(default=Path("/home/blake/memory/db"))
    db_path: Path = Field(default=Path("/home/blake/memory/db/memory.db"))
    
    # MCP Server
    mcp_port: int = Field(default=9932)
    mcp_host: str = Field(default="localhost")
    
    # API endpoints
    llm_api_url: str = Field(default="http://localhost:9931")
    embedding_api_url: str = Field(default="http://localhost:8081")
    
    # Embedding settings
    embedding_dim: int = Field(default=256)
    embedding_model: str = Field(default="nomic-embed-text-v2-moe-q8_0")
    
    # Memory settings
    session_timeout_minutes: int = Field(default=30)
    consolidation_cron_hour: int = Field(default=3)
    consolidation_cron_minute: int = Field(default=0)
    
    # Decay settings
    default_decay_rate: float = Field(default=0.1)
    decay_threshold: float = Field(default=0.05)
    
    # Search settings
    rrf_k: int = Field(default=60)
    rrf_weight_fts: float = Field(default=1.0)
    rrf_weight_vec: float = Field(default=1.0)
    rrf_weight_graph: float = Field(default=1.0)
    
    # Intent classifier
    intent_threshold: float = Field(default=0.45)
    
    # Consolidation
    causal_confidence: float = Field(default=0.7)
    max_causal_pairs: int = Field(default=50)
    
    # Graph traversal
    max_graph_depth_simple: int = Field(default=2)
    max_graph_depth_complex: int = Field(default=4)
    
    # Deduplication
    dedup_similarity_threshold: float = Field(default=0.92)
    
    # Context budget
    context_budget_facts: float = Field(default=0.40)
    context_budget_entities: float = Field(default=0.25)
    context_budget_temporal: float = Field(default=0.25)
    context_budget_evidence: float = Field(default=0.10)
    
    # Logging
    log_level: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        env_prefix = "MEMORY_"

settings = Settings()
