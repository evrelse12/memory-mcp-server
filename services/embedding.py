# services/embedding.py
import httpx
import numpy as np
from typing import List, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Сервис для получения эмбеддингов через OpenAI-совместимый API."""
    
    def __init__(self, api_url: str = None, model: str = None):
        self.api_url = api_url or settings.embedding_api_url
        self.model = model or settings.embedding_model
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def close(self):
        await self.client.aclose()
        
    async def embed(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        if not texts:
            return []
            
        try:
            response = await self.client.post(
                f"{self.api_url}/v1/embeddings",
                json={
                    "model": self.model,
                    "input": texts
                }
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = []
            
            for item in data['data']:
                emb = np.array(item['embedding'], dtype=np.float32)
                if len(emb) > settings.embedding_dim:
                    emb = emb[:settings.embedding_dim]
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)
                
            logger.debug(f"Получено {len(embeddings)} эмбеддингов")
            return embeddings
            
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов: {e}", exc_info=True)
            return None
            
    async def embed_single(self, text: str) -> Optional[np.ndarray]:
        result = await self.embed([text])
        return result[0] if result else None
        
    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> Optional[List[np.ndarray]]:
        if not texts:
            return []
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await self.embed(batch)
            
            if embeddings is None:
                return None
                
            all_embeddings.extend(embeddings)
            
        return all_embeddings
