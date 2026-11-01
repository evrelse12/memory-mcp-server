# services/hot_path.py
import asyncio
import time
from typing import Optional
import logging
from core.storage import Storage
from services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

class HotPath:
    """Hot path для обработки входящих сообщений."""
    
    def __init__(self, storage: Storage, embedding_service: EmbeddingService):
        self.storage = storage
        self.embedding_service = embedding_service
        self._session_last_episode = {}
        
    async def process_message(self, content: str, session_id: str, message_type: str = 'user') -> str:
        now = int(time.time())
        
        episode_data = {
            'type': 'episodic',
            'content': content,
            'session_id': session_id,
            'event_time': now,
            'confidence': 1.0,
            'decay_rate': 0.0,
            'metadata': {'message_type': message_type}
        }
        
        episode_id = await self.storage.insert_node(episode_data)
        logger.info(f"Создан эпизод: {episode_id} (сессия: {session_id})")
        
        last_episode_id = self._session_last_episode.get(session_id)
        if last_episode_id:
            await self.storage.insert_edge({
                'source_id': episode_id,
                'target_id': last_episode_id,
                'relation_type': 'temporal',
                'predicate': 'follows',
                'weight': 1.0,
                'confidence': 1.0
            })
            
        self._session_last_episode[session_id] = episode_id
        
        asyncio.create_task(self._slow_path(episode_id, content))
        
        return episode_id
        
    async def _slow_path(self, episode_id: str, content: str):
        try:
            embedding = await self.embedding_service.embed_single(content)
            if embedding is not None:
                await self.storage.insert_embedding(episode_id, embedding)
        except Exception as e:
            logger.error(f"Ошибка в slow path: {e}", exc_info=True)
