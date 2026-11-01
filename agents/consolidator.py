# agents/consolidator.py
import json
import time
from typing import List, Dict, Any, Optional
import logging
import httpx
from core.storage import Storage
from services.embedding import EmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)

class Consolidator:
    """Консолидатор сессий."""
    
    CONSOLIDATION_PROMPT = """You are a memory consolidation agent. Extract durable knowledge.

Episodes:
{episodes}

Return JSON:
{{
    "semantic": [{{"content": "fact", "confidence": 0.8, "entities": ["entity1"]}}],
    "procedural": [{{"content": "procedure", "confidence": 0.7}}],
    "opinion": [{{"content": "opinion", "confidence": 0.6}}],
    "entities": [{{"name": "Entity", "type": "person|project|organization|concept|tool"}}]
}}
"""
    
    def __init__(self, storage, embedding_service, llm_api_url=None):
        self.storage = storage
        self.embedding_service = embedding_service
        self.llm_api_url = llm_api_url or settings.llm_api_url
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def close(self):
        await self.client.aclose()
        
    async def consolidate_session(self, session_id: str) -> bool:
        logger.info(f"Начало консолидации сессии: {session_id}")
        
        try:
            if await self.is_session_consolidated(session_id):
                logger.info(f"Сессия {session_id} уже консолидирована")
                return True
                
            episodes = await self.get_session_episodes(session_id)
            if not episodes:
                await self.mark_session_consolidated(session_id)
                return True
                
            extracted = await self._call_llm_for_extraction(episodes)
            if not extracted:
                return False
                
            await self._save_extracted_knowledge(session_id, episodes, extracted)
            await self.extract_and_link_entities(session_id, extracted.get('entities', []))
            await self.mark_session_consolidated(session_id)
            
            logger.info(f"Сессия {session_id} успешно консолидирована")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка консолидации: {e}", exc_info=True)
            return False
            
    async def is_session_consolidated(self, session_id: str) -> bool:
        conn = self.storage.db.get_read_connection()
        query = """
            SELECT COUNT(*) as cnt FROM nodes n
            JOIN edges e ON n.id = e.source_id
            WHERE n.type IN ('semantic', 'procedural', 'opinion')
              AND e.relation_type = 'derived_from'
              AND e.target_id IN (SELECT id FROM nodes WHERE session_id = ? AND type = 'episodic')
              AND n.valid_until IS NULL
        """
        
        async with conn.execute(query, (session_id,)) as cursor:
            row = await cursor.fetchone()
            count = row['cnt'] if row else 0
        return count > 0
        
    async def get_session_episodes(self, session_id: str) -> List[Dict[str, Any]]:
        return await self.storage.get_nodes_by_session(session_id)
        
    async def _call_llm_for_extraction(self, episodes: List[Dict]) -> Optional[Dict]:
        episodes_text = "\n".join([f"[{i+1}] {ep['content']}" for i, ep in enumerate(episodes)])
        prompt = self.CONSOLIDATION_PROMPT.format(episodes=episodes_text)
        
        try:
            response = await self.client.post(
                f"{self.llm_api_url}/v1/chat/completions",
                json={
                    "model": "gemma-4-E4B-it-qat-UD-Q4_K_XL",
                    "messages": [
                        {"role": "system", "content": "You are a memory consolidation agent."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            return self._extract_json(content)
            
        except Exception as e:
            logger.error(f"Ошибка вызова LLM: {e}", exc_info=True)
            return None
            
    def _extract_json(self, text: str) -> Optional[Dict]:
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
        
    async def _save_extracted_knowledge(self, session_id, episodes, extracted):
        episode_ids = [ep['id'] for ep in episodes]
        
        for node_type in ['semantic', 'procedural', 'opinion']:
            facts = extracted.get(node_type, [])
            
            for fact in facts:
                content = fact.get('content', '')
                confidence = fact.get('confidence', 0.8)
                
                if not content:
                    continue
                    
                node_id = await self.storage.insert_node({
                    'type': node_type,
                    'content': content,
                    'session_id': session_id,
                    'confidence': confidence,
                    'decay_rate': settings.default_decay_rate
                })
                
                embedding = await self.embedding_service.embed_single(content)
                if embedding is not None:
                    await self.storage.insert_embedding(node_id, embedding)
                    
                for episode_id in episode_ids:
                    await self.storage.insert_edge({
                        'source_id': node_id,
                        'target_id': episode_id,
                        'relation_type': 'derived_from',
                        'predicate': 'extracted_from',
                        'confidence': confidence
                    })
                    
    async def extract_and_link_entities(self, session_id, entities):
        for entity_data in entities:
            name = entity_data.get('name', '')
            entity_type = entity_data.get('type', 'concept')
            
            if not name:
                continue
                
            existing = await self.storage.find_entity_by_name(name)
            
            if existing:
                await self.storage.increment_entity_mentions(existing['id'])
                entity_id = existing['id']
            else:
                entity_id = await self.storage.insert_entity({
                    'canonical_name': name,
                    'type': entity_type,
                    'aliases': [],
                    'mention_count': 1
                })
                
                embedding = await self.embedding_service.embed_single(name)
                if embedding is not None:
                    await self.storage.insert_entity_embedding(entity_id, embedding)
                    
            conn = self.storage.db.get_read_connection()
            sql = """
                SELECT id FROM nodes
                WHERE session_id = ? AND type IN ('semantic', 'procedural', 'opinion')
                  AND valid_until IS NULL
            """
            
            async with conn.execute(sql, (session_id,)) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    await self.storage.link_node_entity(row['id'], entity_id)
                    
    async def resolve_conflict(self, old_node_id: str, new_node_id: str):
        await self.storage.update_node_fields(
            old_node_id,
            {'confidence': 0.3, 'decay_rate': 0.5}
        )
        await self.storage.soft_delete_node(old_node_id)
        await self.storage.insert_edge({
            'source_id': new_node_id,
            'target_id': old_node_id,
            'relation_type': 'supersedes',
            'predicate': 'replaces',
            'confidence': 0.9
        })
        
    async def mark_session_consolidated(self, session_id: str):
        logger.info(f"Сессия {session_id} отмечена как консолидированная")
