# agents/maintenance.py
import asyncio
import time
import logging
from core.storage import Storage
from services.embedding import EmbeddingService
from services.decay import DecayService
from agents.consolidator import Consolidator
from config.settings import settings

logger = logging.getLogger(__name__)

class MaintenanceService:
    """Ночное обслуживание памяти (cron в 03:00)."""
    
    def __init__(self, storage, consolidator, decay_service, embedding_service, llm_api_url=None):
        self.storage = storage
        self.consolidator = consolidator
        self.decay_service = decay_service
        self.embedding_service = embedding_service
        self.llm_api_url = llm_api_url or settings.llm_api_url
        
    async def run_nightly_maintenance(self):
        logger.info("=== Начало ночного обслуживания ===")
        start_time = time.time()
        
        try:
            await self._consolidate_unconsolidated_sessions()
            await self._apply_decay_and_pruning()
            await self._enrich_entities()
            await self._infer_causal_relationships()
        except Exception as e:
            logger.error(f"Ошибка ночного обслуживания: {e}", exc_info=True)
            
        elapsed = time.time() - start_time
        logger.info(f"=== Ночное обслуживание завершено за {elapsed:.2f}s ===")
        
    async def _consolidate_unconsolidated_sessions(self):
        logger.info("Шаг 1: Консолидация сессий")
        
        conn = self.storage.db.get_read_connection()
        sql = """
            SELECT DISTINCT session_id FROM nodes
            WHERE type = 'episodic' AND valid_until IS NULL AND session_id IS NOT NULL
        """
        
        async with conn.execute(sql) as cursor:
            rows = await cursor.fetchall()
            session_ids = [row['session_id'] for row in rows]
            
        if not session_ids:
            logger.info("Нет сессий для консолидации")
            return
            
        for session_id in session_ids:
            if not await self.consolidator.is_session_consolidated(session_id):
                await self.consolidator.consolidate_session(session_id)
                await asyncio.sleep(1)
                
    async def _apply_decay_and_pruning(self):
        logger.info("Шаг 2: Decay + pruning")
        result = await self.decay_service.apply()
        logger.info(f"Decay: {result['decayed']} обновлено, {result['pruned']} помечено для pruning")
        
    async def _enrich_entities(self):
        logger.info("Шаг 3: Entity enrichment")
        
        conn = self.storage.db.get_read_connection()
        sql = """
            SELECT * FROM entities
            WHERE mention_count >= 3 AND (summary IS NULL OR summary = '')
        """
        
        async with conn.execute(sql) as cursor:
            rows = await cursor.fetchall()
            entities = [dict(row) for row in rows]
            
        if not entities:
            logger.info("Нет сущностей для обогащения")
            return
            
        for entity in entities:
            try:
                summary = await self._generate_entity_summary(entity)
                if summary:
                    await self.storage.update_entity(entity['id'], {'summary': summary})
                    
                    text_for_embedding = f"{entity['canonical_name']}. {summary}"
                    embedding = await self.embedding_service.embed_single(text_for_embedding)
                    
                    if embedding is not None:
                        await self.storage.insert_entity_embedding(entity['id'], embedding)
                        
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Ошибка обогащения сущности: {e}")
                
    async def _generate_entity_summary(self, entity):
        nodes = await self.storage.get_nodes_for_entity(entity['id'])
        context_nodes = nodes[:10]
        context_text = "\n".join([f"- {node['content'][:200]}" for node in context_nodes])
        
        prompt = f"""Generate a concise summary (2-3 sentences) about: {entity['canonical_name']}

Context:
{context_text}

Summary:"""
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.llm_api_url}/v1/chat/completions",
                    json={
                        "model": "gemma-4-E4B-it-qat-UD-Q4_K_XL",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 200
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Ошибка генерации summary: {e}")
            return ""
            
    async def _infer_causal_relationships(self):
        logger.info("Шаг 4: Causal inference")
        
        conn = self.storage.db.get_read_connection()
        sql = """
            SELECT e1.id as id1, e1.content as content1,
                   e2.id as id2, e2.content as content2
            FROM nodes e1
            JOIN edges temp ON e1.id = temp.source_id AND temp.relation_type = 'temporal'
            JOIN nodes e2 ON temp.target_id = e2.id
            WHERE e1.type = 'episodic' AND e2.type = 'episodic'
              AND e1.valid_until IS NULL AND e2.valid_until IS NULL
            ORDER BY e1.event_time DESC LIMIT ?
        """
        
        async with conn.execute(sql, (settings.max_causal_pairs,)) as cursor:
            rows = await cursor.fetchall()
            pairs = [dict(row) for row in rows]
            
        if not pairs:
            logger.info("Нет пар эпизодов для анализа")
            return
            
        for pair in pairs:
            try:
                existing_edges = await self.storage.get_edges_from_node(pair['id1'], 'causal')
                if existing_edges:
                    continue
                    
                is_causal = await self._check_causality(pair['content1'], pair['content2'])
                if is_causal:
                    await self.storage.insert_edge({
                        'source_id': pair['id1'],
                        'target_id': pair['id2'],
                        'relation_type': 'causal',
                        'predicate': 'caused',
                        'confidence': settings.causal_confidence
                    })
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Ошибка анализа каузальности: {e}")
                
    async def _check_causality(self, content1, content2):
        prompt = f"""Does event 1 CAUSE event 2?

Event 1: {content1}
Event 2: {content2}

Answer YES or NO."""
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.llm_api_url}/v1/chat/completions",
                    json={
                        "model": "gemma-4-E4B-it-qat-UD-Q4_K_XL",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 10
                    }
                )
                response.raise_for_status()
                data = response.json()
                answer = data['choices'][0]['message']['content'].strip().upper()
                return 'YES' in answer
        except Exception as e:
            logger.error(f"Ошибка проверки каузальности: {e}")
            return False
