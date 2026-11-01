# tools/memory_tools.py
from typing import List, Dict, Any, Optional
import logging
from core.storage import Storage
from services.search import SearchService
from services.embedding import EmbeddingService
from services.decay import DecayService
from config.settings import settings

logger = logging.getLogger(__name__)

class MemoryTools:
    """10 инструментов для основного агента для работы с памятью."""
    
    def __init__(self, storage, search_service, embedding_service, decay_service):
        self.storage = storage
        self.search_service = search_service
        self.embedding_service = embedding_service
        self.decay_service = decay_service
        
    async def search_memory(self, query: str, node_types: Optional[List[str]] = None,
                           entity_ids: Optional[List[str]] = None,
                           time_from: Optional[int] = None, time_to: Optional[int] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Intent-aware гибридный поиск с фильтрами."""
        query_embedding = await self.embedding_service.embed_single(query)
        
        results = await self.search_service.search(
            query, query_embedding=query_embedding, limit=limit,
            node_types=node_types, entity_ids=entity_ids,
            time_from=time_from, time_to=time_to
        )
        
        for result in results:
            entities = await self.storage.get_entities_for_node(result['id'])
            result['entities'] = [
                {'id': e['id'], 'name': e['canonical_name'], 'type': e['type']}
                for e in entities
            ]
            
        return results
        
    async def remember_fact(self, content: str, node_type: str = 'semantic',
                           confidence: float = 0.9, entities: Optional[List[str]] = None) -> Dict[str, Any]:
        """Явно сохранить факт с дедупликацией."""
        if node_type == 'episodic':
            raise ValueError("Нельзя вручную создать episodic-узел")
            
        embedding = await self.embedding_service.embed_single(content)
        
        if embedding is not None:
            conn = self.storage.db.get_read_connection()
            sql = """
                SELECT node_id, distance FROM vec_nodes
                WHERE embedding MATCH ? ORDER BY distance LIMIT 1
            """
            
            async with conn.execute(sql, (embedding.tobytes(),)) as cursor:
                row = await cursor.fetchone()
                if row and row['distance'] < (1.0 - settings.dedup_similarity_threshold):
                    return {
                        'status': 'duplicate',
                        'existing_id': row['node_id'],
                        'similarity': 1.0 - row['distance']
                    }
                    
        node_id = await self.storage.insert_node({
            'type': node_type,
            'content': content,
            'confidence': confidence,
            'decay_rate': settings.default_decay_rate
        })
        
        if embedding is not None:
            await self.storage.insert_embedding(node_id, embedding)
            
        if entities:
            for entity_name in entities:
                entity = await self.storage.find_entity_by_name(entity_name)
                if entity:
                    await self.storage.link_node_entity(node_id, entity['id'])
                else:
                    entity_id = await self.storage.insert_entity({
                        'canonical_name': entity_name,
                        'type': 'concept',
                        'mention_count': 1
                    })
                    await self.storage.link_node_entity(node_id, entity_id)
                    
        return {
            'status': 'created',
            'node_id': node_id,
            'type': node_type,
            'confidence': confidence
        }
        
    async def correct_fact(self, old_node_id: str, new_content: str,
                          new_confidence: float = 0.9) -> Dict[str, Any]:
        """Заменить устаревший факт: soft-delete + supersedes edge."""
        result = await self.remember_fact(new_content, node_type='semantic', confidence=new_confidence)
        
        if result['status'] == 'duplicate':
            new_node_id = result['existing_id']
        else:
            new_node_id = result['node_id']
            
        await self.storage.insert_edge({
            'source_id': new_node_id,
            'target_id': old_node_id,
            'relation_type': 'supersedes',
            'predicate': 'replaces',
            'confidence': 0.95
        })
        
        await self.storage.soft_delete_node(old_node_id)
        await self.storage.update_node_fields(
            old_node_id,
            {'confidence': 0.3, 'decay_rate': 0.5}
        )
        
        return {
            'status': 'corrected',
            'old_node_id': old_node_id,
            'new_node_id': new_node_id
        }
        
    async def confirm_fact(self, node_id: str) -> Dict[str, Any]:
        """Защитить факт от decay: confidence=1.0, decay_rate=0.0."""
        await self.storage.update_node_fields(
            node_id,
            {'confidence': 1.0, 'decay_rate': 0.0}
        )
        return {'status': 'confirmed', 'node_id': node_id}
        
    async def forget_fact(self, node_id: str) -> Dict[str, Any]:
        """Soft-delete факта по запросу пользователя."""
        await self.storage.soft_delete_node(node_id)
        return {'status': 'forgotten', 'node_id': node_id}
        
    async def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Профиль сущности: summary + связанные факты + timeline."""
        entity = await self.storage.find_entity_by_name(entity_name)
        if not entity:
            return None
            
        nodes = await self.storage.get_nodes_for_entity(entity['id'])
        
        facts_by_type = {}
        for node in nodes:
            node_type = node['type']
            if node_type not in facts_by_type:
                facts_by_type[node_type] = []
            facts_by_type[node_type].append(node)
            
        return {
            'id': entity['id'],
            'name': entity['canonical_name'],
            'type': entity['type'],
            'summary': entity.get('summary'),
            'aliases': entity['aliases'],
            'mention_count': entity['mention_count'],
            'first_seen': entity['first_seen'],
            'last_updated': entity['last_updated'],
            'facts': facts_by_type,
            'total_facts': len(nodes)
        }
        
    async def get_timeline(self, entity_name: Optional[str] = None,
                          time_from: Optional[int] = None, time_to: Optional[int] = None,
                          limit: int = 20) -> List[Dict[str, Any]]:
        """Хронологические события с фильтрами."""
        conn = self.storage.db.get_read_connection()
        
        if entity_name:
            entity = await self.storage.find_entity_by_name(entity_name)
            if not entity:
                return []
                
            sql = """
                SELECT n.* FROM nodes n
                JOIN node_entities ne ON n.id = ne.node_id
                WHERE ne.entity_id = ? AND n.type = 'episodic' AND n.valid_until IS NULL
            """
            params = [entity['id']]
        else:
            sql = "SELECT * FROM nodes WHERE type = 'episodic' AND valid_until IS NULL"
            params = []
            
        if time_from:
            sql += " AND event_time >= ?"
            params.append(time_from)
            
        if time_to:
            sql += " AND event_time <= ?"
            params.append(time_to)
            
        sql += " ORDER BY event_time DESC LIMIT ?"
        params.append(limit)
        
        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
            
    async def memory_stats(self) -> Dict[str, Any]:
        """Метрики графа: узлы/рёбра по типам, сущности, orphans, размер БД."""
        import os
        conn = self.storage.db.get_read_connection()
        
        stats = {}
        
        sql = "SELECT type, COUNT(*) as count FROM nodes WHERE valid_until IS NULL GROUP BY type"
        async with conn.execute(sql) as cursor:
            rows = await cursor.fetchall()
            stats['nodes'] = {row['type']: row['count'] for row in rows}
            
        sql = "SELECT relation_type, COUNT(*) as count FROM edges WHERE valid_until IS NULL GROUP BY relation_type"
        async with conn.execute(sql) as cursor:
            rows = await cursor.fetchall()
            stats['edges'] = {row['relation_type']: row['count'] for row in rows}
            
        sql = "SELECT type, COUNT(*) as count FROM entities GROUP BY type"
        async with conn.execute(sql) as cursor:
            rows = await cursor.fetchall()
            stats['entities'] = {row['type']: row['count'] for row in rows}
            
        sql = """
            SELECT COUNT(*) as count FROM nodes n
            WHERE valid_until IS NULL AND NOT EXISTS (
                SELECT 1 FROM edges WHERE source_id = n.id OR target_id = n.id
            )
        """
        async with conn.execute(sql) as cursor:
            row = await cursor.fetchone()
            stats['orphans'] = row['count'] if row else 0
            
        db_size = os.path.getsize(self.storage.db.db_path)
        stats['db_size_mb'] = round(db_size / (1024 * 1024), 2)
        
        return stats
        
    async def explain_fact(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Провенанс: source episodes, supersedes chain, linked entities."""
        node = await self.storage.get_node(node_id)
        if not node:
            return None
            
        derived_edges = await self.storage.get_edges_from_node(node_id, 'derived_from')
        source_episodes = []
        
        for edge in derived_edges:
            episode = await self.storage.get_node(edge['target_id'])
            if episode:
                source_episodes.append(episode)
                
        supersedes_edges = await self.storage.get_edges_to_node(node_id, 'supersedes')
        superseded_facts = []
        
        for edge in supersedes_edges:
            old_fact = await self.storage.get_node(edge['source_id'])
            if old_fact:
                superseded_facts.append(old_fact)
                
        entities = await self.storage.get_entities_for_node(node_id)
        
        return {
            'fact': node,
            'source_episodes': source_episodes,
            'superseded_facts': superseded_facts,
            'entities': entities,
            'confidence': node['confidence'],
            'decay_rate': node['decay_rate'],
            'access_count': node['access_count'],
            'last_accessed': node['last_accessed']
        }
        
    async def weak_facts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Факты с низким confidence, которым скоро грозит decay."""
        return await self.decay_service.get_weak_facts(limit)
