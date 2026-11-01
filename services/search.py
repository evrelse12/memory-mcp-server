# services/search.py
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from config.settings import settings
from core.storage import Storage
from services.embedding import EmbeddingService
from services.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)

class SearchService:
    """Гибридный поиск с Reciprocal Rank Fusion и fallback на LIKE."""
    
    def __init__(self, storage: Storage, embedding_service: EmbeddingService, intent_classifier: IntentClassifier):
        self.storage = storage
        self.embedding_service = embedding_service
        self.intent_classifier = intent_classifier
        
    async def search(self, query: str, query_embedding: Optional[np.ndarray] = None,
                     limit: int = 10, node_types: Optional[List[str]] = None,
                     entity_ids: Optional[List[str]] = None,
                     time_from: Optional[int] = None, time_to: Optional[int] = None) -> List[Dict[str, Any]]:
        intent = await self.intent_classifier.classify(query, query_embedding)
        is_complex = self._classify_query_complexity(query)
        
        # 1. FTS5 поиск
        fts_results = await self._search_fts(query, limit * 3)
        
        # 2. Vector поиск (если есть эмбеддинг)
        vec_results = await self._search_vector(query_embedding, limit * 3) if query_embedding is not None else []
        
        # 3. Fallback: LIKE поиск, если FTS5 ничего не нашёл
        if not fts_results and not vec_results:
            logger.info(f"FTS5 и vector не нашли результатов, используем LIKE fallback для: {query}")
            like_results = await self._search_like(query, limit * 3)
            fts_results = like_results  # Используем LIKE результаты как основные
        
        # 4. Graph поиск
        graph_results = await self._search_graph(query, intent, is_complex, limit * 3)
        
        # 5. Entity поиск
        entity_results = await self._search_entity(query, intent, limit * 3) if intent in ['who', 'what'] else []
        
        # Применяем фильтры
        if node_types:
            fts_results = [r for r in fts_results if r['type'] in node_types]
            vec_results = [r for r in vec_results if r['type'] in node_types]
            graph_results = [r for r in graph_results if r['type'] in node_types]
            entity_results = [r for r in entity_results if r['type'] in node_types]
            
        if time_from:
            fts_results = [r for r in fts_results if r['event_time'] >= time_from]
            vec_results = [r for r in vec_results if r['event_time'] >= time_from]
            
        if time_to:
            fts_results = [r for r in fts_results if r['event_time'] <= time_to]
            vec_results = [r for r in vec_results if r['event_time'] <= time_to]
            
        # RRF слияние
        results = self._rrf_merge(fts_results, vec_results, limit, graph_results, entity_results)
        
        # Access reinforcement
        for result in results:
            await self.storage.access_node(result['id'])
            
        logger.info(f"Поиск '{query}' вернул {len(results)} результатов")
        return results
        
    async def _search_fts(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """FTS5 полнотекстовый поиск."""
        conn = self.storage.db.get_read_connection()
        
        # Разбиваем запрос на токены и создаём FTS5 запрос
        words = query.split()
        if not words:
            return []
            
        # Для каждого слова создаём префиксный поиск
        fts_parts = []
        for word in words:
            if len(word) >= 3:  # FTS5 минимальная длина токена
                fts_parts.append(f'"{word}" *')
        
        if not fts_parts:
            return []
            
        fts_query = ' '.join(fts_parts)
        
        sql = """
            SELECT n.id, n.type, n.content, n.session_id, n.event_time, n.created_at,
                   n.last_accessed, n.access_count, n.confidence, n.decay_rate,
                   n.valid_until, n.metadata
            FROM nodes n
            JOIN nodes_fts ON n.rowid = nodes_fts.rowid
            WHERE nodes_fts MATCH ? AND n.valid_until IS NULL
            ORDER BY bm25(nodes_fts)
            LIMIT ?
        """
        
        try:
            async with conn.execute(sql, (fts_query, limit)) as cursor:
                rows = await cursor.fetchall()
                results = [dict(row) for row in rows]
                logger.debug(f"FTS5 нашёл {len(results)} результатов для '{fts_query}'")
                return results
        except Exception as e:
            logger.warning(f"FTS5 ошибка: {e}")
            return []
            
    async def _search_like(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback: LIKE поиск по содержимому."""
        conn = self.storage.db.get_read_connection()
        
        # Разбиваем запрос на слова и ищем каждое
        words = query.split()
        if not words:
            return []
            
        # Создаём LIKE условия для каждого слова
        like_conditions = []
        params = []
        for word in words:
            if len(word) >= 2:  # Минимальная длина для LIKE
                like_conditions.append("content LIKE ?")
                params.append(f"%{word}%")
        
        if not like_conditions:
            return []
            
        # Объединяем условия через OR (любое совпадение)
        where_clause = " OR ".join(like_conditions)
        
        sql = f"""
            SELECT id, type, content, session_id, event_time, created_at,
                   last_accessed, access_count, confidence, decay_rate,
                   valid_until, metadata
            FROM nodes
            WHERE ({where_clause}) AND valid_until IS NULL
            ORDER BY confidence DESC, event_time DESC
            LIMIT ?
        """
        params.append(limit)
        
        try:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                results = [dict(row) for row in rows]
                logger.debug(f"LIKE нашёл {len(results)} результатов для '{query}'")
                return results
        except Exception as e:
            logger.warning(f"LIKE ошибка: {e}")
            return []
            
    async def _search_vector(self, query_embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Vector KNN поиск через sqlite-vec."""
        conn = self.storage.db.get_read_connection()
        
        sql = """
            SELECT n.id, n.type, n.content, n.session_id, n.event_time, n.created_at,
                   n.last_accessed, n.access_count, n.confidence, n.decay_rate,
                   n.valid_until, n.metadata, distance
            FROM vec_nodes
            JOIN nodes n ON vec_nodes.node_id = n.id
            WHERE embedding MATCH ? AND n.valid_until IS NULL
            ORDER BY distance
            LIMIT ?
        """
        
        try:
            async with conn.execute(sql, (query_embedding.tobytes(), limit)) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    node = dict(row)
                    del node['distance']
                    results.append(node)
                logger.debug(f"Vector нашёл {len(results)} результатов")
                return results
        except Exception as e:
            logger.warning(f"Vector search недоступен: {e}")
            return []
            
    async def _search_graph(self, query: str, intent: str, is_complex: bool, limit: int) -> List[Dict[str, Any]]:
        """Графовый обход по intent."""
        if intent == 'why':
            max_depth = settings.max_graph_depth_complex if is_complex else settings.max_graph_depth_simple
            return await self._causal_bfs(max_depth, limit)
        elif intent == 'when':
            max_depth = settings.max_graph_depth_complex if is_complex else settings.max_graph_depth_simple
            return await self._temporal_chain(max_depth, limit)
        return []
        
    async def _causal_bfs(self, max_depth: int, limit: int) -> List[Dict[str, Any]]:
        """BFS по causal рёбрам."""
        conn = self.storage.db.get_read_connection()
        
        sql = """
            WITH RECURSIVE causal_chain(node_id, depth) AS (
                SELECT source_id, 1 FROM edges
                WHERE relation_type = 'causal' AND valid_until IS NULL
                UNION ALL
                SELECT e.source_id, cc.depth + 1 FROM edges e
                JOIN causal_chain cc ON e.target_id = cc.node_id
                WHERE e.relation_type = 'causal' AND e.valid_until IS NULL AND cc.depth < ?
            )
            SELECT DISTINCT n.id, n.type, n.content, n.session_id, n.event_time, n.created_at,
                   n.last_accessed, n.access_count, n.confidence, n.decay_rate,
                   n.valid_until, n.metadata
            FROM nodes n
            JOIN causal_chain cc ON n.id = cc.node_id
            WHERE n.valid_until IS NULL
            ORDER BY n.event_time DESC
            LIMIT ?
        """
        
        async with conn.execute(sql, (max_depth, limit)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
            
    async def _temporal_chain(self, max_depth: int, limit: int) -> List[Dict[str, Any]]:
        """Обход по temporal рёбрам."""
        conn = self.storage.db.get_read_connection()
        
        sql = """
            WITH RECURSIVE temporal_chain(node_id, depth) AS (
                SELECT source_id, 1 FROM edges
                WHERE relation_type = 'temporal' AND valid_until IS NULL
                UNION ALL
                SELECT e.source_id, tc.depth + 1 FROM edges e
                JOIN temporal_chain tc ON e.target_id = tc.node_id
                WHERE e.relation_type = 'temporal' AND e.valid_until IS NULL AND tc.depth < ?
            )
            SELECT DISTINCT n.id, n.type, n.content, n.session_id, n.event_time, n.created_at,
                   n.last_accessed, n.access_count, n.confidence, n.decay_rate,
                   n.valid_until, n.metadata
            FROM nodes n
            JOIN temporal_chain tc ON n.id = tc.node_id
            WHERE n.valid_until IS NULL
            ORDER BY n.event_time DESC
            LIMIT ?
        """
        
        async with conn.execute(sql, (max_depth, limit)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
            
    async def _search_entity(self, query: str, intent: str, limit: int) -> List[Dict[str, Any]]:
        """Поиск через entity lookup."""
        conn = self.storage.db.get_read_connection()
        
        sql = """
            SELECT DISTINCT n.id, n.type, n.content, n.session_id, n.event_time, n.created_at,
                   n.last_accessed, n.access_count, n.confidence, n.decay_rate,
                   n.valid_until, n.metadata
            FROM nodes n
            JOIN node_entities ne ON n.id = ne.node_id
            JOIN entities e ON ne.entity_id = e.id
            WHERE (LOWER(e.canonical_name) LIKE LOWER(?) OR LOWER(e.aliases) LIKE LOWER(?))
              AND n.valid_until IS NULL
            ORDER BY n.event_time DESC
            LIMIT ?
        """
        
        search_pattern = f"%{query}%"
        
        async with conn.execute(sql, (search_pattern, search_pattern, limit)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
            
    def _rrf_merge(self, fts_results: List[Dict], vec_results: List[Dict], limit: int,
                   graph_results: Optional[List[Dict]] = None,
                   entity_results: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        all_items: Dict[str, Dict] = {}
        
        for rank, item in enumerate(fts_results, 1):
            nid = item['id']
            scores[nid] = scores.get(nid, 0) + settings.rrf_weight_fts / (settings.rrf_k + rank)
            all_items.setdefault(nid, item)
            
        for rank, item in enumerate(vec_results, 1):
            nid = item['id']
            scores[nid] = scores.get(nid, 0) + settings.rrf_weight_vec / (settings.rrf_k + rank)
            all_items.setdefault(nid, item)
            
        if graph_results:
            for rank, item in enumerate(graph_results, 1):
                nid = item['id']
                scores[nid] = scores.get(nid, 0) + settings.rrf_weight_graph / (settings.rrf_k + rank)
                all_items.setdefault(nid, item)
                
        if entity_results:
            for rank, item in enumerate(entity_results, 1):
                nid = item['id']
                scores[nid] = scores.get(nid, 0) + settings.rrf_weight_graph / (settings.rrf_k + rank)
                all_items.setdefault(nid, item)
                
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        return [all_items[nid] for nid in ranked]
        
    def _classify_query_complexity(self, query: str) -> bool:
        """Классифицировать сложность запроса."""
        words = query.split()
        if len(words) > 10:
            return True
        aggregating_keywords = ['all', 'everything', 'compare', 'analyze', 'summary', 'всё', 'все', 'сравни', 'анализ', 'итого', 'резюме']
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in aggregating_keywords)
