# core/storage.py
import json
import time
import uuid
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Storage:
    """CRUD операции для узлов, рёбер и сущностей."""
    
    def __init__(self, db, write_queue):
        self.db = db
        self.write_queue = write_queue
        
    async def insert_node(self, node_data: Dict[str, Any]) -> str:
        node_id = node_data.get('id', str(uuid.uuid4()))
        now = int(time.time())
        
        sql = """
            INSERT INTO nodes (
                id, type, content, session_id, event_time, created_at,
                last_accessed, access_count, confidence, decay_rate,
                valid_until, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            node_id,
            node_data['type'],
            node_data['content'],
            node_data.get('session_id'),
            node_data.get('event_time', now),
            now,
            now,
            0,
            node_data.get('confidence', 1.0),
            node_data.get('decay_rate', 0.1 if node_data['type'] != 'episodic' else 0.0),
            None,
            json.dumps(node_data.get('metadata', {}))
        )
        
        await self.write_queue.execute(sql, params)
        logger.info(f"Создан узел: {node_id} ({node_data['type']})")
        return node_id
        
    async def insert_embedding(self, node_id: str, embedding: np.ndarray):
        sql = "INSERT INTO vec_nodes (node_id, embedding) VALUES (?, ?)"
        params = (node_id, embedding.tobytes())
        await self.write_queue.execute(sql, params)
        
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        async with conn.execute(
            "SELECT * FROM nodes WHERE id = ? AND valid_until IS NULL",
            (node_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
        return None
        
    async def get_nodes_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        async with conn.execute(
            """SELECT * FROM nodes WHERE session_id = ? AND valid_until IS NULL
               ORDER BY event_time ASC""",
            (session_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
            
    async def update_node_fields(self, node_id: str, updates: Dict[str, Any]):
        if not updates:
            return
            
        set_clauses = []
        params = []
        for key, value in updates.items():
            if key == 'metadata' and isinstance(value, dict):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            params.append(value)
            
        params.append(node_id)
        sql = f"UPDATE nodes SET {', '.join(set_clauses)} WHERE id = ? AND valid_until IS NULL"
        
        await self.write_queue.execute(sql, params)
        
    async def soft_delete_node(self, node_id: str):
        now = int(time.time())
        sql = "UPDATE nodes SET valid_until = ? WHERE id = ? AND valid_until IS NULL"
        await self.write_queue.execute(sql, (now, node_id))
        
    async def soft_delete_nodes(self, node_ids: List[str]):
        if not node_ids:
            return
        for node_id in node_ids:
            await self.soft_delete_node(node_id)
            
    async def batch_update_confidence(self, updates: List[tuple]):
        if not updates:
            return
        for node_id, confidence in updates:
            sql = "UPDATE nodes SET confidence = ? WHERE id = ? AND valid_until IS NULL"
            await self.write_queue.execute(sql, (confidence, node_id))
            
    async def access_node(self, node_id: str):
        now = int(time.time())
        sql = """
            UPDATE nodes 
            SET access_count = access_count + 1,
                last_accessed = ?,
                confidence = MIN(1.0, confidence + 0.05 * LOG(1 + access_count / 20.0))
            WHERE id = ? AND valid_until IS NULL
        """
        await self.write_queue.execute(sql, (now, node_id))
        
    async def get_decayable_nodes(self) -> List[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        async with conn.execute(
            """SELECT * FROM nodes 
               WHERE type != 'episodic' AND decay_rate > 0 AND valid_until IS NULL"""
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
            
    async def insert_edge(self, edge_data: Dict[str, Any]) -> str:
        edge_id = edge_data.get('id', str(uuid.uuid4()))
        now = int(time.time())
        
        sql = """
            INSERT INTO edges (
                id, source_id, target_id, relation_type, predicate,
                weight, confidence, valid_from, valid_until, evidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            edge_id,
            edge_data['source_id'],
            edge_data['target_id'],
            edge_data['relation_type'],
            edge_data.get('predicate'),
            edge_data.get('weight', 1.0),
            edge_data.get('confidence', 1.0),
            edge_data.get('valid_from', now),
            edge_data.get('valid_until'),
            json.dumps(edge_data.get('evidence', [])),
            now
        )
        
        await self.write_queue.execute(sql, params)
        return edge_id
        
    async def get_edges_from_node(self, node_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        query = "SELECT * FROM edges WHERE source_id = ? AND valid_until IS NULL"
        params = [node_id]
        
        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)
            
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
            
    async def get_edges_to_node(self, node_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        query = "SELECT * FROM edges WHERE target_id = ? AND valid_until IS NULL"
        params = [node_id]
        
        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)
            
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
            
    async def insert_entity(self, entity_data: Dict[str, Any]) -> str:
        entity_id = entity_data.get('id', str(uuid.uuid4()))
        now = int(time.time())
        
        sql = """
            INSERT INTO entities (
                id, canonical_name, type, aliases, summary, embedding,
                first_seen, last_updated, mention_count, attributes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entity_id,
            entity_data['canonical_name'],
            entity_data['type'],
            json.dumps(entity_data.get('aliases', [])),
            entity_data.get('summary'),
            entity_data.get('embedding'),
            entity_data.get('first_seen', now),
            now,
            entity_data.get('mention_count', 1),
            json.dumps(entity_data.get('attributes', {}))
        )
        
        await self.write_queue.execute(sql, params)
        return entity_id
        
    async def insert_entity_embedding(self, entity_id: str, embedding: np.ndarray):
        sql = "INSERT INTO vec_entities (entity_id, embedding) VALUES (?, ?)"
        params = (entity_id, embedding.tobytes())
        await self.write_queue.execute(sql, params)
        
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        async with conn.execute(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                entity = dict(row)
                entity['aliases'] = json.loads(entity['aliases'] or '[]')
                entity['attributes'] = json.loads(entity['attributes'] or '{}')
                return entity
        return None
        
    async def find_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        async with conn.execute(
            "SELECT * FROM entities WHERE LOWER(canonical_name) = LOWER(?)",
            (name,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                entity = dict(row)
                entity['aliases'] = json.loads(entity['aliases'] or '[]')
                entity['attributes'] = json.loads(entity['attributes'] or '{}')
                return entity
        return None
        
    async def update_entity(self, entity_id: str, updates: Dict[str, Any]):
        if not updates:
            return
            
        set_clauses = []
        params = []
        for key, value in updates.items():
            if key == 'aliases' and isinstance(value, list):
                value = json.dumps(value)
            elif key == 'attributes' and isinstance(value, dict):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            params.append(value)
            
        params.append(entity_id)
        sql = f"UPDATE entities SET {', '.join(set_clauses)} WHERE id = ?"
        
        await self.write_queue.execute(sql, params)
        
    async def increment_entity_mentions(self, entity_id: str):
        sql = """
            UPDATE entities 
            SET mention_count = mention_count + 1, last_updated = ?
            WHERE id = ?
        """
        await self.write_queue.execute(sql, (int(time.time()), entity_id))
        
    async def link_node_entity(self, node_id: str, entity_id: str):
        sql = "INSERT OR IGNORE INTO node_entities (node_id, entity_id) VALUES (?, ?)"
        await self.write_queue.execute(sql, (node_id, entity_id))
        
    async def get_entities_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        async with conn.execute(
            """SELECT e.* FROM entities e
               JOIN node_entities ne ON e.id = ne.entity_id
               WHERE ne.node_id = ?""",
            (node_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            entities = []
            for row in rows:
                entity = dict(row)
                entity['aliases'] = json.loads(entity['aliases'] or '[]')
                entity['attributes'] = json.loads(entity['attributes'] or '{}')
                entities.append(entity)
            return entities
            
    async def get_nodes_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        conn = self.db.get_read_connection()
        async with conn.execute(
            """SELECT n.* FROM nodes n
               JOIN node_entities ne ON n.id = ne.node_id
               WHERE ne.entity_id = ? AND n.valid_until IS NULL
               ORDER BY n.event_time DESC""",
            (entity_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
