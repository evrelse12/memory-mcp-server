# services/decay.py
import time
import math
from typing import List, Dict
import logging
from core.storage import Storage
from config.settings import settings

logger = logging.getLogger(__name__)

class DecayService:
    """Служба забывания на основе кривой Эббингауза."""
    
    def __init__(self, storage: Storage):
        self.storage = storage
        self.threshold = settings.decay_threshold
        
    async def apply(self) -> Dict[str, int]:
        nodes = await self.storage.get_decayable_nodes()
        
        if not nodes:
            return {'decayed': 0, 'pruned': 0}
            
        now = int(time.time())
        updates = []
        to_prune = []
        
        for node in nodes:
            node_id = node['id']
            confidence = node['confidence']
            decay_rate = node['decay_rate']
            last_accessed = node['last_accessed'] or node['created_at']
            
            days_since = (now - last_accessed) / 86400.0
            confidence_new = confidence * math.exp(
                -decay_rate * (max(0, days_since) ** 0.8)
            )
            
            if confidence_new < self.threshold:
                to_prune.append(node_id)
            else:
                updates.append((node_id, confidence_new))
                
        if updates:
            await self.storage.batch_update_confidence(updates)
            
        if to_prune:
            await self.storage.soft_delete_nodes(to_prune)
            
        return {'decayed': len(updates), 'pruned': len(to_prune)}
        
    async def get_weak_facts(self, limit: int = 20) -> List[Dict]:
        conn = self.storage.db.get_read_connection()
        
        sql = """
            SELECT * FROM nodes
            WHERE type IN ('semantic', 'procedural', 'opinion')
              AND valid_until IS NULL AND confidence < 0.3
            ORDER BY confidence ASC LIMIT ?
        """
        
        async with conn.execute(sql, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
