# services/intent_classifier.py
import re
import json
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Intent-aware классификатор для маршрутизации поиска."""
    
    EXEMPLARS = {
        "why": [
            "why did this happen", "what caused the failure",
            "почему это произошло", "в чем причина"
        ],
        "when": [
            "when did we discuss", "timeline of events",
            "когда мы обсуждали", "хронология событий"
        ],
        "who": [
            "who is responsible", "whose idea",
            "кто отвечает за", "чья это идея"
        ],
        "what": [
            "what do you know about", "tell me everything about",
            "что ты знаешь о", "расскажи всё о"
        ],
    }
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.exemplar_embeddings: Optional[Dict[str, List[np.ndarray]]] = None
        self.cache_file = Path(settings.base_dir) / "intent_cache.json"
        
    async def initialize(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                self.exemplar_embeddings = {}
                for intent, embeddings_list in cache_data.items():
                    self.exemplar_embeddings[intent] = [
                        np.array(emb, dtype=np.float32) for emb in embeddings_list
                    ]
                logger.info(f"Загружены exemplar эмбеддинги из кэша")
                return
            except Exception as e:
                logger.warning(f"Ошибка загрузки кэша: {e}")
                
        self.exemplar_embeddings = {}
        
        for intent, examples in self.EXEMPLARS.items():
            embeddings = await self.embedding_service.embed(examples)
            if embeddings:
                self.exemplar_embeddings[intent] = embeddings
                
        if self.exemplar_embeddings:
            cache_data = {}
            for intent, embeddings in self.exemplar_embeddings.items():
                cache_data[intent] = [emb.tolist() for emb in embeddings]
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
                
    async def classify_embedding(self, query_embedding: np.ndarray) -> str:
        if not self.exemplar_embeddings:
            return 'general'
            
        max_similarity = 0.0
        best_intent = 'general'
        
        for intent, exemplars in self.exemplar_embeddings.items():
            for exemplar in exemplars:
                similarity = float(np.dot(query_embedding, exemplar))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_intent = intent
                    
        if max_similarity >= settings.intent_threshold:
            return best_intent
        else:
            return 'general'
            
    def classify_keyword(self, query: str) -> str:
        q = query.strip().lower()
        
        if re.search(r'\b(why|cause|reason|because|почему|причина)\b', q):
            return 'why'
        if re.search(r'\b(when|after|before|timeline|когда|хронология)\b', q):
            return 'when'
        if re.search(r'\b(who|whom|whose|кто|кого|чей)\b', q):
            return 'who'
        if re.search(r'\b(what|which|everything about|что|опиши)\b', q):
            return 'what'
            
        return 'general'
        
    async def classify(self, query: str, query_embedding: Optional[np.ndarray] = None) -> str:
        if query_embedding is not None and self.exemplar_embeddings:
            intent = await self.classify_embedding(query_embedding)
            if intent != 'general':
                return intent
                
        return self.classify_keyword(query)
