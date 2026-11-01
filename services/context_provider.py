# services/context_provider.py
from typing import List, Dict, Any, Optional
import logging
from core.storage import Storage
from services.search import SearchService
from services.embedding import EmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)

class ContextProvider:
    """Провайдер контекста для инъекции в system prompt."""
    
    def __init__(self, storage: Storage, search_service: SearchService, embedding_service: EmbeddingService):
        self.storage = storage
        self.search_service = search_service
        self.embedding_service = embedding_service
        
    async def get_context(self, prompt: str, token_budget: int = 2000) -> Optional[str]:
        is_complex = self._classify_query_complexity(prompt)
        query_embedding = await self.embedding_service.embed_single(prompt)
        
        results = await self.search_service.search(
            prompt, query_embedding=query_embedding,
            limit=20 if is_complex else 10
        )
        
        if not results:
            return None
            
        context = await self._assemble_context(results, token_budget)
        return context
        
    async def _assemble_context(self, results: List[Dict], token_budget: int) -> str:
        sections = {'facts': [], 'entities': [], 'temporal': [], 'evidence': []}
        
        for result in results:
            node_type = result.get('type')
            
            if node_type in ['semantic', 'procedural', 'opinion']:
                sections['facts'].append(result)
            elif node_type == 'episodic':
                sections['temporal'].append(result)
                
            entities = await self.storage.get_entities_for_node(result['id'])
            if entities:
                for entity in entities:
                    if entity not in sections['entities']:
                        sections['entities'].append(entity)
                        
        context_parts = []
        
        if sections['facts']:
            facts_budget = int(token_budget * settings.context_budget_facts)
            facts_text = self._format_facts(sections['facts'], facts_budget)
            if facts_text:
                context_parts.append(facts_text)
                
        if sections['entities']:
            entities_budget = int(token_budget * settings.context_budget_entities)
            entities_text = self._format_entities(sections['entities'], entities_budget)
            if entities_text:
                context_parts.append(entities_text)
                
        if sections['temporal']:
            temporal_budget = int(token_budget * settings.context_budget_temporal)
            temporal_text = self._format_temporal(sections['temporal'], temporal_budget)
            if temporal_text:
                context_parts.append(temporal_text)
                
        evidence_budget = int(token_budget * settings.context_budget_evidence)
        evidence_text = await self._format_evidence(results[:5], evidence_budget)
        if evidence_text:
            context_parts.append(evidence_text)
            
        return "\n\n".join(context_parts)
        
    def _format_facts(self, facts: List[Dict], budget: int) -> str:
        lines = ["## Known Facts", ""]
        current_length = len("\n".join(lines))
        
        for fact in facts:
            content = fact['content']
            confidence = fact.get('confidence', 1.0)
            line = f"- {content} (confidence: {confidence:.2f})"
            
            if current_length + len(line) > budget:
                break
                
            lines.append(line)
            current_length += len(line) + 1
            
        return "\n".join(lines) if len(lines) > 2 else ""
        
    def _format_entities(self, entities: List[Dict], budget: int) -> str:
        lines = ["## Relevant Entities", ""]
        current_length = len("\n".join(lines))
        
        for entity in entities:
            name = entity['canonical_name']
            entity_type = entity.get('type', 'unknown')
            summary = entity.get('summary', '')
            
            if summary:
                line = f"- **{name}** ({entity_type}): {summary}"
            else:
                line = f"- **{name}** ({entity_type})"
                
            if current_length + len(line) > budget:
                break
                
            lines.append(line)
            current_length += len(line) + 1
            
        return "\n".join(lines) if len(lines) > 2 else ""
        
    def _format_temporal(self, episodes: List[Dict], budget: int) -> str:
        lines = ["## Recent Context", ""]
        current_length = len("\n".join(lines))
        
        for episode in episodes[:5]:
            content = episode['content']
            if len(content) > 200:
                content = content[:197] + "..."
            line = f"- {content}"
            
            if current_length + len(line) > budget:
                break
                
            lines.append(line)
            current_length += len(line) + 1
            
        return "\n".join(lines) if len(lines) > 2 else ""
        
    async def _format_evidence(self, results: List[Dict], budget: int) -> str:
        lines = ["## Evidence", ""]
        current_length = len("\n".join(lines))
        
        for result in results[:3]:
            node_id = result['id']
            edges = await self.storage.get_edges_from_node(node_id, 'derived_from')
            
            if edges:
                evidence = []
                for edge in edges[:2]:
                    source_episode = await self.storage.get_node(edge['target_id'])
                    if source_episode:
                        content = source_episode['content']
                        if len(content) > 100:
                            content = content[:97] + "..."
                        evidence.append(content)
                        
                if evidence:
                    line = f"- {result['content'][:100]}... <- {', '.join(evidence)}"
                    
                    if current_length + len(line) > budget:
                        break
                        
                    lines.append(line)
                    current_length += len(line) + 1
                    
        return "\n".join(lines) if len(lines) > 2 else ""
        
    def _classify_query_complexity(self, query: str) -> bool:
        words = query.split()
        if len(words) > 10:
            return True
            
        aggregating_keywords = ['all', 'everything', 'compare', 'analyze', 'summary',
                                'всё', 'все', 'сравни', 'анализ', 'итого', 'резюме']
        
        query_lower = query.lower()
        for keyword in aggregating_keywords:
            if keyword in query_lower:
                return True
        return False
