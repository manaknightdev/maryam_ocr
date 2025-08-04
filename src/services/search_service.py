"""Main search service implementation."""

from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

from ..models.document import Document, SearchQuery, EntityType
from ..models.search_result import SearchResult
from .vector_store import VectorStore
from .entity_extractor import EntityExtractor
from config.settings import settings


class SearchService:
    """Main search service for semantic search functionality."""
    
    def __init__(self):
        """Initialize search service."""
        self.logger = logging.getLogger(__name__)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_store = VectorStore()
        self.entity_extractor = EntityExtractor()
        
    async def semantic_search(
        self, 
        query: str, 
        limit: int = 10,
        entity_types: Optional[List[EntityType]] = None,
        similarity_threshold: float = None
    ) -> List[SearchResult]:
        """
        Perform semantic search across documents.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            entity_types: Filter by entity types
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        if similarity_threshold is None:
            similarity_threshold = settings.SIMILARITY_THRESHOLD
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search vector store
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                limit=limit,
                threshold=similarity_threshold
            )
            
            # Filter by entity types if specified
            if entity_types:
                results = self._filter_by_entities(results, entity_types)
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    document_id=result["document_id"],
                    title=result.get("title", "Untitled"),
                    content_preview=result.get("content", "")[:200] + "...",
                    similarity_score=result["similarity_score"],
                    entities=result.get("entities", []),
                    metadata=result.get("metadata", {})
                )
                search_results.append(search_result)
                
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            raise
    
    async def search_by_entity(
        self, 
        entity_text: str, 
        entity_type: EntityType,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search for documents containing specific entities.
        
        Args:
            entity_text: Entity text to search for
            entity_type: Type of entity
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Search for documents containing the entity
            results = await self.vector_store.search_by_entity(
                entity_text=entity_text,
                entity_type=entity_type.value,
                limit=limit
            )
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    document_id=result["document_id"],
                    title=result.get("title", "Untitled"),
                    content_preview=result.get("content", "")[:200] + "...",
                    similarity_score=1.0,  # Exact entity match
                    entities=result.get("entities", []),
                    metadata=result.get("metadata", {}),
                    matched_entity=entity_text
                )
                search_results.append(search_result)
                
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in entity search: {str(e)}")
            raise
    
    async def get_entity_relationships(
        self, 
        entity_text: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get relationships for a specific entity.
        
        Args:
            entity_text: Entity to find relationships for
            
        Returns:
            Dictionary of relationship types and related entities
        """
        try:
            # Find documents containing the entity
            entity_docs = await self.search_by_entity(
                entity_text=entity_text,
                entity_type=EntityType.PERSON,  # Default to person
                limit=100
            )
            
            # Extract co-occurring entities
            relationships = {
                "persons": [],
                "places": [],
                "events": [],
                "organizations": []
            }
            
            for doc in entity_docs:
                for entity in doc.entities:
                    if entity["text"].lower() != entity_text.lower():
                        rel_type = self._map_entity_to_relationship(entity["entity_type"])
                        if rel_type in relationships:
                            relationships[rel_type].append({
                                "entity": entity["text"],
                                "type": entity["entity_type"],
                                "document_id": doc.document_id,
                                "confidence": entity.get("confidence", 0.0)
                            })
            
            # Remove duplicates and sort by frequency
            for rel_type in relationships:
                entities = relationships[rel_type]
                unique_entities = {}
                for entity in entities:
                    key = entity["entity"]
                    if key not in unique_entities:
                        unique_entities[key] = entity
                        unique_entities[key]["frequency"] = 1
                    else:
                        unique_entities[key]["frequency"] += 1
                
                relationships[rel_type] = sorted(
                    unique_entities.values(),
                    key=lambda x: x["frequency"],
                    reverse=True
                )[:10]  # Top 10 relationships
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error getting entity relationships: {str(e)}")
            raise
    
    def _filter_by_entities(
        self, 
        results: List[Dict[str, Any]], 
        entity_types: List[EntityType]
    ) -> List[Dict[str, Any]]:
        """Filter search results by entity types."""
        filtered_results = []
        entity_type_values = [et.value for et in entity_types]
        
        for result in results:
            entities = result.get("entities", [])
            if any(entity.get("entity_type") in entity_type_values for entity in entities):
                filtered_results.append(result)
                
        return filtered_results
    
    def _map_entity_to_relationship(self, entity_type: str) -> str:
        """Map entity type to relationship category."""
        mapping = {
            "PERSON": "persons",
            "GPE": "places",  # Geopolitical entity
            "LOC": "places",  # Location
            "EVENT": "events",
            "ORG": "organizations",
            "BUILDING": "places"
        }
        return mapping.get(entity_type, "other")