"""Search result data models."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Search result model."""
    document_id: str
    title: str
    content_preview: str
    similarity_score: float
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    matched_entity: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            float: lambda v: round(v, 4)
        }


class EntitySearchResult(BaseModel):
    """Entity-specific search result."""
    entity_text: str
    entity_type: str
    documents: List[SearchResult]
    total_occurrences: int
    related_entities: List[Dict[str, Any]] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)


class EntityRelationship(BaseModel):
    """Entity relationship model."""
    source_entity: str
    source_type: str
    target_entity: str
    target_type: str
    relationship_type: str
    confidence: float
    document_ids: List[str]
    frequency: int = 1