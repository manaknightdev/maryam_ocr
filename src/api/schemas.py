from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from ..models.document import EntityType, AccessLevel


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., description="Search query string", min_length=1)
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    entity_types: Optional[List[EntityType]] = Field(None, description="Filter by entity types")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class DocumentUploadResponse(BaseModel):
    """Document upload response schema."""
    document_id: str
    filename: str
    status: str
    entities_found: int
    access_level: str
    message: str


class BatchDocumentUploadResponse(BaseModel):
    """Batch document upload response schema."""
    successful_uploads: List[DocumentUploadResponse]
    failed_uploads: List[Dict[str, str]]
    total_processed: int
    message: str


class StatsResponse(BaseModel):
    """System statistics response schema."""
    total_documents: int
    document_types: Dict[str, int]
    access_levels: Dict[str, int]
    languages: Dict[str, int]
    total_entities: int
    average_entities_per_doc: float
    vector_store_stats: Dict[str, Any]
    system_status: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: str
    timestamp: float