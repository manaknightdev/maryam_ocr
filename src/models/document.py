"""Document data models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration."""
    PDF = "pdf"
    XML = "xml"
    TEXT = "text"
    DOCX = "docx"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class AccessLevel(str, Enum):
    """Document access level."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    PRIVATE = "private"


class EntityType(str, Enum):
    """Entity type enumeration."""
    PERSON = "person"
    PLACE = "place"
    EVENT = "event"
    ORGANIZATION = "organization"
    BUILDING = "building"
    DATE = "date"


class Entity(BaseModel):
    """Entity extracted from document."""
    text: str
    label: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    language: str = "en"
    subject: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    file_size: Optional[int] = None
    page_count: Optional[int] = None


class Document(BaseModel):
    """Main document model."""
    id: str
    filename: str
    file_path: str
    document_type: DocumentType
    access_level: AccessLevel
    content: str
    metadata: DocumentMetadata
    entities: List[Entity] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentChunk(BaseModel):
    """Document chunk for processing."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_pos: int
    end_pos: int
    embedding: Optional[List[float]] = None
    entities: List[Entity] = Field(default_factory=list)


class SearchQuery(BaseModel):
    """Search query model."""
    query: str
    entity_types: Optional[List[EntityType]] = None
    document_types: Optional[List[DocumentType]] = None
    access_levels: Optional[List[AccessLevel]] = None
    limit: int = 10
    offset: int = 0
    similarity_threshold: float = 0.2