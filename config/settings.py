"""Configuration settings for the semantic search POC."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Semantic Search POC"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # API
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/semantic_search.db"
    
    # Vector Store
    VECTOR_STORE_TYPE: str = "chroma"  # chroma, faiss
    CHROMA_PERSIST_DIR: str = "./data/embeddings/chroma"
    FAISS_INDEX_PATH: str = "./data/embeddings/faiss"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Entity Extraction
    SPACY_MODEL: str = "en_core_web_sm"
    CUSTOM_NER_MODEL: Optional[str] = None
    
    # Document Processing
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx", ".xml"]
    
    # Search
    MAX_SEARCH_RESULTS: int = 50
    SIMILARITY_THRESHOLD: float = 0.2
    
    # Directories
    DATA_DIR: str = "./data"
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    UPLOAD_DIR: str = "./data/uploads"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()