import time
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

from ..services.search_service import SearchService
from ..services.document_processor import DocumentProcessor
from ..models.document import EntityType, DocumentType, AccessLevel
from ..models.search_result import SearchResponse, EntitySearchResult
from .schemas import SearchRequest, DocumentUploadResponse, StatsResponse
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="Intelligent semantic search engine for archival documents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
static_dir = Path(__file__).parent.parent.parent / "templates" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize services
search_service = SearchService()
document_processor = DocumentProcessor()
logger = logging.getLogger(__name__)


# Dependency to get search service
async def get_search_service():
    return search_service


# Dependency to get document processor
async def get_document_processor():
    return document_processor


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Initializing API services...")
    await search_service.vector_store.initialize()
    logger.info("API services initialized successfully")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page."""
    template_path = Path(__file__).parent.parent.parent / "templates" / "index.html"
    if template_path.exists():
        return FileResponse(str(template_path))
    return HTMLResponse("<h1>Semantic Search API</h1><p>Frontend template not found</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Perform semantic search across documents.
    
    - **query**: Search query string
    - **limit**: Maximum number of results (default: 10)
    - **entity_types**: Filter by entity types (optional)
    - **similarity_threshold**: Minimum similarity score (optional)
    """
    start_time = time.time()
    
    try:
        results = await search_service.semantic_search(
            query=request.query,
            limit=request.limit,
            entity_types=request.entity_types,
            similarity_threshold=request.similarity_threshold
        )
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
            filters_applied={
                "entity_types": [et.value for et in request.entity_types] if request.entity_types else [],
                "similarity_threshold": request.similarity_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/v1/search", response_model=SearchResponse)
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Similarity threshold"),
    search_service: SearchService = Depends(get_search_service)
):
    """GET version of search endpoint for simple queries."""
    
    # Parse entity types
    parsed_entity_types = None
    if entity_types:
        try:
            parsed_entity_types = [EntityType(et.strip()) for et in entity_types.split(",")]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid entity type: {str(e)}")
    
    # Create search request
    request = SearchRequest(
        query=q,
        limit=limit,
        entity_types=parsed_entity_types,
        similarity_threshold=threshold
    )
    
    return await search_documents(request, search_service)


@app.get("/api/v1/entities/{entity_type}", response_model=EntitySearchResult)
async def search_by_entity(
    entity_type: EntityType,
    text: str = Query(..., description="Entity text to search for"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Search for documents containing specific entities.
    
    - **entity_type**: Type of entity (person, place, event, etc.)
    - **text**: Entity text to search for
    - **limit**: Maximum number of results
    """
    try:
        results = await search_service.search_by_entity(
            entity_text=text,
            entity_type=entity_type,
            limit=limit
        )
        
        return EntitySearchResult(
            entity_text=text,
            entity_type=entity_type.value,
            documents=results,
            total_occurrences=len(results)
        )
        
    except Exception as e:
        logger.error(f"Entity search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity search failed: {str(e)}")


@app.get("/api/v1/entities/{entity_text}/relationships")
async def get_entity_relationships(
    entity_text: str,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get relationships for a specific entity.
    
    - **entity_text**: Entity to find relationships for
    """
    try:
        relationships = await search_service.get_entity_relationships(entity_text)
        
        return {
            "entity": entity_text,
            "relationships": relationships,
            "total_relationships": sum(len(entities) for entities in relationships.values())
        }
        
    except Exception as e:
        logger.error(f"Relationship search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Relationship search failed: {str(e)}")


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    access_level: AccessLevel = AccessLevel.PUBLIC,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Upload and process a new document.
    
    - **file**: Document file to upload
    - **access_level**: Access level for the document (public, restricted, private)
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Save uploaded file
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Process document
        document = await document_processor.process_document(str(file_path), access_level)
        
        if not document:
            raise HTTPException(status_code=500, detail="Failed to process document")
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            status="processed",
            entities_found=len(document.entities),
            access_level=document.access_level.value,
            message="Document uploaded and processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/v1/documents/{document_id}")
async def get_document(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Get document details by ID.
    
    - **document_id**: Unique document identifier
    """
    try:
        document = await document_processor.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Return document without full content for performance
        return {
            "id": document.id,
            "filename": document.filename,
            "document_type": document.document_type.value,
            "access_level": document.access_level.value,
            "metadata": document.metadata,
            "entities": [{
                "text": e.text,
                "entity_type": e.entity_type.value,
                "confidence": e.confidence
            } for e in document.entities],
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Delete a document by ID.
    
    - **document_id**: Unique document identifier
    """
    try:
        success = await document_processor.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or deletion failed")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats(
    search_service: SearchService = Depends(get_search_service),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    Get system statistics and metrics.
    """
    try:
        # Get processing stats
        processing_stats = await document_processor.get_processing_stats()
        
        # Get vector store stats
        vector_stats = await search_service.vector_store.get_collection_stats()
        
        return StatsResponse(
            total_documents=processing_stats.get("total_documents", 0),
            document_types=processing_stats.get("document_types", {}),
            access_levels=processing_stats.get("access_levels", {}),
            languages=processing_stats.get("languages", {}),
            total_entities=processing_stats.get("total_entities", 0),
            average_entities_per_doc=processing_stats.get("average_entities_per_doc", 0.0),
            vector_store_stats=vector_stats,
            system_status="operational"
        )
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


@app.get("/api/v1/entity-types")
async def get_entity_types():
    """Get available entity types."""
    return {
        "entity_types": [
            {"value": et.value, "label": et.value.title()} 
            for et in EntityType
        ]
    }


@app.get("/api/v1/document-types")
async def get_document_types():
    """Get supported document types."""
    return {
        "document_types": [
            {"value": dt.value, "label": dt.value.upper()} 
            for dt in DocumentType
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.routes:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )