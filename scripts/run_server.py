import asyncio
import sys
import uvicorn
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


def main():
    """Run the FastAPI server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Semantic Search API Server...")
    
    # Create necessary directories
    directories = [
        settings.DATA_DIR,
        settings.UPLOAD_DIR,
        Path(settings.LOG_FILE).parent,
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"""
          Starting Semantic Search API Server
          Server URL: http://{settings.API_HOST}:{settings.API_PORT}
          API Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs
          Frontend: http://{settings.API_HOST}:{settings.API_PORT}/
          Health Check: http://{settings.API_HOST}:{settings.API_PORT}/health

Press Ctrl+C to stop the server
    """)
    
    # Run the server
    uvicorn.run(
        "src.api.routes:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
