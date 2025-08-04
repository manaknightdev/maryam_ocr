"""Main application entry point."""

import asyncio
import logging
from pathlib import Path

from .services.search_service import SearchService
from .services.document_processor import DocumentProcessor
from .models.document import EntityType
from config.settings import settings


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticSearchPOC:
    """Main POC application class."""
    
    def __init__(self):
        """Initialize the POC application."""
        self.search_service = SearchService()
        self.document_processor = DocumentProcessor()
        
    async def initialize(self):
        """Initialize the application."""
        logger.info("Initializing Semantic Search POC...")
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize services
        await self.search_service.vector_store.initialize()
        
        logger.info("Initialization complete!")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            settings.DATA_DIR,
            settings.RAW_DATA_DIR,
            settings.PROCESSED_DATA_DIR,
            settings.UPLOAD_DIR,
            Path(settings.LOG_FILE).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def process_sample_documents(self):
        """Process sample documents for testing."""
        logger.info("Processing sample documents...")
        
        sample_docs_dir = Path(settings.RAW_DATA_DIR) / "sample_documents"
        if not sample_docs_dir.exists():
            logger.warning(f"Sample documents directory not found: {sample_docs_dir}")
            return
        
        # Process all documents in the sample directory
        for file_path in sample_docs_dir.iterdir():
            if file_path.is_file():
                try:
                    await self.document_processor.process_document(str(file_path))
                    logger.info(f"Processed: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
    
    async def demo_search(self):
        """Demonstrate search functionality."""
        logger.info("Running search demo...")
        
        # Example searches
        search_queries = [
            "Napoleon Bonaparte biography",
            "French Revolution events",
            "Paris buildings and architecture",
            "Military campaigns in Europe"
        ]
        
        for query in search_queries:
            logger.info(f"\nSearching for: '{query}'")
            try:
                results = await self.search_service.semantic_search(
                    query=query,
                    limit=5
                )
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  {i}. {result.title} (Score: {result.similarity_score:.3f})")
                        logger.info(f"     Preview: {result.content_preview[:100]}...")
                else:
                    logger.info("  No results found.")
                    
            except Exception as e:
                logger.error(f"Error searching for '{query}': {str(e)}")
    
    async def demo_entity_search(self):
        """Demonstrate entity-based search."""
        logger.info("Running entity search demo...")
        
        # Example entity searches
        entity_searches = [
            ("Napoleon", EntityType.PERSON),
            ("Paris", EntityType.PLACE),
            ("Revolution", EntityType.EVENT)
        ]
        
        for entity_text, entity_type in entity_searches:
            logger.info(f"\nSearching for {entity_type.value}: '{entity_text}'")
            try:
                results = await self.search_service.search_by_entity(
                    entity_text=entity_text,
                    entity_type=entity_type,
                    limit=3
                )
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  {i}. {result.title}")
                        logger.info(f"     Entities: {len(result.entities)} found")
                else:
                    logger.info("  No results found.")
                    
            except Exception as e:
                logger.error(f"Error searching for entity '{entity_text}': {str(e)}")
    
    async def demo_relationships(self):
        """Demonstrate entity relationship extraction."""
        logger.info("Running relationship demo...")
        
        entity = "Napoleon"
        logger.info(f"\nFinding relationships for: '{entity}'")
        
        try:
            relationships = await self.search_service.get_entity_relationships(entity)
            
            for rel_type, entities in relationships.items():
                if entities:
                    logger.info(f"  {rel_type.upper()}:")
                    for entity_data in entities[:3]:  # Show top 3
                        logger.info(f"    - {entity_data['entity']} (freq: {entity_data['frequency']})")
                        
        except Exception as e:
            logger.error(f"Error finding relationships for '{entity}': {str(e)}")


async def main():
    """Main function to run the POC."""
    logger.info("Starting Semantic Search POC")
    
    # Initialize application
    app = SemanticSearchPOC()
    await app.initialize()
    
    # Process sample documents (if any)
    await app.process_sample_documents()
    
    # Run demonstrations
    await app.demo_search()
    await app.demo_entity_search()
    await app.demo_relationships()
    
    logger.info("POC demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())