"""Setup script to initialize the POC environment."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.vector_store import VectorStore
from src.services.document_processor import DocumentProcessor
from config.settings import settings


async def setup_environment():
    """Set up the POC environment."""
    print("🚀 Setting up Semantic Search POC...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create directories
        directories = [
            settings.DATA_DIR,
            settings.RAW_DATA_DIR,
            settings.PROCESSED_DATA_DIR,
            settings.UPLOAD_DIR,
            f"{settings.RAW_DATA_DIR}/sample_documents",
            f"{settings.RAW_DATA_DIR}/pdfs",
            f"{settings.RAW_DATA_DIR}/xml",
            Path(settings.LOG_FILE).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        # Initialize vector store
        print("\n📚 Initializing vector store...")
        vector_store = VectorStore()
        await vector_store.initialize()
        print("✅ Vector store initialized")
        
        # Create sample documents
        sample_docs_dir = Path(f"{settings.RAW_DATA_DIR}/sample_documents")
        
        # Sample document 1: Napoleon biography
        sample1 = sample_docs_dir / "napoleon_biography.txt"
        if not sample1.exists():
            sample1.write_text("""
Napoleon Bonaparte: A Brief Biography

Napoleon Bonaparte (1769-1821) was a French military general and political leader who rose to prominence during the French Revolution. Born in Corsica, Napoleon became Emperor of the French in 1804.

Early Life and Rise to Power
Napoleon was born in Ajaccio, Corsica, to Charles Buonaparte and Letizia Ramolino Bonaparte. He attended military school in France and quickly distinguished himself as a brilliant strategist.

Military Campaigns
Napoleon led numerous military campaigns across Europe, including:
- The Italian Campaign (1796-1797)
- The Egyptian Campaign (1798-1801)
- The Austerlitz Campaign (1805)
- The Russian Campaign (1812)

Napoleon's forces occupied much of continental Europe at the height of his power. He established the Continental System to weaken Britain economically.

Political Reforms
As Emperor, Napoleon implemented significant reforms:
- The Napoleonic Code (Civil Code)
- Educational reforms
- Infrastructure development
- Administrative reorganization

Exile and Death
After defeat at the Battle of Leipzig in 1813 and subsequent abdication, Napoleon was exiled to Elba. He returned for the Hundred Days but was defeated at Waterloo in 1815. He was then exiled to Saint Helena, where he died in 1821.

Legacy
Napoleon's influence on European law, politics, and military strategy continues to this day. His reforms and conquests shaped the modern European state system.
            """.strip())
            print(f"✅ Created sample document: {sample1.name}")
        
        # Sample document 2: French Revolution overview  
        sample2 = sample_docs_dir / "french_revolution.txt"
        if not sample2.exists():
            sample2.write_text("""
The French Revolution (1789-1799): An Overview

The French Revolution was a period of radical political and societal change in France that began with the Estates-General of 1789 and ended with the formation of the French Consulate in November 1799.

Causes of the Revolution
- Economic crisis and debt
- Social inequality under the Ancien Régime
- Influence of Enlightenment ideas
- Weak leadership under Louis XVI

Key Events and Phases

The Moderate Phase (1789-1792)
- Storming of the Bastille (July 14, 1789)
- Declaration of the Rights of Man and of the Citizen
- Abolition of feudalism
- Civil Constitution of the Clergy

The Radical Phase (1792-1794)
- Execution of Louis XVI (January 21, 1793)
- Reign of Terror under Maximilien Robespierre
- Committee of Public Safety
- Revolutionary Wars against European coalitions

The Thermidorian Reaction (1794-1799)
- Fall of Robespierre (July 27, 1794)
- Directory period
- Rise of Napoleon Bonaparte

Important Figures
- Louis XVI - King of France
- Marie Antoinette - Queen of France
- Maximilien Robespierre - Jacobin leader
- Georges Danton - Revolutionary leader
- Jean-Paul Marat - Radical journalist
- Jacques Necker - Finance Minister

Geographic Centers
The revolution centered around Paris, with key locations including:
- Palace of Versailles
- Tuileries Palace
- Place de la Concorde (formerly Place Louis XV)
- Conciergerie prison

Impact and Legacy
The French Revolution fundamentally changed French society and had lasting effects on European politics, inspiring democratic movements worldwide and establishing principles of popular sovereignty and individual rights.
            """.strip())
            print(f"✅ Created sample document: {sample2.name}")
        
        # Sample document 3: Architecture of Paris
        sample3 = sample_docs_dir / "paris_architecture.txt"
        if not sample3.exists():
            sample3.write_text("""
Architectural Marvels of Paris

Paris, the City of Light, is renowned for its stunning architecture spanning centuries of French history and culture.

Medieval Architecture
- Notre-Dame Cathedral: Gothic masterpiece on Île de la Cité
- Sainte-Chapelle: Royal chapel with magnificent stained glass
- Saint-Germain-des-Prés: Ancient abbey church

Renaissance and Classical Period
- Louvre Palace: Royal residence turned world's largest museum
- Luxembourg Palace: Baroque palace and gardens
- Place des Vosges: Oldest planned square in Paris

Haussmann's Paris (19th Century)
Baron Georges-Eugène Haussmann transformed Paris under Napoleon III:
- Wide boulevards and avenues
- Standardized building heights and facades
- Parks and squares system
- Sewerage and water systems

Notable Haussmannian Buildings:
- Opéra Garnier: Neo-baroque opera house
- Grands Boulevards: Commercial and social centers
- Residential buildings with characteristic iron balconies

Modern and Contemporary Architecture
- Eiffel Tower (1889): Iron lattice tower by Gustave Eiffel
- Centre Pompidou (1977): High-tech architecture
- Louvre Pyramid (1989): I.M. Pei's glass pyramid
- Institut du Monde Arabe: Jean Nouvel's modern interpretation

Architectural Districts
- Marais: Medieval and Renaissance architecture
- Saint-Germain-des-Prés: Literary and artistic quarter
- Montmartre: Village atmosphere with Sacré-Cœur Basilica
- La Défense: Modern business district with Grande Arche

Building Materials and Techniques
Traditional Parisian architecture features:
- Lutetian limestone (Pierre de Paris)
- Mansard roofs with zinc coverings
- Iron work and balconies
- Large windows and shutters

Conservation Efforts
Paris maintains strict building codes to preserve its architectural heritage while allowing for contemporary additions that complement the historic urban fabric.
            """.strip())
            print(f"✅ Created sample document: {sample3.name}")
        
        print(f"\n📝 Created {len(list(sample_docs_dir.glob('*.txt')))} sample documents")
        
        # Create .env file if it doesn't exist
        env_file = Path(".env")
        if not env_file.exists():
            env_content = """# Semantic Search POC Configuration
DEBUG=True
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///./data/semantic_search.db

# Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./data/embeddings/chroma

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Search Settings
MAX_SEARCH_RESULTS=50
SIMILARITY_THRESHOLD=0.2

# File Upload
MAX_FILE_SIZE=52428800
"""
            env_file.write_text(env_content)
            print("Created .env configuration file")
        
        print("\nSetup complete! You can now:")
        print("1. Run: python -m src.main")
        print("2. Or process documents: python scripts/process_documents.py")
        print("3. Or start the API server: python scripts/run_server.py")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        print(f"❌ Setup failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(setup_environment())