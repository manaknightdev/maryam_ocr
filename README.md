# Semantic Search Engine POC

A proof-of-concept intelligent semantic search engine for archival documents, made to show how advanced search can work with different types of files like PDFs, XML files, and more.

## Project Overview

This POC addresses the requirements for a future full-scale semantic search system capable of:

- **Entity-centric search** across persons, places, events, buildings, and organizations
- **Multi-modal document processing** (PDFs, XML, text, images, audio, video)
- **Semantic similarity search** using modern embedding techniques
- **Relationship discovery** between entities across documents
- **Access control** for public vs. restricted documents
- **Scalable architecture** for production deployment

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Entity        │    │   Vector        │
│   Processor     │───▶│   Extractor     │───▶│   Store         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text          │    │   Named Entity  │    │   Embeddings    │
│   Extraction    │    │   Recognition   │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │   Search        │
                      │   Service       │
                      └─────────────────┘
```

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd semantic_search_poc
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Initialize the environment**
```bash
python scripts/setup_data.py
```

5. **Run the POC**
```bash
python -m src.main
```

### Expected Output

The POC will demonstrate:
- Document processing and indexing
- Semantic search across sample documents
- Entity extraction and relationship discovery
- Performance metrics and statistics

## Features

### Document Processing
- **PDF text extraction** using PyPDF2
- **XML parsing** for finding aids
- **DOCX support** for modern documents
- **Metadata extraction** (title, author, creation date, keywords)
- **Multi-language support** (currently optimized for English)

### Entity Recognition
- **Named Entity Recognition** using spaCy
- **Custom entity types**: Person, Place, Event, Organization, Building, Date
- **Relationship extraction** between entities
- **Confidence scoring** for entity matches

### Semantic Search
- **Vector embeddings** using Sentence-BERT (`all-MiniLM-L6-v2`)
- **Similarity search** with configurable thresholds
- **Hybrid search** combining semantic and keyword matching
- **Entity-filtered search** results

### Vector Storage
- **ChromaDB integration** for persistent vector storage
- **Scalable indexing** for large document collections
- **Metadata filtering** and search optimization

## Configuration

Key settings in `config/settings.py`:

```python
# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Search Parameters
MAX_SEARCH_RESULTS = 50
SIMILARITY_THRESHOLD = 0.3

# File Processing
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = [".pdf", ".txt", ".docx", ".xml"]
```

## Project Structure

```
semantic_search_poc/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py              # Configuration settings
├── src/
│   ├── main.py                  # Main application entry point
│   ├── models/
│   │   ├── document.py          # Document data models
│   │   └── search_result.py     # Search result models
│   ├── services/
│   │   ├── document_processor.py # Document processing pipeline
│   │   ├── embedding_service.py  # Embedding generation
│   │   ├── entity_extractor.py   # Named entity recognition
│   │   ├── search_service.py     # Main search functionality
│   │   └── vector_store.py       # Vector database operations
│   └── utils/
│       ├── file_handlers.py      # File processing utilities
│       └── logger.py             # Logging configuration
├── data/
│   ├── raw/                     # Input documents
│   ├── processed/               # Processed document metadata
│   └── embeddings/              # Vector embeddings storage
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks for analysis
└── scripts/                     # Utility scripts
```
