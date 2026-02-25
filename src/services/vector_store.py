import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

import numpy as np
from config.settings import settings


class VectorStore:
    """Vector store for document embeddings using ChromaDB."""

    def __init__(self):
        """Initialize vector store."""
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collection = None
        self.collection_name = "documents"

    async def initialize(self):
        """Initialize the vector store."""
        try:
            persist_dir = Path(settings.CHROMA_PERSIST_DIR)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except NotFoundError:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document embeddings for semantic search"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise

    async def add_document(
        self,
        document_id: str,
        embedding: List[float],
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        try:
            if not self.collection:
                await self.initialize()

            chroma_metadata = {k: str(v) for k, v in metadata.items()}

            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[chroma_metadata],
                ids=[document_id]
            )

            self.logger.info(f"Added document {document_id} to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Error adding document {document_id}: {str(e)}")
            return False

    async def similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            if not self.collection:
                await self.initialize()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where
            )

            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    # ChromaDB default metric is squared L2. For normalized vectors, Cosine Similarity = 1 - (L2^2 / 2)
                    similarity_score = 1.0 - (distance / 2.0)

                    if similarity_score >= threshold:
                        result = {
                            'document_id': results['ids'][0][i],
                            'content': results['documents'][0][i] if results['documents'] else '',
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'similarity_score': similarity_score
                        }

                        if 'entities' in result['metadata']:
                            try:
                                import json
                                result['entities'] = json.loads(result['metadata']['entities'])
                            except:
                                result['entities'] = []
                        else:
                            result['entities'] = []

                        formatted_results.append(result)

            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            self.logger.info(f"Found {len(formatted_results)} results above threshold {threshold}")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            return []

    async def search_by_entity(
        self,
        entity_text: str,
        entity_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            if not self.collection:
                await self.initialize()

            results = self.collection.get()
            filtered_results = []
            entity_lower = entity_text.lower()

            if results['ids']:
                for i in range(len(results['ids'])):
                    document_content = results['documents'][i] if results['documents'] else ''
                    metadata = results['metadatas'][i] if results['metadatas'] else {}

                    if (entity_lower in document_content.lower() or
                        entity_lower in str(metadata).lower()):

                        result = {
                            'document_id': results['ids'][i],
                            'content': document_content,
                            'metadata': metadata,
                            'similarity_score': 1.0
                        }

                        if 'entities' in metadata:
                            try:
                                import json
                                result['entities'] = json.loads(metadata['entities'])
                            except:
                                result['entities'] = []
                        else:
                            result['entities'] = []

                        filtered_results.append(result)

                        if len(filtered_results) >= limit:
                            break

            self.logger.info(f"Found {len(filtered_results)} documents containing entity '{entity_text}'")
            return filtered_results

        except Exception as e:
            self.logger.error(f"Error searching for entity '{entity_text}': {str(e)}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        try:
            if not self.collection:
                await self.initialize()

            self.collection.delete(ids=[document_id])
            self.logger.info(f"Deleted document {document_id} from vector store")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            if not self.collection:
                await self.initialize()

            count = self.collection.count()

            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_dimension': settings.EMBEDDING_DIMENSION
            }

        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {}

    async def reset_collection(self) -> bool:
        try:
            if not self.client:
                await self.initialize()

            try:
                self.client.delete_collection(self.collection_name)
            except NotFoundError:
                pass

            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for semantic search"}
            )

            self.logger.info(f"Reset collection: {self.collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error resetting collection: {str(e)}")
            return False
