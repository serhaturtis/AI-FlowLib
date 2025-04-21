"""ChromaDB vector database provider implementation.

This module provides a concrete implementation of the VectorDBProvider
for ChromaDB, an open-source embedding database.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Collection, Generic, TypeVar, Union
import uuid
import asyncio

from ...core.errors import ProviderError, ErrorContext
from .base import VectorDBProvider, VectorDBProviderSettings, SimilaritySearchResult
from ..decorators import provider
from ..constants import ProviderType
from ..base import Provider # Import Provider base class

# Removed embedding provider imports - not needed here
# from ..embedding.base import EmbeddingProvider
# from ..registry import provider_registry

logger = logging.getLogger(__name__)


# Lazy import chromadb
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.api import Collection
    # Removed unused EmbeddingFunction, Embeddings types
except ImportError:
    logger.warning("ChromaDB package not found. Install with 'pip install chromadb'")
    chromadb = None
    # EmbeddingFunction = None # Removed

# Removed FlowlibChromaEmbeddingFunction wrapper class

class ChromaDBProviderSettings(VectorDBProviderSettings):
    """Settings for ChromaDB provider.
    
    Attributes:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the collection to use (default same as index_name)
        client_type: Type of client to use ("persistent" or "http")
        http_host: Host for HTTP client
        http_port: Port for HTTP client
        http_headers: Headers for HTTP client
        distance_function: Distance function to use (cosine, l2, ip)
        anonymized_telemetry: Whether to anonymize telemetry
    """
    
    persist_directory: Optional[str] = "./chroma_data"
    collection_name: Optional[str] = None
    client_type: str = "persistent"  # "persistent" or "http"
    http_host: Optional[str] = None
    http_port: Optional[int] = None
    http_headers: Optional[Dict[str, str]] = None
    distance_function: str = "cosine"  # cosine, l2, ip
    anonymized_telemetry: bool = False

# Type variable for settings
SettingsType = TypeVar('SettingsType', bound=ChromaDBProviderSettings)

@provider(provider_type=ProviderType.VECTOR_DB, name="chroma")
class ChromaDBProvider(VectorDBProvider[ChromaDBProviderSettings]):
    """ChromaDB implementation of the VectorDBProvider.
    
    This provider implements vector storage, retrieval, and similarity search
    using ChromaDB, an open-source embedding database.
    """
    
    def __init__(self, name: str = "chroma", settings: Optional[Union[Dict[str, Any], ChromaDBProviderSettings]] = None):
        """Initialize ChromaDB provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Initialize VectorDBProvider base - it handles settings creation/assignment
        super().__init__(name=name, settings=settings)
        
        # Base class now holds self.settings as ChromaDBProviderSettings object
        # self._settings can be removed or kept as alias if preferred, but using self.settings is standard
        # self._settings = self.settings # Alias if needed
        
        self._client = None
        self._collections = {}
        # Removed _embedding_provider and _embedding_function attributes
        
    async def initialize(self):
        """Initialize the ChromaDB client and default collection."""
        if self._initialized:
            return
            
        try:
            # Check if ChromaDB is installed
            if chromadb is None:
                raise ProviderError(
                    message="ChromaDB package not installed. Install with 'pip install chromadb'",
                    provider_name=self.name
                )
                
            # Removed embedding provider retrieval logic

            # Create client based on settings
            if self.settings.client_type == "persistent":
                # Ensure persistence directory exists
                if self.settings.persist_directory:
                    os.makedirs(self.settings.persist_directory, exist_ok=True)
                
                self._client = chromadb.PersistentClient(
                    path=self.settings.persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=self.settings.anonymized_telemetry
                    )
                )
            elif self.settings.client_type == "http":
                if not self.settings.http_host or not self.settings.http_port:
                    raise ProviderError(
                        message="HTTP host and port must be provided for HTTP client",
                        provider_name=self.name
                    )
                    
                self._client = chromadb.HttpClient(
                    host=self.settings.http_host,
                    port=self.settings.http_port,
                    headers=self.settings.http_headers or {}
                )
            else:
                # In-memory client as fallback
                self._client = chromadb.Client(
                    settings=ChromaSettings(
                        anonymized_telemetry=self.settings.anonymized_telemetry
                    )
                )
                
            # Create or get default collection (without embedding function)
            await self._get_or_create_collection(self.settings.index_name)
            
            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise ProviderError(
                message=f"Failed to initialize ChromaDB provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def shutdown(self):
        """Close ChromaDB client and release resources."""
        if not self._initialized:
            return
            
        try:
            # ChromaDB client doesn't have a close method, so we just 
            # nullify our reference
            self._client = None
            self._collections = {}
            self._initialized = False
            # Removed embedding provider cleanup
            
            logger.debug(f"{self.name} provider shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during {self.name} provider shutdown: {str(e)}")
            raise ProviderError(
                message=f"Failed to shut down ChromaDB provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
        
    async def _initialize(self) -> None:
        pass
            
    async def _get_or_create_collection(
        self, 
        index_name: str
        # Removed embedding_function parameter
    ) -> Collection:
        """Get or create a ChromaDB collection.
        
        Args:
            index_name: Collection name
            
        Returns:
            ChromaDB collection
            
        Raises:
            ProviderError: If collection creation fails
        """
        if not self._client:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        collection_name = self.settings.collection_name or index_name
        
        try:
            # Check if we already have this collection cached
            if collection_name in self._collections:
                return self._collections[collection_name]
                
            # Check if collection exists
            try:
                collection = self._client.get_collection(name=collection_name)
                logger.debug(f"Using existing collection: {collection_name}")
            except Exception:
                # Create collection if it doesn't exist
                # Pass the embedding function wrapper here
                collection = self._client.create_collection(
                    name=collection_name,
                    metadata={"distance_function": self.settings.distance_function},
                    embedding_function=None # Explicitly None
                )
                logger.debug(f"Created new collection: {collection_name}")
                
            # Cache and return the collection
            self._collections[collection_name] = collection
            return collection
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get or create collection {collection_name}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(collection_name=collection_name),
                cause=e
            )
            
    async def create_index(self, index_name: Optional[str] = None, dimension: Optional[int] = None) -> bool:
        """Create a new vector index/collection.
        
        Args:
            index_name: Index name (default from settings if None)
            dimension: Vector dimension (default from settings if None)
            
        Returns:
            True if index was created successfully
            
        Raises:
            ProviderError: If index creation fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            dimension = dimension or self.settings.vector_dimension
            
            # Get or create collection (no embedding function needed)
            await self._get_or_create_collection(index_name)
            return True
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to create index {index_name}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name, dimension=dimension),
                cause=e
            )
            
    async def delete_index(self, index_name: Optional[str] = None) -> bool:
        """Delete a vector index/collection.
        
        Args:
            index_name: Index name (default from settings if None)
            
        Returns:
            True if index was deleted successfully
            
        Raises:
            ProviderError: If index deletion fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            collection_name = self.settings.collection_name or index_name
            
            # Delete collection
            if not self._client:
                raise ProviderError(
                    message="Provider not initialized",
                    provider_name=self.name
                )
                
            try:
                self._client.delete_collection(name=collection_name)
                logger.debug(f"Deleted collection: {collection_name}")
                
                # Remove from cache
                if collection_name in self._collections:
                    del self._collections[collection_name]
                    
                return True
                
            except ValueError:
                # Collection doesn't exist, treat as success
                logger.debug(f"Collection {collection_name} doesn't exist, nothing to delete")
                return True
                
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete index {index_name}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name),
                cause=e
            )
            
    async def insert(self, vector: List[float], metadata: Dict[str, Any], id: Optional[str] = None,
                   index_name: Optional[str] = None) -> str:
        """Insert a vector with metadata.
        
        Args:
            vector: Vector data (embedding)
            metadata: Vector metadata
            id: Optional vector ID (generated if None)
            index_name: Index name (default from settings if None)
            
        Returns:
            Vector ID
            
        Raises:
            ProviderError: If insertion fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Generate ID if not provided
            id = id or str(uuid.uuid4())
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Clean metadata (ChromaDB doesn't support nested objects)
            cleaned_metadata = self._clean_metadata(metadata)
            
            # Add vector to collection
            collection.add(
                embeddings=[vector],
                metadatas=[cleaned_metadata],
                ids=[id]
            )
            
            logger.debug(f"Inserted vector with ID {id} into {index_name}")
            return id
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to insert vector: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name, id=id),
                cause=e
            )
            
    async def insert_batch(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]],
                         ids: Optional[List[str]] = None, index_name: Optional[str] = None) -> List[str]:
        """Insert multiple vectors with metadata.
        
        Args:
            vectors: List of vector data
            metadatas: List of vector metadata
            ids: Optional list of vector IDs (generated if None)
            index_name: Index name (default from settings if None)
            
        Returns:
            List of vector IDs
            
        Raises:
            ProviderError: If batch insertion fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Validate input lengths
            vector_count = len(vectors)
            if len(metadatas) != vector_count:
                raise ValueError(f"Number of vectors ({vector_count}) must match number of metadatas ({len(metadatas)})")
                
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(vector_count)]
            elif len(ids) != vector_count:
                raise ValueError(f"Number of vectors ({vector_count}) must match number of ids ({len(ids)})")
                
            # Clean metadata (ChromaDB doesn't support nested objects)
            cleaned_metadatas = [self._clean_metadata(metadata) for metadata in metadatas]
            
            # Add vectors to collection
            collection.add(
                embeddings=vectors,
                metadatas=cleaned_metadatas,
                ids=ids
            )
            
            logger.debug(f"Inserted {vector_count} vectors into {index_name}")
            return ids
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to insert vectors in batch: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name, vector_count=len(vectors)),
                cause=e
            )
            
    async def get(self, id: str, include_vector: bool = False,
                index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a vector by ID.
        
        Args:
            id: Vector ID
            include_vector: Whether to include vector data
            index_name: Index name (default from settings if None)
            
        Returns:
            Vector metadata and optionally vector data, or None if not found
            
        Raises:
            ProviderError: If retrieval fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Get vector by ID
            result = collection.get(
                ids=[id],
                include_embeddings=include_vector
            )
            
            # Check if any results were returned
            if not result["ids"]:
                return None
                
            # Construct response
            response = {
                "id": result["ids"][0],
                "metadata": result["metadatas"][0]
            }
            
            # Add vector if requested
            if include_vector and "embeddings" in result:
                response["vector"] = result["embeddings"][0]
                
            return response
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to get vector {id}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name, id=id),
                cause=e
            )
            
    async def delete(self, id: str, index_name: Optional[str] = None) -> bool:
        """Delete a vector by ID.
        
        Args:
            id: Vector ID
            index_name: Index name (default from settings if None)
            
        Returns:
            True if vector was deleted successfully
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Delete vector by ID
            collection.delete(ids=[id])
            logger.debug(f"Deleted vector with ID {id} from {index_name}")
            
            return True
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete vector {id}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name, id=id),
                cause=e
            )
            
    async def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[Dict[str, Any]] = None,
                   include_vectors: bool = False, index_name: Optional[str] = None) -> List[SimilaritySearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector data
            top_k: Number of results to return
            filter: Optional metadata filter
            include_vectors: Whether to include vector data in results
            index_name: Index name (default from settings if None)
            
        Returns:
            List of similarity search results
            
        Raises:
            ProviderError: If search fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Perform similarity search
            query_result = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filter,
                include=['metadatas', 'documents', 'distances'] + (['embeddings'] if include_vectors else [])
            )
            
            # Parse results
            results = []
            if query_result and query_result.get("ids") and query_result["ids"][0]:
                ids = query_result["ids"][0]
                distances = query_result.get("distances", [[]])[0]
                metadatas = query_result.get("metadatas", [[]])[0]
                embeddings = query_result.get("embeddings", [[]])[0]
                documents = query_result.get("documents", [[]])[0]

                for i in range(len(ids)):
                    result = SimilaritySearchResult(
                        id=ids[i],
                        score=distances[i] if i < len(distances) else 0.0,
                        metadata=metadatas[i] if i < len(metadatas) else {},
                        text=documents[i] if i < len(documents) else None
                    )
                    
                    # Add vector if requested and available
                    if include_vectors and embeddings and i < len(embeddings):
                        result.vector = embeddings[i]
                        
                    results.append(result)
                    
            return results
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to search vectors: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name, top_k=top_k),
                cause=e
            )
            
    async def search_by_id(self, id: str, top_k: int = 10, filter: Optional[Dict[str, Any]] = None,
                         include_vectors: bool = False, index_name: Optional[str] = None) -> List[SimilaritySearchResult]:
        """Search for similar vectors using an existing vector ID.
        
        Args:
            id: Vector ID to use as query
            top_k: Number of results to return
            filter: Optional metadata filter
            include_vectors: Whether to include vector data in results
            index_name: Index name (default from settings if None)
            
        Returns:
            List of similarity search results
            
        Raises:
            ProviderError: If search fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get the vector by ID first
            vector_data = await self.get(id, include_vector=True, index_name=index_name)
            
            if not vector_data or "vector" not in vector_data:
                raise ProviderError(
                    message=f"Vector with ID {id} not found",
                    provider_name=self.name,
                    context=ErrorContext.create(index_name=index_name, id=id)
                )
                
            # Now search using the vector
            return await self.search(
                query_vector=vector_data["vector"],
                top_k=top_k,
                filter=filter,
                include_vectors=include_vectors,
                index_name=index_name
            )
            
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to search by ID {id}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name, id=id, top_k=top_k),
                cause=e
            )
            
    async def count(self, filter: Optional[Dict[str, Any]] = None, index_name: Optional[str] = None) -> int:
        """Count vectors in the index.
        
        Args:
            filter: Optional metadata filter
            index_name: Index name (default from settings if None)
            
        Returns:
            Vector count
            
        Raises:
            ProviderError: If count fails
        """
        try:
            # Use settings as defaults
            index_name = index_name or self.settings.index_name
            
            # Get collection (no embedding function needed)
            collection = await self._get_or_create_collection(index_name)
            
            # Count with filter if provided
            if filter:
                result = collection.get(where=filter)
                return len(result["ids"]) if "ids" in result else 0
            else:
                # Get collection info for total count
                return collection.count()
                
        except ProviderError as e:
            # Re-raise existing provider errors
            raise e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to count vectors: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(index_name=index_name),
                cause=e
            )
            
    async def check_connection(self) -> bool:
        """Check if vector database connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        if not self._initialized or not self._client:
            return False
            
        try:
            # Try to access the client's heartbeat method
            self._client.heartbeat()
            return True
        except Exception:
            return False
            
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for ChromaDB compatibility.
        
        ChromaDB only supports simple metadata types (str, int, float, bool).
        This method flattens and converts complex types.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Cleaned metadata
        """
        cleaned = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
                
            # Handle lists and dicts by converting to strings
            if isinstance(value, (list, dict)):
                import json
                try:
                    cleaned[key] = json.dumps(value)
                except TypeError:
                    cleaned[key] = str(value)
            # Handle basic types
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            # Convert other types to strings
            else:
                cleaned[key] = str(value)
                
        return cleaned 