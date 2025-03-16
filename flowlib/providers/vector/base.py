"""Vector database provider base class and related functionality.

This module provides the base class for implementing vector database providers
that share common functionality for storing, retrieving, and searching
vector embeddings with metadata.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union, Tuple
import asyncio
import numpy as np
from pydantic import BaseModel, Field

from ...core.errors import ProviderError, ErrorContext
from ...core.models.settings import ProviderSettings
from ..base import AsyncProvider

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class VectorDBProviderSettings(ProviderSettings):
    """Base settings for vector database providers.
    
    Attributes:
        host: Vector database host address
        port: Vector database port
        api_key: API key for cloud vector databases
        username: Authentication username (if required)
        password: Authentication password (if required)
        index_name: Default vector index/collection name
        vector_dimension: Dimension of vector embeddings
        metric: Distance metric for similarity search
        batch_size: Batch size for bulk operations
    """
    
    # Connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Vector settings
    index_name: str = "default"
    vector_dimension: int = 1536  # Default for OpenAI embeddings
    metric: str = "cosine"  # cosine, euclidean, dot
    
    # Performance settings
    batch_size: int = 100
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class VectorMetadata(BaseModel):
    """Metadata for vector entries.
    
    Attributes:
        id: Unique identifier for the vector
        text: Original text that was embedded (if applicable)
        metadata: Custom metadata for the vector
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    
    id: str
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


class SimilaritySearchResult(BaseModel):
    """Result from a vector similarity search.
    
    Attributes:
        id: Vector ID
        score: Similarity score
        metadata: Vector metadata
        vector: Vector data (if requested)
    """
    
    id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector: Optional[List[float]] = None


class VectorDBProvider(AsyncProvider, Generic[T]):
    """Base class for vector database providers.
    
    This class provides:
    1. Vector storage and retrieval
    2. Similarity search
    3. Metadata storage and filtering
    4. Type-safe operations with Pydantic models
    """
    
    def __init__(self, name: str = "vector", settings: Optional[VectorDBProviderSettings] = None):
        """Initialize vector database provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Pass provider_type="vector" to the parent class
        super().__init__(name=name, settings=settings, provider_type="vector")
        self._initialized = False
        self._client = None
        self._settings = settings or VectorDBProviderSettings()
        
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
        
    async def initialize(self):
        """Initialize the vector database connection.
        
        This method should be implemented by subclasses to establish
        connections to the vector database.
        """
        self._initialized = True
        
    async def shutdown(self):
        """Close all connections and release resources.
        
        This method should be implemented by subclasses to properly
        close connections and clean up resources.
        """
        self._initialized = False
        self._client = None
        
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
        raise NotImplementedError("Subclasses must implement create_index()")
        
    async def delete_index(self, index_name: Optional[str] = None) -> bool:
        """Delete a vector index/collection.
        
        Args:
            index_name: Index name (default from settings if None)
            
        Returns:
            True if index was deleted successfully
            
        Raises:
            ProviderError: If index deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_index()")
        
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
        raise NotImplementedError("Subclasses must implement insert()")
        
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
        raise NotImplementedError("Subclasses must implement insert_batch()")
        
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
        raise NotImplementedError("Subclasses must implement get()")
        
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
        raise NotImplementedError("Subclasses must implement delete()")
        
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
        raise NotImplementedError("Subclasses must implement search()")
        
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
        raise NotImplementedError("Subclasses must implement search_by_id()")
        
    async def search_structured(self, query_vector: List[float], output_type: Type[T], top_k: int = 10,
                              filter: Optional[Dict[str, Any]] = None, index_name: Optional[str] = None) -> List[T]:
        """Search for similar vectors and parse results into structured types.
        
        Args:
            query_vector: Query vector data
            output_type: Pydantic model for parsing results
            top_k: Number of results to return
            filter: Optional metadata filter
            index_name: Index name (default from settings if None)
            
        Returns:
            List of parsed model instances
            
        Raises:
            ProviderError: If search or parsing fails
        """
        try:
            # Perform the search
            results = await self.search(
                query_vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_vectors=False,
                index_name=index_name
            )
            
            # Parse results into the output type
            parsed_results = []
            for result in results:
                # Combine metadata with score
                data = result.metadata.copy()
                data["score"] = result.score
                data["id"] = result.id
                
                # Parse into output type
                parsed_results.append(output_type.parse_obj(data))
                
            return parsed_results
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to perform structured vector search: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    output_type=output_type.__name__,
                    top_k=top_k,
                    filter=filter
                ),
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
        raise NotImplementedError("Subclasses must implement count()")
        
    async def check_connection(self) -> bool:
        """Check if vector database connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_connection()") 