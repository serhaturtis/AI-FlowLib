"""Pinecone vector database provider implementation.

This module provides a concrete implementation of the VectorDBProvider
for Pinecone, a managed vector database service.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import uuid

from ...core.errors import ProviderError, ErrorContext
from .base import VectorDBProvider, VectorDBProviderSettings, VectorMetadata, SimilaritySearchResult
from ..decorators import provider
from ..constants import ProviderType

logger = logging.getLogger(__name__)

try:
    import pinecone
    from pinecone import Pinecone as PineconeClient
except ImportError:
    logger.warning("pinecone-client package not found. Install with 'pip install pinecone-client'")


class PineconeProviderSettings(VectorDBProviderSettings):
    """Settings for Pinecone provider.
    
    Attributes:
        api_key: Pinecone API key
        environment: Pinecone environment
        index_name: Name of the Pinecone index to use
        namespace: Optional namespace for partitioning index
        dimension: Vector dimension (required when creating a new index)
        metric: Distance metric for similarity ('cosine', 'dotproduct', 'euclidean')
        pod_type: Pinecone pod type for index creation
        pod_size: Pod size for the index
        replicas: Number of replicas for the index
        shards: Number of shards for the index
        metadata_config: Metadata configuration for index creation
        api_timeout: API timeout in seconds
    """
    
    # Pinecone connection settings
    api_key: str
    environment: str
    index_name: str
    namespace: Optional[str] = None
    
    # Index settings (used when creating new index)
    dimension: Optional[int] = None
    metric: str = "cosine"  # cosine, dotproduct, euclidean
    pod_type: str = "p1"  # p1, p2, s1
    pod_size: str = "x1"  # x1, x2, x4, x8
    replicas: int = 1
    shards: int = 1
    metadata_config: Optional[Dict[str, Any]] = None
    
    # API settings
    api_timeout: int = 20


@provider(provider_type=ProviderType.VECTOR_DB, name="pinecone")
class PineconeProvider(VectorDBProvider):
    """Pinecone implementation of the VectorDBProvider.
    
    This provider implements vector storage, retrieval, and similarity search
    using Pinecone, a managed vector database service.
    """
    
    def __init__(self, name: str = "pinecone", settings: Optional[PineconeProviderSettings] = None):
        """Initialize Pinecone provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Create settings first to avoid issues with _default_settings() method
        settings = settings or PineconeProviderSettings(api_key="", environment="", index_name="")
        
        # Pass explicit settings to parent class
        super().__init__(name=name, settings=settings)
        
        # Store settings for local use
        self._settings = settings
        self._client = None
        self._index = None
        
    async def _initialize(self) -> None:
        """Initialize Pinecone client and index.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Initialize Pinecone client
            self._client = PineconeClient(api_key=self._settings.api_key, environment=self._settings.environment)
            
            # Check if index exists
            existing_indexes = self._client.list_indexes()
            
            if self._settings.index_name not in existing_indexes.names():
                # Create index if dimension is provided
                if not self._settings.dimension:
                    raise ProviderError(
                        message=f"Index {self._settings.index_name} does not exist and no dimension provided for creation",
                        provider_name=self.name,
                        context=ErrorContext.create(index_name=self._settings.index_name)
                    )
                
                # Create index
                self._client.create_index(
                    name=self._settings.index_name,
                    dimension=self._settings.dimension,
                    metric=self._settings.metric,
                    spec={
                        "pod_type": self._settings.pod_type,
                        "pod_size": self._settings.pod_size,
                        "replicas": self._settings.replicas,
                        "shards": self._settings.shards,
                        "metadata_config": self._settings.metadata_config
                    }
                )
                
                # Wait for index to be ready
                while not self._settings.index_name in self._client.list_indexes().names():
                    await asyncio.sleep(1)
                
                logger.info(f"Created Pinecone index: {self._settings.index_name}")
            
            # Get index
            self._index = self._client.Index(self._settings.index_name)
            
            # Get index stats to verify connection
            self._index.describe_index_stats()
            
            logger.info(f"Connected to Pinecone index: {self._settings.index_name}")
            
        except Exception as e:
            self._client = None
            self._index = None
            raise ProviderError(
                message=f"Failed to connect to Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    api_key="***",  # Don't log API key
                    environment=self._settings.environment,
                    index_name=self._settings.index_name
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close Pinecone connection."""
        self._index = None
        self._client = None
        logger.info(f"Closed Pinecone connection for index: {self._settings.index_name}")
    
    async def create_collection(self, 
                              collection_name: Optional[str] = None, 
                              dimension: Optional[int] = None,
                              metric: Optional[str] = None) -> None:
        """Create a collection (namespace) in Pinecone.
        
        Note: In Pinecone, collections are implemented as namespaces
        within an index. This method doesn't actually create a new
        collection since Pinecone namespaces are created implicitly.
        
        Args:
            collection_name: Collection name (namespace)
            dimension: Vector dimension (ignored, set at index level)
            metric: Distance metric (ignored, set at index level)
            
        Raises:
            ProviderError: If creation fails
        """
        if not self._index:
            await self.initialize()
            
        # Pinecone namespaces are created implicitly when inserting vectors
        # Nothing to do here, but we'll log the intended namespace
        logger.info(f"Pinecone namespace '{collection_name}' will be created when vectors are inserted")
    
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection (namespace) from Pinecone.
        
        Args:
            collection_name: Collection name (namespace)
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._index:
            await self.initialize()
            
        try:
            # Delete all vectors in the namespace
            self._index.delete(
                delete_all=True,
                namespace=collection_name
            )
            
            logger.info(f"Deleted all vectors in Pinecone namespace: {collection_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete Pinecone namespace: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(namespace=collection_name),
                cause=e
            )
    
    async def insert_vectors(self, 
                            vectors: List[List[float]], 
                            metadata: List[VectorMetadata], 
                            collection_name: Optional[str] = None) -> List[str]:
        """Insert vectors into Pinecone.
        
        Args:
            vectors: List of vector embeddings
            metadata: List of metadata for each vector
            collection_name: Optional collection name (namespace)
            
        Returns:
            List of vector IDs
            
        Raises:
            ProviderError: If insertion fails
        """
        if not self._index:
            await self.initialize()
            
        # Use default namespace if not specified
        namespace = collection_name or self._settings.namespace or ""
        
        try:
            # Generate IDs if not provided
            ids = []
            for i, meta in enumerate(metadata):
                if meta.id:
                    ids.append(meta.id)
                else:
                    ids.append(str(uuid.uuid4()))
            
            # Prepare vectors with metadata
            vector_items = []
            for i, vec in enumerate(vectors):
                # Convert metadata to dict
                meta_dict = metadata[i].model_dump() if metadata and i < len(metadata) else {}
                # Remove id from metadata dict (it's used as the vector ID)
                if "id" in meta_dict:
                    del meta_dict["id"]
                
                vector_items.append({
                    "id": ids[i],
                    "values": vec,
                    "metadata": meta_dict
                })
            
            # Split into batches of 100 (Pinecone limit)
            batch_size = 100
            batches = [vector_items[i:i + batch_size] for i in range(0, len(vector_items), batch_size)]
            
            # Insert batches
            for batch in batches:
                self._index.upsert(vectors=batch, namespace=namespace)
            
            logger.info(f"Inserted {len(vectors)} vectors into Pinecone namespace: {namespace}")
            
            return ids
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to insert vectors into Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    namespace=namespace,
                    vector_count=len(vectors)
                ),
                cause=e
            )
    
    async def get_vectors(self, 
                         ids: List[str], 
                         collection_name: Optional[str] = None) -> List[Tuple[List[float], VectorMetadata]]:
        """Get vectors by ID from Pinecone.
        
        Args:
            ids: List of vector IDs
            collection_name: Optional collection name (namespace)
            
        Returns:
            List of (vector, metadata) tuples
            
        Raises:
            ProviderError: If retrieval fails
        """
        if not self._index:
            await self.initialize()
            
        # Use default namespace if not specified
        namespace = collection_name or self._settings.namespace or ""
        
        try:
            # Fetch vectors
            response = self._index.fetch(ids=ids, namespace=namespace)
            
            # Parse response
            results = []
            for vec_id in ids:
                if vec_id in response.vectors:
                    vector_data = response.vectors[vec_id]
                    vector = vector_data.values
                    
                    # Create metadata object with ID
                    metadata_dict = vector_data.metadata or {}
                    metadata_dict["id"] = vec_id
                    metadata = VectorMetadata(**metadata_dict)
                    
                    results.append((vector, metadata))
                else:
                    # Vector not found, add empty result
                    results.append(([], VectorMetadata(id=vec_id)))
            
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get vectors from Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    namespace=namespace,
                    ids=ids
                ),
                cause=e
            )
    
    async def delete_vectors(self, 
                           ids: List[str], 
                           collection_name: Optional[str] = None) -> None:
        """Delete vectors by ID from Pinecone.
        
        Args:
            ids: List of vector IDs
            collection_name: Optional collection name (namespace)
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._index:
            await self.initialize()
            
        # Use default namespace if not specified
        namespace = collection_name or self._settings.namespace or ""
        
        try:
            # Delete vectors
            self._index.delete(ids=ids, namespace=namespace)
            
            logger.info(f"Deleted {len(ids)} vectors from Pinecone namespace: {namespace}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete vectors from Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    namespace=namespace,
                    ids=ids
                ),
                cause=e
            )
    
    async def search_by_vector(self, 
                             vector: List[float], 
                             k: int = 10, 
                             collection_name: Optional[str] = None,
                             filter: Optional[Dict[str, Any]] = None) -> List[SimilaritySearchResult]:
        """Search for similar vectors by vector.
        
        Args:
            vector: Query vector
            k: Number of results to return
            collection_name: Optional collection name (namespace)
            filter: Optional metadata filter
            
        Returns:
            List of search results
            
        Raises:
            ProviderError: If search fails
        """
        if not self._index:
            await self.initialize()
            
        # Use default namespace if not specified
        namespace = collection_name or self._settings.namespace or ""
        
        try:
            # Execute query
            results = self._index.query(
                namespace=namespace,
                vector=vector,
                top_k=k,
                include_values=True,
                include_metadata=True,
                filter=filter
            )
            
            # Parse results
            search_results = []
            for match in results.matches:
                # Create metadata object with ID
                metadata_dict = match.metadata or {}
                metadata_dict["id"] = match.id
                metadata = VectorMetadata(**metadata_dict)
                
                # Create search result
                search_result = SimilaritySearchResult(
                    id=match.id,
                    vector=match.values,
                    metadata=metadata,
                    score=match.score,
                    distance=1.0 - match.score if self._settings.metric == "cosine" else match.score
                )
                
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search vectors in Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    namespace=namespace,
                    k=k,
                    filter=filter
                ),
                cause=e
            )
    
    async def search_by_id(self, 
                          id: str, 
                          k: int = 10, 
                          collection_name: Optional[str] = None,
                          filter: Optional[Dict[str, Any]] = None) -> List[SimilaritySearchResult]:
        """Search for similar vectors by ID.
        
        Args:
            id: ID of the vector to use as query
            k: Number of results to return
            collection_name: Optional collection name (namespace)
            filter: Optional metadata filter
            
        Returns:
            List of search results
            
        Raises:
            ProviderError: If search fails
        """
        if not self._index:
            await self.initialize()
            
        # Use default namespace if not specified
        namespace = collection_name or self._settings.namespace or ""
        
        try:
            # Execute query
            results = self._index.query(
                namespace=namespace,
                id=id,
                top_k=k,
                include_values=True,
                include_metadata=True,
                filter=filter
            )
            
            # Parse results
            search_results = []
            for match in results.matches:
                # Create metadata object with ID
                metadata_dict = match.metadata or {}
                metadata_dict["id"] = match.id
                metadata = VectorMetadata(**metadata_dict)
                
                # Create search result
                search_result = SimilaritySearchResult(
                    id=match.id,
                    vector=match.values,
                    metadata=metadata,
                    score=match.score,
                    distance=1.0 - match.score if self._settings.metric == "cosine" else match.score
                )
                
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search vectors by ID in Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    namespace=namespace,
                    id=id,
                    k=k,
                    filter=filter
                ),
                cause=e
            )
    
    async def search_by_metadata(self, 
                               filter: Dict[str, Any], 
                               k: int = 10, 
                               collection_name: Optional[str] = None) -> List[SimilaritySearchResult]:
        """Search for vectors by metadata.
        
        Args:
            filter: Metadata filter
            k: Number of results to return
            collection_name: Optional collection name (namespace)
            
        Returns:
            List of search results
            
        Raises:
            ProviderError: If search fails
        """
        if not self._index:
            await self.initialize()
            
        # Use default namespace if not specified
        namespace = collection_name or self._settings.namespace or ""
        
        try:
            # Get stats to get dimensionality
            stats = self._index.describe_index_stats()
            dimension = stats.dimension
            
            # Create a zero vector for the query
            # This is a workaround since Pinecone doesn't support pure metadata queries
            zero_vector = [0.0] * dimension
            
            # Execute query with filter
            results = self._index.query(
                namespace=namespace,
                vector=zero_vector,
                top_k=k,
                include_values=True,
                include_metadata=True,
                filter=filter
            )
            
            # Parse results
            search_results = []
            for match in results.matches:
                # Create metadata object with ID
                metadata_dict = match.metadata or {}
                metadata_dict["id"] = match.id
                metadata = VectorMetadata(**metadata_dict)
                
                # Create search result - distance is not meaningful here
                search_result = SimilaritySearchResult(
                    id=match.id,
                    vector=match.values,
                    metadata=metadata,
                    score=0.0,
                    distance=0.0
                )
                
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search vectors by metadata in Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    namespace=namespace,
                    filter=filter,
                    k=k
                ),
                cause=e
            )
    
    async def count_vectors(self, collection_name: Optional[str] = None) -> int:
        """Count vectors in a collection.
        
        Args:
            collection_name: Optional collection name (namespace)
            
        Returns:
            Number of vectors
            
        Raises:
            ProviderError: If count fails
        """
        if not self._index:
            await self.initialize()
            
        # Use default namespace if not specified
        namespace = collection_name or self._settings.namespace or ""
        
        try:
            # Get index stats
            stats = self._index.describe_index_stats()
            
            # Get namespace count
            if namespace:
                if namespace in stats.namespaces:
                    return stats.namespaces[namespace].vector_count
                return 0
            
            # Get total count for all namespaces
            return stats.total_vector_count
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to count vectors in Pinecone: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(namespace=namespace),
                cause=e
            ) 