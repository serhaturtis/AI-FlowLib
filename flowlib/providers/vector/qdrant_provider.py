"""Qdrant vector database provider implementation.

This module provides a concrete implementation of the VectorDBProvider
for Qdrant, a vector similarity search engine.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
import uuid

from ...core.errors import ProviderError, ErrorContext
from .base import VectorDBProvider, VectorDBProviderSettings, VectorMetadata, SimilaritySearchResult
from ..decorators import provider
from ..constants import ProviderType

# Import embedding provider base and registry
from ..embedding.base import EmbeddingProvider
from ..registry import provider_registry

logger = logging.getLogger(__name__)

# Define dummy models for type annotations when qdrant-client is not installed
class DummyModels:
    class Distance:
        COSINE = "cosine"
        EUCLID = "euclid"
        DOT = "dot"
    
    class Filter:
        pass

models = DummyModels()

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http import models as rest_models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    logger.warning("qdrant-client package not found. Install with 'pip install qdrant-client'")


class QdrantProviderSettings(VectorDBProviderSettings):
    """Settings for Qdrant provider.
    
    Attributes:
        url: Qdrant server URL (e.g., 'http://localhost:6333')
        api_key: Optional API key for authentication
        collection_name: Default collection name
        prefer_grpc: Whether to use gRPC instead of HTTP
        timeout: Timeout for requests in seconds
        host: Qdrant server host (alternative to URL)
        port: Qdrant server port (alternative to URL)
        grpc_port: Qdrant gRPC port (if different from REST port)
        prefer_local: Whether to use local mode if available
        path: Path to local database (for local mode)
        embedding_provider_name: Name of the embedding provider
    """
    
    # Qdrant connection settings
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: Optional[str] = None
    prefer_grpc: bool = True
    timeout: float = 10.0
    
    # Alternative connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    grpc_port: Optional[int] = None
    
    # Local mode settings
    prefer_local: bool = False
    path: Optional[str] = None
    embedding_provider_name: Optional[str] = "default_embedding"


SettingsType = TypeVar('SettingsType', bound=QdrantProviderSettings)

@provider(provider_type=ProviderType.VECTOR_DB, name="qdrant")
class QdrantProvider(VectorDBProvider[QdrantProviderSettings]):
    """Qdrant implementation of the VectorDBProvider.
    
    This provider implements vector storage, retrieval, and similarity search
    using Qdrant, a vector similarity search engine.
    """
    
    def __init__(self, name: str = "qdrant", settings: Optional[Union[Dict[str, Any], QdrantProviderSettings]] = None):
        """Initialize Qdrant provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Initialize VectorDBProvider base - it handles settings creation/assignment
        super().__init__(name=name, settings=settings)
        
        # Base class now holds self.settings as QdrantProviderSettings object

        # Store settings for local use
        self._settings = self.settings
        self._client = None
        self._collection_info = {}
        self._embedding_provider: Optional[EmbeddingProvider] = None
        
    async def _initialize(self) -> None:
        """Initialize Qdrant client and embedding provider.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Create client based on provided settings
            if self._settings.prefer_local and self._settings.path:
                # Use local mode
                self._client = QdrantClient(
                    path=self._settings.path,
                    timeout=self._settings.timeout
                )
                logger.info(f"Connected to local Qdrant database at: {self._settings.path}")
            elif self._settings.url:
                # Use URL
                self._client = QdrantClient(
                    url=self._settings.url,
                    api_key=self._settings.api_key,
                    prefer_grpc=self._settings.prefer_grpc,
                    timeout=self._settings.timeout
                )
                logger.info(f"Connected to Qdrant server at: {self._settings.url}")
            else:
                # Use host and port
                self._client = QdrantClient(
                    host=self._settings.host,
                    port=self._settings.port,
                    grpc_port=self._settings.grpc_port,
                    prefer_grpc=self._settings.prefer_grpc,
                    api_key=self._settings.api_key,
                    timeout=self._settings.timeout
                )
                logger.info(f"Connected to Qdrant server at: {self._settings.host}:{self._settings.port}")
                
            # Get collection info for default collection if specified
            if self._settings.collection_name:
                try:
                    collection_info = self._client.get_collection(self._settings.collection_name)
                    self._collection_info[self._settings.collection_name] = collection_info
                    logger.info(f"Using default Qdrant collection: {self._settings.collection_name}")
                except Exception as e:
                    logger.warning(f"Default collection not found: {self._settings.collection_name}. It will be created when needed.")
            
            # Get and initialize the embedding provider
            if self._settings.embedding_provider_name:
                self._embedding_provider = await provider_registry.get(
                    ProviderType.EMBEDDING,
                    self._settings.embedding_provider_name
                )
                if not self._embedding_provider:
                    raise ProviderError(f"Embedding provider '{self._settings.embedding_provider_name}' not found")
                # Ensure it's initialized (registry should handle this, but double-check)
                if not self._embedding_provider.initialized:
                    await self._embedding_provider.initialize()
                logger.info(f"Using embedding provider: {self._settings.embedding_provider_name}")
            else:
                # Unlike Chroma, Qdrant requires vectors, so an external embedding provider is mandatory
                raise ConfigurationError("embedding_provider_name must be specified in Qdrant settings")
            
        except Exception as e:
            self._client = None
            raise ProviderError(
                message=f"Failed to connect to Qdrant: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    url=self._settings.url,
                    host=self._settings.host,
                    port=self._settings.port
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._collection_info = {}
            self._embedding_provider = None
            logger.info("Closed Qdrant connection")
    
    def _get_distance_metric(self, metric: Optional[str]) -> models.Distance:
        """Convert string metric to Qdrant Distance enum.
        
        Args:
            metric: Distance metric name
            
        Returns:
            Qdrant Distance enum value
        """
        metric = metric or "cosine"
        metric_lower = metric.lower()
        
        if metric_lower == "cosine":
            return models.Distance.COSINE
        elif metric_lower == "euclid" or metric_lower == "euclidean" or metric_lower == "l2":
            return models.Distance.EUCLID
        elif metric_lower == "dot" or metric_lower == "dotproduct":
            return models.Distance.DOT
        else:
            raise ProviderError(
                message=f"Unsupported distance metric: {metric}",
                provider_name=self.name,
                context=ErrorContext.create(
                    supported_metrics=["cosine", "euclidean", "dotproduct"]
                )
            )
    
    def _get_collection_name(self, collection_name: Optional[str]) -> str:
        """Get collection name from parameter or settings.
        
        Args:
            collection_name: Optional collection name
            
        Returns:
            Collection name
            
        Raises:
            ProviderError: If collection name is not specified
        """
        name = collection_name or self._settings.collection_name
        if not name:
            raise ProviderError(
                message="Collection name not specified",
                provider_name=self.name,
                context=ErrorContext.create(
                    help="Specify collection_name parameter or set default in settings"
                )
            )
        return name
    
    async def create_collection(self, 
                              collection_name: str, 
                              dimension: int,
                              metric: Optional[str] = None) -> None:
        """Create a collection in Qdrant.
        
        Args:
            collection_name: Collection name
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            
        Raises:
            ProviderError: If creation fails
        """
        if not self._client:
            await self.initialize()
            
        try:
            # Check if collection already exists
            try:
                self._client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists")
                return
            except Exception:
                # Collection doesn't exist, create it
                pass
            
            # Get distance metric
            distance = self._get_distance_metric(metric)
            
            # Create collection
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=distance
                )
            )
            
            # Get and store collection info
            collection_info = self._client.get_collection(collection_name)
            self._collection_info[collection_name] = collection_info
            
            logger.info(f"Created Qdrant collection: {collection_name} with dimension={dimension}, metric={distance.name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to create Qdrant collection: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection_name=collection_name,
                    dimension=dimension,
                    metric=metric
                ),
                cause=e
            )
    
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from Qdrant.
        
        Args:
            collection_name: Collection name
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._client:
            await self.initialize()
            
        try:
            # Delete collection
            self._client.delete_collection(collection_name=collection_name)
            
            # Remove from collection info
            if collection_name in self._collection_info:
                del self._collection_info[collection_name]
            
            logger.info(f"Deleted Qdrant collection: {collection_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete Qdrant collection: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(collection_name=collection_name),
                cause=e
            )
    
    def _metadata_to_payload(self, metadata: VectorMetadata) -> Dict[str, Any]:
        """Convert metadata to Qdrant payload.
        
        Args:
            metadata: Vector metadata
            
        Returns:
            Qdrant payload
        """
        # Convert metadata to dict
        payload = metadata.model_dump()
        
        # Remove id field (handled separately in Qdrant)
        if "id" in payload:
            del payload["id"]
            
        return payload
    
    def _payload_to_metadata(self, payload: Dict[str, Any], id: str) -> VectorMetadata:
        """Convert Qdrant payload to metadata.
        
        Args:
            payload: Qdrant payload
            id: Vector ID
            
        Returns:
            Vector metadata
        """
        # Add ID to payload
        payload = payload.copy()
        payload["id"] = id
        
        # Create metadata object
        return VectorMetadata(**payload)
    
    async def insert_vectors(self, 
                            vectors: List[List[float]], 
                            metadata: List[VectorMetadata], 
                            collection_name: Optional[str] = None) -> List[str]:
        """Insert multiple vectors into Qdrant.
        
        Args:
            vectors: List of vector embeddings
            metadata: List of metadata for each vector
            collection_name: Optional collection name
            
        Returns:
            List of vector IDs
            
        Raises:
            ProviderError: If insertion fails
        """
        if not self._client:
            await self.initialize()
            
        collection_name = self._get_collection_name(collection_name)
        
        # Ensure collection exists (or check dimension consistency)
        if collection_name not in self._collection_info:
             # Check dimension consistency if possible before raising error
             if vectors and self._embedding_provider and hasattr(self._embedding_provider, 'embedding_dim'):
                 expected_dim = self._embedding_provider.embedding_dim
                 if expected_dim and len(vectors[0]) != expected_dim:
                     raise ProviderError(
                         f"Vector dimension mismatch. Expected {expected_dim}, got {len(vectors[0])}.",
                         provider_name=self.name,
                         context=ErrorContext.create(collection_name=collection_name)
                     )

             # If collection doesn't exist, raise error (Qdrant upsert doesn't auto-create)
             raise ProviderError(
                 f"Collection '{collection_name}' does not exist. Create it first.",
                 provider_name=self.name,
                 context=ErrorContext.create(collection_name=collection_name)
             )

        # Prepare points for Qdrant
        points = []
        for i, vec in enumerate(vectors):
            # Convert metadata to payload
            payload = self._metadata_to_payload(metadata[i]) if i < len(metadata) else {}
            
            # Create point
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload
                )
            )
        
        # Insert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        logger.info(f"Inserted {len(vectors)} vectors into Qdrant collection: {collection_name}")
        
        return [str(point.id) for point in points]
    
    async def get_vectors(self, 
                         ids: List[str], 
                         collection_name: Optional[str] = None) -> List[Tuple[List[float], VectorMetadata]]:
        """Get vectors by ID from Qdrant.
        
        Args:
            ids: List of vector IDs
            collection_name: Optional collection name
            
        Returns:
            List of (vector, metadata) tuples
            
        Raises:
            ProviderError: If retrieval fails
        """
        if not self._client:
            await self.initialize()
            
        # Get collection name
        coll_name = self._get_collection_name(collection_name)
        
        try:
            # Get points by IDs
            response = self._client.retrieve(
                collection_name=coll_name,
                ids=ids,
                with_vectors=True,
                with_payload=True
            )
            
            # Parse response
            result = []
            id_to_point = {str(point.id): point for point in response}
            
            for id in ids:
                if id in id_to_point:
                    point = id_to_point[id]
                    # Convert payload to metadata
                    metadata = self._payload_to_metadata(point.payload, str(point.id))
                    result.append((point.vector, metadata))
                else:
                    # Vector not found
                    result.append(([], VectorMetadata(id=id)))
            
            return result
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get vectors from Qdrant: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection_name=coll_name,
                    ids=ids
                ),
                cause=e
            )
    
    async def delete_vectors(self, 
                           ids: List[str], 
                           collection_name: Optional[str] = None) -> None:
        """Delete vectors by ID from Qdrant.
        
        Args:
            ids: List of vector IDs
            collection_name: Optional collection name
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._client:
            await self.initialize()
            
        # Get collection name
        coll_name = self._get_collection_name(collection_name)
        
        try:
            # Delete points
            self._client.delete(
                collection_name=coll_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            
            logger.info(f"Deleted {len(ids)} vectors from Qdrant collection: {coll_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete vectors from Qdrant: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection_name=coll_name,
                    ids=ids
                ),
                cause=e
            )
    
    def _filter_to_qdrant(self, filter: Dict[str, Any]) -> Optional[models.Filter]:
        """Convert generic filter to Qdrant filter.
        
        Args:
            filter: Generic filter dict
            
        Returns:
            Qdrant filter object
        """
        if not filter:
            return None
            
        # Convert simple key-value filters to equals condition
        conditions = []
        for key, value in filter.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            elif isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            elif isinstance(value, dict):
                # Handle range queries
                if "$lt" in value or "$gt" in value or "$lte" in value or "$gte" in value:
                    range_params = {}
                    if "$lt" in value:
                        range_params["lt"] = value["$lt"]
                    if "$gt" in value:
                        range_params["gt"] = value["$gt"]
                    if "$lte" in value:
                        range_params["lte"] = value["$lte"]
                    if "$gte" in value:
                        range_params["gte"] = value["$gte"]
                    
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(**range_params)
                        )
                    )
        
        if conditions:
            return models.Filter(
                must=conditions
            )
        
        return None
    
    async def search_by_vector(self, 
                             vector: List[float], 
                             k: int = 10, 
                             collection_name: Optional[str] = None,
                             filter: Optional[Dict[str, Any]] = None) -> List[SimilaritySearchResult]:
        """Search for similar vectors by vector.
        
        Args:
            vector: Query vector
            k: Number of results to return
            collection_name: Optional collection name
            filter: Optional metadata filter
            
        Returns:
            List of search results
            
        Raises:
            ProviderError: If search fails
        """
        if not self._client:
            await self.initialize()
            
        collection_name = self._get_collection_name(collection_name)
        qdrant_filter = self._filter_to_qdrant(filter)
        
        try:
            # Perform search using query vector
            search_result = self._client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=k
            )
            
            # Parse results
            results = []
            for scored_point in search_result:
                # Convert payload to metadata
                metadata = self._payload_to_metadata(scored_point.payload, str(scored_point.id))
                
                # Create search result
                result = SimilaritySearchResult(
                    id=str(scored_point.id),
                    vector=scored_point.vector,
                    metadata=metadata,
                    score=scored_point.score,
                    distance=scored_point.score  # Qdrant returns similarity score
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search vectors in Qdrant: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection_name=collection_name,
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
            collection_name: Optional collection name
            filter: Optional metadata filter
            
        Returns:
            List of search results
            
        Raises:
            ProviderError: If search fails
        """
        if not self._client:
            await self.initialize()
            
        # Get collection name
        coll_name = self._get_collection_name(collection_name)
        
        try:
            # Convert filter
            qdrant_filter = self._filter_to_qdrant(filter)
            
            # Perform search
            search_result = self._client.search(
                collection_name=coll_name,
                query_id=id,
                limit=k,
                with_vectors=True,
                with_payload=True,
                filter=qdrant_filter
            )
            
            # Parse results
            results = []
            for scored_point in search_result:
                # Convert payload to metadata
                metadata = self._payload_to_metadata(scored_point.payload, str(scored_point.id))
                
                # Create search result
                result = SimilaritySearchResult(
                    id=str(scored_point.id),
                    vector=scored_point.vector,
                    metadata=metadata,
                    score=scored_point.score,
                    distance=scored_point.score  # Qdrant returns similarity score
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search vectors by ID in Qdrant: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection_name=coll_name,
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
            collection_name: Optional collection name
            
        Returns:
            List of search results
            
        Raises:
            ProviderError: If search fails
        """
        if not self._client:
            await self.initialize()
            
        # Get collection name
        coll_name = self._get_collection_name(collection_name)
        
        try:
            # Convert filter
            qdrant_filter = self._filter_to_qdrant(filter)
            
            if not qdrant_filter:
                raise ProviderError(
                    message="Empty filter for metadata search",
                    provider_name=self.name,
                    context=ErrorContext.create(
                        collection_name=coll_name,
                        filter=filter
                    )
                )
            
            # Perform scroll search (doesn't need a query vector)
            scroll_result = self._client.scroll(
                collection_name=coll_name,
                limit=k,
                with_vectors=True,
                with_payload=True,
                filter=qdrant_filter
            )
            
            # Parse results
            results = []
            for point in scroll_result[0]:
                # Convert payload to metadata
                metadata = self._payload_to_metadata(point.payload, str(point.id))
                
                # Create search result with placeholder score
                result = SimilaritySearchResult(
                    id=str(point.id),
                    vector=point.vector,
                    metadata=metadata,
                    score=1.0,  # Placeholder score
                    distance=0.0  # Placeholder distance
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search vectors by metadata in Qdrant: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection_name=coll_name,
                    filter=filter,
                    k=k
                ),
                cause=e
            )
    
    async def count_vectors(self, collection_name: Optional[str] = None) -> int:
        """Count vectors in a collection.
        
        Args:
            collection_name: Optional collection name
            
        Returns:
            Number of vectors
            
        Raises:
            ProviderError: If count fails
        """
        if not self._client:
            await self.initialize()
            
        # Get collection name
        coll_name = self._get_collection_name(collection_name)
        
        try:
            # Get collection info
            collection_info = self._client.get_collection(coll_name)
            
            # Return vector count
            return collection_info.vectors_count
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to count vectors in Qdrant: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(collection_name=coll_name),
                cause=e
            )
    
    async def embed_and_search(self, 
                               query_text: str, 
                               k: int = 10, 
                               collection_name: Optional[str] = None,
                               filter: Optional[Dict[str, Any]] = None) -> List[SimilaritySearchResult]:
        """Generate embedding for query text and search Qdrant.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            collection_name: Collection name
            filter: Metadata filter
            
        Returns:
            List of search results
            
        Raises:
            ProviderError: If embedding or search fails
        """
        if not self._embedding_provider:
            raise ProviderError(
                "Embedding provider not available for text search.", 
                provider_name=self.name
            )
            
        # Generate embedding for the query text
        try:
            embedding_list = await self._embedding_provider.embed(query_text)
            if not embedding_list:
                raise ProviderError("Failed to generate embedding for query text.")
            query_vector = embedding_list[0] # Use the first (only) embedding
        except Exception as e:
            raise ProviderError(
                f"Failed to generate query embedding: {e}", 
                provider_name=self.name, 
                cause=e
            )
            
        # Perform search using the generated vector
        return await self.search_by_vector(
            vector=query_vector,
            k=k,
            collection_name=collection_name,
            filter=filter
        ) 