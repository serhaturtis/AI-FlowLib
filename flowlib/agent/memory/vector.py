"""
Vector memory implementation.

This module provides a vector-based memory system for semantic search.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ..core.errors import MemoryError, ProviderError
from ..models.memory import MemoryItem, MemorySearchResult
from .base import BaseMemory

logger = logging.getLogger(__name__)


class VectorMemory(BaseMemory):
    """Vector-based memory system for semantic search.
    
    Uses vector embeddings to enable semantic search capabilities.
    Requires a vector provider (e.g., Chroma) and an embedding provider.
    """
    
    def __init__(
        self,
        provider_name: Optional[str] = "chroma",
        embedding_provider_name: Optional[str] = None,
        name: str = "vector_memory"
    ):
        """Initialize vector memory.
        
        Args:
            provider_name: Name of the vector provider to use
            embedding_provider_name: Name of the embedding provider to use
            name: Component name
        """
        super().__init__(name)
        
        self._provider_name = provider_name
        self._embedding_provider_name = embedding_provider_name
        
        # These will be initialized in _initialize_impl
        self._vector_provider = None
        self._embedding_provider = None
        
    async def _initialize_impl(self) -> None:
        """Initialize vector memory components."""
        # Import here to avoid circular imports
        from flowlib.providers.registry import provider_registry
        from flowlib.providers.constants import ProviderType
        
        try:
            # Initialize vector provider
            if not self._provider_name:
                raise MemoryError("No vector provider specified")
                
            self._vector_provider = await provider_registry.get(
                ProviderType.VECTOR_DB, 
                self._provider_name
            )
            
            if not self._vector_provider:
                raise ProviderError(f"Vector provider not found: {self._provider_name}")
                
            # Initialize embedding provider if specified
            if self._embedding_provider_name:
                # Note: If there's a specific ProviderType for embeddings, use that instead
                self._embedding_provider = await provider_registry.get(
                    ProviderType.EMBEDDING,  # Changed to EMBEDDING type
                    self._embedding_provider_name
                )
                
                if not self._embedding_provider:
                    raise ProviderError(
                        f"Embedding provider not found: {self._embedding_provider_name}"
                    )
            
            # Use vector provider's embedding provider if none is specified
            elif hasattr(self._vector_provider, 'embedding_provider'):
                self._embedding_provider = self._vector_provider.embedding_provider
            else:
                logger.warning(
                    f"No embedding provider specified and vector provider "
                    f"does not have an embedded one. Semantic search may be limited."
                )
                
            # Initialize the vector provider if it has its own initialize method
            if hasattr(self._vector_provider, 'initialize'):
                await self._vector_provider.initialize()
                
            # Initialize the embedding provider if it has its own initialize method
            if self._embedding_provider and hasattr(self._embedding_provider, 'initialize'):
                await self._embedding_provider.initialize()
                
            logger.debug(
                f"Initialized {self.name} with provider={self._provider_name}, "
                f"embedding_provider={self._embedding_provider_name or 'embedded'}"
            )
                
        except Exception as e:
            raise MemoryError(f"Failed to initialize vector memory: {str(e)}") from e
    
    async def _shutdown_impl(self) -> None:
        """Shutdown vector memory components."""
        # Shutdown vector provider if it has its own shutdown method
        if self._vector_provider and hasattr(self._vector_provider, 'shutdown'):
            await self._vector_provider.shutdown()
            
        # Shutdown embedding provider if it has its own shutdown method
        if (self._embedding_provider and 
            self._embedding_provider != self._vector_provider and
            hasattr(self._embedding_provider, 'shutdown')):
            await self._embedding_provider.shutdown()
            
        self._vector_provider = None
        self._embedding_provider = None
        
        logger.debug(f"Shut down {self.name}")
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as a list of floats
        """
        if not self._embedding_provider:
            raise MemoryError("No embedding provider available")
            
        try:
            # Call the embedding provider to get the embedding
            # EmbeddingProvider base class guarantees the embed method
                return await self._embedding_provider.embed(text)
        except Exception as e:
            raise MemoryError(f"Failed to get embedding for text: {str(e)}") from e
    
    def _process_value_for_storage(self, value: Any) -> str:
        """Process a value for storage in the vector database.
        
        Args:
            value: Value to process
            
        Returns:
            String representation of the value
        """
        # Handle different value types appropriately
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)
        elif hasattr(value, 'model_dump'):
            # Pydantic models
            try:
                import json
                return json.dumps(value.model_dump())
            except Exception:
                return str(value)
        elif isinstance(value, dict):
            # Dictionary values
            try:
                import json
                return json.dumps(value)
            except Exception:
                return str(value)
        elif hasattr(value, '__dict__'):
            # Objects with __dict__
            try:
                import json
                return json.dumps(value.__dict__)
            except Exception:
                return str(value)
        else:
            # Fall back to string representation
            return str(value)
    
    async def _store_impl(
        self, 
        key: str, 
        value: Any, 
        context: str, 
        importance: float = 0.5,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> None:
        """Store a value in vector memory.
        
        Args:
            key: Key to store the value under
            value: Value to store
            context: Context path
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata
            **kwargs: Additional storage parameters
        """
        # Process value for storage
        processed_value = self._process_value_for_storage(value)
        
        # Prepare metadata
        full_metadata = {
            "key": key,
            "context": context,
            "importance": importance,
            "content_type": type(value).__name__,
        }
        
        # Add user-provided metadata
        if metadata:
            full_metadata.update(metadata)
            
        # Add additional kwargs to metadata
        full_metadata.update({k: v for k, v in kwargs.items() if k not in ["id", "embedding"]})
        
        try:
            # Generate embedding if the provider doesn't handle it
            embedding = None
            # Most vector DB providers likely won't auto-embed using our separate provider
            # We will almost always need to generate the embedding here.
            # if not hasattr(self._vector_provider, 'auto_embed') or not self._vector_provider.auto_embed:
            embedding_list = await self._get_embedding(processed_value)
            # _get_embedding returns List[List[float]], we need List[float] for single item
            if embedding_list:
                embedding = embedding_list[0]
            
            # Store in vector database
            await self._vector_provider.add_item(
                id=f"{context}:{key}",
                text=processed_value,
                metadata=full_metadata,
                embedding=embedding
            )
            
            logger.debug(f"Stored item with key '{key}' in context '{context}' in vector memory")
            
        except Exception as e:
            raise MemoryError(
                f"Failed to store in vector memory: {key} in context {context}: {str(e)}"
            ) from e
    
    async def _retrieve_impl(
        self, 
        key: str, 
        context: str,
        **kwargs
    ) -> Any:
        """Retrieve a value from vector memory.
        
        Args:
            key: Key to retrieve
            context: Context path
            **kwargs: Additional retrieval parameters
            
        Returns:
            Retrieved value or None if not found
        """
        try:
            # Attempt exact retrieval by ID
            item_id = f"{context}:{key}"
            result = await self._vector_provider.get_item(item_id)
            
            if result:
                logger.debug(f"Retrieved item with key '{key}' from context '{context}' in vector memory")
                return result.get("text")
                
            # If not found, try a similarity search with high threshold
            search_results = await self._vector_provider.search(
                query=key,
                metadata_filter={"context": context, "key": key},
                limit=1,
                min_score=0.95
            )
            
            if search_results and len(search_results) > 0:
                logger.debug(
                    f"Retrieved item with key '{key}' from context '{context}' "
                    f"in vector memory via search"
                )
                return search_results[0].get("text")
                
            logger.debug(f"Item with key '{key}' not found in context '{context}' in vector memory")
            return None
            
        except Exception as e:
            logger.warning(
                f"Failed to retrieve from vector memory: {key} in context {context}: {str(e)}"
            )
            return None
    
    async def _search_impl(
        self, 
        query: str, 
        context: str,
        limit: int = 10,
        min_score: float = 0.0,
        **kwargs
    ) -> List[MemorySearchResult]:
        """Search for relevant memories in vector memory.
        
        Args:
            query: Search query
            context: Context path
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0 to 1.0)
            **kwargs: Additional search parameters
            
        Returns:
            List of matching memory items
        """
        try:
            # Prepare metadata filter for the context
            metadata_filter = {"context": context}
            
            # Add any additional filters from kwargs
            filters = kwargs.get("filters", {})
            if filters:
                metadata_filter.update(filters)
                
            # Generate query embedding
            query_embedding_list = await self._get_embedding(query)
            if not query_embedding_list:
                raise MemoryError("Failed to generate embedding for search query")
            query_embedding = query_embedding_list[0] # Get the single query embedding
            
            # Search using vector provider
            search_results = await self._vector_provider.search(
                query_vector=query_embedding,  # Changed from query=query_embedding
                top_k=limit,  # Changed from limit=limit
                filter=metadata_filter,  # Changed from metadata_filter=metadata_filter
                include_vectors=False,
                index_name=context
            )
            
            # Convert to MemorySearchResult objects
            results = []
            for result in search_results:
                # Extract data from the search result
                item_metadata = result.get("metadata", {})
                item_key = item_metadata.get("key", "unknown")
                item_context = item_metadata.get("context", context)
                item_importance = item_metadata.get("importance", 0.5)
                
                # Create a MemoryItem
                memory_item = MemoryItem(
                    key=item_key,
                    value=result.get("text"),
                    context=item_context,
                    importance=item_importance,
                    metadata=item_metadata
                )
                
                # Create a MemorySearchResult
                search_result = MemorySearchResult(
                    item=memory_item,
                    score=result.get("score", 0.0),
                    metadata={"result_type": "vector_search"}
                )
                
                results.append(search_result)
                
            logger.debug(
                f"Found {len(results)} results for query '{query}' "
                f"in context '{context}' in vector memory"
            )
            return results
            
        except Exception as e:
            logger.warning(
                f"Failed to search vector memory with query '{query}' "
                f"in context {context}: {str(e)}"
            )
            return []
    
    async def _wipe_context_impl(self, context: str) -> None:
        """Wipe all items in a specific context.
        
        Args:
            context: Context path to wipe
        """
        try:
            # Delete all items with matching context
            await self._vector_provider.delete_items(
                metadata_filter={"context": context}
            )
            
            logger.debug(f"Wiped context '{context}' from vector memory")
            
        except Exception as e:
            raise MemoryError(f"Failed to wipe context {context}: {str(e)}") from e
    
    async def _wipe_all_impl(self) -> None:
        """Wipe all memory contents."""
        try:
            # Delete all items
            await self._vector_provider.delete_all()
            
            logger.debug("Wiped all contents from vector memory")
            
        except Exception as e:
            raise MemoryError(f"Failed to wipe all vector memory: {str(e)}") from e 