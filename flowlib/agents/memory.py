"""Agent memory implementation using provider system.

This module provides memory management for agents using the existing
provider system, with support for short-term and long-term memory.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field

import flowlib as fl
from ..core.errors import ProviderError, ErrorContext
from ..core.registry.constants import ProviderType

logger = logging.getLogger(__name__)

class MemoryItem(BaseModel):
    """Represents an item in agent memory."""
    content: str = Field(..., description="The actual content to remember")
    source: str = Field(..., description="Source of the memory (e.g., flow name or user)")
    timestamp: float = Field(default_factory=time.time, description="When the memory was created")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentMemory:
    """Memory manager for agents using provider system.
    
    This class provides:
    1. Short-term memory using a cache provider
    2. Long-term memory using a vector database provider
    3. Memory retrieval by recency or semantic similarity
    """
    
    def __init__(
        self,
        short_term_provider: str = "memory-cache",
        long_term_provider: str = "chroma",
        embeddings_provider: Optional[str] = None
    ):
        """Initialize memory with provider names.
        
        Args:
            short_term_provider: Cache provider name for short-term memory
            long_term_provider: Vector DB provider name for long-term memory
            embeddings_provider: Optional embeddings provider name
        """
        self.short_term_provider_name = short_term_provider
        self.long_term_provider_name = long_term_provider
        self.embeddings_provider_name = embeddings_provider
        
        self.short_term = None
        self.long_term = None
        self.embeddings = None
        
        self.item_count = 0
        self._initialized = False
    
    async def initialize(self):
        """Initialize memory providers.
        
        Raises:
            ProviderError: If provider initialization fails
        """
        if self._initialized:
            return
            
        try:
            # Initialize short-term memory (cache provider)
            self.short_term = await fl.provider_registry.get(
                ProviderType.CACHE,
                self.short_term_provider_name
            )
            
            # Initialize long-term memory (vector DB provider)
            self.long_term = await fl.provider_registry.get(
                ProviderType.VECTOR_DB,
                self.long_term_provider_name
            )
            
            # Create vector index for long-term memory
            await self.long_term.create_index("agent-memory")
            
            # Initialize embeddings provider if specified
            if self.embeddings_provider_name:
                self.embeddings = await fl.provider_registry.get(
                    ProviderType.LLM,
                    self.embeddings_provider_name
                )
            
            self._initialized = True
            logger.info("Agent memory initialized successfully")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to initialize agent memory: {str(e)}",
                provider_name="agent-memory",
                context=ErrorContext.create(
                    short_term=self.short_term_provider_name,
                    long_term=self.long_term_provider_name
                ),
                cause=e
            )
    
    def _check_initialized(self):
        """Check if memory is initialized.
        
        Raises:
            RuntimeError: If memory is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Agent memory is not initialized. Call initialize() first.")
    
    async def add(self, item: Union[MemoryItem, str]) -> str:
        """Add an item to memory.
        
        Args:
            item: Memory item or string content
            
        Returns:
            Item ID
            
        Raises:
            ProviderError: If memory operation fails
        """
        self._check_initialized()
        
        # Convert string to MemoryItem if needed
        if isinstance(item, str):
            item = MemoryItem(
                content=item,
                source="agent",
                timestamp=time.time()
            )
            
        # Set timestamp if not provided
        if not item.timestamp:
            item.timestamp = time.time()
            
        try:
            # Store in short-term memory
            key = f"memory:{int(item.timestamp)}"
            await self.short_term.set(
                key=key,
                value=item.dict() if hasattr(item, 'dict') else item.model_dump(),
                ttl=3600  # 1 hour TTL for short-term memory
            )
            
            # Store in long-term memory if important enough
            if item.importance > 0.3:
                # Get embeddings for the content
                embeddings = await self._get_embeddings(item.content)
                
                # Store in vector DB
                item_dict = item.dict() if hasattr(item, 'dict') else item.model_dump()
                id = await self.long_term.insert(
                    vector=embeddings,
                    metadata=item_dict,
                    index_name="agent-memory"
                )
                
                # Update item count
                self.item_count += 1
                return id
            
            # Update item count
            self.item_count += 1
            return key
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add item to memory: {str(e)}",
                provider_name="agent-memory",
                context=ErrorContext.create(
                    item_content=item.content[:100] + "..." if len(item.content) > 100 else item.content
                ),
                cause=e
            )
    
    async def retrieve_recent(self, limit: int = 10) -> List[MemoryItem]:
        """Retrieve recent items from short-term memory.
        
        Args:
            limit: Maximum number of items to retrieve
            
        Returns:
            List of memory items
            
        Raises:
            ProviderError: If memory operation fails
        """
        self._check_initialized()
        
        try:
            # This is a simplified implementation that assumes the cache provider
            # doesn't have a way to list or scan keys. A real implementation would
            # need to adapt to the specific cache provider capabilities.
            
            # For in-memory cache, we could scan all keys and sort by timestamp
            items = []
            
            # Return empty list for now
            # In a real implementation, this would retrieve items from the cache
            return items
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to retrieve recent memory items: {str(e)}",
                provider_name="agent-memory",
                cause=e
            )
    
    async def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search long-term memory using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of items to retrieve
            
        Returns:
            List of memory items matching the query
            
        Raises:
            ProviderError: If memory operation fails
        """
        self._check_initialized()
        
        try:
            # Get embeddings for the query
            query_embeddings = await self._get_embeddings(query)
            
            # Search vector DB
            results = await self.long_term.search(
                query_vector=query_embeddings,
                top_k=limit,
                index_name="agent-memory"
            )
            
            # Convert results to MemoryItems
            items = []
            for result in results:
                # Extract metadata from result
                metadata = result.metadata
                
                # Create MemoryItem
                item = MemoryItem(
                    content=metadata.get("content", ""),
                    source=metadata.get("source", "unknown"),
                    timestamp=metadata.get("timestamp", 0),
                    importance=metadata.get("importance", 0.5),
                    metadata=metadata.get("metadata", {})
                )
                items.append(item)
                
            return items
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search memory: {str(e)}",
                provider_name="agent-memory",
                context=ErrorContext.create(query=query),
                cause=e
            )
    
    async def _get_embeddings(self, text: str) -> List[float]:
        """Get vector embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embeddings
            
        Raises:
            ProviderError: If embedding generation fails
        """
        # If no embeddings provider, use a simple placeholder
        # In a real implementation, this would use an embedding model
        if not self.embeddings:
            # Return a placeholder embedding vector (all zeros)
            # A real implementation would use a proper embedding model
            return [0.0] * 384  # 384-dimensional embedding
            
        try:
            # Use the embeddings provider to generate embeddings
            # The actual implementation depends on the provider's API
            # This is just a placeholder
            embeddings = await self.embeddings.generate_embeddings(text)
            return embeddings
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to generate embeddings: {str(e)}",
                provider_name="agent-memory",
                cause=e
            )