"""Enhanced memory management system for agents.

This module provides a unified memory management system with working, short-term,
and long-term memory capabilities, organized by memory contexts.
"""

import logging
import time
import json
import uuid
from typing import List, Dict, Any, Optional, Union, Set, Tuple, Type

from pydantic import BaseModel, Field

import flowlib as fl
from ..core.errors import ProviderError, ErrorContext
from ..core.registry.constants import ProviderType

logger = logging.getLogger(__name__)


class MemoryContext:
    """Defines a namespace for memory operations.
    
    Each context (e.g., "conversation", "task-123", "planning") has its
    own isolated namespace to prevent key collisions.
    """
    
    def __init__(self, name: str, parent: Optional["MemoryContext"] = None):
        """Initialize a memory context.
        
        Args:
            name: Name of the context
            parent: Optional parent context
        """
        self.name = name
        self.parent = parent
        
    def get_full_path(self) -> str:
        """Get the full hierarchical path for this context."""
        if self.parent:
            return f"{self.parent.get_full_path()}:{self.name}"
        return self.name
        
    def __eq__(self, other):
        if not isinstance(other, MemoryContext):
            return False
        return self.get_full_path() == other.get_full_path()
        
    def __hash__(self):
        return hash(self.get_full_path())


class MemoryItem(BaseModel):
    """Represents an item in agent memory."""
    content: Any = Field(..., description="The content to remember")
    content_type: str = Field("text", description="Type of content (text, json, binary)")
    context: str = Field(..., description="Memory context this item belongs to")
    source: str = Field(..., description="Source of the memory (e.g., flow name or user)")
    key: Optional[str] = Field(None, description="Retrieval key (namespace_key:item_key)")
    timestamp: float = Field(default_factory=time.time, description="When the memory was created")
    expires_at: Optional[float] = Field(None, description="When this memory expires")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Tags for retrieval")
    
    class Config:
        arbitrary_types_allowed = True


class MemoryManager:
    """Unified memory management system for agents.
    
    Provides organized access to different memory types:
    - Working Memory: Current context and operational data
    - Short-Term Memory: Recently accessed information
    - Long-Term Memory: Persistent, semantically retrievable information
    """
    
    def __init__(
        self,
        working_memory_provider: str = "memory-cache",
        short_term_provider: str = "memory-cache", 
        long_term_provider: str = "chroma",
        embeddings_provider: Optional[str] = None
    ):
        """Initialize memory manager with provider configuration.
        
        Args:
            working_memory_provider: Provider for working memory
            short_term_provider: Provider for short-term memory
            long_term_provider: Provider for long-term memory
            embeddings_provider: Provider for generating embeddings
        """
        # Initialize memory providers
        self.working_memory_provider_name = working_memory_provider
        self.short_term_provider_name = short_term_provider
        self.long_term_provider_name = long_term_provider
        self.embeddings_provider_name = embeddings_provider
        
        # Will be initialized later
        self.working_memory = None
        self.short_term = None  
        self.long_term = None
        self.embeddings = None
        
        # Track memory contexts
        self._active_contexts = set()
        self._initialized = False
    
    async def initialize(self):
        """Initialize all memory systems."""
        if self._initialized:
            return
            
        try:
            # Initialize working memory (for current context)
            self.working_memory = await fl.provider_registry.get(
                ProviderType.CACHE,
                self.working_memory_provider_name
            )
            
            # Initialize short-term memory
            self.short_term = await fl.provider_registry.get(
                ProviderType.CACHE,
                self.short_term_provider_name
            )
            
            # Initialize long-term memory
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
            logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to initialize memory manager: {str(e)}",
                provider_name="memory-manager",
                context=ErrorContext.create(
                    working_memory=self.working_memory_provider_name,
                    short_term=self.short_term_provider_name,
                    long_term=self.long_term_provider_name
                ),
                cause=e
            )
    
    # Context Management
    
    def create_context(self, name: str, parent: Optional[MemoryContext] = None) -> MemoryContext:
        """Create a new memory context.
        
        Args:
            name: Name of the context
            parent: Optional parent context
            
        Returns:
            Memory context object
        """
        context = MemoryContext(name, parent)
        self._active_contexts.add(context.get_full_path())
        return context
        
    async def clear_context(self, context: Union[str, MemoryContext]) -> None:
        """Clear all memory associated with a context.
        
        Args:
            context: Context to clear
        """
        self._check_initialized()
        
        context_path = context.get_full_path() if isinstance(context, MemoryContext) else context
        
        # Clear working memory with this context prefix
        keys = await self._list_keys_with_prefix(self.working_memory, f"{context_path}:")
        for key in keys:
            await self.working_memory.delete(key)
            
        # Remove from active contexts
        if context_path in self._active_contexts:
            self._active_contexts.remove(context_path)
    
    # Working Memory Operations
    
    async def store(self, key: str, value: Any, context: Union[str, MemoryContext], 
                   ttl: Optional[int] = 3600, to_long_term: bool = False,
                   importance: float = 0.5) -> str:
        """Store an item in working memory.
        
        Args:
            key: Item key within the context
            value: Value to store
            context: Memory context
            ttl: Time-to-live in seconds
            to_long_term: Whether to also store in long-term memory
            importance: Importance score (used for long-term storage decisions)
            
        Returns:
            Full storage key
        """
        self._check_initialized()
        
        # Format context path
        context_path = context.get_full_path() if isinstance(context, MemoryContext) else context
        
        # Create full key
        full_key = f"{context_path}:{key}"
        
        # Convert to MemoryItem if not already
        if not isinstance(value, MemoryItem):
            # Detect content type
            content_type = "json" if isinstance(value, (dict, list)) else "text"
            
            # Create a memory item
            memory_item = MemoryItem(
                content=value,
                content_type=content_type,
                context=context_path,
                source="agent",
                key=full_key,
                timestamp=time.time(),
                expires_at=time.time() + ttl if ttl else None,
                importance=importance
            )
        else:
            memory_item = value
            
        # Store in working memory
        item_data = memory_item.dict() if hasattr(memory_item, 'dict') else memory_item.model_dump()
        await self.working_memory.set(
            key=full_key,
            value=item_data,
            ttl=ttl
        )
        
        # Optionally store in long-term memory
        if to_long_term or memory_item.importance > 0.7:
            await self._store_in_long_term(memory_item)
            
        return full_key
    
    async def retrieve(self, key: str, context: Union[str, MemoryContext], 
                     model_class: Optional[Type] = None) -> Optional[Any]:
        """Retrieve an item from working memory.
        
        Args:
            key: Item key within the context
            context: Memory context
            model_class: Optional Pydantic model class to deserialize data into
            
        Returns:
            Retrieved value or None if not found
        """
        self._check_initialized()
        
        # Format context path
        context_path = context.get_full_path() if isinstance(context, MemoryContext) else context
        
        # Create full key
        full_key = f"{context_path}:{key}"
        
        # Get from working memory
        value = await self.working_memory.get(full_key)
        if not value:
            return None
            
        # If we have a memory item, extract the content
        if isinstance(value, dict) and "content" in value:
            content = value["content"]
            content_type = value.get("content_type", "text")
            
            # Handle special cases for known keys
            if key == "conversation_input":
                # Special case for conversation input
                try:
                    from ..agents.flows import MessageInput
                    if isinstance(content, dict):
                        return MessageInput(**content)
                except ImportError:
                    logger.warning("Could not import MessageInput, returning raw content")
                    return content
            
            # If model_class is provided, try to deserialize into that class
            if model_class and isinstance(content, dict):
                try:
                    return model_class(**content)
                except Exception as e:
                    logger.warning(f"Failed to deserialize into {model_class.__name__}: {str(e)}")
            
            return content
        
        # Not a memory item, return as is
        if model_class and isinstance(value, dict):
            try:
                return model_class(**value)
            except Exception as e:
                logger.warning(f"Failed to deserialize raw value into {model_class.__name__}: {str(e)}")
                
        return value
    
    # Long-term Memory Operations
    
    async def remember(self, content: str, context: Union[str, MemoryContext], 
                      source: str, importance: float = 0.5, 
                      metadata: Optional[Dict[str, Any]] = None,
                      tags: Optional[List[str]] = None) -> str:
        """Store information in long-term memory.
        
        Args:
            content: Content to remember
            context: Memory context
            source: Source of the memory
            importance: Importance score (0-1)
            metadata: Additional metadata
            tags: Tags for retrieval
            
        Returns:
            Memory ID
        """
        self._check_initialized()
        
        # Format context path
        context_path = context.get_full_path() if isinstance(context, MemoryContext) else context
        
        # Create memory item
        memory_item = MemoryItem(
            content=content,
            content_type="text",
            context=context_path,
            source=source,
            timestamp=time.time(),
            importance=importance,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Store in long-term memory
        return await self._store_in_long_term(memory_item)
    
    async def recall(self, query: str, context: Optional[Union[str, MemoryContext]] = None,
                    limit: int = 5) -> List[MemoryItem]:
        """Recall information from long-term memory using semantic search.
        
        Args:
            query: Search query
            context: Optional context to filter results
            limit: Maximum number of results
            
        Returns:
            List of memory items matching the query
        """
        self._check_initialized()
        
        # Get embeddings for the query
        query_embeddings = await self._get_embeddings(query)
        
        # Prepare filter if context is provided
        filter_dict = None
        if context:
            context_path = context.get_full_path() if isinstance(context, MemoryContext) else context
            filter_dict = {"context": context_path}
        
        # Search vector DB
        results = await self.long_term.search(
            query_vector=query_embeddings,
            top_k=limit,
            filter=filter_dict,
            index_name="agent-memory"
        )
        
        # Convert results to MemoryItems
        items = []
        for result in results:
            # Extract metadata from result
            metadata = result.metadata
            
            # Create MemoryItem
            item = MemoryItem(**metadata)
            items.append(item)
            
        return items
    
    # Memory Maintenance
    
    async def cleanup_expired(self) -> int:
        """Remove expired items from working and short-term memory.
        
        Returns:
            Number of items cleaned up
        """
        # This implementation depends on the specific provider capabilities
        # For now, we'll just return 0 as most cache providers handle expiration automatically
        return 0
        
    async def persist_working_memory(self, context: Union[str, MemoryContext], 
                                   min_importance: float = 0.5) -> int:
        """Move important items from working memory to long-term memory.
        
        Args:
            context: Context to persist
            min_importance: Minimum importance threshold
            
        Returns:
            Number of items persisted
        """
        self._check_initialized()
        context_path = context.get_full_path() if isinstance(context, MemoryContext) else context
        
        # This is a simplistic implementation that would be enhanced
        # based on the specific provider's capabilities to list keys
        # Since most cache providers don't have this capability, we'll
        # just return 0 for now
        return 0
    
    # Helper methods
    
    def _check_initialized(self) -> None:
        """Check if memory manager is initialized."""
        if not self._initialized:
            raise RuntimeError("Memory manager is not initialized. Call initialize() first.")
    
    async def _store_in_long_term(self, item: MemoryItem) -> str:
        """Store a memory item in long-term memory.
        
        Args:
            item: Memory item to store
            
        Returns:
            Item ID in long-term storage
        """
        # Get embeddings for the content
        if isinstance(item.content, (dict, list)):
            # Convert to string for embedding
            content_str = json.dumps(item.content)
        else:
            content_str = str(item.content)
            
        embeddings = await self._get_embeddings(content_str)
        
        # Store in vector DB
        item_dict = item.dict() if hasattr(item, 'dict') else item.model_dump()
        id = await self.long_term.insert(
            vector=embeddings,
            metadata=item_dict,
            index_name="agent-memory"
        )
        
        return id
        
    async def _get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embeddings
        """
        if self.embeddings:
            # Use configured embeddings provider
            return await self.embeddings.get_embeddings(text)
        else:
            # Use a placeholder embedding method that generates 384-dimensional vectors
            # This is just a placeholder and should be replaced with a real embedding
            # implementation in a production system
            import hashlib
            import numpy as np
            
            # Create a seed from the text hash
            text_bytes = text.encode()
            hash_obj = hashlib.sha256(text_bytes)
            seed = int.from_bytes(hash_obj.digest()[:4], byteorder='big')
            
            # Use the seed to generate a deterministic but reasonable embedding
            # that's 384 dimensions (to match common embedding models like in ChromaDB)
            rng = np.random.RandomState(seed)
            raw_embedding = rng.randn(384)
            
            # Normalize to unit length as most embedding spaces use normalized vectors
            normalized = raw_embedding / np.linalg.norm(raw_embedding)
            return normalized.tolist()
    
    async def _list_keys_with_prefix(self, provider, prefix: str) -> List[str]:
        """List all keys with a given prefix.
        
        This is a placeholder - implementation depends on the provider's capabilities.
        Some providers might not support listing keys at all.
        """
        # This is a simplified implementation that works only for providers
        # that have list_keys or scan methods
        if hasattr(provider, "list_keys"):
            all_keys = await provider.list_keys()
            return [key for key in all_keys if key.startswith(prefix)]
        
        # If the provider doesn't support listing keys, return empty list
        return [] 