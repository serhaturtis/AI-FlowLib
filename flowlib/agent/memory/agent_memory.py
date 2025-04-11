"""
Agent memory system using flowlib's Context.

This module provides a simplified memory system for agents by leveraging
flowlib's Context system rather than implementing a separate solution.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Type

from ...core.context import Context
from ..core.base import BaseComponent
from ..models.config import MemoryConfig
from .models import (
    MemoryItem, 
    MemoryStoreRequest, 
    MemoryRetrieveRequest, 
    MemorySearchRequest,
    MemorySearchResult,
    MemoryContext
)
from .interfaces import MemoryInterface

logger = logging.getLogger(__name__)

class AgentMemory(BaseComponent, MemoryInterface):
    """Simplified agent memory system using flowlib's Context.
    
    Instead of implementing complex multiple memory types, this uses
    the Context class from flowlib as the primary storage mechanism.
    
    This class implements the MemoryInterface protocol.
    """
    
    def __init__(
        self, 
        config: Optional[MemoryConfig] = None,
        name: str = "agent_memory"
    ):
        """Initialize the memory manager.
        
        Args:
            config: Memory configuration
            name: Component name
        """
        super().__init__(name)
        
        # Configuration
        self.config = config or MemoryConfig()
        
        # Context storage
        self._contexts = {}
        self._default_context = "agent"
    
    async def _initialize_impl(self) -> None:
        """Initialize memory components."""
        # Create default context
        self._contexts[self._default_context] = Context()
        logger.debug(f"Initialized {self.name} with default context")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown memory components."""
        # Nothing to clean up
        pass
    
    async def store_with_model(self, request: MemoryStoreRequest) -> None:
        """Store a value in memory using a structured request.
        
        Args:
            request: Memory store request with parameters
        """
        ctx_name = request.context or self._default_context
        
        # Create context if it doesn't exist
        if ctx_name not in self._contexts:
            self._contexts[ctx_name] = Context()
        
        # Store value in context
        ctx = self._contexts[ctx_name]
        ctx.set(request.key, request.value)
        
        # Store metadata if provided
        if request.metadata:
            meta_key = f"{request.key}_metadata"
            metadata = request.metadata.copy()
            # Add importance to metadata if provided
            if request.importance is not None:
                metadata["importance"] = request.importance
            # Add TTL to metadata if provided
            if request.ttl is not None:
                metadata["ttl"] = request.ttl
            
            ctx.set(meta_key, metadata)
    
    async def retrieve_with_model(self, request: MemoryRetrieveRequest) -> Any:
        """Retrieve a value from memory using a structured request.
        
        Args:
            request: Memory retrieve request
            
        Returns:
            Retrieved value or default if not found
        """
        ctx_name = request.context or self._default_context
        
        # Check if context exists
        if ctx_name not in self._contexts:
            return request.default
        
        # Retrieve value from context
        ctx = self._contexts[ctx_name]
        value = ctx.get(request.key)
        
        # Return default if value not found
        if value is None:
            return request.default
            
        # If metadata only is requested, return the metadata
        if request.metadata_only:
            meta_key = f"{request.key}_metadata"
            return ctx.get(meta_key, {})
            
        return value
    
    async def retrieve_relevant(self, query: str, context: str = None, limit: int = 5) -> List[str]:
        """Retrieve relevant memories based on a query.
        
        This method uses semantic search to find memories relevant to the query.
        
        Args:
            query: Search query
            context: Memory context (uses default if None)
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memory content as strings
            
        Notes:
            This implementation uses basic keyword matching since the underlying
            AgentMemory doesn't support true semantic search. More sophisticated
            implementations can be found in specialized memory classes like VectorMemory.
        """
        logger.debug(f"Retrieving relevant memories for query: {query} in context: {context}")
        
        # Create a search request
        search_request = MemorySearchRequest(
            query=query,
            context=context or self._default_context,
            limit=limit
        )
        
        # Perform the search
        result = await self.search_with_model(search_request)
        
        # Extract content from results
        memories = []
        for item in result.items:
            if isinstance(item.content, str):
                memories.append(item.content)
            else:
                # Handle non-string content by converting to string representation
                try:
                    if hasattr(item.content, "model_dump"):
                        memories.append(str(item.content.model_dump()))
                    else:
                        memories.append(str(item.content))
                except Exception as e:
                    logger.warning(f"Failed to convert memory content to string: {str(e)}")
        
        logger.debug(f"Retrieved {len(memories)} relevant memories")
        return memories
    
    async def search_with_model(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search memory using a structured request.
        
        Args:
            request: Memory search request
            
        Returns:
            Memory search result with matching items
        """
        ctx_name = request.context or self._default_context
        
        # Check if context exists
        if ctx_name not in self._contexts:
            return MemorySearchResult(
                items=[],
                count=0,
                query=request.query,
                context=ctx_name
            )
        
        # Get context data
        ctx = self._contexts[ctx_name]
        
        # Basic search: find keys containing the query
        items = []
        for key in ctx.keys():
            # Skip metadata keys
            if key.endswith("_metadata"):
                continue
                
            # Get value and metadata
            value = ctx.get(key)
            metadata = ctx.get(f"{key}_metadata", {})
            
            # Skip if value is None
            if value is None:
                continue
                
            # Check if key or string value contains query
            match = False
            if request.query.lower() in key.lower():
                match = True
            elif isinstance(value, str) and request.query.lower() in value.lower():
                match = True
                
            # If match found, add to results
            if match:
                item = MemoryItem(
                    key=key,
                    content=value,
                    context=ctx_name,
                    metadata=metadata,
                    created_at=metadata.get("timestamp", ""),
                    importance=metadata.get("importance", 0.5)
                )
                items.append(item)
                
        # Sort by importance (descending)
        items.sort(key=lambda x: x.importance, reverse=True)
        
        # Apply limit
        if request.limit > 0:
            items = items[:request.limit]
            
        # Create result
        result = MemorySearchResult(
            items=items,
            count=len(items),
            query=request.query,
            context=ctx_name
        )
        
        return result
    
    def create_context(
        self,
        context_name: str,
        parent: Optional[Union[str, MemoryContext]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new memory context.
        
        Args:
            context_name: Name for the new context
            parent: Optional parent context or context path
            metadata: Optional metadata for the context
            
        Returns:
            Full context path
        """
        # Get parent context path
        parent_path = ""
        if parent:
            if isinstance(parent, str):
                parent_path = parent
            else:
                parent_path = parent.path
        
        # Create context path
        context_path = f"{parent_path}/{context_name}" if parent_path else context_name
        
        # Create context
        self._contexts[context_path] = Context()
        
        # Store metadata if provided
        if metadata:
            self._contexts[context_path].set("__metadata__", metadata)
            
        return context_path
    
    def get_context_model(self, context_path: str) -> Optional[MemoryContext]:
        """Get a memory context model by path.
        
        Args:
            context_path: Context path
            
        Returns:
            Memory context model or None if not found
        """
        # Check if context exists
        if context_path not in self._contexts:
            return None
            
        # Get metadata
        metadata = self._contexts[context_path].get("__metadata__", {})
        
        # Create context model
        return MemoryContext(
            path=context_path,
            metadata=metadata
        )
    
    async def wipe(
        self,
        context: Optional[str] = None
    ) -> None:
        """Wipe memory contents.
        
        Args:
            context: Optional context to wipe (wipes all if None)
        """
        if context:
            # Wipe specific context
            if context in self._contexts:
                self._contexts[context] = Context()
                logger.debug(f"Wiped context: {context}")
        else:
            # Wipe all contexts
            self._contexts = {
                self._default_context: Context()
            }
            logger.debug("Wiped all contexts")
            
    def to_context(self, context_path: str = None) -> Optional[Context]:
        """Convert a memory context to a Context object.
        
        Args:
            context_path: Context path (defaults to default context)
            
        Returns:
            Context object or None if not found
        """
        ctx_name = context_path or self._default_context
        return self._contexts.get(ctx_name) 