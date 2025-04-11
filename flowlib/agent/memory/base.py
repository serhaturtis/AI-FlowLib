"""
Base memory implementation.

This module provides the foundation for all memory types in the agent system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..core.base import BaseComponent
from ..core.errors import MemoryError, NotInitializedError
from .interfaces import MemoryInterface
from .models import (
    MemoryItem, 
    MemoryStoreRequest, 
    MemoryRetrieveRequest, 
    MemorySearchRequest,
    MemorySearchResult,
    MemoryContext
)

logger = logging.getLogger(__name__)


class BaseMemory(BaseComponent, MemoryInterface, ABC):
    """Base class for all memory implementations.
    
    Implements common functionality and enforces the MemoryInterface protocol.
    Subclasses must implement the abstract methods.
    """
    
    def __init__(
        self, 
        name: str,
        contexts: Optional[Dict[str, MemoryContext]] = None
    ):
        """Initialize the base memory.
        
        Args:
            name: Name of the memory component
            contexts: Optional dictionary of pre-defined contexts
        """
        super().__init__(name)
        self._contexts = contexts or {}
        self._default_context = "agent"
        
        # Ensure the base context exists
        if "agent" not in self._contexts:
            self._contexts["agent"] = MemoryContext(
                name="agent",
                path="agent",
                parent=None,
                metadata={"builtin": True}
            )
    
    def create_context(
        self, 
        context_name: str, 
        parent: Optional[Union[str, MemoryContext]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new memory context.
        
        Args:
            context_name: Name for the new context
            parent: Optional parent context (string path or MemoryContext object)
            metadata: Optional metadata for the context
            
        Returns:
            Full context path
        """
        # Verify initialization
        self._check_initialized()
        
        # Sanitize name (replace slashes with underscores)
        sanitized_name = context_name.replace('/', '_').strip()
        
        # Handle parent being a MemoryContext object or a string
        parent_path = None
        if parent:
            if isinstance(parent, MemoryContext):
                parent_path = parent.path
            else:
                parent_path = parent
            
            # Verify parent exists
            if parent_path not in self._contexts:
                raise MemoryError(f"Parent context '{parent_path}' does not exist")
        
        # Build path
        if parent_path:
            path = f"{parent_path}/{sanitized_name}"
        else:
            path = sanitized_name
            
        # Check if context already exists
        if path in self._contexts:
            logger.warning(f"Context '{path}' already exists, returning existing path")
            return path
            
        # Create and store context
        self._contexts[path] = MemoryContext(
            name=sanitized_name,
            path=path,
            parent=parent_path,
            metadata=metadata or {}
        )
        
        logger.debug(f"Created memory context: {path}")
        return path
    
    def get_context_model(self, context_path: str) -> Optional[MemoryContext]:
        """Get a memory context by path.
        
        Args:
            context_path: Context path
            
        Returns:
            Memory context or None if not found
        """
        self._check_initialized()
        return self._contexts.get(context_path)
    
    def get_context(self, context_path: str) -> Optional[MemoryContext]:
        """Get a memory context by path (alias for get_context_model).
        
        Args:
            context_path: Context path
            
        Returns:
            Memory context or None if not found
        """
        return self.get_context_model(context_path)
    
    def list_contexts(self, parent: Optional[str] = None) -> List[MemoryContext]:
        """List available memory contexts.
        
        Args:
            parent: Optional parent context to filter by
            
        Returns:
            List of memory contexts
        """
        self._check_initialized()
        
        if parent:
            return [
                ctx for ctx in self._contexts.values()
                if ctx.parent == parent
            ]
        return list(self._contexts.values())
    
    def _resolve_context(self, context: Optional[Union[str, MemoryContext]] = None) -> str:
        """Resolve context path, using default if None.
        
        Args:
            context: Context path, MemoryContext object, or None
            
        Returns:
            Resolved context path
            
        Raises:
            MemoryError: If the context path is invalid or does not exist
        """
        self._check_initialized()
        
        # Use provided context or default
        if context is None:
            resolved = self._default_context
        elif isinstance(context, MemoryContext):
            resolved = context.path
        else:
            resolved = str(context)  # Ensure string type
            
        # Fail fast if path is empty or invalid
        if not resolved:
            raise MemoryError(
                message="Context path cannot be empty",
                context=resolved
            )
        
        # Verify context exists - fail if not found
        if resolved not in self._contexts:
            raise MemoryError(
                message=f"Context '{resolved}' does not exist. Contexts must be explicitly created before use.",
                context=resolved
            )
            
        # Context exists, return it
        return resolved
    
    async def store_with_model(self, request: MemoryStoreRequest) -> None:
        """Store a value in memory using a structured request.
        
        Args:
            request: Memory store request with parameters
            
        Raises:
            NotInitializedError: If memory is not initialized
            MemoryError: If storage fails or parameters are invalid
        """
        self._check_initialized()
        
        # Validate key
        if not request.key:
            raise MemoryError(
                message="Memory key cannot be empty",
                operation="store_with_model"
            )
            
        # Validate importance
        if request.importance is not None and (request.importance < 0.0 or request.importance > 1.0):
            raise MemoryError(
                message=f"Importance must be between 0.0 and 1.0, got {request.importance}",
                operation="store_with_model",
                key=request.key
            )
            
        # Resolve context
        context_path = self._resolve_context(request.context)
        
        # Initialize metadata dict if not provided
        metadata = request.metadata or {}
        
        # Add timestamp if not already present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
            
        # Add importance to metadata if provided
        if request.importance is not None:
            metadata["importance"] = request.importance
            
        # Add TTL to metadata if provided
        if request.ttl is not None:
            metadata["ttl"] = request.ttl
        
        try:
            # Call implementation
            await self._store_impl(
                key=request.key,
                value=request.value,
                context=context_path,
                importance=request.importance or 0.5,
                metadata=metadata,
                **request.dict(exclude={"key", "value", "context", "importance", "metadata"})
            )
            logger.debug(f"Stored '{request.key}' in context '{context_path}'")
        except Exception as e:
            # Wrap exception
            raise MemoryError(
                message=f"Failed to store '{request.key}': {str(e)}",
                operation="store_with_model",
                key=request.key,
                context=context_path,
                cause=e
            ) from e
    
    async def retrieve_with_model(self, request: MemoryRetrieveRequest) -> Any:
        """Retrieve a value from memory using a structured request.
        
        Args:
            request: Memory retrieve request
            
        Returns:
            Retrieved value or default if not found
            
        Raises:
            NotInitializedError: If memory is not initialized
            MemoryError: If retrieval fails
        """
        self._check_initialized()
        
        # Validate key
        if not request.key:
            raise MemoryError(
                message="Memory key cannot be empty",
                operation="retrieve_with_model"
            )
            
        # Resolve context
        context_path = self._resolve_context(request.context)
        
        try:
            # Call implementation with additional parameters
            value = await self._retrieve_impl(
                key=request.key,
                context=context_path,
                **request.dict(exclude={"key", "context", "default", "metadata_only"})
            )
            
            # Return default if value not found
            if value is None:
                return request.default
                
            # If metadata only is requested, return metadata
            if request.metadata_only:
                # Try to get metadata key
                meta_key = f"{request.key}_metadata"
                try:
                    metadata = await self._retrieve_impl(meta_key, context_path)
                    return metadata or {}
                except Exception:
                    return {}
                
            return value
        except Exception as e:
            # Special case: don't wrap exceptions for None results
            if "not found" in str(e).lower():
                return request.default
                
            # Wrap exception
            raise MemoryError(
                message=f"Failed to retrieve '{request.key}': {str(e)}",
                operation="retrieve_with_model",
                key=request.key,
                context=context_path,
                cause=e
            ) from e
    
    async def search_with_model(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search memory using a structured request.
        
        Args:
            request: Memory search request with parameters
            
        Returns:
            Memory search result with matching items
            
        Raises:
            NotInitializedError: If memory is not initialized
            MemoryError: If search fails
        """
        self._check_initialized()
        
        # Validate query
        if not request.query:
            raise MemoryError(
                message="Search query cannot be empty",
                operation="search_with_model"
            )
            
        # Resolve context
        context_path = self._resolve_context(request.context)
        
        try:
            # Call implementation
            results = await self._search_impl(
                query=request.query,
                context=context_path,
                limit=request.limit,
                min_score=request.threshold,
                **request.dict(exclude={"query", "context", "limit", "threshold"})
            )
            
            # Convert to MemorySearchResult
            return MemorySearchResult(
                items=results,
                count=len(results),
                query=request.query,
                context=context_path
            )
        except Exception as e:
            # Wrap exception
            raise MemoryError(
                message=f"Failed to search for '{request.query}': {str(e)}",
                operation="search_with_model",
                context=context_path,
                cause=e
            ) from e
    
    async def wipe(
        self,
        context: Optional[str] = None
    ) -> None:
        """Wipe memory contents.
        
        Args:
            context: Optional context to wipe (wipes all if None)
            
        Raises:
            NotInitializedError: If memory is not initialized
            MemoryError: If wipe fails
        """
        self._check_initialized()
        
        try:
            if context:
                # Resolve and wipe specific context
                context_path = self._resolve_context(context)
                await self._wipe_context_impl(context_path)
                logger.info(f"Wiped context '{context_path}'")
            else:
                # Wipe all contexts
                await self._wipe_all_impl()
                logger.info("Wiped all memory contexts")
        except Exception as e:
            # Wrap exception
            context_str = context or "all"
            raise MemoryError(
                message=f"Failed to wipe context '{context_str}': {str(e)}",
                operation="wipe",
                context=context_str,
                cause=e
            ) from e
    
    @abstractmethod
    async def _store_impl(
        self, 
        key: str, 
        value: Any, 
        context: str, 
        importance: float = 0.5,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> None:
        """Implementation-specific method to store a value.
        
        Args:
            key: Key to store value under
            value: Value to store
            context: Resolved context path
            importance: Importance score (0.0 to 1.0)
            metadata: Metadata for the memory
            **kwargs: Additional implementation-specific parameters
        """
        ...
    
    @abstractmethod
    async def _retrieve_impl(
        self, 
        key: str, 
        context: str,
        **kwargs
    ) -> Any:
        """Implementation-specific method to retrieve a value.
        
        Args:
            key: Key to retrieve
            context: Resolved context path
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            Retrieved value or None if not found
        """
        ...
    
    @abstractmethod
    async def _search_impl(
        self, 
        query: str, 
        context: str,
        limit: int = 10,
        min_score: float = 0.0,
        **kwargs
    ) -> List[MemoryItem]:
        """Implementation-specific method to search for memories.
        
        Args:
            query: Search query
            context: Resolved context path
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0.0 to 1.0)
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of memory items with scores
        """
        ...
    
    @abstractmethod
    async def _wipe_context_impl(self, context: str) -> None:
        """Implementation-specific method to wipe a specific context.
        
        Args:
            context: Resolved context path to wipe
        """
        ...
    
    @abstractmethod
    async def _wipe_all_impl(self) -> None:
        """Implementation-specific method to wipe all contexts."""
        ...
        
    def _check_initialized(self) -> None:
        """Check if component is initialized.
        
        Raises:
            NotInitializedError: If component is not initialized
        """
        if not self.initialized:
            raise NotInitializedError(f"{self.name} is not initialized")
