"""
Working Memory Component.

Provides a simple, potentially time-limited, key-value store for short-term memory.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from pydantic import Field

from ..core.errors import MemoryError
from .base import BaseMemory
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

# Simple in-memory store for now
_working_memory_store: Dict[str, Dict[str, MemoryItem]] = {}
_working_memory_ttl: Dict[str, datetime] = {}

class WorkingMemory(BaseMemory):
    """Simple in-memory working memory with optional TTL."""
    
    # TODO: Add config model for TTL, cleanup interval etc.
    def __init__(
        self,
        default_ttl_seconds: Optional[int] = 3600, # Default to 1 hour
        name: str = "working_memory"
    ):
        """Initialize working memory."""
        super().__init__(name)
        self._default_ttl = timedelta(seconds=default_ttl_seconds) if default_ttl_seconds else None
        self._store = _working_memory_store
        self._ttl_map = _working_memory_ttl
        # TODO: Add background task for periodic cleanup
        logger.info(f"Initialized {self.name} with default TTL: {self._default_ttl}")
        
    async def _initialize_impl(self) -> None:
        """Initialize working memory store (if needed)."""
        # For in-memory, nothing specific to do, but could load from snapshot
        logger.debug(f"{self.name} initialization complete.")
        pass 
    
    async def _shutdown_impl(self) -> None:
        """Shutdown working memory (if needed)."""
        # For in-memory, nothing specific to do, but could save snapshot
        logger.debug(f"{self.name} shutdown complete.")
        pass

    def _get_context_store(self, context: str) -> Dict[str, MemoryItem]:
        """Get or create the dictionary for a given context."""
        if context not in self._store:
            self._store[context] = {}
        return self._store[context]
    
    async def _store_impl(
        self, 
        key: str, 
        value: Any, 
        context: str, 
        metadata: Dict[str, Any] = None,
        importance: float = 0.5, # Not used currently
        ttl_seconds: Optional[int] = None,
        **kwargs
    ) -> None:
        """Store item in working memory."""
        self._cleanup_expired() # Perform cleanup before storing
        context_store = self._get_context_store(context)
        
        metadata = metadata or {}
        metadata['stored_at'] = datetime.utcnow().isoformat()

        item = MemoryItem(key=key, value=value, context=context, metadata=metadata)
        context_store[key] = item
        
        # Set TTL if provided or use default
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds is not None else self._default_ttl
        if ttl:
            expiry_time = datetime.utcnow() + ttl
            self._ttl_map[f"{context}::{key}"] = expiry_time
            metadata['expires_at'] = expiry_time.isoformat()
            
        logger.debug(f"Stored key '{key}' in working memory context '{context}'. TTL: {ttl}")
    
    async def _retrieve_impl(
        self, 
        key: str, 
        context: str,
        **kwargs
    ) -> Optional[MemoryItem]:
        """Retrieve item from working memory by key."""
        self._cleanup_expired() # Perform cleanup before retrieving
        context_store = self._get_context_store(context)
        item = context_store.get(key)
        
        if item:
            # Check TTL if it exists for this item
            ttl_key = f"{context}::{key}"
            if ttl_key in self._ttl_map and datetime.utcnow() > self._ttl_map[ttl_key]:
                logger.debug(f"Key '{key}' found in working memory context '{context}' but expired. Removing.")
                del context_store[key]
                del self._ttl_map[ttl_key]
                return None # Expired
            else:
                logger.debug(f"Retrieved key '{key}' from working memory context '{context}'.")
                # Return the value field directly? Interface expects Any
                # Let's return the value for now, consistent with AgentMemory perhaps?
                # BaseMemory retrieve returns MemoryItem though. Let's stick to MemoryItem.
                return item
        else:
            logger.debug(f"Key '{key}' not found in working memory context '{context}'.")
        return None
    
    async def _search_impl(
        self, 
        query: str, 
        context: str,
        limit: int = 10,
        **kwargs
    ) -> List[MemoryItem]:
        """Search working memory (simple substring match for now)."""
        self._cleanup_expired() # Perform cleanup before searching
        context_store = self._get_context_store(context)
        results = []
        query_lower = query.lower()
        
        # Iterate through non-expired items
        valid_keys = set(context_store.keys())
        for key, item in list(context_store.items()): # Iterate copy in case of cleanup
            ttl_key = f"{context}::{key}"
            if ttl_key in self._ttl_map and datetime.utcnow() > self._ttl_map[ttl_key]:
                 if key in valid_keys:
                      del context_store[key]
                      del self._ttl_map[ttl_key]
                 continue # Skip expired
                 
            # Simple search: check key and string representation of value
            try:
                value_str = str(item.value).lower()
                if query_lower in key.lower() or query_lower in value_str:
                    results.append(item)
            except Exception:
                 # Ignore items that cannot be easily stringified for search
                 pass
                 
            if len(results) >= limit:
                 break
                 
        logger.debug(f"Found {len(results)} items matching query '{query}' in working memory context '{context}'.")
        return results
    
    async def _wipe_context_impl(self, context: str) -> None:
        """Wipe a specific context from working memory."""
        if context in self._store:
            del self._store[context]
            # Remove associated TTL entries
            prefix = f"{context}::"
            keys_to_remove = [k for k in self._ttl_map if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._ttl_map[k]
            logger.info(f"Wiped working memory context: {context}")
        else:
            # BaseMemory.wipe already checks if context exists, but log here too
            logger.warning(f"Attempted to wipe non-existent working memory context: {context}")

    async def _wipe_all_impl(self) -> None:
        """Wipe all contexts from working memory."""
        self._store.clear()
        self._ttl_map.clear()
        logger.info("Wiped all working memory contexts.")

    def _cleanup_expired(self) -> None:
        """Remove expired items from the store."""
        now = datetime.utcnow()
        expired_ttl_keys = [k for k, expiry in self._ttl_map.items() if now > expiry]
        if not expired_ttl_keys:
            return
            
        logger.debug(f"Cleaning up {len(expired_ttl_keys)} expired items from working memory...")
        for ttl_key in expired_ttl_keys:
            try:
                 context, key = ttl_key.split('::', 1)
                 if context in self._store and key in self._store[context]:
                      del self._store[context][key]
                 del self._ttl_map[ttl_key]
            except (ValueError, KeyError):
                 # Handle potential errors if key format is wrong or item already removed
                 logger.warning(f"Error cleaning up TTL key: {ttl_key}", exc_info=True)
                 # Ensure TTL entry is removed even if store deletion failed
                 if ttl_key in self._ttl_map:
                      del self._ttl_map[ttl_key]
                      
        logger.debug("Working memory cleanup complete.")

    # Override context methods to indicate they are not used
    def create_context(self, context_name: str, **kwargs) -> str:
        logger.debug("WorkingMemory does not use explicit context creation.")
        # Return the name, as BaseMemory expects a string path
        return context_name 

    def get_context_model(self, context_path: str) -> Optional[MemoryContext]:
        logger.debug("WorkingMemory does not manage context models.")
        return None # Return None instead of raising NotImplementedError
        
    # Override search_with_model to bypass BaseMemory's context resolution
    async def search_with_model(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search working memory using a structured request, bypassing context resolution.

        Overrides BaseMemory.search_with_model to avoid checking the context registry,
        as WorkingMemory handles context implicitly.
        
        Args:
            request: Memory search request with parameters
            
        Returns:
            Memory search result with matching items
            
        Raises:
            NotInitializedError: If memory is not initialized
            MemoryError: If search fails or query is empty
        """
        self._check_initialized() # Still check if initialized
        
        # Validate query
        if not request.query:
            raise MemoryError(
                message="Search query cannot be empty",
                operation="search_with_model",
                component_name=self.name
            )
            
        # Use provided context or default, DO NOT RESOLVE/VALIDATE with _resolve_context
        context_to_use = request.context or self._default_context 
        if not context_to_use:
             # This should ideally not happen if _default_context is set
             raise MemoryError(
                 message="Search requires a context, and no default context is set.",
                 operation="search_with_model",
                 component_name=self.name
             )
             
        logger.debug(f"WorkingMemory searching in implicit context: {context_to_use}")

        try:
            # Directly call the internal search implementation
            # Pass relevant parameters from the request
            results: List[MemoryItem] = await self._search_impl(
                query=request.query,
                context=context_to_use,
                limit=request.limit,
                # Pass other potential kwargs if _search_impl uses them
                # Currently _search_impl doesn't seem to use more kwargs
                **request.dict(exclude={"query", "context", "limit"})
            )
            
            # Return results in the standard format
            return MemorySearchResult(
                query=request.query,
                items=results,
                context=context_to_use # Report the context used
            )
        except Exception as e:
            # Wrap exception
            raise MemoryError(
                message=f"WorkingMemory search failed for '{request.query}': {str(e)}",
                operation="search_with_model",
                context=context_to_use,
                component_name=self.name,
                cause=e
            ) from e 