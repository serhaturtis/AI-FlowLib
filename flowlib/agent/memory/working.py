"""
Working memory implementation.

This module provides an in-memory cache with TTL for temporary storage.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.errors import MemoryError
from ..models.memory import MemoryContext, MemoryItem, MemorySearchResult
from .base import BaseMemory

logger = logging.getLogger(__name__)


class WorkingMemory(BaseMemory):
    """In-memory cache with TTL for temporary storage.
    
    This component provides a simple cache for frequently accessed
    memory items with automatic expiration.
    """
    
    def __init__(
        self,
        ttl: Optional[int] = 3600,  # 1 hour default
        max_items: int = 1000,
        name: str = "working_memory"
    ):
        """Initialize working memory.
        
        Args:
            ttl: Default TTL in seconds (or None for no expiration)
            max_items: Maximum number of items to store
            name: Component name
        """
        super().__init__(name)
        
        self.default_ttl = ttl
        self.max_items = max_items
        self._store: Dict[Tuple[str, str], MemoryItem] = {}
        self._lock = asyncio.Lock()
        
    async def _initialize_impl(self) -> None:
        """Initialize the working memory."""
        self._store = {}
        logger.debug(f"Initialized {self.name} with max_items={self.max_items}")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the working memory."""
        self._store.clear()
        logger.debug(f"Shut down {self.name}")
    
    async def _cleanup_expired(self) -> None:
        """Remove expired items from memory."""
        now = datetime.now()
        keys_to_remove = []
        
        async with self._lock:
            for (ctx, key), item in self._store.items():
                if item.expires_at and item.expires_at <= now:
                    keys_to_remove.append((ctx, key))
                    
            for key in keys_to_remove:
                del self._store[key]
                
            if keys_to_remove:
                logger.debug(f"Removed {len(keys_to_remove)} expired items from {self.name}")
    
    async def _store_impl(
        self, 
        key: str, 
        value: Any, 
        context: str, 
        importance: float = 0.5,
        metadata: Dict[str, Any] = None,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """Store a value in working memory.
        
        Args:
            key: Key to store the value under
            value: Value to store
            context: Context path
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata
            ttl: Optional time-to-live in seconds
            **kwargs: Additional storage parameters
        """
        await self._cleanup_expired()
        
        # Calculate expiration time if TTL is provided
        expires_at = None
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        if effective_ttl is not None:
            expires_at = datetime.now() + timedelta(seconds=effective_ttl)
        
        # Combine metadata
        combined_metadata = {**kwargs}
        if metadata:
            combined_metadata.update(metadata)
        
        # Create memory item
        item = MemoryItem(
            key=key,
            value=value,
            context=context,
            expires_at=expires_at,
            importance=importance,
            metadata=combined_metadata
        )
        
        # Store the item
        async with self._lock:
            # Enforce maximum items limit
            if len(self._store) >= self.max_items:
                # Find least important items
                items = sorted(
                    self._store.items(),
                    key=lambda x: (x[1].importance, x[1].created_at)
                )
                
                # Remove up to 10% of items
                items_to_remove = max(
                    1,
                    min(
                        int(self.max_items * 0.1),
                        len(self._store) - self.max_items + 1
                    )
                )
                
                for i in range(items_to_remove):
                    del self._store[items[i][0]]
                    
                logger.debug(
                    f"Removed {items_to_remove} items from {self.name} due to capacity limit"
                )
            
            # Store the new item
            self._store[(context, key)] = item
            logger.debug(f"Stored item with key '{key}' in context '{context}'")
    
    async def _retrieve_impl(
        self, 
        key: str, 
        context: str,
        **kwargs
    ) -> Any:
        """Retrieve a value from working memory.
        
        Args:
            key: Key to retrieve
            context: Context path
            **kwargs: Additional retrieval parameters
            
        Returns:
            Retrieved value or None if not found
        """
        await self._cleanup_expired()
        
        # First check for exact match
        item = self._store.get((context, key))
        if item and not item.is_expired:
            logger.debug(f"Retrieved item with key '{key}' from context '{context}'")
            return item.value
            
        # If not in exact context, check parent and child contexts
        if context != "agent":  # Skip this if already searching in root context
            for (ctx, k), item in self._store.items():
                if k == key and not item.is_expired:
                    # Check for parent/child relationship between contexts
                    if (ctx.startswith(f"{context}/") or  # Context is parent of item
                        context.startswith(f"{ctx}/")):   # Item is parent of context
                        logger.debug(f"Retrieved item with key '{key}' from related context '{ctx}'")
                        return item.value
            
        logger.debug(f"Item with key '{key}' not found in context '{context}' or related contexts")
        return None
    
    async def _search_impl(
        self, 
        query: str, 
        context: str,
        limit: int = 10,
        min_score: float = 0.0,
        **kwargs
    ) -> List[MemorySearchResult]:
        """Search for relevant memories in working memory.
        
        Args:
            query: Search query
            context: Context path
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0 to 1.0)
            **kwargs: Additional search parameters
            
        Returns:
            List of matching memory items
        """
        await self._cleanup_expired()
        
        # Simple string matching for working memory
        results = []
        query_lower = query.lower()
        
        logger.debug(f"Searching for '{query}' in context '{context}' (total items: {len(self._store)})")
        
        for (ctx, key), item in self._store.items():
            # Skip expired items
            if item.is_expired:
                continue
                
            # Check if context matches or is a parent/child relationship
            if context != "agent":  # When searching from "agent", include all contexts
                # Context matching rules:
                # 1. Exact match
                # 2. Context is parent of item (item_ctx starts with context/)
                # 3. Item is parent of context (context starts with item_ctx/)
                is_match = (
                    ctx == context or 
                    ctx.startswith(f"{context}/") or 
                    context.startswith(f"{ctx}/")
                )
                
                if not is_match:
                    logger.debug(f"Skipping item with key '{key}' in context '{ctx}' - not related to search context '{context}'")
                    continue
                
                logger.debug(f"Including item with key '{key}' in context '{ctx}' for search in context '{context}'")
            
            # Calculate a simple match score based on string content
            score = 0.0
            match_found = False
            
            # Check key match
            if query_lower in key.lower():
                score = max(score, 0.8)  # Key matches are highly relevant
                match_found = True
                logger.debug(f"Key match found for '{key}' (score: {score})")
                
            # Check value match if it can be converted to string
            if hasattr(item.value, '__str__'):
                value_str = str(item.value).lower()
                
                if query_lower in value_str:
                    # Simple scoring based on length of match relative to content
                    content_score = min(0.9, 0.5 + len(query_lower) / len(value_str))
                    score = max(score, content_score)
                    match_found = True
                    logger.debug(f"Value match found for '{key}' in value '{value_str}' (score: {score})")
                    
            # Add to results if we have a match with sufficient score
            if match_found and score >= min_score:
                results.append(MemorySearchResult(
                    item=item,
                    score=score,
                    metadata={"match_type": "string_match"}
                ))
                logger.debug(f"Added item with key '{key}' to search results with score {score}")
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: (x.score, x.item.importance), reverse=True)
        if len(results) > limit:
            results = results[:limit]
        
        logger.debug(f"Found {len(results)} results for query '{query}' in context '{context}'")
        return results
    
    async def _wipe_context_impl(self, context: str) -> None:
        """Wipe all items in a specific context.
        
        Args:
            context: Context path to wipe
        """
        keys_to_remove = []
        
        async with self._lock:
            # Find all keys in this context or subcontexts
            for (ctx, key) in self._store.keys():
                if ctx == context or ctx.startswith(f"{context}/"):
                    keys_to_remove.append((ctx, key))
                    
            # Remove the keys
            for key in keys_to_remove:
                del self._store[key]
                
        logger.debug(f"Wiped {len(keys_to_remove)} items from context '{context}'")
    
    async def _wipe_all_impl(self) -> None:
        """Wipe all memory contents."""
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            
        logger.debug(f"Wiped all memory contents ({count} items)") 