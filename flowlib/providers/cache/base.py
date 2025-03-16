"""Cache provider base class and related functionality.

This module provides the base class for implementing cache providers
that share common functionality for storing and retrieving cached data.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
import asyncio
from datetime import timedelta
from pydantic import BaseModel, Field

from ...core.errors import ProviderError, ErrorContext
from ...core.models.settings import ProviderSettings
from ..base import AsyncProvider

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class CacheProviderSettings(ProviderSettings):
    """Base settings for cache providers.
    
    Attributes:
        host: Cache server host
        port: Cache server port
        username: Authentication username
        password: Authentication password
        database: Database/namespace index
        key_prefix: Prefix for all cache keys
        default_ttl: Default time-to-live for cache entries
    """
    
    # Connection settings (for distributed caches)
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Cache behavior settings
    default_ttl: int = 300  # 5 minutes
    max_size: Optional[int] = None
    eviction_policy: str = "lru"  # lru, lfu, fifo, random
    use_compression: bool = False
    serialize_method: str = "json"
    namespace: str = "default"
    
    # Performance settings
    pool_size: int = 5
    timeout: float = 5.0
    retry_on_timeout: bool = True
    max_retries: int = 3


class CacheProvider(AsyncProvider, Generic[T]):
    """Base class for cache providers.
    
    This class provides:
    1. Common caching operations (get, set, delete)
    2. Type-safe operations with Pydantic models
    3. TTL and eviction policy support
    4. Connection management for distributed caches
    """
    
    def __init__(self, name: str = "cache", settings: Optional[CacheProviderSettings] = None):
        """Initialize cache provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Pass provider_type="cache" to the parent class
        super().__init__(name=name, settings=settings, provider_type="cache")
        self._initialized = False
        self._connection = None
        self._settings = settings or CacheProviderSettings()
        
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
        
    async def initialize(self):
        """Initialize the cache provider.
        
        This method should be implemented by subclasses to establish
        connections to distributed caches or initialize in-memory storage.
        """
        self._initialized = True
        
    async def shutdown(self):
        """Close all connections and release resources.
        
        This method should be implemented by subclasses to properly
        close connections and clean up resources.
        """
        self._initialized = False
        self._connection = None
        
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            ProviderError: If retrieval fails
        """
        raise NotImplementedError("Subclasses must implement get()")
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            True if value was cached successfully
            
        Raises:
            ProviderError: If caching fails
        """
        raise NotImplementedError("Subclasses must implement set()")
        
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was deleted or didn't exist
            
        Raises:
            ProviderError: If deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete()")
        
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
            
        Raises:
            ProviderError: If check fails
        """
        raise NotImplementedError("Subclasses must implement exists()")
        
    async def ttl(self, key: str) -> Optional[int]:
        """Get the remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
            
        Raises:
            ProviderError: If TTL check fails
        """
        raise NotImplementedError("Subclasses must implement ttl()")
        
    async def clear(self) -> bool:
        """Clear all values from the cache.
        
        Returns:
            True if cache was cleared successfully
            
        Raises:
            ProviderError: If clearing fails
        """
        raise NotImplementedError("Subclasses must implement clear()")
        
    async def get_structured(self, key: str, output_type: Type[T]) -> Optional[T]:
        """Get a structured value from the cache.
        
        Args:
            key: Cache key
            output_type: Pydantic model to parse the value into
            
        Returns:
            Parsed model instance or None if not found
            
        Raises:
            ProviderError: If retrieval or parsing fails
        """
        try:
            # Get the value from cache
            value = await self.get(key)
            
            # Return None if not found
            if value is None:
                return None
                
            # Parse the value into the model
            return output_type.parse_obj(value)
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to get structured value: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    key=key,
                    output_type=output_type.__name__
                ),
                cause=e
            )
            
    async def set_structured(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> bool:
        """Set a structured value in the cache.
        
        Args:
            key: Cache key
            value: Pydantic model instance to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            True if value was cached successfully
            
        Raises:
            ProviderError: If caching fails
        """
        try:
            # Convert model to dict
            value_dict = value.dict()
            
            # Set the value in cache
            return await self.set(key, value_dict, ttl)
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to set structured value: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    key=key,
                    model_type=type(value).__name__
                ),
                cause=e
            )
            
    async def check_connection(self) -> bool:
        """Check if cache connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_connection()")
        
    def make_namespaced_key(self, key: str) -> str:
        """Create a namespaced key to avoid collisions.
        
        Args:
            key: Original key
            
        Returns:
            Namespaced key
        """
        if not self._settings.namespace:
            return key
        return f"{self._settings.namespace}:{key}" 