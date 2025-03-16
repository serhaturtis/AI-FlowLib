"""Redis-based implementation of the CacheProvider.

This module provides a concrete implementation of the CacheProvider
using Redis as the backend for distributed caching.
"""

import logging
import json
import pickle
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
import asyncio

logger = logging.getLogger(__name__)

# For type annotations only
ConnectionPool = Any

try:
    import redis.asyncio as redis
    from redis.asyncio.connection import ConnectionPool
    from redis.exceptions import RedisError
except ImportError:
    logger.warning("redis package not found. Install with 'pip install redis'")
    # Define a placeholder for RedisError for type checking
    class RedisError(Exception):
        pass

from ...core.errors import ProviderError, ErrorContext
from ...core.models.settings import ProviderSettings
from ...core.registry.decorators import provider
from ...core.registry.constants import ProviderType
from .base import CacheProvider, CacheProviderSettings


class RedisCacheProviderSettings(CacheProviderSettings):
    """Redis-specific cache provider settings.
    
    Extends the base CacheProviderSettings with Redis-specific options.
    
    Attributes:
        db: Redis database number
        socket_timeout: Socket timeout in seconds
        socket_connect_timeout: Socket connection timeout in seconds
        socket_keepalive: Whether to use socket keepalive
        socket_keepalive_options: Socket keepalive options
        encoding: Redis response encoding
        encoding_errors: How to handle encoding errors
        decode_responses: Whether to decode responses to strings
        sentinel_kwargs: Additional options for Redis Sentinel
        sentinel: List of Redis Sentinel nodes
        sentinel_master: Name of the Redis Sentinel master
    """
    
    # Redis-specific connection settings
    db: int = 0
    socket_timeout: Optional[float] = None
    socket_connect_timeout: Optional[float] = None
    socket_keepalive: bool = False
    socket_keepalive_options: Optional[Dict[int, Union[int, bytes]]] = None
    
    # Encoding settings
    encoding: str = "utf-8"
    encoding_errors: str = "strict"
    decode_responses: bool = False
    
    # Sentinel settings
    sentinel_kwargs: Dict[str, Any] = {}
    sentinel: Optional[List[str]] = None
    sentinel_master: Optional[str] = None


@provider(provider_type=ProviderType.CACHE, name="redis-cache")
class RedisCacheProvider(CacheProvider):
    """Redis implementation of the CacheProvider.
    
    This provider uses Redis for caching, supporting all standard
    cache operations with TTL support, atomic operations, and
    distributed locking mechanisms.
    """
    
    def __init__(self, name: str = "redis_cache", settings: Optional[RedisCacheProviderSettings] = None):
        """Initialize Redis cache provider.
        
        Args:
            name: Unique provider name
            settings: Optional Redis-specific provider settings
        """
        # Use Redis-specific settings or create default
        redis_settings = settings or RedisCacheProviderSettings()
        super().__init__(name=name, settings=redis_settings)
        self._redis_settings = redis_settings
        self._pool = None
        self._redis = None
        
    async def initialize(self):
        """Initialize the Redis connection pool.
        
        Raises:
            ProviderError: If Redis connection cannot be established
        """
        try:
            # Create a connection pool
            self._pool = ConnectionPool(
                host=self._redis_settings.host or "localhost",
                port=self._redis_settings.port or 6379,
                db=self._redis_settings.db,
                username=self._redis_settings.username,
                password=self._redis_settings.password,
                socket_timeout=self._redis_settings.socket_timeout,
                socket_connect_timeout=self._redis_settings.socket_connect_timeout,
                socket_keepalive=self._redis_settings.socket_keepalive,
                socket_keepalive_options=self._redis_settings.socket_keepalive_options,
                encoding=self._redis_settings.encoding,
                encoding_errors=self._redis_settings.encoding_errors,
                decode_responses=self._redis_settings.decode_responses,
                max_connections=self._redis_settings.pool_size,
            )
            
            # Create Redis client
            self._redis = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            if not await self.check_connection():
                raise ProviderError(
                    message="Failed to connect to Redis server",
                    provider_name=self.name,
                    context=ErrorContext.create(
                        host=self._redis_settings.host,
                        port=self._redis_settings.port
                    )
                )
                
            # Mark as initialized
            await super().initialize()
            logger.info(f"Redis cache provider '{self.name}' initialized successfully")
            
        except RedisError as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis initialization error: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    host=self._redis_settings.host,
                    port=self._redis_settings.port
                ),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to initialize Redis provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def shutdown(self):
        """Close Redis connections and release resources."""
        # Close pool
        if self._pool:
            self._pool.disconnect()
            self._pool = None
            
        # Clear client
        self._redis = None
        
        # Mark as not initialized
        await super().shutdown()
        logger.info(f"Redis cache provider '{self.name}' shut down")
        
    async def check_connection(self) -> bool:
        """Check if Redis connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        try:
            # Simple PING command
            return await self._redis.ping()
        except Exception:
            return False
            
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            ProviderError: If retrieval fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)
            
            # Get value from Redis
            value = await self._redis.get(ns_key)
            
            # Return None if not found
            if value is None:
                return None
                
            # Deserialize value based on serialization method
            if self._redis_settings.serialize_method == "json":
                if isinstance(value, bytes):
                    value = value.decode(self._redis_settings.encoding)
                return json.loads(value)
            elif self._redis_settings.serialize_method == "pickle":
                return pickle.loads(value)
            else:
                return value
                
        except RedisError as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis get error: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to get value from Redis: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            True if value was cached successfully
            
        Raises:
            ProviderError: If caching fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)
            
            # Set TTL to default if not specified
            if ttl is None:
                ttl = self._redis_settings.default_ttl
                
            # Serialize value based on serialization method
            if self._redis_settings.serialize_method == "json":
                serialized = json.dumps(value)
            elif self._redis_settings.serialize_method == "pickle":
                serialized = pickle.dumps(value)
            else:
                serialized = value
                
            # Set value in Redis with TTL
            result = await self._redis.set(ns_key, serialized, ex=ttl)
            return result is True
            
        except RedisError as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis set error: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to set value in Redis: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
            
    async def delete(self, key: str) -> bool:
        """Delete a value from Redis.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was deleted
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)
            
            # Delete value from Redis
            result = await self._redis.delete(ns_key)
            return result > 0
            
        except RedisError as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis delete error: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to delete value from Redis: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
            
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
            
        Raises:
            ProviderError: If check fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)
            
            # Check if key exists in Redis
            result = await self._redis.exists(ns_key)
            return result > 0
            
        except RedisError as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis exists error: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to check key existence in Redis: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
            
    async def ttl(self, key: str) -> Optional[int]:
        """Get the remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
            
        Raises:
            ProviderError: If TTL check fails
        """
        try:
            # Create namespaced key
            ns_key = self.make_namespaced_key(key)
            
            # Get TTL from Redis
            ttl = await self._redis.ttl(ns_key)
            
            # Redis returns -2 if key doesn't exist, -1 if no TTL set
            if ttl == -2:
                return None
            elif ttl == -1:
                return -1  # No expiration
            else:
                return ttl
                
        except RedisError as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis TTL error: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to get TTL from Redis: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(key=key),
                cause=e
            )
            
    async def clear(self) -> bool:
        """Clear all values from the cache with the current namespace.
        
        Returns:
            True if cache was cleared successfully
            
        Raises:
            ProviderError: If clearing fails
        """
        try:
            # If a namespace is set, only clear keys in that namespace
            if self._redis_settings.namespace:
                pattern = f"{self._redis_settings.namespace}:*"
                # Get all keys matching the pattern
                cursor = b'0'
                deleted_count = 0
                
                while cursor:
                    cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                    
                    if keys:
                        # Delete keys in batches
                        result = await self._redis.delete(*keys)
                        deleted_count += result
                        
                    # Exit when no more keys
                    if cursor == b'0':
                        break
                        
                return True
            else:
                # Clear all keys (dangerous!)
                await self._redis.flushdb()
                return True
                
        except RedisError as e:
            # Wrap Redis errors
            raise ProviderError(
                message=f"Redis clear error: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(namespace=self._redis_settings.namespace),
                cause=e
            )
        except Exception as e:
            # Re-raise other errors
            raise ProviderError(
                message=f"Failed to clear cache in Redis: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(namespace=self._redis_settings.namespace),
                cause=e
            ) 