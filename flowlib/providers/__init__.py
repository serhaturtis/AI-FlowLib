"""Provider package for interacting with various services.

This package contains providers for various services, organized by type:
- LLM providers (llm): Large Language Model providers
- Database providers (db): SQL and NoSQL database providers
- Message queue providers (mq): Message queue and event bus providers
- Cache providers (cache): In-memory and distributed caching providers
- Vector database providers (vector): Vector embedding storage and search
- Storage providers (storage): Object storage and file management
- API providers (api): External API integration providers
"""

# Core imports
from .base import Provider, AsyncProvider
from .factory import create_provider

# Import from the registry system
from ..core.registry import provider_registry
from ..core.registry import ProviderType
from ..core.registry import provider, llm_provider, db_provider, vector_db_provider
from ..core.registry import cache_provider, storage_provider, message_queue_provider

# Re-export common provider types
from .llm import LLMProvider
from .llm.llama_cpp_provider import LlamaCppProvider
from .db import DBProvider, DBProviderSettings, PostgreSQLProvider, PostgreSQLProviderSettings
from .db.mongodb_provider import MongoDBProvider, MongoDBProviderSettings
from .db.sqlite_provider import SQLiteProvider, SQLiteProviderSettings
from .mq import MQProvider, MQProviderSettings, MessageMetadata
from .mq.rabbitmq_provider import RabbitMQProvider, RabbitMQProviderSettings
from .mq.kafka_provider import KafkaProvider, KafkaProviderSettings
from .cache import CacheProvider, CacheProviderSettings
from .cache.redis_provider import RedisCacheProvider, RedisCacheProviderSettings
from .cache.memory_provider import MemoryCacheProvider, InMemoryCacheProviderSettings
from .vector import VectorDBProvider, VectorDBProviderSettings, VectorMetadata, SimilaritySearchResult, ChromaDBProvider, ChromaDBProviderSettings
from .vector.pinecone_provider import PineconeProvider, PineconeProviderSettings
from .vector.qdrant_provider import QdrantProvider, QdrantProviderSettings
from .storage import StorageProvider, StorageProviderSettings, FileMetadata
from .storage.local_provider import LocalStorageProvider, LocalStorageProviderSettings
from .storage.s3_provider import S3Provider, S3ProviderSettings

__all__ = [
    # Base classes
    "Provider",
    "AsyncProvider",
    
    # Factory function
    "create_provider",
    
    # Registry access
    "provider_registry",
    
    # Provider type constants
    "ProviderType",
    
    # Registration decorators
    "provider",
    "llm_provider",
    "db_provider", 
    "vector_db_provider",
    "cache_provider",
    "storage_provider",
    "message_queue_provider",
    
    # LLM providers
    "LLMProvider",
    "LlamaCppProvider",
    
    # Database providers
    "DBProvider",
    "DBProviderSettings",
    "PostgreSQLProvider", 
    "PostgreSQLProviderSettings",
    "MongoDBProvider",
    "MongoDBProviderSettings",
    "SQLiteProvider",
    "SQLiteProviderSettings",
    
    # Message queue providers
    "MQProvider",
    "MQProviderSettings",
    "MessageMetadata",
    "RabbitMQProvider",
    "RabbitMQProviderSettings",
    "KafkaProvider",
    "KafkaProviderSettings",
    
    # Cache providers
    "CacheProvider",
    "CacheProviderSettings",
    "RedisCacheProvider",
    "RedisCacheProviderSettings",
    "MemoryCacheProvider",
    "InMemoryCacheProviderSettings",
    
    # Vector database providers
    "VectorDBProvider",
    "VectorDBProviderSettings",
    "VectorMetadata",
    "SimilaritySearchResult",
    "ChromaDBProvider",
    "ChromaDBProviderSettings",
    "PineconeProvider",
    "PineconeProviderSettings",
    "QdrantProvider",
    "QdrantProviderSettings",
    
    # Storage providers
    "StorageProvider",
    "StorageProviderSettings",
    "FileMetadata",
    "S3Provider",
    "S3ProviderSettings",
    "LocalStorageProvider",
    "LocalStorageProviderSettings"
]
