"""Provider factory for creating different types of providers.

This module provides a factory for creating different types of providers
based on the provider_type specified in the configuration.
"""

import logging
from typing import Dict, Type, Any, Optional

from ..core.errors import ProviderError, ErrorContext
from .constants import ProviderType
from .registry import provider_registry
from .base import Provider

logger = logging.getLogger(__name__)

# Registry of provider specific implementations
# This is separate from the provider_registry and helps with factory creation
PROVIDER_IMPLEMENTATIONS: Dict[str, Dict[str, Type[Provider]]] = {
    ProviderType.LLM: {
        "llamacpp": None,  # Will be populated on import
        "llama": None,     # Will be populated on import
    },
    ProviderType.DATABASE: {
        "postgres": None,
        "postgresql": None,
        "mongodb": None,
        "mongo": None,
        "sqlite": None,
        "sqlite3": None,
    },
    ProviderType.MESSAGE_QUEUE: {
        "rabbitmq": None,
        "rabbit": None,
        "kafka": None,
    },
    ProviderType.CACHE: {
        "redis": None,
        "memory": None,
        "inmemory": None,
    },
    ProviderType.VECTOR_DB: {
        "chroma": None,
        "chromadb": None,
        "pinecone": None,
        "qdrant": None,
    },
    ProviderType.STORAGE: {
        "s3": None,
        "aws": None,
        "local": None,
        "localfile": None,
        "file": None,
    },
}

def create_provider(
    provider_type: str,
    name: str,
    implementation: Optional[str] = None,
    register: bool = True,
    **kwargs
) -> Provider:
    """Create a provider based on the provider_type and optional implementation.
    
    Args:
        provider_type: Type of provider (e.g., ProviderType.LLM, ProviderType.DATABASE)
        name: Unique name for the provider instance
        implementation: Optional specific implementation (e.g., "postgres" for db)
        register: Whether to register the provider in the registry
        **kwargs: Additional arguments to pass to the provider constructor
        
    Returns:
        Provider instance of the appropriate type
        
    Raises:
        ProviderError: If the specified provider_type or implementation is not supported
    """
    # First try to get the provider from the registry
    # This will catch providers registered with decorators
    if provider_registry.contains(provider_type, name):
        logger.info(f"Provider '{name}' of type '{provider_type}' already registered, returning existing instance")
        return provider_registry.get_sync(provider_type, name)
    
    # If implementation is specified, try to get specific provider class
    provider_class = None
    
    if implementation and provider_type in PROVIDER_IMPLEMENTATIONS:
        impl_registry = PROVIDER_IMPLEMENTATIONS[provider_type]
        if implementation.lower() in impl_registry:
            provider_class = impl_registry[implementation.lower()]
            
            # Lazy import provider classes to avoid import issues
            if provider_class is None:
                provider_class = _import_provider_class(provider_type, implementation.lower())
                impl_registry[implementation.lower()] = provider_class
        
        if provider_class is None:
            supported_implementations = list(impl_registry.keys())
            raise ProviderError(
                message=f"Unsupported implementation {implementation} for provider type {provider_type}",
                context=ErrorContext.create(
                    provider_type=provider_type,
                    implementation=implementation,
                    supported=supported_implementations
                )
            )
        
    if provider_class is None:
        raise ProviderError(
            message=f"No provider found for type {provider_type}",
            context=ErrorContext.create(
                provider_type=provider_type,
                supported_types=list(PROVIDER_IMPLEMENTATIONS.keys())
            )
        )
        
    try:
        # Create provider instance with specified name
        provider = provider_class(name=name, **kwargs)
        
        # Register if requested
        if register:
            provider_registry.register(provider)
            
        return provider
    except Exception as e:
        raise ProviderError(
            message=f"Failed to create provider '{name}' of type '{provider_type}': {str(e)}",
            context=ErrorContext.create(
                provider_type=provider_type,
                implementation=implementation or "default",
                name=name
            ),
            cause=e
        )

async def create_and_initialize_provider(
    provider_type: str,
    name: str,
    implementation: Optional[str] = None,
    register: bool = True,
    **kwargs
) -> Provider:
    """Create and initialize a provider.
    
    This is a convenience function that combines create_provider with initialization.
    
    Args:
        provider_type: Type of provider
        name: Unique name for the provider
        implementation: Optional specific implementation
        register: Whether to register the provider
        **kwargs: Additional arguments for the provider
        
    Returns:
        Initialized provider instance
        
    Raises:
        ProviderError: If provider creation or initialization fails
    """
    # First check if provider already exists and is initialized
    if provider_registry.contains(provider_type, name):
        return await provider_registry.get_provider_async(provider_type, name)
    
    # Create the provider
    provider = create_provider(
        provider_type=provider_type,
        name=name,
        implementation=implementation,
        register=register,
        **kwargs
    )
    
    # Initialize and return
    try:
        await provider.initialize()
        return provider
    except Exception as e:
        raise ProviderError(
            message=f"Failed to initialize provider '{name}' of type '{provider_type}': {str(e)}",
            provider_name=name,
            context={
                "provider_type": provider_type,
                "implementation": implementation if implementation else "default"
            },
            cause=e
        )

def _import_provider_class(provider_type: str, implementation: str) -> Type[Provider]:
    """Import a provider class by type and implementation.
    
    Args:
        provider_type: Type of provider
        implementation: Specific implementation
        
    Returns:
        Provider class
        
    Raises:
        ImportError: If provider class cannot be imported
    """
    # Map of provider types and implementations to provider classes
    provider_map = {
        ProviderType.LLM: {
            "llamacpp": "LlamaCppProvider",
        },
        ProviderType.DATABASE: {
            "postgresql": "PostgreSQLProvider",
            "mongodb": "MongoDBProvider",
            "sqlite": "SQLiteProvider",
        },
        ProviderType.VECTOR_DB: {
            "chromadb": "ChromaDBProvider",
            "pinecone": "PineconeProvider",
            "qdrant": "QdrantProvider",
        },
    }
    
    # Map of provider types to modules
    module_map = {
        ProviderType.LLM: "flowlib.providers.llm",
        ProviderType.DATABASE: "flowlib.providers.db",
        ProviderType.VECTOR_DB: "flowlib.providers.vector",
        ProviderType.MESSAGE_QUEUE: "flowlib.providers.mq",
        ProviderType.CACHE: "flowlib.providers.cache",
        ProviderType.STORAGE: "flowlib.providers.storage",
    }
    
    # Check if we have a mapping for this provider type and implementation
    if provider_type not in provider_map or implementation not in provider_map[provider_type]:
        raise ImportError(f"No provider mapping for {provider_type}/{implementation}")
    
    # Get the module path and class name
    module_path = module_map[provider_type]
    class_name = provider_map[provider_type][implementation]
    
    try:
        # Use direct imports for each provider type/implementation
        if provider_type == ProviderType.LLM:
            if implementation == "llamacpp":
                from ..providers.llm.llama_cpp_provider import LlamaCppProvider
                return LlamaCppProvider
        elif provider_type == ProviderType.DATABASE:
            if implementation == "postgresql":
                from ..providers.db.postgres_provider import PostgreSQLProvider
                return PostgreSQLProvider
            elif implementation == "mongodb":
                from ..providers.db.mongodb_provider import MongoDBProvider
                return MongoDBProvider
            elif implementation == "sqlite":
                from ..providers.db.sqlite_provider import SQLiteProvider
                return SQLiteProvider
        elif provider_type == ProviderType.VECTOR_DB:
            if implementation == "chromadb":
                from ..providers.vector.chroma_provider import ChromaDBProvider
                return ChromaDBProvider
            elif implementation == "pinecone":
                from ..providers.vector.pinecone_provider import PineconeProvider
                return PineconeProvider
            elif implementation == "qdrant":
                from ..providers.vector.qdrant_provider import QdrantProvider
                return QdrantProvider
        
        # If we get here, something went wrong with our mappings
        raise ImportError(f"Provider import failed for {provider_type}/{implementation}")
    except ImportError as e:
        logger.error(f"Failed to import provider class for {provider_type}/{implementation}: {str(e)}")
        raise

def _import_provider_type(provider_type: str) -> Type[Provider]:
    """Dynamically import a base provider class by type.
    
    Args:
        provider_type: Type of provider
        
    Returns:
        Provider class
        
    Raises:
        ImportError: If provider class cannot be imported
    """
    # Map of provider types to import paths
    import_map = {
        ProviderType.LLM: "from ..providers.llm.base import LLMProvider; return LLMProvider",
        ProviderType.DATABASE: "from ..providers.db.base import DBProvider; return DBProvider",
        ProviderType.MESSAGE_QUEUE: "from ..providers.mq.base import MQProvider; return MQProvider",
        ProviderType.CACHE: "from ..providers.cache.base import CacheProvider; return CacheProvider",
        ProviderType.VECTOR_DB: "from ..providers.vector.base import VectorDBProvider; return VectorDBProvider",
        ProviderType.STORAGE: "from ..providers.storage.base import StorageProvider; return StorageProvider",
    }
    
    # Get import statement
    if provider_type in import_map:
        import_statement = import_map[provider_type]
        try:
            # Execute import statement
            local_vars = {}
            exec(import_statement, globals(), local_vars)
            return local_vars["return"]
        except ImportError as e:
            logger.error(f"Failed to import provider class for {provider_type}: {str(e)}")
            raise
    
    raise ImportError(f"No import mapping for {provider_type}") 