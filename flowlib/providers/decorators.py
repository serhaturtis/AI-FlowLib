import inspect

from .constants import ProviderType
from .registry import provider_registry

def provider(name: str, provider_type: str = ProviderType.LLM, **metadata):
    """Register a class as a provider factory.
    
    This decorator:
    1. Registers a provider factory in the provider registry
    2. Handles common metadata population
    3. Provides consistent naming
    
    Args:
        name: Unique name for the provider
        provider_type: Type of provider (llm, db, etc.)
        **metadata: Additional metadata about the provider
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Ensure registry is available
        if provider_registry is None:
            raise RuntimeError("Provider registry not initialized")
            
        # Create factory function
        def factory():
            # Try to find appropriate settings for this provider
            settings = None
            
            # Look for provider-specific settings class
            settings_class_name = f"{cls.__name__}Settings"
            module = inspect.getmodule(cls)
            if module and hasattr(module, settings_class_name):
                settings_class = getattr(module, settings_class_name)
                settings = settings_class()
            
            # Instantiate the class with the provider name and settings
            return cls(name=name, settings=settings)
        
        # Register the factory
        provider_registry.register_factory(
            name=name,
            factory=factory,
            provider_type=provider_type,
            **metadata
        )
        
        # Add registration info to class metadata
        cls.__provider_name__ = name
        cls.__provider_type__ = provider_type
        cls.__provider_metadata__ = metadata
        
        return cls
    
    return decorator

# Specialized provider decorators for convenience
def llm_provider(name: str, **metadata):
    """Register a class as an LLM provider factory."""
    return provider(name, ProviderType.LLM, **metadata)

def db_provider(name: str, **metadata):
    """Register a class as a database provider factory."""
    return provider(name, ProviderType.DATABASE, **metadata)

def vector_db_provider(name: str, **metadata):
    """Register a class as a vector database provider factory."""
    return provider(name, ProviderType.VECTOR_DB, **metadata)

def cache_provider(name: str, **metadata):
    """Register a class as a cache provider factory."""
    return provider(name, ProviderType.CACHE, **metadata)

def storage_provider(name: str, **metadata):
    """Register a class as a storage provider factory."""
    return provider(name, ProviderType.STORAGE, **metadata)

def message_queue_provider(name: str, **metadata):
    """Register a class as a message queue provider factory."""
    return provider(name, ProviderType.MESSAGE_QUEUE, **metadata) 