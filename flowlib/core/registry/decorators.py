"""Registration decorators for resources and providers.

This module provides decorators that simplify the registration of
resources and providers in the registry system.
"""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

from .constants import ResourceType, ProviderType

# We'll set these after the registry module is initialized to avoid circular imports
resource_registry = None
provider_registry = None

def resource(name: str, resource_type: str = ResourceType.MODEL, **metadata):
    """Register a class or function as a resource.
    
    This decorator:
    1. Registers a class or function in the resource registry
    2. Handles common metadata population
    3. Provides consistent naming
    
    Args:
        name: Unique name for the resource
        resource_type: Type of resource (model, prompt, etc.)
        **metadata: Additional metadata about the resource
        
    Returns:
        Decorator function
    """
    def decorator(obj):
        # Ensure registry is available
        if resource_registry is None:
            raise RuntimeError("Resource registry not initialized")
            
        # Register the resource
        resource_registry.register(
            name=name,
            obj=obj,
            resource_type=resource_type,
            **metadata
        )
        
        # Add registration info to object metadata
        obj.__resource_name__ = name
        obj.__resource_type__ = resource_type
        obj.__resource_metadata__ = metadata
        
        return obj
    
    return decorator

def model(name: str, **metadata):
    """Register a class as a model resource.
    
    This decorator is a specialized version of @resource for models.
    
    Args:
        name: Unique name for the model
        **metadata: Additional metadata about the model
        
    Returns:
        Decorator function
    """
    return resource(name, ResourceType.MODEL, **metadata)

def prompt(name: str, **metadata):
    """Register a string or function as a prompt resource.
    
    This decorator is a specialized version of @resource for prompts.
    
    Args:
        name: Unique name for the prompt
        **metadata: Additional metadata about the prompt
        
    Returns:
        Decorator function
    """
    return resource(name, ResourceType.PROMPT, **metadata)

def config(name: str, **metadata):
    """Register a class as a configuration resource.
    
    This decorator is a specialized version of @resource for configs.
    
    Args:
        name: Unique name for the config
        **metadata: Additional metadata about the config
        
    Returns:
        Decorator function
    """
    return resource(name, ResourceType.CONFIG, **metadata)

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
            # Instantiate the class directly with the provider name
            return cls(name=name)
        
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