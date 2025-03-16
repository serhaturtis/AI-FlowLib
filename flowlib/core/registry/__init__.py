"""Registry system for flowlib.

This package provides a centralized registry system for managing resources
and providers with clean interfaces, clear separation of concerns, and
enhanced type safety.
"""

from .base import BaseRegistry
from .constants import ResourceType, ProviderType
from .resource_registry import ResourceRegistry
from .provider_registry import ProviderRegistry
from .decorators import (
    resource, model, prompt, config,
    provider, llm_provider, db_provider, vector_db_provider,
    cache_provider, storage_provider, message_queue_provider
)

# Create registry instances
resource_registry = ResourceRegistry()
provider_registry = ProviderRegistry()

# Initialize the decorators with registry references
import sys
from . import decorators
decorators.resource_registry = resource_registry
decorators.provider_registry = provider_registry

# Simplify importing for common types
from .constants import ResourceType, ProviderType

__all__ = [
    # Registry classes
    "BaseRegistry",
    "ResourceRegistry",
    "ProviderRegistry",
    
    # Registry instances
    "resource_registry",
    "provider_registry",
    
    # Resource type constants
    "ResourceType",
    "ProviderType",
    
    # Decorators
    "resource",
    "model",
    "prompt",
    "config",
    "provider",
    "llm_provider",
    "db_provider",
    "vector_db_provider",
    "cache_provider",
    "storage_provider",
    "message_queue_provider"
] 