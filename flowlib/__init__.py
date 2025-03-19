"""Consolidated Flowlib Framework with enhanced usability and type safety.

This package provides a unified, type-safe framework for building and executing flows,
using providers, and creating agents with clean error handling and validation.

Key features:
1. Declarative resource management with decorators for models, providers, and prompts
2. Lazy initialization of LLM providers with automatic model loading
3. Type-safe flow execution with validation
4. Structured error handling across all components
"""

# Core models
from .core.models.result import FlowResult, FlowStatus
from .core.models.context import Context
from .core.models.settings import FlowSettings

# Error handling
from .core.errors import BaseError, ValidationError, ExecutionError, ResourceError

# Registry system
from .core.registry import (
    resource_registry, provider_registry,
    ResourceType, ProviderType,
    resource, model, prompt, config,
    provider, llm_provider, db_provider, vector_db_provider,
    cache_provider, storage_provider, message_queue_provider
)

# Flows
from .flows import Flow, Stage, CompositeFlow, FlowBuilder, StandaloneStage
from .flows import flow, stage, standalone, pipeline, stage_registry

# Providers
from .providers import Provider, create_provider


__version__ = "0.1.0"

__all__ = [
    # Core models
    "FlowResult",
    "FlowStatus",
    "Context",
    "FlowSettings",
    
    # Errors
    "BaseError",
    "ValidationError",
    "ExecutionError",
    "ResourceError",
    
    # Registry
    "resource_registry",
    "provider_registry",
    
    # Resource type constants
    "ResourceType",
    "ProviderType",
    
    # Resource Decorators
    "resource",
    "model",
    "prompt",
    "config",
    
    # Provider Decorators
    "provider",
    "llm_provider",
    "db_provider", 
    "vector_db_provider",
    "cache_provider", 
    "storage_provider",
    "message_queue_provider",
    
    # Flows
    "Flow",
    "Stage",
    "CompositeFlow",
    "FlowBuilder",
    "StandaloneStage",
    "flow",
    "stage",
    "standalone",
    "pipeline",
    "stage_registry",
    
    # Providers
    "Provider",
    "create_provider",
    
    # Version
    "__version__"
]
