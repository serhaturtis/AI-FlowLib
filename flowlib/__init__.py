"""Flowlib Framework.

This package provides a unified, type-safe framework for building and executing flows,
using providers, and creating agents with clean error handling and validation.

Key features:
1. Declarative resource management with decorators for models, providers, and prompts
2. Lazy initialization of LLM providers with automatic model loading
3. Type-safe flow execution with validation
4. Structured error handling across all components
"""

from .core.context import Context

from .flows.base import FlowStatus, FlowSettings, Flow
from .flows.composite import CompositeFlow
from .flows.builder import FlowBuilder
from .flows.stage import Stage
from .flows.standalone import StandaloneStage
from .flows.stage import Stage
from .flows.decorators import flow, stage, standalone, pipeline, stage_registry

from .flows.results import FlowResult

from .core.errors import BaseError, ValidationError, ExecutionError, ResourceError

from .resources.registry import resource_registry
from .providers.registry import provider_registry

from .resources.constants import ResourceType
from .providers.constants import ProviderType

from .resources.decorators import resource, model, prompt, config
from .providers import Provider, create_provider
from .providers.decorators import provider, llm_provider, db_provider, vector_db_provider, cache_provider, storage_provider, message_queue_provider


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
