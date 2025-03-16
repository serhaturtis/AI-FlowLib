"""Core module for unified flow framework.

This package provides fundamental components for building and
executing flows, with enhanced result handling and error management.
"""

# Export core components
from .models.result import FlowResult, FlowStatus, result_from_value, error_result
from .models.context import Context
from .models.settings import FlowSettings, ProviderSettings, LLMProviderSettings, AgentSettings, create_settings

# Export error handling
from .errors import (
    BaseError,
    ErrorContext,
    ValidationError,
    ExecutionError,
    StateError,
    ConfigurationError,
    ResourceError,
    ProviderError,
    ErrorManager,
    default_manager as error_manager
)

# Export validation
from .validation import validate_data, validate_with_schema, create_dynamic_model, validate_function

# Export resource management
from .registry import ResourceRegistry, resource_registry

__all__ = [
    # Result handling
    "FlowResult",
    "FlowStatus",
    "result_from_value",
    "error_result",
    
    # Context
    "Context",
    
    # Settings
    "FlowSettings",
    "ProviderSettings",
    "LLMProviderSettings",
    "AgentSettings",
    "create_settings",
    
    # Error handling
    "BaseError",
    "ErrorContext",
    "ValidationError",
    "ExecutionError",
    "StateError", 
    "ConfigurationError",
    "ResourceError",
    "ProviderError",
    "ErrorManager",
    "error_manager",
    
    # Validation
    "validate_data",
    "validate_with_schema",
    "create_dynamic_model",
    "validate_function",
    
    # Resource management
    "ResourceRegistry",
    "resource_registry"
]
