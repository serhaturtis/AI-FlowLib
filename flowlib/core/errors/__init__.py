"""Error handling for the flow framework.

This package provides a comprehensive error handling system for the flow framework,
including structured error types, error context, and error management.
"""

from .base import (
    BaseError,
    ValidationError,
    ExecutionError,
    StateError,
    ConfigurationError,
    ResourceError,
    ProviderError,
    ErrorContext,
    ErrorManager,
    default_manager,
)

__all__ = [
    "BaseError",
    "ValidationError",
    "ExecutionError",
    "StateError",
    "ConfigurationError",
    "ResourceError",
    "ProviderError",
    "ErrorContext",
    "ErrorManager",
    "default_manager",
]
