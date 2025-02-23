"""Error handling system for the framework."""

from .base import (
    BaseError,
    ValidationError,
    ResourceError,
    ErrorContext,
)
from .handlers import (
    ErrorHandler,
    RetryHandler, 
    FallbackHandler,
    with_error_handling
)
from .manager import ErrorManager

__all__ = [
    # Base errors
    'BaseError',
    'ValidationError', 
    'ResourceError',
    
    # Error context
    'ErrorContext',
    
    # Error handlers
    'ErrorHandler',
    'RetryHandler',
    'FallbackHandler',
    'with_error_handling',
    
    # Error manager
    'ErrorManager'
] 