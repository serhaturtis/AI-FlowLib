"""Error context utilities."""

from typing import Any
from datetime import datetime

from ...core.errors.base import BaseError, ErrorContext

def create_error_context(
    flow_name: str,
    **details: Any
) -> ErrorContext:
    """Create error context with standard fields.
    
    Args:
        flow_name: Name of flow
        **details: Additional context details
        
    Returns:
        Error context instance
    """
    return ErrorContext.create(
        flow_name=flow_name,
        timestamp=datetime.now(),
        **details
    )

def chain_error(
    original: BaseError,
    new_error: Exception,
    flow_name: str,
    **context_details: Any
) -> BaseError:
    """Chain a new error to an existing one.
    
    Args:
        original: Original error
        new_error: New error to chain
        flow_name: Name of flow
        **context_details: Additional context details
        
    Returns:
        Updated error chain
    """
    if not isinstance(new_error, BaseError):
        from .handling import ErrorHandling
        new_error = ErrorHandling.wrap_error(
            new_error,
            flow_name,
            **context_details
        )
    
    return original.chain(new_error) 