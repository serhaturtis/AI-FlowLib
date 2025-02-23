"""Core error handling utilities."""

from typing import Any, Type
from datetime import datetime

from ...core.errors.base import BaseError, ExecutionError, ErrorContext

class ErrorHandling:
    """Centralized error handling utilities."""
    
    @staticmethod
    def wrap_error(
        error: Exception,
        flow_name: str,
        error_type: Type[BaseError] = ExecutionError,
        **context_details: Any
    ) -> BaseError:
        """Wrap an exception in a framework error type.
        
        Args:
            error: Original exception
            flow_name: Name of flow where error occurred
            error_type: Type of error to create
            **context_details: Additional error context details
            
        Returns:
            Wrapped error
        """
        if isinstance(error, BaseError):
            return error
            
        context = ErrorContext.create(
            flow_name=flow_name,
            error_type=type(error).__name__,
            timestamp=datetime.now(),
            **context_details
        )
        
        return error_type(
            message=str(error),
            context=context,
            cause=error
        )
    
    @staticmethod
    def execution_error(
        message: str,
        flow_name: str,
        **context_details: Any
    ) -> ExecutionError:
        """Create a standard execution error.
        
        Args:
            message: Error message
            flow_name: Name of flow
            **context_details: Additional context details
            
        Returns:
            Execution error instance
        """
        return ExecutionError(
            message=message,
            context=ErrorContext.create(
                flow_name=flow_name,
                timestamp=datetime.now(),
                **context_details
            )
        ) 