"""Base error classes with enhanced error context and management.

This module provides the foundation for the error handling system,
including structured error types, error context, and error management.
"""

import asyncio
import contextlib
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable, Set

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ErrorContext:
    """Enhanced error context with structured information.
    
    This class provides:
    1. Structured context information for errors
    2. Clean serialization for logging and reporting
    3. Immutable context with builder pattern
    """
    
    def __init__(self, context_data: Dict[str, Any] = None):
        """Initialize error context.
        
        Args:
            context_data: Optional initial context data
        """
        self._data = context_data or {}
        self._timestamp = datetime.now()
    
    @classmethod
    def create(cls, **kwargs) -> 'ErrorContext':
        """Create a new error context with the given data.
        
        Args:
            **kwargs: Context data
            
        Returns:
            New ErrorContext instance
        """
        return cls(kwargs)
    
    def add(self, **kwargs) -> 'ErrorContext':
        """Create a new context with additional data.
        
        Args:
            **kwargs: Additional context data
            
        Returns:
            New ErrorContext instance with combined data
        """
        new_data = dict(self._data)
        new_data.update(kwargs)
        return ErrorContext(new_data)
    
    def add_data(self, data: Any) -> 'ErrorContext':
        """Create a new context with additional data object.
        
        This method is useful when adding non-dictionary data to the context.
        
        Args:
            data: An object to add to the context
            
        Returns:
            New ErrorContext instance with added data
        """
        # If data is a mapping-like object, extract items
        if hasattr(data, 'items') and callable(data.items):
            return self.add(**{k: v for k, v in data.items()})
        
        # For Pydantic models, convert to dict
        if hasattr(data, 'dict') and callable(data.dict):
            return self.add(data=data.dict())
            
        # For other objects, store as 'data'
        return self.add(data=data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context data.
        
        Args:
            key: Key to look up
            default: Default value if key not found
            
        Returns:
            Value associated with key or default
        """
        return self._data.get(key, default)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the context data dictionary."""
        return dict(self._data)
    
    @property
    def timestamp(self) -> datetime:
        """Get the context creation timestamp."""
        return self._timestamp
    
    def __str__(self) -> str:
        """String representation."""
        return f"ErrorContext({self._data})"


class BaseError(Exception):
    """Base class for all framework errors with enhanced context.
    
    This class provides:
    1. Structured error information with context
    2. Clean serialization for logging and reporting
    3. Cause tracking for nested errors
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize error.
        
        Args:
            message: Error message
            context: Optional error context
            cause: Optional cause exception
        """
        self.message = message
        self.context = context or ErrorContext()
        self.cause = cause
        self.timestamp = datetime.now()
        self.traceback = self._capture_traceback()
        self.result = None  # Can be set by error handlers
        
        # Initialize with message
        super().__init__(message)
    
    def _capture_traceback(self) -> str:
        """Capture the current traceback."""
        return traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary.
        
        Returns:
            Dictionary representation of error
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.data,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback
        }
    
    def __str__(self) -> str:
        """String representation."""
        cause_str = f" (caused by: {self.cause})" if self.cause else ""
        return f"{self.__class__.__name__}: {self.message}{cause_str}"


class ValidationError(BaseError):
    """Error raised when validation fails.
    
    This class provides:
    1. Structured validation error information
    2. Clean access to validation error details
    """
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            validation_errors: Optional list of validation error details
            context: Optional error context
            cause: Optional cause exception
        """
        self.validation_errors = validation_errors or []
        
        # Add validation errors to context
        if context:
            context = context.add(validation_errors=self.validation_errors)
        else:
            context = ErrorContext.create(validation_errors=self.validation_errors)
        
        super().__init__(message, context, cause)
    
    def __str__(self) -> str:
        """String representation."""
        base_str = super().__str__()
        if self.validation_errors:
            errors_str = "; ".join(
                f"{e.get('location', 'unknown')}: {e.get('message', 'unknown')}"
                for e in self.validation_errors[:3]
            )
            if len(self.validation_errors) > 3:
                errors_str += f" (and {len(self.validation_errors) - 3} more)"
            return f"{base_str} - {errors_str}"
        return base_str


class ExecutionError(BaseError):
    """Error raised when flow execution fails.
    
    This class provides:
    1. Structured execution error information
    2. Clean access to execution context
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize execution error.
        
        Args:
            message: Error message
            context: Optional error context
            cause: Optional cause exception
        """
        super().__init__(message, context, cause)


class StateError(BaseError):
    """Error raised when state operations fail.
    
    This class provides:
    1. Structured state error information
    2. Clean access to state context
    """
    
    def __init__(
        self,
        message: str,
        state_name: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize state error.
        
        Args:
            message: Error message
            state_name: Optional state name
            context: Optional error context
            cause: Optional cause exception
        """
        # Add state info to context
        if context and state_name:
            context = context.add(state_name=state_name)
        elif state_name:
            context = ErrorContext.create(state_name=state_name)
        
        super().__init__(message, context, cause)


class ConfigurationError(BaseError):
    """Error raised when configuration is invalid.
    
    This class provides:
    1. Structured configuration error information
    2. Clean access to configuration context
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Optional configuration key
            context: Optional error context
            cause: Optional cause exception
        """
        # Add configuration info to context
        if context and config_key:
            context = context.add(config_key=config_key)
        elif config_key:
            context = ErrorContext.create(config_key=config_key)
        
        super().__init__(message, context, cause)


class ResourceError(BaseError):
    """Error raised when resource operations fail.
    
    This class provides:
    1. Structured resource error information
    2. Clean access to resource context
    """
    
    def __init__(
        self,
        message: str,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize resource error.
        
        Args:
            message: Error message
            resource_id: Optional resource ID
            resource_type: Optional resource type
            context: Optional error context
            cause: Optional cause exception
        """
        # Add resource info to context
        if context:
            if resource_id:
                context = context.add(resource_id=resource_id)
            if resource_type:
                context = context.add(resource_type=resource_type)
        else:
            context_data = {}
            if resource_id:
                context_data["resource_id"] = resource_id
            if resource_type:
                context_data["resource_type"] = resource_type
            context = ErrorContext(context_data)
        
        super().__init__(message, context, cause)


class ProviderError(BaseError):
    """Error raised when provider operations fail.
    
    This class provides:
    1. Structured provider error information
    2. Clean access to provider context
    """
    
    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize provider error.
        
        Args:
            message: Error message
            provider_name: Optional provider name
            context: Optional error context
            cause: Optional cause exception
        """
        # Add provider info to context
        if context and provider_name:
            context = context.add(provider_name=provider_name)
        elif provider_name:
            context = ErrorContext.create(provider_name=provider_name)
        
        super().__init__(message, context, cause)


class ErrorManager:
    """Enhanced error manager with structured error handling.
    
    This class provides:
    1. Centralized error handling and management
    2. Error boundary context manager
    3. Type-based error handler registration
    """
    
    def __init__(self):
        """Initialize error manager."""
        self._handlers: Dict[Type[BaseError], List[Callable]] = {}
        self._global_handlers: List[Callable] = []
    
    def register(self, error_type: Type[BaseError], handler: Callable) -> None:
        """Register a handler for a specific error type.
        
        Args:
            error_type: Error type to handle
            handler: Handler function or callable
        """
        if error_type not in self._handlers:
            self._handlers[error_type] = []
        self._handlers[error_type].append(handler)
        logger.debug(f"Registered handler for {error_type.__name__}")
    
    def register_global(self, handler: Callable) -> None:
        """Register a global handler for all errors.
        
        Args:
            handler: Handler function or callable
        """
        self._global_handlers.append(handler)
        logger.debug("Registered global error handler")
    
    def handle_error(self, error: BaseError, context_data: Any = None) -> None:
        """Handle an error with registered handlers.
        
        Args:
            error: Error to handle
            context_data: Optional additional context data (can be any object)
        """
        # Add context data if provided
        if context_data is not None:
            try:
                error.context = error.context.add_data(context_data)
            except Exception as e:
                logger.warning(f"Failed to add context data to error: {str(e)}")
        
        # Call type-specific handlers
        for error_type, handlers in self._handlers.items():
            if isinstance(error, error_type):
                for handler in handlers:
                    try:
                        handler(error)
                    except Exception as e:
                        logger.error(f"Error in handler {handler}: {e}")
        
        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(error)
            except Exception as e:
                logger.error(f"Error in global handler {handler}: {e}")
    
    @contextlib.contextmanager
    def error_boundary(self, context_data: Any = None):
        """Create an error boundary context manager.
        
        Args:
            context_data: Optional context data to add to errors
            
        Yields:
            None
            
        Raises:
            BaseError: Re-raises handled errors
        """
        try:
            yield
        except BaseError as e:
            self.handle_error(e, context_data)
            raise
        except Exception as e:
            # Convert to BaseError and handle
            error = ExecutionError(
                message=str(e),
                context=ErrorContext.create(),
                cause=e
            )
            self.handle_error(error, context_data)
            raise error from e
    
    @contextlib.asynccontextmanager
    async def async_error_boundary(self, context_data: Any = None):
        """Create an async error boundary context manager.
        
        Args:
            context_data: Optional context data to add to errors
            
        Yields:
            None
            
        Raises:
            BaseError: Re-raises handled errors
        """
        try:
            yield
        except BaseError as e:
            self.handle_error(e, context_data)
            raise
        except Exception as e:
            # Convert to BaseError and handle
            error = ExecutionError(
                message=str(e),
                context=ErrorContext.create(),
                cause=e
            )
            self.handle_error(error, context_data)
            raise error from e


# Create a default error manager instance
default_manager = ErrorManager()

# Register a default logging handler
def default_logging_handler(error: BaseError) -> None:
    """Default logging handler for errors.
    
    Args:
        error: Error to log
    """
    logger.error(f"{error.__class__.__name__}: {error.message}")
    if error.cause:
        logger.debug(f"Caused by: {error.cause}")
    logger.debug(f"Context: {error.context.data}")

default_manager.register_global(default_logging_handler) 