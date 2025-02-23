# src/core/errors/base.py

from datetime import datetime
from typing import Optional, Dict, Any, List, Type, TypeVar
from dataclasses import dataclass, field
from traceback import extract_tb

T = TypeVar('T', bound='ErrorContext')

@dataclass
class ErrorContext:
    """
    Standardized error context that provides structured error information.
    
    This class maintains a standardized way to capture and manage error context
    information throughout the framework. It includes:
    - Detailed error information
    - Source of the error
    - Timestamp of when the error occurred
    - Stack trace information
    - Flow execution context
    """
    details: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def create(cls: Type[T], **kwargs: Any) -> T:
        """
        Create a new ErrorContext with the provided details.
        
        This is the preferred way to create an ErrorContext as it properly
        structures all provided information into the details dictionary.
        
        Args:
            **kwargs: Key-value pairs to include in the context details
            
        Returns:
            New ErrorContext instance with provided details
        """
        return cls(details=kwargs)
    
    def add(self, **kwargs: Any) -> 'ErrorContext':
        """
        Add additional context details.
        
        Creates a new ErrorContext instance with merged details.
        
        Args:
            **kwargs: Additional key-value pairs to add to context
            
        Returns:
            New ErrorContext with updated details
        """
        return ErrorContext(
            details={**self.details, **kwargs},
            source=self.source,
            timestamp=self.timestamp
        )
    
    def with_source(self, source: str) -> 'ErrorContext':
        """
        Create new context with updated source.
        
        Args:
            source: New source identifier
            
        Returns:
            New ErrorContext with updated source
        """
        return ErrorContext(
            details=self.details.copy(),
            source=source,
            timestamp=self.timestamp
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from context details.
        
        Args:
            key: Detail key to retrieve
            default: Default value if key not found
            
        Returns:
            Value if found, default otherwise
        """
        return self.details.get(key, default)
    
    def __str__(self) -> str:
        """String representation of error context."""
        parts = []
        if self.source:
            parts.append(f"source='{self.source}'")
        if self.details:
            parts.append(f"details={self.details}")
        return f"ErrorContext({', '.join(parts)})"

class BaseError(Exception):
    """Base error class for all framework errors."""
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.context = context or ErrorContext()
        self.timestamp = datetime.now()
        self.cause = cause
        super().__init__(message)

    def chain(self, error: 'BaseError') -> 'BaseError':
        """Chain another error as the cause of this one."""
        self.cause = error
        return self

    def with_context(self, **kwargs: Any) -> 'BaseError':
        """Add additional context to the error."""
        self.context = self.context.add(**kwargs)
        return self

    def __str__(self) -> str:
        """String representation including context and cause."""
        result = self.message
        if self.context and self.context.details:
            result += f" (Context: {self.context.details})"
        if self.cause:
            result += f"\nCaused by: {str(self.cause)}"
        return result

class FrameworkError(BaseError):
    """General framework error."""
    pass

class ResourceError(FrameworkError):
    """Error for resource operations."""
    pass

class ValidationError(FrameworkError):
    """Error for validation failures."""
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []

class ExecutionError(FrameworkError):
    """Error for execution failures."""
    pass

class FlowError(FrameworkError):
    """Error for flow-related issues."""
    pass

class StateError(FrameworkError):
    """Error for state management issues."""
    pass

class ConfigurationError(FrameworkError):
    """Error for configuration issues."""
    pass

