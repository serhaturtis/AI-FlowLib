"""Error manager with enhanced error handling capabilities.

This module provides a robust error management system with error boundaries
and customizable error handlers.
"""

from typing import Dict, Any, Optional, List, Union, Type, TypeVar, Callable, Awaitable, cast
from contextlib import asynccontextmanager
import logging
import traceback

from .base import BaseError, ErrorContext

logger = logging.getLogger(__name__)

ErrorHandlerFunc = Callable[[BaseError, Dict[str, Any]], Union[None, Dict[str, Any]]]
AsyncErrorHandlerFunc = Callable[[BaseError, Dict[str, Any]], Awaitable[Union[None, Dict[str, Any]]]]

ErrorHandler = Union[ErrorHandlerFunc, AsyncErrorHandlerFunc]

class ErrorManager:
    """Centralized error management with customizable handlers.
    
    This class provides:
    1. Error boundary capabilities
    2. Customizable error handlers
    3. Error logging and reporting
    4. Context preservation
    """
    
    def __init__(self):
        """Initialize error manager."""
        self._handlers: Dict[Type[BaseError], List[ErrorHandler]] = {}
        self._global_handlers: List[ErrorHandler] = []
    
    def register(self, error_type: Type[BaseError], handler: ErrorHandler) -> None:
        """Register an error handler for a specific error type.
        
        Args:
            error_type: Type of error to handle
            handler: Handler function or coroutine
        """
        if error_type not in self._handlers:
            self._handlers[error_type] = []
        
        self._handlers[error_type].append(handler)
    
    def register_global(self, handler: ErrorHandler) -> None:
        """Register a global error handler that processes all errors.
        
        Args:
            handler: Handler function or coroutine
        """
        self._global_handlers.append(handler)
    
    async def _handle_error(self, error: BaseError, context: Dict[str, Any]) -> None:
        """Handle an error with registered handlers.
        
        Args:
            error: Error to handle
            context: Context data
            
        Returns:
            Optional handler result
        """
        # Collect all applicable handlers
        handlers = []
        
        # Add specific handlers for this error type
        for error_type, type_handlers in self._handlers.items():
            if isinstance(error, error_type):
                handlers.extend(type_handlers)
        
        # Add global handlers
        handlers.extend(self._global_handlers)
        
        # Execute handlers
        for handler in handlers:
            try:
                import inspect
                
                if inspect.iscoroutinefunction(handler):
                    # Async handler
                    await handler(error, context)
                else:
                    # Sync handler
                    handler(error, context)
                    
            except Exception as e:
                # Log handler errors but don't propagate
                logger.error(f"Error in error handler: {e}")
                logger.error(traceback.format_exc())
    
    @asynccontextmanager
    async def error_boundary(self, context_data: Dict[str, Any] = None):
        """Create an error boundary that handles errors with registered handlers.
        
        Args:
            context_data: Optional context data to include
            
        Yields:
            None
            
        Raises:
            BaseError: Propagated after handling
        """
        # Prepare context
        context = context_data or {}
        
        try:
            # Yield control to the wrapped code
            yield
            
        except BaseError as e:
            # Handle framework error
            await self._handle_error(e, context)
            
            # Re-raise after handling
            raise
            
        except Exception as e:
            # Convert and handle non-framework error
            from .base import ExecutionError
            
            error_context = ErrorContext.create(
                exception_type=type(e).__name__,
                **context
            )
            
            error = ExecutionError(
                message=str(e),
                context=error_context,
                cause=e
            )
            
            # Handle converted error
            await self._handle_error(error, context)
            
            # Re-raise converted error
            raise error from e

class LoggingHandler:
    """Error handler that logs errors with configurable verbosity.
    
    This class provides detailed logging of errors, including
    context information and causal chains.
    """
    
    def __init__(
        self,
        level: int = logging.ERROR,
        include_context: bool = True,
        include_traceback: bool = True,
        logger_name: Optional[str] = None
    ):
        """Initialize logging handler.
        
        Args:
            level: Logging level
            include_context: Whether to include context in logs
            include_traceback: Whether to include traceback in logs
            logger_name: Optional custom logger name
        """
        self.level = level
        self.include_context = include_context
        self.include_traceback = include_traceback
        self.logger = logging.getLogger(logger_name or __name__)
    
    def __call__(self, error: BaseError, context: Dict[str, Any]) -> None:
        """Handle error by logging it.
        
        Args:
            error: Error to handle
            context: Context data
        """
        # Format error message
        message = f"{type(error).__name__}: {error.message}"
        
        # Add context if enabled
        if self.include_context and error.context:
            message += f"\nContext: {error.context}"
        
        # Add traceback if enabled
        if self.include_traceback and error.cause:
            cause_tb = "".join(traceback.format_exception(
                type(error.cause),
                error.cause,
                error.cause.__traceback__
            ))
            message += f"\nCaused by: {cause_tb}"
        
        # Log the error
        self.logger.log(self.level, message)

class MetricsHandler:
    """Error handler that records error metrics.
    
    This class provides error tracking for monitoring and alerting.
    """
    
    def __init__(self, metrics_client: Any):
        """Initialize metrics handler.
        
        Args:
            metrics_client: Client for recording metrics
        """
        self.metrics_client = metrics_client
    
    def __call__(self, error: BaseError, context: Dict[str, Any]) -> None:
        """Handle error by recording metrics.
        
        Args:
            error: Error to handle
            context: Context data
        """
        # Record error count
        self.metrics_client.increment(
            "flow.errors",
            tags={
                "error_type": type(error).__name__,
                "flow_name": error.context.get("flow_name", "unknown")
            }
        ) 