# src/core/errors/handlers.py

import functools
import asyncio
from typing import Type, Callable, Tuple, Optional, Dict, Any

from .base import BaseError

class ErrorHandler:
    """Base class for error handlers."""
    
    async def handle(
        self,
        error: BaseError,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Handle error occurrence.
        
        Args:
            error: Error that occurred
            context: Error context
            
        Returns:
            Optional result from error handling
        """
        pass

class RetryHandler(ErrorHandler):
    """Handler that retries operations on failure."""
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retries
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay after each retry
            exceptions: Optional tuple of exceptions to handle
        """
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions or (Exception,)
    
    async def handle(
        self,
        error: BaseError,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Handle error with retries.
        
        Args:
            error: Error that occurred
            context: Error context with retry function
            
        Returns:
            Result from successful retry or None
        """
        if not isinstance(error.cause, self.exceptions):
            return None
            
        retry_fn = context.get('retry_fn')
        if not retry_fn:
            return None
            
        current_delay = self.delay
        for i in range(self.max_retries):
            try:
                await asyncio.sleep(current_delay)
                if asyncio.iscoroutinefunction(retry_fn):
                    return await retry_fn()
                return retry_fn()
            except Exception:
                current_delay *= self.backoff
        return None

class FallbackHandler(ErrorHandler):
    """Handler that provides fallback behavior."""
    
    def __init__(self, fallback_fn: Callable):
        """Initialize fallback handler.
        
        Args:
            fallback_fn: Function to call for fallback
        """
        self.fallback_fn = fallback_fn
    
    async def handle(
        self,
        error: BaseError,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Handle error with fallback.
        
        Args:
            error: Error that occurred
            context: Error context
            
        Returns:
            Result from fallback function
        """
        if asyncio.iscoroutinefunction(self.fallback_fn):
            return await self.fallback_fn(error, context)
        return self.fallback_fn(error, context)

def with_error_handling(
    *handlers: ErrorHandler
) -> Callable:
    """Decorator for adding error handling to functions.
    
    Args:
        *handlers: Error handlers to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                if not isinstance(e, BaseError):
                    # Wrap in BaseError if needed
                    e = BaseError(str(e), cause=e)
                
                # Try each handler
                context = {
                    'args': args,
                    'kwargs': kwargs,
                    'retry_fn': lambda: func(*args, **kwargs)
                }
                
                for handler in handlers:
                    result = await handler.handle(e, context)
                    if result is not None:
                        return result
                        
                # Re-raise if no handler succeeded
                raise
                
        return wrapper
    return decorator

