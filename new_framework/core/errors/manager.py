# src/core/errors/manager.py

from typing import Any, List, Dict, Type, Optional
from contextlib import asynccontextmanager

from .base import BaseError
from .handlers import ErrorHandler

class ErrorManager:
    """Manages error handling strategies."""
    
    def __init__(self):
        self._handlers: Dict[Type[BaseError], List[ErrorHandler]] = {}

    def register(
        self,
        error_type: Type[BaseError],
        handler: ErrorHandler
    ) -> None:
        """Register error handler."""
        if error_type not in self._handlers:
            self._handlers[error_type] = []
        self._handlers[error_type].append(handler)

    async def handle(
        self,
        error: BaseError,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Handle error using registered handlers.
        
        Args:
            error: Error to handle
            context: Execution context
            
        Returns:
            Handler result if successful, None otherwise
        """
        # Find matching handlers
        handlers = []
        for error_type, type_handlers in self._handlers.items():
            if isinstance(error, error_type):
                handlers.extend(type_handlers)
                
        # Try each handler
        for handler in handlers:
            result = await handler.handle(error, context)
            if result is not None:
                return result
                
        return None

    @asynccontextmanager
    async def error_boundary(self, context: Dict[str, Any]):
        """
        Context manager for error handling.
        
        Args:
            context: Execution context
        """
        try:
            yield
        except BaseError as e:
            # Try to handle error
            result = await self.handle(e, context)
            if result is None:
                raise
            yield result  # Return the handled result if available