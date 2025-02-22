from typing import TypeVar, Generic
from contextlib import AsyncExitStack
from abc import ABC, abstractmethod

from ..errors.base import StateError, ErrorContext

T = TypeVar('T')

class ManagedResource(ABC, Generic[T]):
    """Base class for resources that need lifecycle management."""
    
    def __init__(self):
        """Initialize managed resource."""
        self._is_initialized = False
        self._exit_stack = AsyncExitStack()
    
    @property
    def is_initialized(self) -> bool:
        """Check if resource is initialized."""
        return self._is_initialized
    
    async def __aenter__(self) -> 'ManagedResource[T]':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()
    
    def check_initialized(self) -> None:
        """Check if resource is initialized.
        
        Raises:
            StateError: If resource is not initialized
        """
        if not self._is_initialized:
            raise StateError(
                "Resource not initialized",
                ErrorContext.create(
                    resource_type=self.__class__.__name__,
                    state="uninitialized"
                )
            )
    
    @abstractmethod
    async def initialize(self, config: T) -> None:
        """Initialize the resource.
        
        Args:
            config: Resource configuration
            
        Raises:
            Exception: If initialization fails
        """
        pass
    
    async def cleanup(self) -> None:
        """Clean up resource."""
        await self._exit_stack.aclose()
        self._is_initialized = False 