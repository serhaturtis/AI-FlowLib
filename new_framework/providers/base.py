from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
from contextlib import AsyncExitStack

from ..core.errors.base import ResourceError, ErrorContext

class Provider(ABC):
    """Base class for all providers."""
    
    def __init__(self, name: str):
        """Initialize provider.
        
        Args:
            name: Provider name
        """
        self.name = name
        self._exit_stack = AsyncExitStack()
        self._initialized = False
    
    async def __aenter__(self) -> 'Provider':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize provider resources."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        await self._exit_stack.aclose()
        self._initialized = False
    
    def check_initialized(self) -> None:
        """Check if provider is initialized.
        
        Raises:
            ResourceError: If provider is not initialized
        """
        if not self._initialized:
            raise ResourceError(
                "Provider not initialized",
                ErrorContext.create(
                    provider_name=self.name,
                    provider_type=self.__class__.__name__
                )
            )
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
