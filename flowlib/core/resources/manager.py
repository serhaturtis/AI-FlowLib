"""Resource management system."""

from typing import Any, Dict, Optional, Type, TypeVar, Callable, AsyncGenerator
from contextlib import asynccontextmanager
import inspect
from functools import wraps

from ...providers.llm import LLMProvider

T = TypeVar('T')

class ResourceManager:
    """Manager for handling resource lifecycle."""
    
    def __init__(self):
        """Initialize resource manager."""
        self._resources: Dict[str, Any] = {}
        self._cleanup_fns: Dict[str, Callable] = {}
        
    async def cleanup(self):
        """Clean up all managed resources."""
        for name, cleanup_fn in self._cleanup_fns.items():
            if inspect.iscoroutinefunction(cleanup_fn):
                await cleanup_fn(self._resources[name])
            else:
                cleanup_fn(self._resources[name])
        self._resources.clear()
        self._cleanup_fns.clear()
        
    def register(
        self,
        name: str,
        resource: Any,
        cleanup_fn: Optional[Callable] = None
    ) -> None:
        """Register a resource for management.
        
        Args:
            name: Resource name
            resource: Resource instance
            cleanup_fn: Optional cleanup function
        """
        self._resources[name] = resource
        if cleanup_fn:
            self._cleanup_fns[name] = cleanup_fn
            
    def get(self, name: str) -> Optional[Any]:
        """Get a registered resource.
        
        Args:
            name: Resource name
            
        Returns:
            Resource if found, None otherwise
        """
        return self._resources.get(name)
    
    async def remove(self, name: str) -> None:
        """Remove a resource from management.
        
        Args:
            name: Resource name
        """
        if name in self._resources:
            if name in self._cleanup_fns:
                cleanup_fn = self._cleanup_fns.pop(name)
                if inspect.iscoroutinefunction(cleanup_fn):
                    await cleanup_fn(self._resources[name])
                else:
                    cleanup_fn(self._resources[name])
            del self._resources[name]
    
    @asynccontextmanager
    async def managed_resource(
        self,
        name: str,
        resource_type: Type[T],
        *args: Any,
        **kwargs: Any
    ) -> AsyncGenerator[T, None]:
        """Context manager for resource lifecycle.
        
        Args:
            name: Resource name
            resource_type: Resource class
            *args: Arguments for resource creation
            **kwargs: Keyword arguments for resource creation
            
        Returns:
            Managed resource instance
        """
        resource = resource_type(*args, **kwargs)
        self.register(name, resource)
        try:
            yield resource
        finally:
            await self.remove(name)
    
    def resource_factory(
        self,
        name: str,
        resource_type: Type[T],
        cleanup_fn: Optional[Callable] = None
    ) -> Callable[..., T]:
        """Create a factory function for resource creation.
        
        Args:
            name: Resource name
            resource_type: Resource class
            cleanup_fn: Optional cleanup function
            
        Returns:
            Factory function
        """
        def factory(*args: Any, **kwargs: Any) -> T:
            resource = resource_type(*args, **kwargs)
            self.register(name, resource, cleanup_fn)
            return resource
        return factory
    
    def llm(
        self,
        name: str,
        model_configs: Dict[str, Any],
        max_models: int = 2
    ) -> LLMProvider:
        """Create an LLM provider.
        
        Args:
            name: Provider name
            model_configs: Model configurations
            max_models: Maximum number of models
            
        Returns:
            LLM provider instance
        """
        provider = LLMProvider(name, model_configs, max_models)
        self.register(name, provider, provider.cleanup)
        return provider 