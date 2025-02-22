"""Resource management system."""

from typing import Any, Dict, Optional, Type, TypeVar, Callable
from contextlib import asynccontextmanager
import inspect
from functools import wraps

from .providers.llm import LLMProvider

T = TypeVar('T')

class ResourceManager:
    """Manager for handling resource lifecycle."""
    
    def __init__(self):
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
    ):
        """Register a resource for management.
        
        Args:
            name: Resource name
            resource: Resource instance
            cleanup_fn: Optional cleanup function
        """
        self._resources[name] = resource
        if cleanup_fn:
            self._cleanup_fns[name] = cleanup_fn
            
    def get(self, name: str) -> Any:
        """Get a managed resource.
        
        Args:
            name: Resource name
            
        Returns:
            Managed resource
        
        Raises:
            KeyError: If resource not found
        """
        return self._resources[name]

class ManagedResource:
    """Base class for resources that need lifecycle management."""
    
    def __init__(self):
        self._manager = ResourceManager()
        
    async def cleanup(self):
        """Clean up managed resources."""
        await self._manager.cleanup()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

def managed(cls: Type[T]) -> Type[T]:
    """Decorator to add resource management to a class.
    
    Example:
        @managed
        class DocumentAnalyzer:
            def __init__(self):
                self.provider = managed.llm("gpt3")
                self.cache = managed.cache("redis")
    
    Args:
        cls: Class to add resource management to
        
    Returns:
        Managed class
    """
    # Store original init
    orig_init = cls.__init__
    
    @wraps(orig_init)
    def init(self, *args, **kwargs):
        # Initialize resource manager
        self._manager = ResourceManager()
        
        # Call original init
        orig_init(self, *args, **kwargs)
    
    # Replace init
    cls.__init__ = init
    
    # Add async context manager if not defined
    if not hasattr(cls, '__aenter__'):
        async def __aenter__(self):
            return self
        cls.__aenter__ = __aenter__
        
    if not hasattr(cls, '__aexit__'):
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._manager.cleanup()
        cls.__aexit__ = __aexit__
        
    # Add cleanup method if not defined
    if not hasattr(cls, 'cleanup'):
        async def cleanup(self):
            await self._manager.cleanup()
        cls.cleanup = cleanup
        
    return cls

class ResourceFactory:
    """Factory for creating managed resources."""
    
    def __init__(self, manager: ResourceManager):
        self._manager = manager
        
    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        cleanup_fn: Optional[Callable] = None
    ):
        """Register a resource factory.
        
        Args:
            name: Resource type name
            factory: Factory function
            cleanup_fn: Optional cleanup function
        """
        def wrapper(*args, **kwargs):
            resource = factory(*args, **kwargs)
            self._manager.register(name, resource, cleanup_fn)
            return resource
        setattr(self, name, wrapper)

# Create global resource factory
managed.factory = ResourceFactory(ResourceManager())

# Register common resource types
@asynccontextmanager
async def cleanup_provider(provider):
    try:
        yield
    finally:
        await provider.cleanup()
        
managed.factory.register(
    "llm",
    lambda name, model_configs, max_models: LLMProvider(
        name=name,
        model_configs=model_configs,
        max_models=max_models
    ),
    lambda p: p.cleanup()
) 