from typing import Any, Dict, Optional, Callable
from contextlib import asynccontextmanager

from .manager import ResourceManager
from .managed_resource import managed

from ...providers.llm import LLMProvider

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