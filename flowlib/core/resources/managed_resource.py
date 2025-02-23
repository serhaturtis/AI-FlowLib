from typing import Type, TypeVar
from functools import wraps

from .manager import ResourceManager

T = TypeVar('T')

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