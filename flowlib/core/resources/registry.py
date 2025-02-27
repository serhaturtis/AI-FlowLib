from typing import Dict, Any, Optional, Type, TypeVar, Callable
from functools import wraps
import inspect
from pydantic import BaseModel

T = TypeVar('T')

class ResourceRegistry:
    """Global registry for resources."""
    
    _instance = None
    _resources: Dict[str, Dict[str, Any]] = {}
    _resource_types: Dict[str, Type[Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register_resource(cls, resource_type: str, name: str, config: Any) -> None:
        """Register a resource configuration.
        
        Args:
            resource_type: Type of resource (e.g., 'provider', 'model')
            name: Name of the resource
            config: Resource configuration
        """
        if resource_type not in cls._resources:
            cls._resources[resource_type] = {}
        cls._resources[resource_type][name] = config
    
    @classmethod
    def get_resource(cls, resource_type: str, name: str) -> Optional[Any]:
        """Get a registered resource.
        
        Args:
            resource_type: Type of resource
            name: Name of the resource
            
        Returns:
            Resource if found, None otherwise
        """
        return cls._resources.get(resource_type, {}).get(name)
    
    @classmethod
    def register_resource_type(cls, name: str, resource_type: Type[Any]) -> None:
        """Register a resource type.
        
        Args:
            name: Name of the resource type
            resource_type: Resource type class
        """
        cls._resource_types[name] = resource_type
    
    @classmethod
    def get_resource_type(cls, name: str) -> Optional[Type[Any]]:
        """Get a registered resource type.
        
        Args:
            name: Name of the resource type
            
        Returns:
            Resource type if found, None otherwise
        """
        return cls._resource_types.get(name)

def resource(resource_type: str):
    """Decorator to register a resource configuration.
    
    Args:
        resource_type: Type of resource (e.g., 'provider', 'model')
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Ensure class is a Pydantic model
        if not issubclass(cls, BaseModel):
            raise TypeError(f"{cls.__name__} must be a Pydantic model")
        
        # Get resource name from class
        name = getattr(cls, 'name', cls.__name__)
        
        # Register the resource type
        ResourceRegistry.register_resource_type(resource_type, cls)
        
        # Register an instance with default values
        instance = cls()
        ResourceRegistry.register_resource(resource_type, name, instance)
        
        return cls
    return decorator

# Convenience decorators for common resource types
def provider(name: str = None):
    """Decorator to register a provider configuration."""
    def decorator(cls: Type[T]) -> Type[T]:
        if name is not None:
            cls.name = name
        return resource('provider')(cls)
    return decorator

def model(name: str = None):
    """Decorator to register a model configuration."""
    def decorator(cls: Type[T]) -> Type[T]:
        if name is not None:
            cls.name = name
        return resource('model')(cls)
    return decorator 