from typing import Dict, Any, Protocol
from .constants import ResourceType
from .registry import resource_registry


class PromptTemplate(Protocol):
    """Protocol defining the interface for prompt templates.
    
    Any class decorated with @prompt must implement this interface.
    """
    template: str
    config: Dict[str, Any]


def resource(name: str, resource_type: str = ResourceType.MODEL, **metadata):
    """Register a class or function as a resource.
    
    This decorator:
    1. Registers a class or function in the resource registry
    2. Handles common metadata population
    3. Provides consistent naming
    
    Args:
        name: Unique name for the resource
        resource_type: Type of resource (model, prompt, etc.)
        **metadata: Additional metadata about the resource
        
    Returns:
        Decorator function
    """
    def decorator(obj):
        # Ensure registry is available
        if resource_registry is None:
            raise RuntimeError("Resource registry not initialized")
            
        # Register the resource
        resource_registry.register(
            name=name,
            obj=obj,
            resource_type=resource_type,
            **metadata
        )
        
        # Add registration info to object metadata
        obj.__resource_name__ = name
        obj.__resource_type__ = resource_type
        obj.__resource_metadata__ = metadata
        
        return obj
    
    return decorator

def model(name: str, **metadata):
    """Register a class as a model resource.
    
    This decorator is a specialized version of @resource for models.
    
    Args:
        name: Unique name for the model
        **metadata: Additional metadata about the model
        
    Returns:
        Decorator function
    """
    return resource(name, ResourceType.MODEL, **metadata)

def prompt(name: str, **metadata):
    """Register a class as a prompt resource.
    
    This decorator ensures the decorated class adheres to the PromptTemplate protocol
    by requiring a 'template' attribute and adding a default 'config' attribute if missing.
    
    Args:
        name: Unique name for the prompt
        **metadata: Additional metadata about the prompt
        
    Returns:
        Decorator function that returns a class conforming to PromptTemplate
        
    Raises:
        ValueError: If the decorated object does not have a 'template' attribute
                   or if 'config' is not present after decoration
    """
    def decorator(obj):
        # Check if template exists before registration
        if not hasattr(obj, 'template'):
            raise ValueError(f"Prompt '{name}' must have a 'template' attribute")
        
        # Call the resource decorator
        decorated_obj = resource(name, ResourceType.PROMPT, **metadata)(obj)
        
        # Add default config only if it doesn't already exist
        if not hasattr(decorated_obj, 'config'):
            decorated_obj.config = {
                "max_tokens": 2048,
                "temperature": 0.5,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        
        # Verify config exists (either from the original object or our default)
        if not hasattr(decorated_obj, 'config'):
            raise ValueError(f"Prompt '{name}' must have a 'config' attribute")
        
        # This object now conforms to PromptTemplate protocol
        return decorated_obj
    
    return decorator

def config(name: str, **metadata):
    """Register a class as a configuration resource.
    
    This decorator is a specialized version of @resource for configs.
    
    Args:
        name: Unique name for the config
        **metadata: Additional metadata about the config
        
    Returns:
        Decorator function
    """
    return resource(name, ResourceType.CONFIG, **metadata)

