"""Resource registry implementation for non-provider resources.

This module provides a concrete implementation of the BaseRegistry for
managing non-provider resources like models, prompts, and configurations.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

from ..errors import ResourceError, ErrorContext
from .base import BaseRegistry
from .constants import ResourceType

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ResourceRegistry(BaseRegistry[T]):
    """Registry for non-provider resources.
    
    This class implements the BaseRegistry interface for managing resources
    like models, prompts, and configurations with type safety and validation.
    """
    
    def __init__(self):
        """Initialize resource registry."""
        # Main storage for resources: (resource_type, name) -> resource
        self._resources: Dict[tuple, Any] = {}
        # Storage for resource metadata
        self._metadata: Dict[tuple, Dict[str, Any]] = {}
        # Collection of resource types in use
        self._resource_types: set = set()
        
    def register(self, name: str, obj: Any, resource_type: str = ResourceType.MODEL, **metadata) -> None:
        """Register a resource with the registry.
        
        Args:
            name: Unique name for the resource
            obj: The resource to register
            resource_type: Type of the resource
            **metadata: Additional metadata about the resource
            
        Raises:
            ValueError: If resource with same name/type already exists
        """
        key = (resource_type, name)
        
        if key in self._resources:
            raise ValueError(f"Resource '{name}' of type '{resource_type}' already exists")
        
        # Store resource and metadata
        self._resources[key] = obj
        self._metadata[key] = metadata
        self._resource_types.add(resource_type)
        
        logger.debug(f"Registered {resource_type} '{name}'")
        
    def get(self, name: str, resource_type: Optional[str] = None, expected_type: Optional[Type] = None) -> Any:
        """Get a resource by name.
        
        Args:
            name: Resource name
            resource_type: Optional resource type for filtering
            expected_type: Optional expected type for validation
            
        Returns:
            The resource
            
        Raises:
            KeyError: If resource not found
            ValueError: If resource_type doesn't match
            TypeError: If expected_type doesn't match
        """
        if name not in self._resources:
            raise KeyError(f"Resource '{name}' not found")
            
        resource = self._resources[name]
        
        # Check resource type if specified
        if resource_type and self._resource_types.get(name) != resource_type:
            raise ValueError(
                f"Resource '{name}' is not of type '{resource_type}'. " +
                f"It is of type '{self._resource_types.get(name)}'."
            )
            
        # Type check if specified
        if expected_type and not isinstance(resource, expected_type):
            raise TypeError(
                f"Resource '{name}' is not of expected type {expected_type.__name__}. " +
                f"It is of type {type(resource).__name__}."
            )
            
        return resource
        
    def get_sync(self, name: str, resource_type: str = ResourceType.MODEL, expected_type: Optional[Type] = None) -> Any:
        """Get a resource by name and type (synchronous version).
        
        Args:
            name: Name of the resource
            resource_type: Type of the resource
            expected_type: Optional Python type for validation
            
        Returns:
            The requested resource
            
        Raises:
            KeyError: If resource doesn't exist
            TypeError: If resource doesn't match expected type
        """
        key = (resource_type, name)
        
        if key not in self._resources:
            raise KeyError(f"Resource '{name}' of type '{resource_type}' not found")
            
        resource = self._resources[key]
        
        # Type checking if expected_type is provided
        if expected_type and not isinstance(resource, expected_type):
            raise TypeError(f"Resource '{name}' is not of expected type {expected_type.__name__}")
            
        return resource
        
    async def get(self, name: str, resource_type: str = ResourceType.MODEL, expected_type: Optional[Type] = None) -> Any:
        """Get a resource by name (async version for compatibility with ProviderRegistry).
        
        This async version is provided for API compatibility with ProviderRegistry's async get,
        allowing code to use 'await registry.get()' regardless of registry type. For resources,
        this simply delegates to the synchronous version as resources don't need initialization.
        
        Args:
            name: Name of the resource
            resource_type: Type of the resource
            expected_type: Optional Python type for validation
            
        Returns:
            The requested resource
            
        Raises:
            KeyError: If resource doesn't exist
            TypeError: If resource doesn't match expected type
        """
        # Just delegate to the synchronous version
        return self.get_sync(name, resource_type, expected_type)
    
    def get_metadata(self, name: str, resource_type: str = ResourceType.MODEL) -> Dict[str, Any]:
        """Get metadata for a resource.
        
        Args:
            name: Name of the resource
            resource_type: Type of the resource
            
        Returns:
            Metadata dictionary for the resource
            
        Raises:
            KeyError: If resource doesn't exist
        """
        key = (resource_type, name)
        
        if key not in self._resources:
            raise KeyError(f"Resource '{name}' of type '{resource_type}' not found")
            
        return self._metadata.get(key, {})
    
    def contains(self, name: str, resource_type: str = ResourceType.MODEL) -> bool:
        """Check if a resource exists.
        
        Args:
            name: Name to check
            resource_type: Type of the resource
            
        Returns:
            True if the resource exists, False otherwise
        """
        key = (resource_type, name)
        return key in self._resources
    
    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List registered resources matching criteria.
        
        Args:
            filter_criteria: Optional criteria to filter results
                - resource_type: Filter by resource type
                
        Returns:
            List of resource names matching the criteria
        """
        filter_type = None
        if filter_criteria and 'resource_type' in filter_criteria:
            filter_type = filter_criteria['resource_type']
        
        result = []
        for (rt, name) in self._resources.keys():
            if filter_type is None or rt == filter_type:
                result.append(name)
                
        return result
    
    def list_types(self) -> List[str]:
        """List all resource types in the registry.
        
        Returns:
            List of resource types
        """
        return list(self._resource_types)
    
    def get_by_type(self, resource_type: str) -> Dict[str, Any]:
        """Get all resources of a specific type.
        
        Args:
            resource_type: Type of resources to retrieve
            
        Returns:
            Dictionary of resource names to resources
        """
        result = {}
        for (rt, name), resource in self._resources.items():
            if rt == resource_type:
                result[name] = resource
        return result
    
    def get_typed(self, name: str, expected_type: Type[T], resource_type: Optional[str] = None) -> T:
        """Get a resource with type validation and casting.
        
        Args:
            name: Name of the resource
            expected_type: Expected Python type
            resource_type: Optional resource type (tries all types if None)
            
        Returns:
            The requested resource cast to the expected type
            
        Raises:
            KeyError: If resource doesn't exist
            TypeError: If resource doesn't match expected type
        """
        if resource_type:
            # Try specific resource type
            resource = self.get(name, resource_type, expected_type)
            return cast(T, resource)
        else:
            # Try all resource types
            for rt in self._resource_types:
                key = (rt, name)
                if key in self._resources:
                    resource = self._resources[key]
                    if isinstance(resource, expected_type):
                        return cast(T, resource)
            
            # Not found or wrong type
            raise KeyError(f"Resource '{name}' of expected type {expected_type.__name__} not found") 