"""Base registry interface for the flowlib registry system.

This module defines the abstract base class for all registry types in the 
flowlib system, providing a common interface for registration and retrieval.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic

T = TypeVar('T')

class BaseRegistry(ABC, Generic[T]):
    """Abstract base class for all registry types.
    
    This class defines the common interface that all registries must implement,
    establishing a consistent pattern for registration and retrieval operations.
    """
    
    @abstractmethod
    def register(self, name: str, obj: T, **metadata) -> None:
        """Register an object with the registry.
        
        Args:
            name: Unique name for the object
            obj: The object to register
            **metadata: Additional metadata about the object
        """
        pass
        
    @abstractmethod
    def get(self, name: str, expected_type: Optional[Type] = None) -> T:
        """Get an object by name with optional type checking.
        
        Args:
            name: Name of the object to retrieve
            expected_type: Optional type for type checking
            
        Returns:
            The registered object
            
        Raises:
            KeyError: If the object doesn't exist
            TypeError: If the object doesn't match the expected type
        """
        pass
        
    @abstractmethod
    def contains(self, name: str) -> bool:
        """Check if an object exists in the registry.
        
        Args:
            name: Name to check
            
        Returns:
            True if the object exists, False otherwise
        """
        pass
        
    @abstractmethod
    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List registered objects matching criteria.
        
        Args:
            filter_criteria: Optional criteria to filter results
            
        Returns:
            List of object names matching the criteria
        """
        pass 