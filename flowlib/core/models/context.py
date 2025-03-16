"""Enhanced context model with attribute-based access and improved state management.

This module provides a Context class for managing execution state with 
attribute-based access, snapshot capabilities, and clean validation.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type, cast
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)

class Context(Generic[T]):
    """Enhanced execution context with improved state management.
    
    This class provides:
    1. Attribute-based access to context data
    2. Clean state management with snapshots and rollbacks
    3. Type-safe validation of data
    4. Deep copying for isolation
    """
    
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        model_type: Optional[Type[T]] = None
    ):
        """Initialize context.
        
        Args:
            data: Initial data dictionary
            model_type: Optional Pydantic model type for validation
        """
        self._data = data or {}
        self._model_type = model_type
        self._snapshots: List[Dict[str, Any]] = []
        
        # Validate initial data if model type is provided
        if model_type and data:
            self._validate(data)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the data dictionary."""
        return self._data
    
    def __getattr__(self, name: str) -> Any:
        """Enable attribute-based access to context data.
        
        Args:
            name: Attribute name to access
            
        Returns:
            Attribute value
            
        Raises:
            AttributeError: If attribute not found
        """
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context data.
        
        Args:
            key: Key to look up
            default: Default value if key not found
            
        Returns:
            Value associated with key or default
        """
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> 'Context':
        """Set a value in context data.
        
        Args:
            key: Key to set
            value: Value to associate with key
            
        Returns:
            Self for chaining
        """
        self._data[key] = value
        return self
    
    def update(self, data: Dict[str, Any]) -> 'Context':
        """Update context data with dictionary.
        
        Args:
            data: Dictionary of values to update
            
        Returns:
            Self for chaining
        """
        if self._model_type:
            # Validate update if model type is provided
            self._validate(data)
        self._data.update(data)
        return self
    
    def _validate(self, data: Dict[str, Any]) -> None:
        """Validate data against model type.
        
        Args:
            data: Data to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not self._model_type:
            return
            
        try:
            # Only validate the fields that are being updated
            # Create a partial model with just the fields in data
            field_values = {}
            for key, value in data.items():
                if key in self._model_type.__annotations__:
                    field_values[key] = value
            
            if field_values:
                # Validate the partial model
                self._model_type(**field_values)
        except Exception as e:
            raise ValueError(f"Context data validation failed: {str(e)}")
    
    def create_snapshot(self) -> int:
        """Create a snapshot of current state.
        
        Returns:
            Snapshot ID
        """
        self._snapshots.append(deepcopy(self._data))
        return len(self._snapshots) - 1
    
    def rollback(self, snapshot_id: Optional[int] = None) -> 'Context':
        """Rollback to a previous snapshot.
        
        Args:
            snapshot_id: Optional snapshot ID (defaults to last snapshot)
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If snapshot ID is invalid
        """
        if not self._snapshots:
            raise ValueError("No snapshots available for rollback")
        
        if snapshot_id is None:
            # Default to last snapshot
            snapshot_id = len(self._snapshots) - 1
        
        if snapshot_id < 0 or snapshot_id >= len(self._snapshots):
            raise ValueError(f"Invalid snapshot ID: {snapshot_id}")
        
        # Restore data from snapshot
        self._data = deepcopy(self._snapshots[snapshot_id])
        
        # Remove this and any later snapshots
        self._snapshots = self._snapshots[:snapshot_id]
        
        return self
    
    def clear_snapshots(self) -> 'Context':
        """Clear all snapshots.
        
        Returns:
            Self for chaining
        """
        self._snapshots = []
        return self
    
    def copy(self) -> 'Context':
        """Create a deep copy of the context.
        
        Returns:
            New Context instance with copied data
        """
        return Context(
            data=deepcopy(self._data),
            model_type=self._model_type
        )
    
    def as_model(self) -> Optional[T]:
        """Convert context data to model instance.
        
        Returns:
            Model instance or None if no model type
        """
        if not self._model_type:
            return None
        
        try:
            return cast(T, self._model_type(**self._data))
        except Exception as e:
            raise ValueError(f"Failed to convert context to {self._model_type.__name__}: {str(e)}")
    
    def __str__(self) -> str:
        """String representation."""
        model_name = self._model_type.__name__ if self._model_type else "None"
        return f"Context(model_type={model_name}, data_keys={list(self._data.keys())}, snapshots={len(self._snapshots)})"
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in context data."""
        return key in self._data 