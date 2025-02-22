# src/core/context.py

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from copy import deepcopy

class ContextState:
    """Snapshot of context state."""
    
    def __init__(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
        timestamp: datetime
    ):
        """
        Initialize context state.
        
        Args:
            data: Context data
            metadata: Context metadata
            timestamp: State timestamp
        """
        self.data = deepcopy(data)
        self.metadata = deepcopy(metadata)
        self.timestamp = timestamp

    def restore(self, context: 'Context') -> None:
        """
        Restore this state to a context.
        
        Args:
            context: Context to restore to
        """
        context.data = deepcopy(self.data)
        context.metadata = deepcopy(self.metadata)

class Context:
    """
    Manages data and state during flow execution.
    Provides hierarchical state management and history tracking.
    """
    
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional['Context'] = None,
        copy_history: bool = False
    ):
        """
        Initialize context.
        
        Args:
            data: Initial data dictionary
            metadata: Initial metadata dictionary
            parent: Optional parent context
            copy_history: Whether to copy history when creating new context
        """
        self.id = str(uuid4())
        self.data = data or {}
        self.metadata = metadata or {}
        self.parent = parent
        self._history: List[ContextState] = []
        self._temp_data: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.last_modified = self.created_at

    def copy(self, deep: bool = True, copy_history: bool = False) -> 'Context':
        """
        Create a copy of this context.
        
        Args:
            deep: Whether to perform a deep copy of data
            copy_history: Whether to copy execution history
            
        Returns:
            New context instance with copied data
        """
        if deep:
            data = deepcopy(self.data)
            metadata = deepcopy(self.metadata)
            temp_data = deepcopy(self._temp_data)
        else:
            data = dict(self.data)
            metadata = dict(self.metadata)
            temp_data = dict(self._temp_data)
            
        # Create new context with copied data
        new_context = Context(
            data=data,
            metadata=metadata,
            parent=self.parent,  # Maintain same parent relationship
            copy_history=copy_history
        )
        
        # Copy temporary data
        new_context._temp_data = temp_data
        
        # Copy history if requested
        if copy_history:
            new_context._history = [
                ContextState(
                    data=deepcopy(state.data) if deep else dict(state.data),
                    metadata=deepcopy(state.metadata) if deep else dict(state.metadata),
                    timestamp=state.timestamp
                )
                for state in self._history
            ]
        
        # Copy timestamps
        new_context.created_at = self.created_at
        new_context.last_modified = self.last_modified
        
        return new_context

    def get(
        self,
        key: str,
        default: Any = None,
        include_parent: bool = True
    ) -> Any:
        """
        Get value from context.
        
        Args:
            key: Data key
            default: Default value if key not found
            include_parent: Whether to check parent context
            
        Returns:
            Value if found, default otherwise
        """
        # Check local data
        if key in self.data:
            return self.data[key]
            
        # Check temp data
        if key in self._temp_data:
            return self._temp_data[key]
            
        # Check parent if allowed
        if include_parent and self.parent is not None:
            return self.parent.get(key, default)
            
        return default

    def set(
        self,
        key: str,
        value: Any,
        temporary: bool = False
    ) -> None:
        """
        Set value in context.
        
        Args:
            key: Data key
            value: Value to store
            temporary: Whether to store in temporary storage
        """
        self._save_state()
        
        if temporary:
            self._temp_data[key] = value
        else:
            self.data[key] = value
            
        self.last_modified = datetime.now()

    def update(
        self,
        data: Dict[str, Any],
        temporary: bool = False
    ) -> None:
        """
        Update multiple values.
        
        Args:
            data: Dictionary of updates
            temporary: Whether to store in temporary storage
        """
        self._save_state()
        
        if temporary:
            self._temp_data.update(data)
        else:
            self.data.update(data)
            
        self.last_modified = datetime.now()

    def delete(
        self,
        key: str,
        include_temp: bool = True
    ) -> None:
        """
        Delete value from context.
        
        Args:
            key: Key to delete
            include_temp: Whether to check temporary storage
        """
        self._save_state()
        
        if key in self.data:
            del self.data[key]
            
        if include_temp and key in self._temp_data:
            del self._temp_data[key]
            
        self.last_modified = datetime.now()

    def _save_state(self) -> None:
        """Save current state to history."""
        self._history.append(
            ContextState(
                data=self.data,
                metadata=self.metadata,
                timestamp=datetime.now()
            )
        )

    def rollback(self) -> None:
        """Rollback to previous state."""
        if self._history:
            previous_state = self._history.pop()
            previous_state.restore(self)
            self.last_modified = datetime.now()

    def create_child(
        self,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Context':
        """
        Create a new context inheriting from this one.
        
        Args:
            data: Optional initial data
            metadata: Optional initial metadata
            
        Returns:
            New child context
        """
        return Context(
            data=data,
            metadata={
                **(metadata or {}),
                "parent_context_id": self.id
            },
            parent=self
        )

    def merge(self, other: 'Context') -> None:
        """
        Merge another context into this one.
        
        Args:
            other: Context to merge
        """
        self._save_state()
        self.data.update(other.data)
        self.metadata.update(other.metadata)
        self._temp_data.update(other._temp_data)
        self.last_modified = datetime.now()

    def clear_temp(self) -> None:
        """Clear temporary storage."""
        self._temp_data.clear()

    def get_history(self) -> List[ContextState]:
        """Get state history."""
        return self._history.copy()

    def contains(
        self,
        key: str,
        include_parent: bool = True,
        include_temp: bool = True
    ) -> bool:
        """
        Check if key exists in context.
        
        Args:
            key: Key to check
            include_parent: Whether to check parent context
            include_temp: Whether to check temporary storage
            
        Returns:
            True if key exists, False otherwise
        """
        if key in self.data:
            return True
            
        if include_temp and key in self._temp_data:
            return True
            
        if include_parent and self.parent is not None:
            return self.parent.contains(key)
            
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "id": self.id,
            "data": deepcopy(self.data),
            "metadata": deepcopy(self.metadata),
            "temp_data": deepcopy(self._temp_data),
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat()
        }

    def __str__(self) -> str:
        """String representation."""
        return f"Context(id={self.id}, keys={list(self.data.keys())})"