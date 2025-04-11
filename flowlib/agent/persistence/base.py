"""
Base implementation for state persistence.

This module provides a base implementation of the state persistence interface
with common functionality that can be extended by specific implementations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..core.base import BaseComponent
from ..core.errors import StatePersistenceError
from ..models.state import AgentState
from .interfaces import StatePersistenceInterface

logger = logging.getLogger(__name__)


class BaseStatePersister(BaseComponent, StatePersistenceInterface):
    """Base implementation of state persistence.
    
    Provides common functionality for state persistence implementations.
    Specific implementations should inherit from this class and override
    the storage-specific methods.
    """
    
    def __init__(self, name: str = "base_state_persister"):
        """Initialize base state persister.
        
        Args:
            name: Component name
        """
        super().__init__(name)
    
    async def _initialize_impl(self) -> None:
        """Initialize the persister.
        
        This method should be overridden by implementations to perform
        any necessary initialization.
        """
        pass
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the persister.
        
        This method should be overridden by implementations to perform
        any necessary cleanup.
        """
        pass
    
    async def save_state(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Save agent state.
        
        This method updates the state's timestamp and delegates to the
        implementation-specific _save_state_impl method.
        
        Args:
            state: Agent state to save
            metadata: Optional metadata to save with the state
            
        Returns:
            True if state was saved successfully
            
        Raises:
            StatePersistenceError: If saving fails
        """
        if not state.task_id:
            logger.warning("Cannot save state without task_id")
            return False
        
        try:
            # Update timestamp
            state.updated_at = datetime.now()
            
            # Delegate to implementation
            return await self._save_state_impl(state, metadata)
            
        except Exception as e:
            error_msg = f"Error saving state: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="save",
                task_id=state.task_id,
                cause=e
            )
    
    async def load_state(
        self,
        task_id: str
    ) -> Optional[AgentState]:
        """Load agent state.
        
        This method delegates to the implementation-specific _load_state_impl method.
        
        Args:
            task_id: Task ID to load state for
            
        Returns:
            Loaded state or None if not found
            
        Raises:
            StatePersistenceError: If loading fails
        """
        try:
            # Delegate to implementation
            return await self._load_state_impl(task_id)
            
        except Exception as e:
            error_msg = f"Error loading state: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="load",
                task_id=task_id,
                cause=e
            )
    
    async def delete_state(
        self,
        task_id: str
    ) -> bool:
        """Delete agent state.
        
        This method delegates to the implementation-specific _delete_state_impl method.
        
        Args:
            task_id: Task ID to delete state for
            
        Returns:
            True if state was deleted successfully
            
        Raises:
            StatePersistenceError: If deletion fails
        """
        try:
            # Delegate to implementation
            return await self._delete_state_impl(task_id)
            
        except Exception as e:
            error_msg = f"Error deleting state: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="delete",
                task_id=task_id,
                cause=e
            )
    
    async def list_states(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """List available states.
        
        This method delegates to the implementation-specific _list_states_impl method.
        
        Args:
            filter_criteria: Optional criteria to filter by
            
        Returns:
            List of state metadata dictionaries
            
        Raises:
            StatePersistenceError: If listing fails
        """
        try:
            # Delegate to implementation
            return await self._list_states_impl(filter_criteria)
            
        except Exception as e:
            error_msg = f"Error listing states: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="list",
                cause=e
            )
    
    # Implementation-specific methods to be overridden
    
    async def _save_state_impl(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Implementation-specific method to save state.
        
        Args:
            state: Agent state to save
            metadata: Optional metadata to save with the state
            
        Returns:
            True if state was saved successfully
        """
        raise NotImplementedError("_save_state_impl must be implemented by subclasses")
    
    async def _load_state_impl(
        self,
        task_id: str
    ) -> Optional[AgentState]:
        """Implementation-specific method to load state.
        
        Args:
            task_id: Task ID to load state for
            
        Returns:
            Loaded state or None if not found
        """
        raise NotImplementedError("_load_state_impl must be implemented by subclasses")
    
    async def _delete_state_impl(
        self,
        task_id: str
    ) -> bool:
        """Implementation-specific method to delete state.
        
        Args:
            task_id: Task ID to delete state for
            
        Returns:
            True if state was deleted successfully
        """
        raise NotImplementedError("_delete_state_impl must be implemented by subclasses")
    
    async def _list_states_impl(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Implementation-specific method to list states.
        
        Args:
            filter_criteria: Optional criteria to filter by
            
        Returns:
            List of state metadata dictionaries
        """
        raise NotImplementedError("_list_states_impl must be implemented by subclasses") 