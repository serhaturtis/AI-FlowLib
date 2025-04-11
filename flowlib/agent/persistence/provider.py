"""
Provider-based state persistence.

This module provides a provider-based implementation of the state persister
interface, allowing for flexible storage backends.
"""

import logging
from typing import Dict, List, Optional

from ..core.errors import StatePersistenceError
from ..models.state import AgentState
from .base import BaseStatePersister

logger = logging.getLogger(__name__)


class ProviderStatePersister(BaseStatePersister):
    """Provider-based implementation of state persistence.
    
    Uses a registered provider to store and retrieve agent states.
    """
    
    def __init__(self, provider_name: str):
        """Initialize provider state persister.
        
        Args:
            provider_name: Name of the provider to use
        """
        super().__init__("provider_state_persister")
        self.provider_name = provider_name
        self.provider = None
    
    async def _initialize_impl(self) -> None:
        """Initialize persister by getting the provider from registry."""
        try:
            from ..core.registry import get_provider
            self.provider = get_provider(self.provider_name)
            if not self.provider:
                raise StatePersistenceError(
                    message=f"Provider not found: {self.provider_name}",
                    operation="initialize"
                )
            logger.debug(f"Using provider: {self.provider_name}")
        except Exception as e:
            error_msg = f"Error initializing provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="initialize",
                cause=e
            )
    
    async def _save_state_impl(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Save agent state using provider.
        
        Args:
            state: Agent state to save
            metadata: Optional metadata to save with the state
            
        Returns:
            True if state was saved successfully
        """
        try:
            # Convert state to dict
            state_dict = state.model_dump()
            
            # Save state
            await self.provider.save_state(state_dict, metadata)
            
            logger.debug(f"State saved using provider: {state.task_id}")
            return True
            
        except Exception as e:
            error_msg = f"Error saving state using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="save",
                task_id=state.task_id,
                cause=e
            )
    
    async def _load_state_impl(
        self,
        task_id: str
    ) -> Optional[AgentState]:
        """Load agent state using provider.
        
        Args:
            task_id: Task ID to load state for
            
        Returns:
            Loaded state or None if not found
        """
        try:
            # Load state from provider
            state_dict = await self.provider.load_state(task_id)
            
            if not state_dict:
                logger.warning(f"State not found: {task_id}")
                return None
            
            # Create AgentState
            return AgentState.model_validate(state_dict)
            
        except Exception as e:
            error_msg = f"Error loading state using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="load",
                task_id=task_id,
                cause=e
            )
    
    async def _delete_state_impl(
        self,
        task_id: str
    ) -> bool:
        """Delete agent state using provider.
        
        Args:
            task_id: Task ID to delete state for
            
        Returns:
            True if state was deleted successfully
        """
        try:
            # Delete state from provider
            await self.provider.delete_state(task_id)
            
            logger.debug(f"State deleted using provider: {task_id}")
            return True
            
        except Exception as e:
            error_msg = f"Error deleting state using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="delete",
                task_id=task_id,
                cause=e
            )
    
    async def _list_states_impl(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """List available states using provider.
        
        Args:
            filter_criteria: Optional criteria to filter by
            
        Returns:
            List of state metadata dictionaries
        """
        try:
            # List states from provider
            return await self.provider.list_states(filter_criteria)
            
        except Exception as e:
            error_msg = f"Error listing states using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="list",
                cause=e
            ) 