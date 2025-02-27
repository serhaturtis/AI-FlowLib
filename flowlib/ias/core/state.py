"""State management system for the Integrated Agent System."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type
from uuid import UUID

from ..models.base import DomainState, StateMetadata, StateType, ValidationResult
from ..utils.errors import (
    StateError, StateLockError, StateNotFoundError,
    StateValidationError, StateVersionError
)

logger = logging.getLogger(__name__)

class StateManager:
    """Manages state for all domains in the system."""
    
    def __init__(self):
        """Initialize state manager."""
        self._states: Dict[UUID, DomainState] = {}
        self._locks: Dict[UUID, asyncio.Lock] = {}
        self._history: Dict[UUID, List[DomainState]] = {}
        self._subscribers: Dict[str, Set[callable]] = {}
        
    async def create_state(
        self,
        domain: str,
        state_type: Type[DomainState],
        data: Dict[str, Any]
    ) -> DomainState:
        """Create a new state object.
        
        Args:
            domain: Domain creating the state
            state_type: Type of state to create
            data: Initial state data
            
        Returns:
            Created state object
            
        Raises:
            StateError: If state creation fails
        """
        try:
            metadata = StateMetadata(
                created_by=domain,
                updated_by=domain
            )
            
            state = state_type(
                type=state_type,
                metadata=metadata,
                data=data
            )
            
            self._states[state.id] = state
            self._history[state.id] = [state]
            self._locks[state.id] = asyncio.Lock()
            
            await self._notify_subscribers(domain, StateType.CREATED, state)
            return state
            
        except Exception as e:
            raise StateError(f"Failed to create state: {str(e)}") from e
            
    async def get_state(self, state_id: UUID) -> DomainState:
        """Get a state object by ID.
        
        Args:
            state_id: ID of state to get
            
        Returns:
            State object
            
        Raises:
            StateNotFoundError: If state doesn't exist
        """
        if state_id not in self._states:
            raise StateNotFoundError(f"State {state_id} not found")
            
        return self._states[state_id]
        
    async def update_state(
        self,
        domain: str,
        state_id: UUID,
        data: Dict[str, Any],
        expected_version: Optional[int] = None
    ) -> DomainState:
        """Update an existing state object.
        
        Args:
            domain: Domain updating the state
            state_id: ID of state to update
            data: New state data
            expected_version: Expected current version (for optimistic locking)
            
        Returns:
            Updated state object
            
        Raises:
            StateNotFoundError: If state doesn't exist
            StateLockError: If state is locked by another domain
            StateVersionError: If version doesn't match expected
        """
        if state_id not in self._states:
            raise StateNotFoundError(f"State {state_id} not found")
            
        state = self._states[state_id]
        lock = self._locks[state_id]
        
        if state.metadata.is_locked and state.metadata.lock_holder != domain:
            raise StateLockError(
                f"State {state_id} is locked by {state.metadata.lock_holder}"
            )
            
        if expected_version and state.metadata.version != expected_version:
            raise StateVersionError(
                f"State {state_id} version mismatch: "
                f"expected {expected_version}, got {state.metadata.version}"
            )
            
        async with lock:
            # Create new version
            new_state = state.model_copy(deep=True)
            new_state.data = data
            new_state.metadata.version += 1
            new_state.metadata.updated_at = datetime.utcnow()
            new_state.metadata.updated_by = domain
            
            # Update state and history
            self._states[state_id] = new_state
            self._history[state_id].append(new_state)
            
            await self._notify_subscribers(domain, StateType.UPDATED, new_state)
            return new_state
            
    async def validate_state(
        self,
        domain: str,
        state_id: UUID,
        validation: ValidationResult
    ) -> DomainState:
        """Add validation result to state.
        
        Args:
            domain: Domain performing validation
            state_id: ID of state to validate
            validation: Validation result
            
        Returns:
            Updated state object
            
        Raises:
            StateNotFoundError: If state doesn't exist
            StateValidationError: If validation fails
        """
        if state_id not in self._states:
            raise StateNotFoundError(f"State {state_id} not found")
            
        state = self._states[state_id]
        lock = self._locks[state_id]
        
        async with lock:
            try:
                # Create new version with validation
                new_state = state.model_copy(deep=True)
                new_state.metadata.validations.append(validation)
                new_state.metadata.version += 1
                new_state.metadata.updated_at = datetime.utcnow()
                new_state.metadata.updated_by = domain
                
                # Update state and history
                self._states[state_id] = new_state
                self._history[state_id].append(new_state)
                
                event_type = (
                    StateType.VALIDATED if validation.is_valid
                    else StateType.INVALIDATED
                )
                await self._notify_subscribers(domain, event_type, new_state)
                return new_state
                
            except Exception as e:
                raise StateValidationError(
                    f"Failed to validate state {state_id}: {str(e)}"
                ) from e
                
    async def lock_state(
        self,
        domain: str,
        state_id: UUID
    ) -> DomainState:
        """Lock state for exclusive access.
        
        Args:
            domain: Domain requesting lock
            state_id: ID of state to lock
            
        Returns:
            Locked state object
            
        Raises:
            StateNotFoundError: If state doesn't exist
            StateLockError: If state is already locked
        """
        if state_id not in self._states:
            raise StateNotFoundError(f"State {state_id} not found")
            
        state = self._states[state_id]
        lock = self._locks[state_id]
        
        if state.metadata.is_locked:
            raise StateLockError(
                f"State {state_id} is already locked by {state.metadata.lock_holder}"
            )
            
        async with lock:
            # Create new version with lock
            new_state = state.model_copy(deep=True)
            new_state.metadata.is_locked = True
            new_state.metadata.lock_holder = domain
            new_state.metadata.version += 1
            new_state.metadata.updated_at = datetime.utcnow()
            new_state.metadata.updated_by = domain
            
            # Update state and history
            self._states[state_id] = new_state
            self._history[state_id].append(new_state)
            
            await self._notify_subscribers(domain, StateType.LOCKED, new_state)
            return new_state
            
    async def unlock_state(
        self,
        domain: str,
        state_id: UUID
    ) -> DomainState:
        """Unlock state.
        
        Args:
            domain: Domain releasing lock
            state_id: ID of state to unlock
            
        Returns:
            Unlocked state object
            
        Raises:
            StateNotFoundError: If state doesn't exist
            StateLockError: If state is not locked by the domain
        """
        if state_id not in self._states:
            raise StateNotFoundError(f"State {state_id} not found")
            
        state = self._states[state_id]
        lock = self._locks[state_id]
        
        if not state.metadata.is_locked:
            return state
            
        if state.metadata.lock_holder != domain:
            raise StateLockError(
                f"State {state_id} is locked by {state.metadata.lock_holder}, "
                f"not {domain}"
            )
            
        async with lock:
            # Create new version without lock
            new_state = state.model_copy(deep=True)
            new_state.metadata.is_locked = False
            new_state.metadata.lock_holder = None
            new_state.metadata.version += 1
            new_state.metadata.updated_at = datetime.utcnow()
            new_state.metadata.updated_by = domain
            
            # Update state and history
            self._states[state_id] = new_state
            self._history[state_id].append(new_state)
            
            await self._notify_subscribers(domain, StateType.UNLOCKED, new_state)
            return new_state
            
    def subscribe(
        self,
        domain: str,
        callback: callable
    ) -> None:
        """Subscribe to state changes for a domain.
        
        Args:
            domain: Domain to subscribe to
            callback: Callback function for state changes
        """
        if domain not in self._subscribers:
            self._subscribers[domain] = set()
        self._subscribers[domain].add(callback)
        
    def unsubscribe(
        self,
        domain: str,
        callback: callable
    ) -> None:
        """Unsubscribe from state changes for a domain.
        
        Args:
            domain: Domain to unsubscribe from
            callback: Callback function to remove
        """
        if domain in self._subscribers:
            self._subscribers[domain].discard(callback)
            
    async def _notify_subscribers(
        self,
        domain: str,
        event_type: StateType,
        state: DomainState
    ) -> None:
        """Notify subscribers of state changes.
        
        Args:
            domain: Domain that changed the state
            event_type: Type of state change
            state: Changed state object
        """
        if domain in self._subscribers:
            for callback in self._subscribers[domain]:
                try:
                    await callback(event_type, state)
                except Exception as e:
                    logger.error(
                        f"Error in state change callback for domain {domain}: {str(e)}"
                    ) 