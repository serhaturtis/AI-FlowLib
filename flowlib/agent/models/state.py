"""
Agent state models based on flowlib's Context.

This module provides state tracking for agents by extending flowlib's Context
system rather than reimplementing similar functionality.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from ...core.context import Context
from ..planning.models import Plan

class ExecutionHistoryEntry(BaseModel):
    """Record of a flow execution."""
    cycle: int = Field(..., description="Agent cycle number during execution")
    flow_name: str = Field(..., description="Name of the executed flow")
    inputs: Dict[str, Any] = Field(..., description="Inputs provided to the flow")
    result: Dict[str, Any] = Field(..., description="Result dictionary returned by the flow")
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentStateModel(BaseModel):
    """Pydantic model representing the complete state of an agent task."""
    task_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the task")
    task_description: str = Field(..., description="Description of the overall task")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the state was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the state was last updated")
    cycle: int = Field(0, description="Current execution cycle number")
    is_complete: bool = Field(False, description="Whether the task is considered complete")
    completion_reason: Optional[str] = Field(None, description="Reason for task completion, if applicable")
    progress: int = Field(0, ge=0, le=100, description="Task progress percentage (0-100)")
    user_messages: List[str] = Field(default_factory=list, description="Chronological list of user messages")
    system_messages: List[str] = Field(default_factory=list, description="Chronological list of system/assistant messages")
    execution_history: List[ExecutionHistoryEntry] = Field(default_factory=list, description="Record of executed flows and their results")
    errors: List[str] = Field(default_factory=list, description="Log of errors encountered during execution")
    # Add any other state variables needed, e.g., specific data gathered
    current_entities: Dict[str, Any] = Field(default_factory=dict)
    # --- Fields for Multi-Step Plan Execution ---
    current_plan: Optional['Plan'] = Field(None, description="The currently active multi-step plan.")
    current_step_index: int = Field(0, description="The index of the next step to execute in the current plan.")
    # --------------------------------------------

    # Model config can be added if needed, e.g., for validation
    # class Config:
    #     validate_assignment = True

class AgentState(Context[AgentStateModel]):
    """Agent state providing methods to interact with the underlying AgentStateModel.
    
    This class extends Context and holds an AgentStateModel instance.
    It provides convenient properties and methods for accessing and modifying the state,
    while ensuring the underlying data is managed via the model.
    """
    
    def __init__(
        self, 
        task_description: str = "",
        task_id: Optional[str] = None,
        # Allow initializing from an existing model instance or dictionary (for loading)
        initial_state_data: Optional[Union[AgentStateModel, Dict[str, Any]]] = None
    ):
        """Initialize agent state using AgentStateModel.
        
        Args:
            task_description: Initial description of the agent's task (used if no initial_state_data).
            task_id: Optional unique ID (used if no initial_state_data or if data lacks it).
            initial_state_data: Optional existing AgentStateModel instance or a dictionary 
                                 to load state from (e.g., from persistence).
        """
        state_model: AgentStateModel
        
        if isinstance(initial_state_data, AgentStateModel):
            state_model = initial_state_data
            # Override task_id/description if explicitly provided
            if task_id:
                state_model.task_id = task_id
            if task_description:
                 state_model.task_description = task_description
            state_model.updated_at = datetime.now() # Mark as updated on load/init

        elif isinstance(initial_state_data, dict):
            # Attempt to load from dictionary
            try:
                # Ensure provided task_id/description are included if missing in dict
                if "task_id" not in initial_state_data and task_id:
                     initial_state_data["task_id"] = task_id
                if "task_description" not in initial_state_data and task_description:
                     initial_state_data["task_description"] = task_description
                
                # Add/update timestamp on load
                initial_state_data["updated_at"] = datetime.now()
                
                state_model = AgentStateModel(**initial_state_data)
            except Exception as e:
                # Log error and fallback to default state?
                # Adhering to no fallbacks: raise error.
                raise ValueError(f"Failed to initialize AgentState from dictionary: {e}")
        else:
            # Create a new default state model
            state_model = AgentStateModel(
                task_id=task_id or str(uuid4()), # Generate UUID if no task_id given
                task_description=task_description or "Default Task" # Provide a default description
            )
        
        # Initialize the base Context with the AgentStateModel instance
        super().__init__(data=state_model)

    # --- Properties (Accessors) --- 
    # These now primarily access the internal _data dict, which is a dump of the model
    
    @property
    def task_id(self) -> str:
        return self._data.get("task_id", "")
    
    @property
    def task_description(self) -> str:
        return self._data.get("task_description", "")
    
    @property
    def progress(self) -> int:
        return self._data.get("progress", 0)
    
    @property
    def is_complete(self) -> bool:
        return self._data.get("is_complete", False)
    
    @property
    def completion_reason(self) -> Optional[str]:
        return self._data.get("completion_reason")
    
    @property
    def execution_history(self) -> List[ExecutionHistoryEntry]:
        # Note: This returns the raw list of dicts from the model dump.
        # If actual ExecutionHistoryEntry instances are needed, use self.as_model().execution_history
        raw_history = self._data.get("execution_history", [])
        # Attempt to reconstruct models for type hint consistency, though expensive
        try:
            return [ExecutionHistoryEntry(**entry) for entry in raw_history]
        except Exception:
             # Log warning? Return raw list if reconstruction fails
             return raw_history 
    
    @property
    def errors(self) -> List[str]:
        return self._data.get("errors", [])
    
    @property
    def cycles(self) -> int:
        return self._data.get("cycle", 0)
    
    @property
    def user_messages(self) -> List[str]:
        return self._data.get("user_messages", [])
    
    @property
    def system_messages(self) -> List[str]:
        return self._data.get("system_messages", [])
        
    # --- Mutator Methods --- 
    # These methods modify the internal _data dictionary directly
    # and update the timestamp. Direct modification bypasses Pydantic 
    # validation unless we reconstruct the model on each change (expensive).
    
    def _update_timestamp(self):
        """Helper to update the timestamp in the internal data."""
        self._data["updated_at"] = datetime.now()
        
    @task_description.setter
    def task_description(self, value: str) -> None:
        self._data["task_description"] = value
        self._update_timestamp()

    @progress.setter
    def progress(self, value: int) -> None:
        # Apply validation logic before setting
        progress_value = max(0, min(100, value))
        self._data["progress"] = progress_value
        self._update_timestamp()
        
    def add_user_message(self, message: str) -> None:
        # Ensure list exists
        if "user_messages" not in self._data:
            self._data["user_messages"] = []
        self._data["user_messages"].append(message)
        # History tracking might need adjustment if add_to_history changes
        self.add_to_history(
            flow_name="conversation",
            inputs={"type": "user_message"},
            result={"message": message} 
            # elapsed_time is missing in add_to_history signature now?
        )
        self._update_timestamp()
    
    def add_system_message(self, message: str) -> None:
        if "system_messages" not in self._data:
            self._data["system_messages"] = []
        self._data["system_messages"].append(message)
        self.add_to_history(
            flow_name="conversation",
            inputs={"type": "system_message"},
            result={"message": message}
        )
        self._update_timestamp()
    
    def increment_cycle(self) -> int:
        current = self._data.get("cycle", 0)
        self._data["cycle"] = current + 1
        self._update_timestamp()
        return self._data["cycle"]
    
    def set_complete(self, reason: str = "Task completed") -> None:
        self._data["is_complete"] = True
        self._data["completion_reason"] = reason
        self._update_timestamp()
    
    def add_execution_result(
        self,
        flow_name: str,
        inputs: Dict[str, Any],
        result: Dict[str, Any] # Result here is likely already a dict from FlowResult
    ) -> None:
        # Use the current cycle count from the internal data
        current_cycle = self._data.get("cycle", 0)
        entry = ExecutionHistoryEntry(
            cycle=current_cycle,
            flow_name=flow_name,
            inputs=inputs,
            result=result # Assuming result is already a dict
        )
        if "execution_history" not in self._data:
            self._data["execution_history"] = []
        # Append the dictionary representation of the entry
        self._data["execution_history"].append(entry.model_dump())
        self._update_timestamp()
    
    def add_error(self, error: str) -> None:
        if "errors" not in self._data:
            self._data["errors"] = []
        self._data["errors"].append(error)
        self._update_timestamp()

    # Simplified add_to_history based on how add_user/system_message calls it
    # The original had elapsed_time, needs confirmation if still needed.
    def add_to_history(
        self,
        flow_name: str,
        inputs: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
         """Adds an entry to execution history (simplified)."""
         # This method is essentially the same as add_execution_result now.
         # Consider consolidating or refactoring later.
         self.add_execution_result(flow_name, inputs, result)
         # Note: add_execution_result already calls _update_timestamp

    # Keep existing convenience methods if they operate on the internal dict
    def to_dict(self) -> Dict[str, Any]:
        """Return the internal state dictionary."""
        # Reconstruct model and dump? Or just return _data?
        # Returning _data is faster but might be slightly stale if setters missed something.
        # Let's return _data for now.
        return self._data.copy()
        
    def model_dump(self) -> Dict[str, Any]:
        """Dump the current state as a dictionary (consistent with Pydantic)."""
        # Reconstruct the model from _data and dump it to ensure consistency
        current_model = self.as_model()
        if current_model:
             return current_model.model_dump()
        else:
             # Should not happen if initialized correctly
             return self._data.copy() 
        
    def model_dump_json(self) -> str:
        """Dump the current state as a JSON string."""
        current_model = self.as_model()
        if current_model:
            return current_model.model_dump_json()
        else:
            import json
            return json.dumps(self._data) # Fallback, less safe

    # __str__ representation can use the base Context one or be specific
    def __str__(self) -> str:
        return f"AgentState(task_id='{self.task_id}', cycle={self.cycles}, complete={self.is_complete})"
