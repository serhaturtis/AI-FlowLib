"""
Agent state models based on flowlib's Context.

This module provides state tracking for agents by extending flowlib's Context
system rather than reimplementing similar functionality.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from ...core.context import Context

class ExecutionHistoryEntry(BaseModel):
    """Record of a flow execution."""
    cycle: int
    flow_name: str
    inputs: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentState(Context):
    """Agent state extending flowlib's Context.
    
    Instead of reimplementing state tracking, this class extends
    Context to leverage its attribute-based access and state management.
    """
    
    def __init__(
        self, 
        task_description: str = "",
        task_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """Initialize agent state.
        
        Args:
            task_description: Description of the agent's task
            task_id: Unique ID for the task (generated if not provided)
            data: Initial state data
        """
        # Initialize the base Context
        state_data = data or {}
        
        # Add agent-specific fields with defaults
        state_data.update({
            "task_id": task_id or str(uuid4()),
            "task_description": task_description,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "cycle": 0,
            "is_complete": False,
            "completion_reason": None,
            "execution_history": [],
            "errors": [],
            "progress": 0,
            "user_messages": [],
            "system_messages": []
        })
        
        super().__init__(data=state_data)
    
    @property
    def task_id(self) -> str:
        """Get task ID."""
        return self.data.get("task_id", "")
    
    @property
    def task_description(self) -> str:
        """Get task description."""
        return self.data.get("task_description", "")
    
    @task_description.setter
    def task_description(self, value: str) -> None:
        """Set task description.
        
        Args:
            value: New task description
        """
        self.data["task_description"] = value
        self.data["updated_at"] = datetime.now()
    
    @property
    def progress(self) -> int:
        """Get task progress percentage (0-100)."""
        return self.data.get("progress", 0)
    
    @progress.setter
    def progress(self, value: int) -> None:
        """Set task progress percentage.
        
        Args:
            value: Progress percentage (0-100)
        """
        # Ensure progress is between 0 and 100
        progress_value = max(0, min(100, value))
        self.data["progress"] = progress_value
        self.data["updated_at"] = datetime.now()
    
    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.data.get("is_complete", False)
    
    @property
    def completion_reason(self) -> Optional[str]:
        """Get the reason for task completion."""
        return self.data.get("completion_reason")
    
    @property
    def execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.data.get("execution_history", [])
    
    @property
    def errors(self) -> List[str]:
        """Get error messages."""
        return self.data.get("errors", [])
    
    @property
    def cycles(self) -> int:
        """Get the current cycle count."""
        return self.data.get("cycle", 0)
    
    @property
    def user_messages(self) -> List[str]:
        """Get list of user messages in conversation history."""
        if "user_messages" not in self.data:
            self.data["user_messages"] = []
        return self.data["user_messages"]
    
    @property
    def system_messages(self) -> List[str]:
        """Get list of system messages in conversation history."""
        if "system_messages" not in self.data:
            self.data["system_messages"] = []
        return self.data["system_messages"]
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation history.
        
        Args:
            message: User message text
        """
        if "user_messages" not in self.data:
            self.data["user_messages"] = []
        self.data["user_messages"].append(message)
        
        # Add to execution history for tracking
        self.add_to_history(
            flow_name="conversation",
            inputs={"type": "user_message"},
            result={"message": message},
            elapsed_time=0.0
        )
        self.data["updated_at"] = datetime.now()
    
    def add_system_message(self, message: str) -> None:
        """Add a system message to the conversation history.
        
        Args:
            message: System message text
        """
        if "system_messages" not in self.data:
            self.data["system_messages"] = []
        self.data["system_messages"].append(message)
        
        # Add to execution history for tracking
        self.add_to_history(
            flow_name="conversation",
            inputs={"type": "system_message"},
            result={"message": message},
            elapsed_time=0.0
        )
        self.data["updated_at"] = datetime.now()
    
    def increment_cycle(self) -> int:
        """Increment the cycle counter.
        
        Returns:
            New cycle number
        """
        current = self.data.get("cycle", 0)
        self.data["cycle"] = current + 1
        self.data["updated_at"] = datetime.now()
        return self.data["cycle"]
    
    def set_complete(self, reason: str = "Task completed") -> None:
        """Mark the task as complete.
        
        Args:
            reason: Reason for completion
        """
        self.data["is_complete"] = True
        self.data["completion_reason"] = reason
        self.data["updated_at"] = datetime.now()
    
    def add_execution_result(
        self,
        flow_name: str,
        inputs: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Add an execution result to history.
        
        Args:
            flow_name: Name of the executed flow
            inputs: Flow inputs
            result: Flow result
        """
        entry = ExecutionHistoryEntry(
            cycle=self.data.get("cycle", 0),
            flow_name=flow_name,
            inputs=inputs,
            result=result
        )
        
        if "execution_history" not in self.data:
            self.data["execution_history"] = []
            
        self.data["execution_history"].append(entry.model_dump())
        self.data["updated_at"] = datetime.now()
    
    def add_error(self, error: str) -> None:
        """Add an error message.
        
        Args:
            error: Error message
        """
        if "errors" not in self.data:
            self.data["errors"] = []
            
        self.data["errors"].append(error)
        self.data["updated_at"] = datetime.now()
    
    def add_to_history(
        self,
        flow_name: str,
        inputs: Dict[str, Any],
        result: Dict[str, Any],
        elapsed_time: float = 0.0
    ) -> None:
        """Add an entry to the execution history.
        
        This method is called by the agent engine to track execution history.
        
        Args:
            flow_name: Name of the executed flow
            inputs: Flow inputs
            result: Flow result dictionary
            elapsed_time: Execution time in seconds
        """
        # Create a history entry
        history_item = {
            "flow_name": flow_name,
            "inputs": inputs,
            "result": result,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat(),
            "cycle": self.data.get("cycle", 0)
        }
        
        # Ensure the execution_history list exists
        if "execution_history" not in self.data:
            self.data["execution_history"] = []
            
        # Add the history item
        self.data["execution_history"].append(history_item)
        self.data["updated_at"] = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary.
        
        Returns:
            Dictionary representation of agent state
        """
        return self.data.copy()
        
    def model_dump(self) -> Dict[str, Any]:
        """Convert agent state to dictionary for serialization.
        
        This method ensures flow descriptions are properly serialized.
        
        Returns:
            Dictionary representation of agent state
        """
        state_dict = self.data.copy()
        
        # Handle special serialization for flow descriptions
        if "flow_descriptions" in state_dict:
            flow_descriptions = state_dict["flow_descriptions"]
            
            # Convert each dynamic description to static
            static_descriptions = {}
            for name, desc in flow_descriptions.items():
                if hasattr(desc, "to_static"):
                    # Convert to static version
                    static_desc = desc.to_static()
                    static_descriptions[name] = static_desc.model_dump()
                else:
                    # Already static or not a FlowDescription instance
                    static_descriptions[name] = desc
                    
            # Replace with static versions
            state_dict["flow_descriptions"] = static_descriptions
            
        return state_dict
        
    def model_dump_json(self) -> str:
        """Convert agent state to JSON string.
        
        Returns:
            JSON string representation of agent state
        """
        import json
        return json.dumps(self.model_dump())
