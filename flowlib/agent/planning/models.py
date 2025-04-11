"""
Planning models for the agent system.

This module defines the data models used in planning operations, including:
- Task context models for execution state and history
- Planning-specific models for results, validation, and explanations
"""

from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from pydantic import BaseModel, Field

# Import the original FlowMetadata to avoid duplication
from ...flows.metadata import FlowMetadata

# Use StageRegistry for flow operations
from ...flows.registry import StageRegistry

# --- Task Context Models ---

class TaskState(BaseModel):
    """Current state of a task.
    
    Attributes:
        task_id: Unique identifier for the task
        task_description: Description of the task
        cycle: Current execution cycle number
        progress: Task progress percentage (0-100)
        is_complete: Whether the task is complete
        completion_reason: Reason for task completion if complete
        last_updated: Timestamp of last state update
    """
    task_id: str = Field(..., description="Unique identifier for the task")
    task_description: str = Field(..., description="Description of the task")
    cycle: int = Field(0, description="Current execution cycle number")
    progress: int = Field(0, description="Task progress percentage (0-100)")
    is_complete: bool = Field(False, description="Whether the task is complete")
    completion_reason: Optional[str] = Field(None, description="Reason for task completion if complete")
    last_updated: datetime = Field(default_factory=datetime.now, description="Timestamp of last state update")


class MessageHistory(BaseModel):
    """History of messages for a task.
    
    Attributes:
        user_messages: List of user messages with timestamps
        system_messages: List of system messages with timestamps
    """
    class Message(BaseModel):
        """A message with timestamp and content."""
        timestamp: datetime = Field(default_factory=datetime.now)
        content: str = Field(..., description="Message content")
        
    user_messages: List[Message] = Field(default_factory=list, description="List of user messages")
    system_messages: List[Message] = Field(default_factory=list, description="List of system messages")


class ErrorLog(BaseModel):
    """Log of errors for a task.
    
    Attributes:
        errors: List of error entries with timestamps
    """
    class ErrorEntry(BaseModel):
        """An error entry with timestamp, message, and source."""
        timestamp: datetime = Field(default_factory=datetime.now)
        message: str = Field(..., description="Error message")
        source: str = Field(..., description="Component that raised the error")
        traceback: Optional[str] = Field(None, description="Error traceback if available")
        
    errors: List[ErrorEntry] = Field(default_factory=list, description="List of error entries")


class ExecutionContext(BaseModel):
    """Complete execution context for a task.
    
    This model combines all context information needed for planning:
    - Current task state
    - Message history
    - Error log
    
    Attributes:
        state: Current state of the task
        messages: History of messages for the task
        errors: Log of errors for the task
    """
    state: TaskState = Field(..., description="Current state of the task")
    messages: MessageHistory = Field(default_factory=MessageHistory, description="History of messages for the task")
    errors: ErrorLog = Field(default_factory=ErrorLog, description="Log of errors for the task")


# --- Planning-Specific Models ---

class PlanningExplanation(BaseModel):
    """Human-readable explanation of a plan.
    
    Attributes:
        explanation: Text explaining the planning decisions
        rationale: Optional rationale for the decisions
        decision_factors: Factors that influenced the decision
    """
    explanation: str = Field(..., description="Text explaining the planning decisions")
    rationale: str = Field(None, description="Rationale for the decisions")
    decision_factors: List[str] = Field(default_factory=list, description="Factors that influenced the decision")

class PlanningResult(BaseModel):
    """Result of a planning operation.
    
    Attributes:
        selected_flow: Name of the selected flow
        inputs: Inputs for the selected flow
        metadata: Metadata about the planning decision
    """
    selected_flow: str = Field(..., description="Name of the selected flow")
    reasoning: PlanningExplanation = Field(..., description="Reasoning behind the planning decision")


class PlanningValidation(BaseModel):
    """Result of plan validation.
    
    Attributes:
        is_valid: Whether the plan is valid
        errors: List of validation errors if any
    """
    is_valid: bool = Field(..., description="Whether the plan is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors if any")


