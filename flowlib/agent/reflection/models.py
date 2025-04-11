from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ...flows.results import FlowResult


class ReflectionResult(BaseModel):
    """Model for structured reflection results."""
    reflection: str = Field(description="A detailed analysis of what happened and why")
    progress: int = Field(default=0, description="An estimate of overall task progress (0-100)")
    is_complete: bool = Field(default=False, description="Whether the task is complete")
    completion_reason: Optional[str] = Field(default=None, description="If is_complete is true, the reason the task is complete")
    insights: Optional[List[str]] = Field(default=None, description="Key insights or lessons learned from this execution")


class ReflectionInput(BaseModel):
    """Standardized input model for reflection process."""
    task_description: str = Field(description="Description of the overall task")
    flow_name: str = Field(description="Name of the executed flow")
    flow_status: str = Field(description="Status of the flow execution")
    flow_result: FlowResult = Field(description="Result from the flow execution as a FlowResult model")
    flow_inputs: BaseModel = Field(description="Inputs provided to the flow as a Pydantic model")
    state_summary: str = Field(description="Summary of current state")
    execution_history_text: str = Field(description="Formatted execution history")
    planning_rationale: str = Field(description="Rationale from the planning phase")
    cycle: int = Field(description="Current execution cycle number")
    progress: int = Field(default=0, description="Current progress percentage (0-100)")
    memory_context: Optional[str] = Field(default=None, description="Memory context for this reflection")