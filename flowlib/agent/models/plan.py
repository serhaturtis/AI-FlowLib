"""Pydantic models for agent execution outcomes.""" # Renamed purpose

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
# Removed import uuid

# Import FlowResult for typing
from ...flows.results import FlowResult
# Import StepReflectionResult for typing
from ..reflection.models import StepReflectionResult

# Removed PlanStep class definition

# --- Model for Plan Execution Outcome --- 
class PlanExecutionOutcome(BaseModel):
    """Represents the outcome of executing a multi-step plan."""
    status: str = Field(..., description="The final status of the plan execution attempt (e.g., SUCCESS, ERROR, NO_ACTION_NEEDED)")
    result: Optional[FlowResult] = Field(None, description="The FlowResult of the last executed step, or a dummy result on failure.")
    error: Optional[str] = Field(None, description="Error message if the plan execution failed.")
    step_reflections: List[StepReflectionResult] = Field(default_factory=list, description="List of reflections collected after each step.")

    class Config:
        extra = "forbid"
        # Allow FlowResult type
        arbitrary_types_allowed = True