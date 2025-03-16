"""Agent system models.

This module defines the core models used by the agent system for 
planning, reflection, and state management.
"""

from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field

class PlanningResponse(BaseModel):
    """Response from planning the next action."""
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    selected_flow: str = Field(..., description="Name of the selected flow to execute")
    is_complete: bool = Field(False, description="Whether the task is complete")
    completion_reason: Optional[str] = Field(None, description="Reason for task completion if complete")

class ReflectionResponse(BaseModel):
    """Response from reflecting on flow execution results."""
    reflection: str = Field(..., description="Detailed reflection on the flow execution")
    progress: int = Field(0, ge=0, le=100, description="Estimated progress toward task completion (0-100)")
    is_complete: bool = Field(False, description="Whether the task is complete")
    completion_reason: Optional[str] = Field(None, description="Reason for task completion if complete")
    new_information: List[str] = Field(default_factory=list, description="New information to remember")

class FlowDescription(BaseModel):
    """Description of a flow for agent planning."""
    name: str = Field(..., description="Name of the flow")
    flow: Any = Field(..., description="Reference to the flow object")
    input_schema: Optional[Any] = Field(None, description="Input schema of the flow (Python class)")
    output_schema: Optional[Any] = Field(None, description="Output schema of the flow (Python class)")
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class AgentState(BaseModel):
    """State of the agent during execution."""
    task_description: Optional[str] = Field(None, description="Description of the agent's task")
    is_complete: bool = Field(False, description="Whether the task is complete")
    completion_reason: Optional[str] = Field(None, description="Reason for task completion")
    progress: int = Field(0, ge=0, le=100, description="Estimated progress toward task completion (0-100)")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of flow executions")
    memory: List[str] = Field(default_factory=list, description="Agent's memory items")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during execution")
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    provider_name: str = Field("llamacpp", description="Name of LLM provider to use")
    planner_model: str = Field("default", description="Model to use for planning")
    input_generator_model: str = Field("default", description="Model to use for input generation")
    reflection_model: str = Field("default", description="Model to use for reflection")
    max_retries: int = Field(3, description="Maximum number of retries for LLM calls")
    default_system_prompt: str = Field("", description="Default system prompt for the agent")
    stop_on_error: bool = Field(False, description="Whether to stop execution on error") 