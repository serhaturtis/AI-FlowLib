"""Base models for agent system."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class AgentAction(BaseModel):
    """Represents an action an agent can take"""
    flow_name: str = Field(..., description="Name of the flow to execute")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters for the flow")
    reasoning: str = Field(..., description="Reasoning behind choosing this flow")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in this action"
    )

class AgentMemory(BaseModel):
    """Agent's memory storage"""
    short_term: Dict[str, Any] = Field(
        default_factory=dict,
        description="Recent events and context"
    )
    working: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current task-specific information"
    )
    long_term: Dict[str, Any] = Field(
        default_factory=dict,
        description="Persistent knowledge and patterns"
    )

class AgentState(BaseModel):
    """Represents agent's current state"""
    memory: AgentMemory = Field(
        default_factory=AgentMemory,
        description="Agent's memory storage"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current execution context"
    )
    completed_flows: List[str] = Field(
        default_factory=list,
        description="Names of completed flows"
    )
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results and outputs from executed flows"
    )
    current_task: Optional[str] = Field(
        default=None,
        description="Description of the current task"
    ) 