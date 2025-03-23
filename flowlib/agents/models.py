"""Agent system models.

This module defines the core models used by the agent system for 
planning, reflection, and state management.
"""

from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field
import time

class MemoryItem(BaseModel):
    """Structured memory item for agent's memory system."""
    key: str = Field(..., description="Descriptive key for this memory item (e.g., 'user_name', 'user_preference')")
    value: str = Field(..., description="The actual information to remember")
    relevant_keys: List[str] = Field(default_factory=list, description="Related memory keys this might be connected to")
    importance: float = Field(0.7, ge=0.0, le=1.0, description="Importance score (0-1)")
    source: str = Field("reflection", description="Source of the memory")
    context: Optional[str] = Field(None, description="Additional context about this memory item")

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

class AgentState(BaseModel):
    """State of the agent during execution."""
    task_description: Optional[str] = Field(None, description="Description of the agent's task")
    is_complete: bool = Field(False, description="Whether the task is complete")
    completion_reason: Optional[str] = Field(None, description="Reason for task completion")
    progress: int = Field(0, ge=0, le=100, description="Estimated progress toward task completion (0-100)")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of flow executions")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during execution")
    last_response: Optional[str] = Field(None, description="Last response generated by the agent")
    
    class Config:
        arbitrary_types_allowed = True

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = Field("agent", description="Name of the agent")
    provider_name: str = Field("llamacpp", description="Name of LLM provider to use")
    planner_model: str = Field("default", description="Model to use for planning")
    input_generator_model: str = Field("default", description="Model to use for input generation")
    reflection_model: str = Field("default", description="Model to use for reflection")
    working_memory: str = Field("memory-cache", description="Provider to use for working memory")
    short_term_memory: str = Field("memory-cache", description="Provider to use for short-term memory")
    long_term_memory: str = Field("chroma", description="Provider to use for long-term memory")
    memory_cleanup_interval: int = Field(300, description="Seconds between memory cleanup operations")
    max_execution_history: int = Field(100, description="Maximum execution history items to keep")
    max_retries: int = Field(3, description="Maximum number of retries for LLM calls")
    default_system_prompt: str = Field("", description="Default system prompt for the agent")
    stop_on_error: bool = Field(False, description="Whether to stop execution on error")

class FlowDescription(BaseModel):
    """Description of a flow for agent planning."""
    name: str = Field(..., description="Name of the flow")
    input_schema: Optional[str] = Field(None, description="Input schema of the flow")
    output_schema: Optional[str] = Field(None, description="Output schema of the flow")
    description: Optional[str] = Field(None, description="Description of the flow")
    
    class Config:
        arbitrary_types_allowed = True