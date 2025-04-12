"""
Reflection interface definitions.

This module defines the protocols and interfaces for reflection operations.
"""

from typing import Dict, Protocol, Any, Optional
from pydantic import BaseModel
from ...flows.results import FlowResult
from ..models.state import AgentState
from .models import ReflectionResult


class ReflectionInterface(Protocol):
    """Interface for reflection operations.
    
    Defines the methods for analyzing results and updating state.
    """
    
    async def reflect(
        self,
        state: AgentState,
        flow_name: str,
        flow_inputs: BaseModel,
        flow_result: FlowResult,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> ReflectionResult:
        """Analyze execution results and update state.
        
        Args:
            state: Current agent state
            flow_name: Name of the flow that was executed
            flow_inputs: Inputs provided to the flow as a Pydantic model
            flow_result: Result from the flow execution
            memory_context: Optional memory context for reflection
            **kwargs: Additional reflection arguments
            
        Returns:
            ReflectionResult with analysis and updated state
        """
        ... 