"""
Reflection interface definitions.

This module defines the protocols and interfaces for reflection operations.
"""

from typing import Dict, Protocol
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
        flow_inputs: Dict[str, str],
        flow_result: FlowResult,
        **kwargs
    ) -> ReflectionResult:
        """Analyze execution results and update state.
        
        Args:
            state: Current agent state
            flow_name: Name of the flow that was executed
            flow_inputs: Inputs that were used for the flow
            flow_result: Result from the flow execution
            **kwargs: Additional reflection arguments
            
        Returns:
            ReflectionResult with analysis and updated state
        """
        ... 