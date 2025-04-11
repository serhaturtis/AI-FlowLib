"""
Planning interface definitions.

This module defines the interfaces used for planning operations, ensuring
consistent method signatures across different planning implementations.
"""

from typing import Dict, Any, Optional, Protocol, List, Type
from ..planning.models import (
    PlanningResult,
    PlanningValidation,
    ExecutionContext
)

class PlanningInterface(Protocol):
    """Interface for planning operations.
    
    This interface defines the methods that must be implemented by any
    planning component used in the agent system.
    
    Implementations must ensure:
    1. All methods handle errors by raising appropriate exceptions
    2. External dependencies like registries are properly documented
    3. Model providers are properly configured
    """
    
    async def plan(self, context: ExecutionContext) -> PlanningResult:
        """Generate a plan based on the current context and available flows.
        
        Args:
            context: Current execution context with state, history, etc.
            
        Returns:
            PlanningResult with selected flow and metadata
            
        Raises:
            PlanningError: If planning fails for any reason
            NotInitializedError: If the planner is not properly initialized
        """
        ...
    
    async def validate_plan(self, plan: PlanningResult) -> PlanningValidation:
        """Validate a generated plan against available flows.
        
        Args:
            plan: The plan to validate
            
        Returns:
            PlanningValidation indicating if the plan is valid
            
        Raises:
            PlanningError: If validation fails for any reason
        """
        ...
    
    async def generate_inputs(
        self,
        state: 'AgentState',
        flow_name: str,
        planning_result: Dict[str, Any],
        memory_context: str,
        flow: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate inputs for a flow based on state and planning result.
        
        This method requires access to a flow registry and model provider,
        which should be set up in the implementation.
        
        Args:
            state: Agent state object containing task details and history
            flow_name: Name of the flow to generate inputs for
            planning_result: Result from planning phase
            memory_context: Memory context for this cycle
            flow: Flow instance (optional)
            
        Returns:
            Generated inputs for the flow as a dictionary
            
        Raises:
            PlanningError: If input generation fails
            NotInitializedError: If the planner is not properly initialized
        """
        ... 