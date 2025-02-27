"""Base agent implementation."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from pydantic import BaseModel

from ..models.base import AgentState, AgentAction
from ..tools.flow_tool import FlowTool
from ...core.errors.base import ResourceError, ErrorContext, ValidationError

logger = logging.getLogger(__name__)

class Agent(ABC):
    """Base agent class."""
    
    def __init__(
        self,
        tools: List[FlowTool],
        config: Dict[str, Any]
    ):
        """Initialize agent.
        
        Args:
            tools: List of flow tools available to the agent
            config: Agent configuration
        """
        self.tools = {tool.name: tool for tool in tools}
        self.config = config
        self.state = AgentState()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
    
    async def solve_task(self, task_description: str) -> Dict[str, Any]:
        """Main method to solve a task using available flows.
        
        Args:
            task_description: Description of the task to solve
            
        Returns:
            Dictionary containing task results
        """
        logger.info(f"Starting task: {task_description}")
        self.state.current_task = task_description
        self.state.context["task"] = task_description
        
        try:
            while True:
                # 1. Plan next action
                next_action = await self._plan_next_action()
                
                if next_action is None:
                    logger.info("Task complete")
                    break
                
                # 2. Get the tool for the chosen flow
                flow_name = next_action.flow_name
                if flow_name not in self.tools:
                    raise ResourceError(
                        f"Tool '{flow_name}' not found",
                        ErrorContext.create(available_tools=list(self.tools.keys()))
                    )
                
                flow_tool = self.tools[flow_name]
                
                logger.info(f"Executing flow: {flow_name}")
                logger.info(f"Reasoning: {next_action.reasoning}")
                
                try:
                    # 3. Generate proper inputs if needed
                    input_data = None
                    if flow_tool.requires_inputs:
                        input_data = await self._generate_flow_inputs(flow_tool)
                        if not isinstance(input_data, BaseModel):
                            raise ValidationError(
                                f"Generated inputs for flow {flow_name} must be a Pydantic model",
                                ErrorContext.create(flow_name=flow_name)
                            )
                    
                    # 4. Execute flow
                    result = await flow_tool.execute(input_data)
                    
                    # 5. Update state with serializable results
                    self.state.completed_flows.append(flow_name)
                    if isinstance(result, BaseModel):
                        self.state.artifacts[flow_name] = result.model_dump()
                    else:
                        self.state.artifacts[flow_name] = result
                    
                    # 6. Reflect on result
                    await self._reflect_on_result(flow_name, result)
                    
                except Exception as e:
                    logger.error(f"Error executing flow {flow_name}: {str(e)}")
                    # Store error in artifacts
                    self.state.artifacts[f"{flow_name}_error"] = str(e)
                    # Raise to break the loop
                    raise
            
            return self.state.artifacts
            
        except Exception as e:
            logger.error(f"Task failed: {str(e)}")
            return self.state.artifacts
    
    @abstractmethod
    async def _plan_next_action(self) -> Optional[AgentAction]:
        """Plan next action based on current state.
        
        Returns:
            Next action to take or None if task is complete
        """
        pass
    
    @abstractmethod
    async def _generate_flow_inputs(
        self,
        flow_tool: FlowTool
    ) -> Optional[BaseModel]:
        """Generate valid inputs for a flow.
        
        Args:
            flow_tool: Flow tool to generate inputs for
            
        Returns:
            Input data as a Pydantic model, or None if no inputs required
        """
        pass
    
    @abstractmethod
    async def _reflect_on_result(
        self,
        flow_name: str,
        result: BaseModel
    ) -> None:
        """Update agent's understanding based on flow result.
        
        Args:
            flow_name: Name of the executed flow
            result: Flow execution result as a Pydantic model
        """
        pass 