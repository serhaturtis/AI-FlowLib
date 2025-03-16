"""Base agent implementation for flowlib.

This module provides the base Agent class that serves as the foundation
for all agent implementations in flowlib. The Agent class defines the
core functionality for managing flows, planning actions, and executing tasks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type, Union, get_type_hints

from pydantic import BaseModel, create_model, Field

from ..core.models import Context
from ..core.errors import ResourceError, ExecutionError, ErrorContext
from ..core.registry import ResourceRegistry, provider_registry
from ..core.registry.constants import ProviderType
from ..flows import Flow

from .models import AgentState, AgentConfig, PlanningResponse, ReflectionResponse, FlowDescription

logger = logging.getLogger(__name__)

class Agent:
    """Base agent implementation for executing flows with reasoning.
    
    The Agent class provides the core functionality for:
    1. Managing a collection of flows
    2. Planning which flows to execute based on the current state
    3. Generating inputs for flows
    4. Executing flows
    5. Reflecting on flow results
    6. Updating agent state
    
    This class is designed to be extended by specific agent implementations
    that provide concrete implementations of the planning, input generation,
    and reflection methods.
    """
    
    def __init__(
        self, 
        flows: List[Flow], 
        config: Optional[AgentConfig] = None,
        task_description: Optional[str] = None
    ):
        """Initialize the agent with flows and configuration.
        
        Args:
            flows: List of flows the agent can use
            config: Agent configuration
            task_description: Description of the task to perform
        """
        # Set configuration
        if config is None:
            self.config = AgentConfig()
        else:
            self.config = config
            
        # Initialize state
        self.state = AgentState(task_description=task_description)
        self.last_result = None
        
        # Register flows
        self.flows = {}
        for flow in flows:
            self._register_flow(flow)
            
        # Provider will be initialized asynchronously when needed
        self.provider = None
            
    async def initialize_provider(self):
        """Initialize the LLM provider asynchronously.
        
        This method should be called before using the provider.
        """
        if self.provider is None:
            provider_name = self.config.provider_name
            try:
                # Use async get method which handles provider initialization
                self.provider = await provider_registry.get(ProviderType.LLM.value, provider_name)
                if not self.provider:
                    raise ResourceError(
                        f"Provider {provider_name} not found",
                        ErrorContext.create(provider_name=provider_name)
                    )
                logger.info(f"Initialized provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {str(e)}")
                raise ResourceError(
                    f"Failed to initialize provider {provider_name}: {str(e)}",
                    ErrorContext.create(provider_name=provider_name)
                )
            
    def _register_flow(self, flow: Flow) -> None:
        """Register a flow with the agent.
        
        This method registers a flow and extracts its input/output schemas
        for use in planning and execution.
        
        Args:
            flow: Flow to register
        """
        # Initialize schemas as None
        input_schema = None
        output_schema = None
        
        # Try to extract schemas from the flow
        try:
            # First check if the flow has input_schema and output_schema attributes
            if hasattr(flow, 'input_schema') and hasattr(flow, 'output_schema'):
                input_schema = flow.input_schema
                output_schema = flow.output_schema
            # Otherwise try to extract from the first stage
            elif hasattr(flow, 'get_stage') and callable(getattr(flow, 'get_stage')):
                stages = flow.get_stages()
                if stages:
                    first_stage = stages[0]
                    if hasattr(first_stage, 'input_model') and hasattr(first_stage, 'output_model'):
                        input_schema = first_stage.input_model
                        output_schema = first_stage.output_model
        except Exception as e:
            logger.warning(f"Failed to extract schemas from flow {flow.name}: {str(e)}")
        
        # Register the flow with its schemas
        self.flows[flow.name] = FlowDescription(
            name=flow.name,
            flow=flow,
            input_schema=input_schema,
            output_schema=output_schema
        )
        
        logger.info(f"Registered flow: {flow.name}")
    
    async def execute(self) -> AgentState:
        """Execute the agent's task.
        
        This method implements the main task execution loop:
        1. Plan the next action
        2. Generate inputs for the selected flow
        3. Execute the flow
        4. Reflect on the results
        5. Update the agent's state
        6. Repeat until the task is complete
        
        Returns:
            Final agent state after task completion
        """
        logger.info(f"Starting agent execution for task: {self.state.task_description}")
        
        # Ensure provider is initialized
        await self.initialize_provider()
        
        # Main execution loop
        while not self.state.is_complete:
            try:
                # Plan next action
                planning_response = await self._plan_next_action()
                
                # Check if task is complete
                if planning_response.is_complete:
                    self.state.is_complete = True
                    self.state.completion_reason = planning_response.completion_reason
                    logger.info(f"Task complete: {planning_response.completion_reason}")
                    break
                
                # Get selected flow
                flow_name = planning_response.selected_flow
                if flow_name not in self.flows:
                    raise ExecutionError(
                        f"Selected flow '{flow_name}' not found",
                        ErrorContext.create(flow_name=flow_name, available_flows=list(self.flows.keys()))
                    )
                
                flow_desc = self.flows[flow_name]
                
                # Generate inputs for the flow
                flow_inputs = await self._generate_flow_inputs(flow_desc, planning_response)
                
                # Execute the flow
                logger.info(f"Executing flow: {flow_name}")
                context = Context(data=flow_inputs)
                result = await flow_desc.flow.execute(context)
                self.last_result = result
                
                # Reflect on the results
                reflection = await self._reflect_on_result(flow_desc, flow_inputs, result)
                
                # Update state
                self._update_state(planning_response, flow_desc, flow_inputs, result, reflection)
                
            except Exception as e:
                logger.error(f"Error during agent execution: {str(e)}")
                self.state.errors.append(str(e))
                
                # Add to execution history
                self.state.execution_history.append({
                    "step": len(self.state.execution_history) + 1,
                    "action": "error",
                    "error": str(e)
                })
                
                # Check if we should stop on error
                if self.config.stop_on_error:
                    logger.info("Stopping execution due to error")
                    self.state.is_complete = True
                    self.state.completion_reason = f"Error: {str(e)}"
                    break
        
        return self.state
    
    def _update_state(
        self,
        planning: PlanningResponse,
        flow_desc: FlowDescription,
        inputs: Any,
        result: Context,
        reflection: ReflectionResponse
    ) -> None:
        """Update the agent's state after a flow execution.
        
        Args:
            planning: Planning response
            flow_desc: Flow description
            inputs: Flow inputs
            result: Flow execution result
            reflection: Reflection on the result
        """
        # Add to execution history
        self.state.execution_history.append({
            "step": len(self.state.execution_history) + 1,
            "action": "execute_flow",
            "flow": flow_desc.name,
            "reasoning": planning.reasoning,
            "inputs": inputs,
            "outputs": result.data,
            "reflection": reflection.reflection
        })
        
        # Update memory with new information
        if reflection.new_information:
            self.state.memory.extend(reflection.new_information)
            
        # Update task progress
        self.state.progress = reflection.progress
        
        # Check if task is complete based on reflection
        if reflection.is_complete:
            self.state.is_complete = True
            self.state.completion_reason = reflection.completion_reason
    
    async def _plan_next_action(self) -> PlanningResponse:
        """Plan the next action based on the current state.
        
        This method should be implemented by subclasses to provide
        agent-specific planning logic.
        
        Returns:
            Planning response with the selected flow and reasoning
        """
        raise NotImplementedError("Subclasses must implement _plan_next_action")
    
    async def _generate_flow_inputs(
        self, 
        flow_desc: FlowDescription,
        planning: PlanningResponse
    ) -> Any:
        """Generate inputs for a selected flow.
        
        This method should be implemented by subclasses to provide
        agent-specific input generation logic.
        
        Args:
            flow_desc: Description of the selected flow
            planning: Planning response with context
            
        Returns:
            Generated inputs for the flow
        """
        raise NotImplementedError("Subclasses must implement _generate_flow_inputs")
    
    async def _reflect_on_result(
        self,
        flow_desc: FlowDescription,
        inputs: Any,
        result: Context
    ) -> ReflectionResponse:
        """Reflect on flow execution results.
        
        This method should be implemented by subclasses to provide
        agent-specific reflection logic.
        
        Args:
            flow_desc: Description of the executed flow
            inputs: Inputs provided to the flow
            result: Result of the flow execution
            
        Returns:
            Reflection on the results
        """
        raise NotImplementedError("Subclasses must implement _reflect_on_result")
    
    def get_flow_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all registered flows.
        
        Returns:
            List of flow descriptions
        """
        return [
            {
                "name": desc.name,
                "input_schema": desc.input_schema.__name__ if desc.input_schema else "None",
                "output_schema": desc.output_schema.__name__ if desc.output_schema else "None"
            }
            for desc in self.flows.values()
        ]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's current state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            "task": self.state.task_description,
            "is_complete": self.state.is_complete,
            "completion_reason": self.state.completion_reason,
            "progress": self.state.progress,
            "steps_executed": len(self.state.execution_history),
            "memory_items": len(self.state.memory),
            "errors": len(self.state.errors)
        } 