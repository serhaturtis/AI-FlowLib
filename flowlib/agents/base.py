"""Base agent implementation with enhanced memory and discovery.

This module provides the core Agent class with improved planning, memory,
and dynamic flow discovery capabilities.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Type, Union, get_type_hints

from ..core.models import Context
from ..core.errors import ResourceError, ExecutionError, ErrorContext
from ..core.registry import provider_registry
from ..core.registry.constants import ProviderType
from ..flows import Flow

from .models import AgentState, AgentConfig, PlanningResponse, ReflectionResponse, FlowDescription
from .memory_manager import MemoryManager, MemoryContext, MemoryItem
from .discovery import FlowDiscovery

logger = logging.getLogger(__name__)


class Agent:
    """Enhanced base agent implementation with memory and flow discovery.
    
    This agent provides:
    1. Dynamic flow discovery at runtime
    2. Working, short-term, and long-term memory using the memory manager
    3. LLM-powered planning, input generation, and reflection
    4. Comprehensive state tracking and error handling
    """
    
    def __init__(
        self, 
        flows: List[Flow] = None, 
        config: Optional[AgentConfig] = None,
        task_description: Optional[str] = None,
        flow_paths: List[str] = ["./flows"]
    ):
        """Initialize the agent with flows and configuration.
        
        Args:
            flows: List of flows the agent can use
            config: Agent configuration
            task_description: Description of the task to perform
            flow_paths: Paths to scan for flow definitions
        """
        # Set configuration
        if config is None:
            self.config = AgentConfig()
        else:
            self.config = config
            
        # Initialize state
        self.state = AgentState(task_description=task_description)
        self.last_result = None
        
        # Initialize flow collections
        self.flows = {}
        self.flow_descriptions = {}
        
        # Register flows
        if flows:
            for flow in flows:
                self._register_flow(flow)
                
        # Create memory manager with configured providers
        self.memory = MemoryManager(
            working_memory_provider=self.config.working_memory, 
            short_term_provider=self.config.short_term_memory,
            long_term_provider=self.config.long_term_memory
        )
        
        # Memory contexts will be created during initialization
        self.base_context = None
        self.execution_context = None
        self.task_context = None
        
        # Initialize flow discovery
        self.flow_discovery = FlowDiscovery(flow_paths)
            
        # Provider will be initialized asynchronously when needed
        self.provider = None
            
    async def initialize(self):
        """Initialize the agent components."""
        # Initialize provider
        await self.initialize_provider()

        # Initialize memory
        await self.memory.initialize()
        
        # Create standard memory contexts
        self.base_context = self.memory.create_context(f"agent:{self.config.name}")
        self.execution_context = self.memory.create_context("execution", self.base_context)
        self.task_context = self.memory.create_context("task", self.base_context)
        
        # Store task description if available
        if self.state.task_description:
            await self.memory.store(
                "description", 
                self.state.task_description, 
                self.task_context,
                importance=0.8
            )

        # Discover flows via the registry and flow discovery
        await self._discover_flows()

        logger.info(f"Initialized agent with {len(self.flows)} flows")

    async def _discover_flows(self):
        """Discover flows from registry and directories."""
        # First check the registry for flows registered via decorators
        from ..flows.registry import stage_registry

        # Get flow names from registry
        registered_flows = stage_registry.get_flows()
        for flow_name in registered_flows:
            if flow_name not in self.flows:
                # Try to get the actual flow instance from registry
                flow_instance = stage_registry.get_flow(flow_name)
                
                if flow_instance:
                    # Register the flow instance with the agent
                    self._register_flow(flow_instance)
                    logger.info(f"Discovered flow from registry: {flow_name}")
                else:
                    # Just note the flow exists if we don't have an instance
                    logger.debug(f"Found flow name in registry but no instance: {flow_name}")
                    self.flow_descriptions[flow_name] = FlowDescription(
                        name=flow_name,
                        input_schema=None,
                        output_schema=None,
                        description=None
                    )
        
        # Then scan directories for additional flows
        discovered_flows = await self.flow_discovery.refresh_flows()
        for name, flow in discovered_flows.items():
            if name not in self.flows:
                self._register_flow(flow)
                logger.info(f"Discovered flow from directory scan: {name}")
            
    async def initialize_provider(self):
        """Initialize the LLM provider asynchronously.
        
        This method should be called before using the provider.
        """
        if self.provider is None:
            provider_name = self.config.provider_name
            try:
                # Use async get method which handles provider initialization
                self.provider = await provider_registry.get(ProviderType.LLM, provider_name)
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
        # Get flow name
        flow_name = getattr(flow, "name", flow.__class__.__name__)
        
        # Store flow
        self.flows[flow_name] = flow
        
        # Extract input/output schemas
        input_schema = None
        output_schema = None
        description = None
        
        # Get from flow attributes if available
        if hasattr(flow, "input_schema"):
            input_schema = str(flow.input_schema)
        
        if hasattr(flow, "output_schema"):
            output_schema = str(flow.output_schema)
            
        if hasattr(flow, "description"):
            description = flow.description
        
        # Store flow description
        self.flow_descriptions[flow_name] = FlowDescription(
            name=flow_name,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description
        )
        
        logger.info(f"Registered flow: {flow_name}")
    
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
        
        # Initialize components if not already initialized
        if not self.provider:
            await self.initialize()
            
        # Create execution context for this run
        execution_id = str(uuid.uuid4())
        run_context = self.memory.create_context(f"run:{execution_id}", self.execution_context)
        
        # Store execution start info
        await self.memory.store(
            "start_info", 
            {
                "task": self.state.task_description,
                "start_time": time.time(),
                "execution_id": execution_id
            },
            run_context,
            ttl=3600
        )
        
        # Always refresh flows at the beginning of execution
        logger.info("Refreshing flows before execution")
        await self._discover_flows()
        
        # Track execution iterations for debugging
        execution_iterations = 0
        
        try:
            # Main execution loop
            while not self.state.is_complete:
                execution_iterations += 1
                step_context = self.memory.create_context(f"step:{execution_iterations}", run_context)
                
                logger.info(f"=== Execution loop iteration {execution_iterations} ===")
                
                try:
                    # Refresh flows to discover any new ones
                    discovered_flows = await self.flow_discovery.refresh_flows()
                    for name, flow in discovered_flows.items():
                        if name not in self.flows:
                            self._register_flow(flow)
                    
                    # Plan next action
                    logger.info(f"Planning next action (iteration {execution_iterations})")
                    planning_response = await self._plan_next_action()
                    
                    # Store planning in memory
                    await self.memory.store(
                        "planning",
                        {
                            "reasoning": planning_response.reasoning,
                            "selected_flow": planning_response.selected_flow,
                            "is_complete": planning_response.is_complete,
                            "completion_reason": planning_response.completion_reason
                        },
                        step_context,
                        ttl=3600,
                        importance=0.6
                    )
                    
                    # Check if task is complete
                    if planning_response.is_complete:
                        self.state.is_complete = True
                        self.state.completion_reason = planning_response.completion_reason
                        logger.info(f"Planning marked task as complete: {planning_response.completion_reason}")
                        break
                    
                    # Get selected flow
                    flow_name = planning_response.selected_flow
                    
                    # Clean up flow name again (in case it was modified after planning)
                    if flow_name:
                        flow_name = flow_name.strip("'\"")
                    
                    if flow_name not in self.flows:
                        raise ExecutionError(
                            f"Selected flow '{flow_name}' not found",
                            ErrorContext.create(flow_name=flow_name, available_flows=list(self.flows.keys()))
                        )
                    
                    flow = self.flows[flow_name]
                    
                    # Generate inputs for the flow
                    flow_inputs = await self._generate_flow_inputs(flow_name, planning_response)
                    
                    # Store flow inputs in memory
                    await self.memory.store(
                        "inputs",
                        flow_inputs,
                        step_context,
                        ttl=3600
                    )
                    
                    # Execute the flow
                    logger.info(f"Executing flow: {flow_name}")
                    context = Context(data=flow_inputs)
                    result = await flow.execute(context)
                    self.last_result = result
                    
                    # Store flow results in memory
                    result_data = result.data.dict() if hasattr(result.data, 'dict') else result.data.model_dump() if hasattr(result.data, 'model_dump') else result.data
                    await self.memory.store(
                        "result",
                        result_data,
                        step_context,
                        ttl=3600
                    )
                    
                    # Special handling for conversation flow
                    if flow_name == "conversation-flow" and hasattr(result.data, "response"):
                        # Store the response directly in the state
                        self.state.last_response = result.data.response
                        logger.info("Stored conversation response in agent state")
                    
                    # Reflect on the results
                    logger.info(f"Reflecting on results from flow: {flow_name}")
                    reflection = await self._reflect_on_result(flow_name, flow_inputs, result)
                    
                    # Store reflection in memory
                    await self.memory.store(
                        "reflection",
                        reflection.dict() if hasattr(reflection, 'dict') else reflection.model_dump(),
                        step_context, 
                        ttl=3600,
                        importance=0.7
                    )
                    
                    # Update state
                    self._update_state(planning_response, flow_name, flow_inputs, result, reflection)
                    
                    # Check if the task was marked as complete by the reflection
                    if self.state.is_complete:
                        logger.info(f"Reflection marked task as complete: {self.state.completion_reason}")
                    else:
                        logger.info(f"Reflection did not mark task as complete - continuing execution loop")
                    
                    # Store new information in long-term memory
                    print("Checking new information")
                    for info in reflection.new_information:
                        print(f"{info.key}: {info.value}")
                        await self.memory.remember(
                            content=info.value,
                            context=self.task_context,
                            source=f"{info.source or 'flow:' + flow_name}",
                            importance=info.importance,
                            metadata={"key": info.key, "relevant_keys": info.relevant_keys, "context": info.context}
                        )
                    
                except Exception as e:
                    logger.error(f"Error during step execution: {str(e)}")
                    self.state.errors.append(str(e))
                    
                    # Store error in memory
                    try:
                        await self.memory.store(
                            "error",
                            str(e),
                            step_context,
                            ttl=3600,
                            importance=0.8
                        )
                        
                        # Also add to long-term memory
                        await self.memory.remember(
                            content=f"Error occurred during execution: {str(e)}",
                            context=self.task_context,
                            source="error",
                            importance=0.8,
                            tags=["error"]
                        )
                    except Exception as mem_error:
                        logger.warning(f"Failed to store error in memory: {str(mem_error)}")
                    
                    # Check if we should stop on error
                    if self.config.stop_on_error:
                        logger.info("Stopping execution due to error")
                        self.state.is_complete = True
                        self.state.completion_reason = f"Error: {str(e)}"
                        break
                
                # Cleanup old memory if needed
                if execution_iterations % 5 == 0:
                    try:
                        await self.memory.cleanup_expired()
                    except Exception as e:
                        logger.warning(f"Memory cleanup failed: {str(e)}")
        
        finally:
            # Store execution completion info
            await self.memory.store(
                "completion_info",
                {
                    "end_time": time.time(),
                    "is_complete": self.state.is_complete,
                    "completion_reason": self.state.completion_reason,
                    "iterations": execution_iterations,
                    "errors": len(self.state.errors)
                },
                run_context,
                ttl=3600,
                importance=0.7
            )
            
            # Persist important working memory to long-term
            await self.memory.persist_working_memory(run_context, min_importance=0.6)
                
        return self.state
    
    def _update_state(
        self,
        planning: PlanningResponse,
        flow_name: str,
        inputs: Any,
        result: Context,
        reflection: ReflectionResponse
    ) -> None:
        """Update the agent's state after a flow execution.
        
        Args:
            planning: Planning response
            flow_name: Name of the executed flow
            inputs: Flow inputs
            result: Flow result
            reflection: Reflection response
        """
        # Limit execution history length based on configuration
        if len(self.state.execution_history) >= self.config.max_execution_history:
            # Remove oldest items
            excess = len(self.state.execution_history) - self.config.max_execution_history + 1
            self.state.execution_history = self.state.execution_history[excess:]
            
        # Add planning information
        self.state.execution_history.append({
            "step": len(self.state.execution_history) + 1,
            "action": "plan",
            "flow": planning.selected_flow,
            "reasoning": planning.reasoning
        })
        
        # Add execution information
        self.state.execution_history.append({
            "step": len(self.state.execution_history) + 1,
            "action": "execute",
            "flow": flow_name,
            "inputs": inputs
        })
        
        # Add reflection information
        self.state.execution_history.append({
            "step": len(self.state.execution_history) + 1,
            "action": "reflect",
            "flow": flow_name,
            "reflection": reflection.reflection,
            "progress": reflection.progress,
            "new_information": reflection.new_information
        })
        
        # Update progress
        self.state.progress = reflection.progress
        
        # Update completion status
        if reflection.is_complete:
            self.state.is_complete = True
            self.state.completion_reason = reflection.completion_reason
            
    async def _plan_next_action(self) -> PlanningResponse:
        """Plan the next action to take.
        
        This method queries the LLM to decide what flow to execute next based
        on the current state, task description, and available flows.
        
        Returns:
            Planning response
        """
        # Get flows for planning
        available_flows = [self.flow_descriptions[name].dict() for name in self.flows]
        
        # Get state and history for planning
        state_dict = self.state.dict() if hasattr(self.state, 'dict') else self.state.model_dump()
        
        # Use planning flow if available
        planning_flow_name = "agent-planning-flow"
        
        if planning_flow_name in self.flows:
            flow = self.flows[planning_flow_name]
            
            # Create input for planning flow
            from .flows import PlanningInput
            planning_input = PlanningInput(
                task_description=self.state.task_description or "",
                available_flows=available_flows,
                current_state=state_dict,
                execution_history=self.state.execution_history,
                model_name=self.config.planner_model
            )
            
            # Execute planning flow
            context = Context(data=planning_input)
            result = await flow.execute(context)
            
            # Return planning response
            return result.data
        else:
            # Fallback implementation for when planning flow is not available
            raise NotImplementedError("Planning flow not available and fallback not implemented")
            
    async def _generate_flow_inputs(self, flow_name: str, planning: PlanningResponse) -> Any:
        """Generate inputs for a flow.
        
        This method queries the LLM to generate inputs for a flow based on
        the current state, task description, and flow schema.
        
        Args:
            flow_name: Name of the flow
            planning: Planning response
            
        Returns:
            Flow inputs
        """
        # Check for existing input in memory - this allows conversation handler
        # or other callers to provide pre-generated inputs
        if self.base_context:
            conversation_context = self.memory.create_context("conversation", self.base_context)
            
            # Special handling for conversation flow
            if flow_name == "conversation-flow":
                # Try to get conversation input from memory with proper model class
                try:
                    from .flows import ConversationInput
                    conversation_input = await self.memory.retrieve(
                        "conversation_input", 
                        conversation_context,
                        model_class=ConversationInput
                    )
                    if conversation_input:
                        logger.info("Using pre-stored conversation input from memory")
                        return conversation_input
                except ImportError:
                    # Fall back to standard retrieve if import fails
                    conversation_input = await self.memory.retrieve("conversation_input", conversation_context)
                    if conversation_input:
                        logger.info("Using pre-stored conversation input from memory")
                        return conversation_input
        
        # Get flow description and schemas
        flow_desc = self.flow_descriptions.get(flow_name)
        flow = self.flows.get(flow_name)
        
        # Use input generation flow if available
        input_gen_flow_name = "agent-input-generation-flow"
        
        if input_gen_flow_name in self.flows:
            flow_gen = self.flows[input_gen_flow_name]
            
            # Create input for generation flow
            from .flows import InputGenerationInput
            input_schema = flow_desc.input_schema if flow_desc and flow_desc.input_schema else "{}"
            
            # Create input for input generation flow
            input_gen_input = InputGenerationInput(
                flow_name=flow_name,
                input_schema=input_schema,
                task_description=self.state.task_description or "",
                current_state=self.state.dict() if hasattr(self.state, 'dict') else self.state.model_dump(),
                planning_reasoning=planning.reasoning,
                model_name=self.config.input_generator_model,
                output_type=flow.__class__.__name__ if flow else "dict"
            )
            
            # Execute input generation flow
            context = Context(data=input_gen_input)
            result = await flow_gen.execute(context)
            
            # Return generated inputs
            return result.data
        else:
            # Fallback implementation for when input generation flow is not available
            raise NotImplementedError("Input generation flow not available and fallback not implemented")
            
    async def _reflect_on_result(self, flow_name: str, inputs: Any, result: Context) -> ReflectionResponse:
        """Reflect on the result of a flow execution.
        
        This method queries the LLM to reflect on the result of a flow execution
        and update the agent's understanding of the task.
        
        Args:
            flow_name: Name of the executed flow
            inputs: Flow inputs
            result: Flow result
            
        Returns:
            Reflection response
        """
        # Use reflection flow if available
        reflection_flow_name = "agent-reflection-flow"
        
        if reflection_flow_name in self.flows:
            flow = self.flows[reflection_flow_name]
            
            # Create input for reflection flow
            from .flows import ReflectionInput
            
            # Convert inputs and outputs to string formats
            if hasattr(inputs, 'dict'):
                inputs_str = str(inputs.dict())
            elif hasattr(inputs, 'model_dump'):
                inputs_str = str(inputs.model_dump())
            else:
                inputs_str = str(inputs)
                
            if hasattr(result.data, 'dict'):
                outputs_str = str(result.data.dict())
            elif hasattr(result.data, 'model_dump'):
                outputs_str = str(result.data.model_dump())
            else:
                outputs_str = str(result.data)
            
            # Create reflection input
            reflection_input = ReflectionInput(
                flow_name=flow_name,
                flow_inputs=inputs_str,
                flow_outputs=outputs_str,
                task_description=self.state.task_description or "",
                current_state=self.state.dict() if hasattr(self.state, 'dict') else self.state.model_dump(),
                model_name=self.config.reflection_model
            )
            
            # Execute reflection flow
            context = Context(data=reflection_input)
            result = await flow.execute(context)
            
            # Return reflection response
            return result.data
        else:
            # Fallback implementation for when reflection flow is not available
            raise NotImplementedError("Reflection flow not available and fallback not implemented")
    
    def get_flow_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all available flows.
        
        Returns:
            List of flow descriptions
        """
        return [self.flow_descriptions[name].dict() if hasattr(self.flow_descriptions[name], 'dict') else self.flow_descriptions[name].model_dump() for name in self.flows]