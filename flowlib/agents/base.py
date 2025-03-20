"""Base agent class for the unified flow framework.

This module provides the Agent base class that integrates LLM-based planning,
input generation, and reflection for executing flows.
"""

import asyncio
import inspect
import logging
import json
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Type, TYPE_CHECKING

from pydantic import BaseModel, Field

import flowlib as fl
from flowlib.core.registry.constants import ProviderType, ResourceType
from flowlib.core.errors import ExecutionError, ErrorContext, ResourceError
from flowlib.core.models.context import Context
from flowlib.flows.base import Flow

from .models import AgentConfig, AgentState, PlanningResponse, ReflectionResponse, FlowDescription
from .discovery import FlowDiscovery
from .memory_manager import MemoryManager, MemoryContext

# Import entity-centric memory components
from .memory.manager import HybridMemoryManager
from .memory.models import Entity

# We'll use lazy imports for memory flows to avoid circular dependencies
# These will be imported when needed in the class methods
# from .flows.memory_flows import (
#     ConversationInput, MemorySearchInput, RetrievedMemories, 
#     ExtractedEntities, MemoryExtractionFlow, MemoryRetrievalFlow
# )

logger = logging.getLogger(__name__)

# Use typing.TYPE_CHECKING for forward references
if TYPE_CHECKING:
    from .flows.memory_flows import (
        ConversationInput, RetrievedMemories, ExtractedEntities
    )

# Forward declare the types for type hints
ConversationInput = TypeVar('ConversationInput')
MemorySearchInput = TypeVar('MemorySearchInput')
MemoryExtractionFlow = TypeVar('MemoryExtractionFlow')
MemoryRetrievalFlow = TypeVar('MemoryRetrievalFlow')

class Agent:
    """Enhanced base agent implementation with memory and flow discovery.
    
    This agent provides:
    1. Dynamic flow discovery at runtime
    2. Working, short-term, and long-term memory using the memory manager
    3. LLM-powered planning, input generation, and reflection
    4. Comprehensive state tracking and error handling
    5. Entity-centric memory with hybrid vector and graph capabilities
    """
    
    def __init__(
        self, 
        flows: List[Flow] = None, 
        config: Optional[AgentConfig] = None,
        task_description: Optional[str] = None,
        flow_paths: List[str] = ["./flows"],
        use_hybrid_memory: bool = False
    ):
        """Initialize the agent with flows and configuration.
        
        Args:
            flows: List of flows the agent can use
            config: Agent configuration
            task_description: Description of the task to perform
            flow_paths: Paths to scan for flow definitions
            use_hybrid_memory: Whether to use the hybrid entity-centric memory system
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
                
        # Setup memory management
        self.use_hybrid_memory = use_hybrid_memory
        
        if use_hybrid_memory:
            # Initialize hybrid memory manager for entity-centric memory
            self.hybrid_memory = HybridMemoryManager(
                graph_provider_name="memory-graph",  # Use in-memory graph by default
                vector_provider_name=self.config.long_term_memory if hasattr(self.config, 'long_term_memory') else None
            )
            
        # Create traditional memory manager as well
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
        print("\n===== DEBUGGING AGENT INITIALIZATION =====")
        
        # Initialize provider
        print("Initializing provider")
        await self.initialize_provider()
        print("Provider initialized")

        # Initialize memory
        print("Initializing memory")
        await self.memory.initialize()
        print("Memory initialized")
        
        # Initialize hybrid memory if enabled
        if self.use_hybrid_memory:
            print("Initializing hybrid memory")
            await self.hybrid_memory.initialize()
            print("Hybrid memory initialized")
        
        # Create standard memory contexts
        print("Creating memory contexts")
        self.base_context = self.memory.create_context(f"agent:{self.config.name}")
        self.execution_context = self.memory.create_context("execution", self.base_context)
        self.task_context = self.memory.create_context("task", self.base_context)
        print(f"Created memory contexts: base={self.base_context.get_full_path()}, execution={self.execution_context.get_full_path()}, task={self.task_context.get_full_path()}")
        
        # Store task description if available
        if self.state.task_description:
            print(f"Storing task description: {self.state.task_description[:100]}...")
            await self.memory.store(
                "description", 
                self.state.task_description, 
                self.task_context,
                importance=0.8
            )
            print("Task description stored")

        # Discover flows via the registry and flow discovery
        print("Starting flow discovery")
        await self._discover_flows()
        
        # Check for essential flows after discovery
        print("\nChecking for essential flows after discovery:")
        essential_flows = ["memory-retrieval", "memory-extraction", "conversation-flow", "agent-planning-flow", "agent-input-generation-flow", "agent-reflection-flow"]
        for flow_name in essential_flows:
            is_available = flow_name in self.flows
            print(f"  {flow_name}: {'Available' if is_available else 'Not available'}")
            if is_available:
                flow_instance = self.flows[flow_name]
                print(f"    Type: {type(flow_instance)}")
                methods = [method for method in dir(flow_instance) if not method.startswith('_') and callable(getattr(flow_instance, method))]
                print(f"    Methods: {methods}")

        logger.info(f"Initialized agent with {len(self.flows)} flows")
        print("===== END AGENT INITIALIZATION DEBUGGING =====\n")

    async def _discover_flows(self):
        """Discover flows from registry and directories."""
        print("\n===== DEBUGGING FLOW DISCOVERY =====")
        
        # First check the registry for flows registered via decorators
        from ..flows.registry import stage_registry
        print(f"Loaded stage_registry: {stage_registry}")

        # Get flow names from registry
        registered_flows = stage_registry.get_flows()
        print(f"Registered flows from registry: {registered_flows}")
        
        for flow_name in registered_flows:
            if flow_name not in self.flows:
                # Try to get the actual flow instance from registry
                print(f"Getting flow instance for '{flow_name}'")
                flow_instance = stage_registry.get_flow(flow_name)
                
                if flow_instance:
                    # Register the flow instance with the agent
                    print(f"Registering flow instance: {flow_instance}")
                    self._register_flow(flow_instance)
                    logger.info(f"Discovered flow from registry: {flow_name}")
                else:
                    # Just note the flow exists if we don't have an instance
                    print(f"No instance for flow: {flow_name}")
                    logger.debug(f"Found flow name in registry but no instance: {flow_name}")
                    self.flow_descriptions[flow_name] = FlowDescription(
                        name=flow_name,
                        input_schema=None,
                        output_schema=None,
                        description=None
                    )
        
        print(f"After registry checks, self.flows: {list(self.flows.keys())}")
        print(f"After registry checks, self.flow_descriptions: {list(self.flow_descriptions.keys())}")
        
        # Then scan directories for additional flows
        print("Starting directory scan for flows")
        discovered_flows = await self.flow_discovery.refresh_flows()
        print(f"Discovered flows from directories: {list(discovered_flows.keys())}")
        
        for name, flow in discovered_flows.items():
            if name not in self.flows:
                print(f"Registering discovered flow: {name}")
                self._register_flow(flow)
                logger.info(f"Discovered flow from directory scan: {name}")
                
        print(f"Final self.flows: {list(self.flows.keys())}")
        print("===== END FLOW DISCOVERY DEBUGGING =====\n")
            
    async def initialize_provider(self):
        """Initialize the LLM provider asynchronously.
        
        This method should be called before using the provider.
        """
        if self.provider is None:
            provider_name = self.config.provider_name
            try:
                # Use async get method which handles provider initialization
                self.provider = await fl.provider_registry.get(ProviderType.LLM, provider_name)
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
        
        # Use the new standard interface methods first
        if hasattr(flow, "get_pipeline_input_model"):
            input_schema_model = flow.get_pipeline_input_model()
            if input_schema_model:
                try:
                    # Format the schema for better display in prompts
                    if hasattr(input_schema_model, "__fields__"):
                        # Pydantic v1 style
                        fields = input_schema_model.__fields__
                        fields_str = ", ".join([f"{name}: {field.type_.__name__}" for name, field in fields.items()])
                        input_schema = f"{input_schema_model.__name__} ({fields_str})"
                    elif hasattr(input_schema_model, "model_fields"):
                        # Pydantic v2 style
                        fields = input_schema_model.model_fields
                        fields_str = ", ".join([f"{name}: {str(field.annotation.__name__)}" for name, field in fields.items() 
                                               if hasattr(field.annotation, "__name__")])
                        input_schema = f"{input_schema_model.__name__} ({fields_str})"
                    # Special handling for RootModel
                    elif hasattr(input_schema_model, "__origin__") and input_schema_model.__origin__.__name__ == "RootModel":
                        # Get root type annotation if possible
                        if hasattr(input_schema_model, "__annotations__") and "root" in input_schema_model.__annotations__:
                            root_type = input_schema_model.__annotations__["root"]
                            root_type_name = getattr(root_type, "__name__", str(root_type))
                            input_schema = f"{input_schema_model.__name__} (Root: {root_type_name})"
                        else:
                            input_schema = f"{input_schema_model.__name__} (RootModel)"
                    # Try custom string method if available
                    elif hasattr(input_schema_model, "__str__"):
                        try:
                            input_schema = str(input_schema_model())
                        except:
                            input_schema = input_schema_model.__name__
                    else:
                        # Fallback - just use the model name
                        input_schema = input_schema_model.__name__
                except (AttributeError, TypeError):
                    # Fallback to string representation
                    input_schema = str(input_schema_model)
        
        if hasattr(flow, "get_pipeline_output_model"):
            output_schema_model = flow.get_pipeline_output_model()
            if output_schema_model:
                try:
                    # Format the schema for better display in prompts
                    if hasattr(output_schema_model, "__fields__"):
                        # Pydantic v1 style
                        fields = output_schema_model.__fields__
                        fields_str = ", ".join([f"{name}: {field.type_.__name__}" for name, field in fields.items()])
                        output_schema = f"{output_schema_model.__name__} ({fields_str})"
                    elif hasattr(output_schema_model, "model_fields"):
                        # Pydantic v2 style
                        fields = output_schema_model.model_fields
                        fields_str = ", ".join([f"{name}: {str(field.annotation.__name__)}" for name, field in fields.items() 
                                               if hasattr(field.annotation, "__name__")])
                        output_schema = f"{output_schema_model.__name__} ({fields_str})"
                    # Special handling for RootModel
                    elif hasattr(output_schema_model, "__origin__") and output_schema_model.__origin__.__name__ == "RootModel":
                        # Get root type annotation if possible
                        if hasattr(output_schema_model, "__annotations__") and "root" in output_schema_model.__annotations__:
                            root_type = output_schema_model.__annotations__["root"]
                            root_type_name = getattr(root_type, "__name__", str(root_type))
                            output_schema = f"{output_schema_model.__name__} (Root: {root_type_name})"
                        else:
                            output_schema = f"{output_schema_model.__name__} (RootModel)"
                    # Try custom string method if available
                    elif hasattr(output_schema_model, "__str__"):
                        try:
                            output_schema = str(output_schema_model())
                        except:
                            output_schema = output_schema_model.__name__
                    else:
                        # Fallback - just use the model name
                        output_schema = output_schema_model.__name__
                except (AttributeError, TypeError):
                    # Fallback to string representation
                    output_schema = str(output_schema_model)
            
        # Fallback to old way of getting description
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
        if input_schema or output_schema:
            logger.debug(f"Flow schema: {flow_name} - Input: {input_schema}, Output: {output_schema}")
    
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
        # Refresh flow schemas before planning
        print("\n===== REFRESHING FLOW SCHEMAS BEFORE PLANNING =====")
        for flow_name, flow in self.flows.items():
            # Extract and update schema information
            input_schema = None
            output_schema = None
            description = None
            
            # First try to get input schema directly from the flow
            input_schema_model = None
            output_schema_model = None
            
            if hasattr(flow, "get_pipeline_input_model"):
                input_schema_model = flow.get_pipeline_input_model()
            if hasattr(flow, "get_pipeline_output_model"):
                output_schema_model = flow.get_pipeline_output_model()
            
            # Format the input schema
            if input_schema_model:
                try:
                    # Format input schema to a readable string representation
                    input_schema = self._format_schema_model(input_schema_model, f"Error formatting input schema for {flow_name}")
                except Exception as e:
                    print(f"Error formatting input schema for {flow_name}: {str(e)}")
                    input_schema = str(input_schema_model)
            
            # Format the output schema
            if output_schema_model:
                try:
                    # Format output schema to a readable string representation
                    output_schema = self._format_schema_model(output_schema_model, f"Error formatting output schema for {flow_name}")
                except Exception as e:
                    print(f"Error formatting output schema for {flow_name}: {str(e)}")
                    output_schema = str(output_schema_model)
            
            # Get description
            if hasattr(flow, "description"):
                description = flow.description
            
            # Update flow description with fresh schema information
            if flow_name in self.flow_descriptions:
                print(f"Updating schema for flow: {flow_name}")
                print(f"  Input schema: {input_schema}")
                print(f"  Output schema: {output_schema}")
                
                # Get current description or use new one
                current_desc = self.flow_descriptions[flow_name]
                if description is None and hasattr(current_desc, "description"):
                    description = current_desc.description
                
                # Create updated flow description
                self.flow_descriptions[flow_name] = FlowDescription(
                    name=flow_name,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    description=description
                )
            else:
                print(f"Adding flow description for: {flow_name}")
                self.flow_descriptions[flow_name] = FlowDescription(
                    name=flow_name,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    description=description
                )
        print("===== SCHEMA REFRESH COMPLETE =====\n")
        
        # Get flows for planning
        available_flows = [self.flow_descriptions[name].dict() if hasattr(self.flow_descriptions[name], 'dict') else self.flow_descriptions[name].model_dump() for name in self.flows]
        
        # Debug: print the flow descriptions
        print("\n===== FLOW DESCRIPTIONS PASSED TO PLANNING =====")
        for flow in available_flows:
            print(f"Flow: {flow.get('name')}")
            print(f"  Description: {flow.get('description')}")
            print(f"  Input schema: {flow.get('input_schema')}")
            print(f"  Output schema: {flow.get('output_schema')}")
        print("===== END FLOW DESCRIPTIONS =====\n")
        
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
            
    def _format_schema_model(self, model, error_context="Error formatting schema"):
        """Format a schema model into a readable string representation.
        
        This helper handles both Pydantic v1 and v2 field formats to extract
        type information safely.
        
        Args:
            model: The Pydantic model class to format
            error_context: Context string for error logging
            
        Returns:
            A formatted string representation of the model schema
        """
        if not model:
            return None
            
        try:
            model_name = model.__name__ if hasattr(model, "__name__") else str(model)
            
            # Try Pydantic v2 style first (model_fields)
            if hasattr(model, "model_fields"):
                fields = model.model_fields
                field_strs = []
                
                for name, field in fields.items():
                    # Handle different ways field types might be stored
                    field_type = None
                    
                    # Try to get annotation directly
                    if hasattr(field, "annotation"):
                        annotation = field.annotation
                        if hasattr(annotation, "__name__"):
                            field_type = annotation.__name__
                        elif hasattr(annotation, "_name"):
                            field_type = annotation._name
                        else:
                            field_type = str(annotation)
                    
                    if field_type:
                        field_strs.append(f"{name}: {field_type}")
                    else:
                        field_strs.append(name)
                        
                if field_strs:
                    return f"{model_name} ({', '.join(field_strs)})"
                
            # Try Pydantic v1 style (__fields__)
            if hasattr(model, "__fields__"):
                fields = model.__fields__
                field_strs = []
                
                for name, field in fields.items():
                    # Handle different ways field types might be stored
                    field_type = None
                    
                    # Try type_ attribute (older Pydantic versions)
                    if hasattr(field, "type_") and hasattr(field.type_, "__name__"):
                        field_type = field.type_.__name__
                    # Try outer_type_ attribute
                    elif hasattr(field, "outer_type_") and hasattr(field.outer_type_, "__name__"):
                        field_type = field.outer_type_.__name__
                    # Try annotation attribute
                    elif hasattr(field, "annotation") and hasattr(field.annotation, "__name__"):
                        field_type = field.annotation.__name__
                    
                    if field_type:
                        field_strs.append(f"{name}: {field_type}")
                    else:
                        field_strs.append(name)
                        
                if field_strs:
                    return f"{model_name} ({', '.join(field_strs)})"
            
            # Special handling for RootModel
            if hasattr(model, "__origin__") and getattr(model.__origin__, "__name__", "") == "RootModel":
                # Get root type annotation if possible
                if hasattr(model, "__annotations__") and "root" in model.__annotations__:
                    root_type = model.__annotations__["root"]
                    root_type_name = getattr(root_type, "__name__", str(root_type))
                    return f"{model_name} (Root: {root_type_name})"
                else:
                    return f"{model_name} (RootModel)"
            
            # Only as a last resort, try custom string method
            if hasattr(model, "__str__"):
                try:
                    custom_str = str(model())
                    if custom_str != model_name:
                        return custom_str
                except:
                    pass  # Fall through if this fails
                    
            # If we can't extract field information, just return the model name
            return model_name
            
        except Exception as e:
            print(f"{error_context}: {str(e)}")
            return str(model)
            
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
                    # Lazy import to avoid circular dependency
                    from .flows import MessageInput
                    conversation_input = await self.memory.retrieve(
                        "conversation_input", 
                        conversation_context,
                        model_class=MessageInput
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

    # Entity-centric memory methods
    
    async def retrieve_memories(self, query: Optional[str] = None, conversation: Optional[ConversationInput] = None) -> "RetrievedMemories":
        """Retrieve relevant memories based on query or conversation context.
        
        This method uses the MemoryRetrievalFlow to retrieve relevant memories from
        the hybrid memory system, either by explicit query or by analyzing conversation.
        
        Args:
            query: Optional explicit query string
            conversation: Optional conversation input for context-based retrieval
            
        Returns:
            Retrieved memories result with entities and formatted context
            
        Raises:
            FlowError: If memory retrieval fails
        """
        print("\n===== DEBUGGING RETRIEVE_MEMORIES =====")
        print(f"Query: {query}")
        print(f"Conversation provided: {conversation is not None}")
        if conversation:
            print(f"Conversation type: {type(conversation)}")
            print(f"Conversation attributes: {dir(conversation)}")
            print(f"Conversation dict: {conversation.dict() if hasattr(conversation, 'dict') else conversation.model_dump() if hasattr(conversation, 'model_dump') else 'No dict/model_dump method'}")
            
        print(f"self.use_hybrid_memory: {self.use_hybrid_memory}")
        if not self.use_hybrid_memory:
            logger.warning("Hybrid memory not enabled. Use Agent(use_hybrid_memory=True) to enable.")
            # Lazy import to avoid circular dependency
            print("Importing RetrievedMemories (hybrid memory not enabled branch)")
            from .flows.memory_flows import RetrievedMemories
            print(f"RetrievedMemories type: {type(RetrievedMemories)}")
            result = RetrievedMemories(entities=[], formatted_context="", query_used="")
            print(f"Created empty RetrievedMemories result: {result}")
            print("===== END DEBUGGING RETRIEVE_MEMORIES =====\n")
            return result
            
        print(f"self.flows keys: {list(self.flows.keys())}")
        if "memory-retrieval" not in self.flows:
            logger.warning("MemoryRetrievalFlow not available")
            # Lazy import to avoid circular dependency
            print("Importing RetrievedMemories (flow not available branch)")
            from .flows.memory_flows import RetrievedMemories
            result = RetrievedMemories(entities=[], formatted_context="", query_used="")
            print(f"Created empty RetrievedMemories result: {result}")
            print("===== END DEBUGGING RETRIEVE_MEMORIES =====\n")
            return result
            
        logger.info(f"Retrieving memories: query='{query}', conversation={conversation is not None}")
        
        # Use the memory-retrieval flow
        print("Getting memory-retrieval flow")
        flow = self.flows["memory-retrieval"]
        print(f"Flow type: {type(flow)}")
        
        # Create input for memory retrieval
        # Lazy import to avoid circular dependency
        print("Importing MemorySearchInput")
        from .flows.memory_flows import MemorySearchInput
        print(f"MemorySearchInput type: {type(MemorySearchInput)}")
        print("Creating search input")
        search_input = MemorySearchInput(
            query=query,
            conversation=conversation
        )
        print(f"Created search_input: {search_input}")
        
        # Execute memory retrieval flow
        print("Creating context for flow execution")
        context = Context(data=search_input)
        print("Executing memory-retrieval flow")
        try:
            result = await flow.execute(context)
            print(f"Flow execution result: {result}")
            print(f"result.data: {result.data}")
            print("===== END DEBUGGING RETRIEVE_MEMORIES =====\n")
            return result.data
        except Exception as e:
            print(f"Error executing memory-retrieval flow: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("===== END DEBUGGING RETRIEVE_MEMORIES =====\n")
            raise
    
    async def extract_and_store_memories(self, conversation: ConversationInput) -> "ExtractedEntities":
        """Extract and store entities from conversation in memory.
        
        This method uses the MemoryExtractionFlow to extract entities, attributes,
        and relationships from conversation text and store them in memory.
        
        Args:
            conversation: Conversation input with history
            
        Returns:
            Extracted entities result
            
        Raises:
            FlowError: If memory extraction fails
        """
        if not self.use_hybrid_memory:
            logger.warning("Hybrid memory not enabled. Use Agent(use_hybrid_memory=True) to enable.")
            # Lazy import to avoid circular dependency
            from .flows.memory_flows import ExtractedEntities
            return ExtractedEntities(entities=[], raw_extraction="")
            
        if "memory-extraction" not in self.flows:
            logger.warning("MemoryExtractionFlow not available")
            # Lazy import to avoid circular dependency
            from .flows.memory_flows import ExtractedEntities
            return ExtractedEntities(entities=[], raw_extraction="")
            
        logger.info("Extracting and storing memories from conversation")
        
        # Use the memory-extraction flow
        flow = self.flows["memory-extraction"]
        
        # Execute memory extraction flow
        context = Context(data=conversation)
        extraction_result = await flow.execute(context)
        
        # Get extracted entities
        extracted_entities = extraction_result.data
        
        # Store entities in memory
        if extracted_entities.entities:
            logger.info(f"Storing {len(extracted_entities.entities)} extracted entities in memory")
            await self.hybrid_memory.store_entities(extracted_entities.entities)
        
        return extracted_entities