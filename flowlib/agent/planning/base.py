"""
Base implementation for planning operations.

This module provides a base implementation for planning components that
can be extended by specific planning strategies.
"""

import logging
from typing import Dict, Any, Optional

from ..core.errors import PlanningError, NotInitializedError
from ..core.base import BaseComponent
from .interfaces import PlanningInterface
from .models import (
    PlanningResult,
    PlanningValidation,
    ExecutionContext
)
from ...flows.registry import stage_registry
from ..models.state import AgentState
from ..models.config import PlannerConfig
from ...utils.pydantic.schema import model_to_simple_json_schema
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class BasePlanning(BaseComponent, PlanningInterface):
    """Base implementation for planning operations.
    
    This class provides a foundation for planning components, implementing
    the PlanningInterface with common functionality and error handling.
    
    External dependencies:
    - Requires a flow registry (stage_registry) for accessing flows
    - Requires a model provider for generating plans and inputs
    """
    
    def __init__(self, name: str):
        """Initialize the planning component.
        
        Args:
            name: Name of the planning component
        """
        super().__init__(name)
        
    async def _initialize_impl(self) -> None:
        """Initialize the planning component.
        
        This method should be overridden by subclasses to perform any
        necessary initialization.
        """
        pass
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the planning component.
        
        This method should be overridden by subclasses to perform any
        necessary cleanup.
        """
        pass
    
    async def plan(
        self,
        context: ExecutionContext
    ) -> PlanningResult:
        """Generate a plan based on the current context and available flows.
        
        Args:
            context: Current execution context
            
        Returns:
            PlanningResult containing the selected flow and inputs
            
        Raises:
            PlanningError: If planning fails
            NotInitializedError: If the planner is not properly initialized
        """
        try:
            self._logger.info(f"Planning with context task_id={context.state.task_id}")
            return await self._plan_impl(context)
        except Exception as e:
            self._logger.error(f"Failed to generate plan: {str(e)}")
            raise PlanningError(f"Failed to generate plan: {str(e)}") from e
    
    async def validate_plan(
        self,
        plan: PlanningResult
    ) -> PlanningValidation:
        """Validate a generated plan against available flows.
        
        Args:
            plan: The plan to validate
            
        Returns:
            PlanningValidation indicating if the plan is valid
            
        Raises:
            PlanningError: If validation fails
        """
        try:
            self._logger.info(f"Validating plan for flow '{plan.selected_flow}'")
            return await self._validate_plan_impl(plan)
        except Exception as e:
            self._logger.error(f"Failed to validate plan: {str(e)}")
            raise PlanningError(f"Failed to validate plan: {str(e)}") from e
    
    async def _plan_impl(
        self,
        context: ExecutionContext
    ) -> PlanningResult:
        """Select a flow based on the current context using LLM.
        
        This implementation requires:
        - A configured model provider (self.llm_provider)
        - Access to stage_registry for flow information
        - A planning template (_planning_template)
        
        Args:
            context: Current execution context
            
        Returns:
            PlanningResult with selected flow and inputs
            
        Raises:
            PlanningError: If planning fails or no flows are available
            NotInitializedError: If required components are not initialized
        """
        # Get flows directly from stage_registry
        if not stage_registry:
            raise PlanningError("No flow registry available")
        
        # If no flows available, raise error
        flow_instances = stage_registry.get_flow_instances()
        if not flow_instances:
            raise PlanningError("No flows available for planning")
        
        # Format available flows text
        available_flows_text = ""
        for flow_name, flow_instance in flow_instances.items():
            # Get flow metadata
            flow_metadata = stage_registry.get_flow_metadata(flow_name)
            if not flow_metadata:
                raise PlanningError(f"Flow '{flow_name}' has no metadata in registry. Flows must be properly registered with metadata.")
            
            # Use the description from metadata directly
            description = flow_metadata.description
            available_flows_text += f"- {flow_name}: {description}\n"
        
        # Format execution history text
        user_messages = context.messages.user_messages if context.messages else []
        system_messages = context.messages.system_messages if context.messages else []
        
        execution_history_text = ""
        for i, msg in enumerate(user_messages):
            execution_history_text += f"User {i+1}: {msg.content}\n"
            # Add corresponding system message if available
            if i < len(system_messages):
                execution_history_text += f"System {i+1}: {system_messages[i].content}\n"
        
        # Get relevant memories
        memory_context = ""
        if self.parent and hasattr(self.parent, "_memory") and self.parent._memory:
            try:
                relevant_memories = await self.parent._memory.retrieve_relevant(
                    query=context.state.task_description,
                    limit=5
                )
                if relevant_memories:
                    memory_context = "Relevant memories:\n"
                    for i, memory in enumerate(relevant_memories):
                        memory_context += f"{i+1}. {memory}\n"
            except Exception as e:
                self._logger.warning(f"Error retrieving relevant memories: {str(e)}")
        
        # Prepare variables for prompt
        prompt_variables = {
            "task_description": context.state.task_description,
            "available_flows_text": available_flows_text,
            "execution_history_text": execution_history_text,
            "memory_context": memory_context,
            "cycle": context.state.cycle
        }
        
        # Ensure we have a model name from config
        if not hasattr(self, "config") or not hasattr(self.config, "model_name"):
            raise PlanningError("Model name not found in planner configuration")
        
        # Generate structured planning response
        result = await self.llm_provider.generate_structured(
            prompt=self._planning_template,
            output_type=PlanningResult,
            prompt_variables=prompt_variables,
            model_name=self.config.model_name
        )
        
        if result is None:
            raise PlanningError("LLM returned None for planning result")
        
        # Verify the selected flow is in the registry
        if result.selected_flow != "none" and result.selected_flow not in flow_instances:
            raise PlanningError(f"Selected flow '{result.selected_flow}' not found in registry")
        
        return result
    
    async def _validate_plan_impl(
        self,
        plan: PlanningResult
    ) -> PlanningValidation:
        """Validate a planning result.
        
        Args:
            plan: The plan to validate
            
        Returns:
            PlanningValidation indicating if the plan is valid
            
        Raises:
            PlanningError: If validation fails
        """
        errors = []
        
        # Check if selected flow exists in stage_registry
        if not stage_registry:
            errors.append("Flow registry is not available")
        elif plan.selected_flow != "none" and plan.selected_flow not in stage_registry.get_flow_instances():
            errors.append(f"Selected flow '{plan.selected_flow}' not found in registry")
        
        # Create validation result
        validation = PlanningValidation(
            is_valid=len(errors) == 0,
            errors=errors
        )
        
        return validation
        
    async def generate_inputs(
        self,
        state: AgentState,
        flow_name: str,
        planning_result: Dict[str, Any],
        memory_context: str,
        flow: Optional[Any] = None
    ) -> BaseModel:
        """Generate inputs for a flow based on state and planning result.
        
        Args:
            state: Agent state object containing task details and history
            flow_name: Name of the flow to generate inputs for
            planning_result: Result from planning phase
            memory_context: Memory context for this cycle
            flow: Flow instance (optional)
            
        Returns:
            Generated input model for the flow as a Pydantic BaseModel instance
            
        Raises:
            PlanningError: If input generation fails
            NotInitializedError: If required components are missing
        """
        if not self._input_generation_template:
            raise NotInitializedError(
                component_name=self._name,
                operation="generate_inputs",
                details="Input generation template is not initialized"
            )
        
        # Get flow metadata to determine input model
        flow_metadata = stage_registry.get_flow_metadata(flow_name)
        if not flow_metadata:
            raise PlanningError(f"Flow '{flow_name}' has no metadata in registry")
        
        input_model = flow_metadata.input_model
        if not input_model:
            raise PlanningError(f"Flow '{flow_name}' has no input model defined")
            
        # Get flow description
        flow_description = flow_metadata.description
        
        # Get input schema using the model's schema method
        input_schema = ""
        try:
            # Use the improved schema utility
            input_schema = model_to_simple_json_schema(input_model)
        except Exception as e:
            self._logger.warning(f"Failed to get schema for {input_model.__name__}: {str(e)}")
            # Fallback to string representation
            input_schema = str(input_model.__name__)
            
        # Format execution history 
        execution_history_text = ""
        if state.execution_history:
            history_items = []
            for i, entry in enumerate(state.execution_history):
                if isinstance(entry, dict):
                    flow = entry.get('flow_name', 'unknown')
                    status = entry.get('status', 'unknown')
                    cycle = entry.get('cycle', 0)
                    history_items.append(f"{i+1}. Cycle {cycle}: Executed {flow} with status {status}")
            execution_history_text = "\n".join(history_items)
        else:
            execution_history_text = "No execution history available"
            
        # Get planning rationale
        planning_rationale = ""
        try:
            if isinstance(planning_result, dict) and "reasoning" in planning_result:
                reasoning = planning_result["reasoning"]
                if isinstance(reasoning, dict):
                    planning_rationale = reasoning.get("explanation", "")
            elif hasattr(planning_result, "reasoning") and planning_result.reasoning:
                planning_rationale = planning_result.reasoning.explanation
        except Exception as e:
            self._logger.warning(f"Failed to extract planning rationale: {str(e)}")
            planning_rationale = "No planning rationale available"
            
        # Get relevant memories
        relevant_memories_text = "No relevant memories available"
        if memory_context and self.parent and hasattr(self.parent, "_memory") and self.parent._memory:
            try:
                # Use the typed memory interface 
                from ..memory.interfaces import MemoryInterface
                memory: MemoryInterface = self.parent._memory
                
                relevant_memories = await memory.retrieve_relevant(
                    query=state.task_description,
                    context=memory_context,
                    limit=5
                )
                
                if relevant_memories:
                    memory_items = []
                    for i, memory in enumerate(relevant_memories):
                        memory_items.append(f"{i+1}. {memory}")
                    relevant_memories_text = "\n".join(memory_items)
            except Exception as e:
                self._logger.warning(f"Error retrieving relevant memories: {str(e)}")
        
        # Prepare variables for prompt
        prompt_variables = {
            "task_description": state.task_description,
            "flow_name": flow_name,
            "flow_description": flow_description,
            "input_schema": input_schema,
            "planning_rationale": planning_rationale,
            "execution_history_text": execution_history_text,
            "relevant_memories_text": relevant_memories_text
        }
        
        # Generate structured input
        inputs = await self.llm_provider.generate_structured(
            prompt=self._input_generation_template,
            output_type=input_model,
            model_name=self.config.model_name,
            prompt_variables=prompt_variables
        )
        
        self._logger.info(f"Generated inputs for flow '{flow_name}': {str(inputs.model_dump())[:100]}...")
        return inputs


class AgentPlanner(BasePlanning):
    """Main planner implementation for the agent system.
    
    This class implements the planning interface with LLM-based planning that
    selects flows based on task state and context.
    
    Required configuration:
    - model_name: Name of the model to use for planning
    - provider_name: Name of the provider to use for planning
    """
    
    def __init__(self, config: PlannerConfig, name: str = "agent_planner"):
        """Initialize the agent planner.
        
        Args:
            config: PlannerConfig object with planning configuration
            name: Name of the planner component
            
        Raises:
            TypeError: If config is not a PlannerConfig object
        """
        super().__init__(name)
        
        # Type check: config must be PlannerConfig
        if not isinstance(config, PlannerConfig):
            raise TypeError(f"config must be a PlannerConfig object, got {type(config).__name__}")
            
        self.config = config
        self._llm_provider = None
        self._planning_template = None
        self._input_generation_template = None
        
    @property
    def llm_provider(self):
        """Get the model provider used for planning.
        
        If not set explicitly, tries to get from parent component.
            
        Returns:
            Model provider instance or None if not available
        """
        if self._llm_provider:
            return self._llm_provider
            
        # Try to get from parent if available
        if self.parent and hasattr(self.parent, "llm_provider"):
            return self.parent.llm_provider
            
        return None
    
    @llm_provider.setter
    def llm_provider(self, provider):
        """Set the model provider for planning.
        
        Args:
            provider: Model provider instance
        """
        self._llm_provider = provider
    
    async def _initialize_impl(self) -> None:
        """Initialize the planner component.
        
        This method creates a model provider if not already available
        and loads the necessary prompt templates.
            
        Raises:
            NotInitializedError: If initialization fails
        """
        self._logger.info("Initializing agent planner")
        
        # Create model provider if not already available
        if not self._llm_provider:
            # Check if provider name is available in config
            if not hasattr(self.config, "provider_name"):
                raise NotInitializedError(
                    component_name=self._name,
                    operation="planning",
                    details="provider_name not found in planner configuration"
                )
            
            # Get provider name from config
            provider_name = self.config.provider_name
            
            # Create provider from registry
            from ...providers import ProviderType
            from ...providers.registry import provider_registry
            from ...providers.llm.base import LLMProvider
            
            provider = await provider_registry.get(
                ProviderType.LLM,
                provider_name
            )
            
            if not provider:
                raise NotInitializedError(
                    component_name=self._name,
                    operation="planning",
                    details=f"Failed to create model provider '{provider_name}'"
                )
                
            self._llm_provider = provider
            self._logger.info(f"Created model provider '{provider_name}' for planning")
        
        # Load prompt templates
        self._planning_template = await self._load_planning_template()
        if not self._planning_template:
            raise NotInitializedError(
                component_name=self._name,
                operation="planning",
                details="Failed to load planning template"
            )
            
        self._input_generation_template = await self._load_input_generation_template()
        if not self._input_generation_template:
            raise NotInitializedError(
                component_name=self._name,
                operation="planning",
                details="Failed to load input generation template"
            )
    
    async def _load_planning_template(self) -> object:
        """Load the planning prompt template.
        
        Returns:
            Planning prompt template
            
        Raises:
            NotInitializedError: If template cannot be loaded
        """
        from ...resources.registry import resource_registry
        from ...resources.constants import ResourceType
        
        # Try to get from resource registry
        if resource_registry.contains("planning_default", ResourceType.PROMPT):
            template = resource_registry.get_sync("planning_default", ResourceType.PROMPT)
            self._logger.info("Loaded default planning template from resource registry")
            return template
            
        # Use the default implementation
        from ..planning.prompts import DefaultPlanningPrompt
        self._logger.info("Using built-in DefaultPlanningPrompt")
        return DefaultPlanningPrompt()
    
    async def _load_input_generation_template(self) -> object:
        """Load the input generation prompt template.
        
        Returns:
            Input generation prompt template
            
        Raises:
            NotInitializedError: If template cannot be loaded
        """
        from ...resources.registry import resource_registry
        from ...resources.constants import ResourceType
        
        # Try to get from resource registry
        if resource_registry.contains("input_generation_default", ResourceType.PROMPT):
            template = resource_registry.get_sync("input_generation_default", ResourceType.PROMPT)
            self._logger.info("Loaded default input generation template from resource registry")
            return template
            
        # Use the default implementation
        from ..planning.prompts import DefaultInputGenerationPrompt
        self._logger.info("Using built-in DefaultInputGenerationPrompt")
        return DefaultInputGenerationPrompt()
            
    async def generate_inputs(
        self,
        state: AgentState,
        flow_name: str,
        planning_result: Dict[str, Any],
        memory_context: str,
        flow: Optional[Any] = None
    ) -> BaseModel:
        """Generate inputs for a flow based on state and planning result.
        
        Args:
            state: Agent state object containing task details and history
            flow_name: Name of the flow to generate inputs for
            planning_result: Result from planning phase
            memory_context: Memory context for this cycle
            flow: Flow instance (optional)
            
        Returns:
            Generated input model for the flow as a Pydantic BaseModel instance
            
        Raises:
            PlanningError: If input generation fails
            NotInitializedError: If required components are missing
        """
        if not self._input_generation_template:
            raise NotInitializedError(
                component_name=self._name,
                operation="generate_inputs",
                details="Input generation template is not initialized"
            )
        
        # Get flow metadata to determine input model
        flow_metadata = stage_registry.get_flow_metadata(flow_name)
        if not flow_metadata:
            raise PlanningError(f"Flow '{flow_name}' has no metadata in registry")
        
        input_model = flow_metadata.input_model
        if not input_model:
            raise PlanningError(f"Flow '{flow_name}' has no input model defined")
            
        # Get flow description
        flow_description = flow_metadata.description
        
        # Get input schema using the model's schema method
        input_schema = ""
        try:
            # Use the improved schema utility
            input_schema = model_to_simple_json_schema(input_model)
        except Exception as e:
            self._logger.warning(f"Failed to get schema for {input_model.__name__}: {str(e)}")
            # Fallback to string representation
            input_schema = str(input_model.__name__)
            
        # Format execution history 
        execution_history_text = ""
        if state.execution_history:
            history_items = []
            for i, entry in enumerate(state.execution_history):
                if isinstance(entry, dict):
                    flow = entry.get('flow_name', 'unknown')
                    status = entry.get('status', 'unknown')
                    cycle = entry.get('cycle', 0)
                    history_items.append(f"{i+1}. Cycle {cycle}: Executed {flow} with status {status}")
            execution_history_text = "\n".join(history_items)
        else:
            execution_history_text = "No execution history available"
            
        # Get planning rationale
        planning_rationale = ""
        try:
            if isinstance(planning_result, dict) and "reasoning" in planning_result:
                reasoning = planning_result["reasoning"]
                if isinstance(reasoning, dict):
                    planning_rationale = reasoning.get("explanation", "")
            elif hasattr(planning_result, "reasoning") and planning_result.reasoning:
                planning_rationale = planning_result.reasoning.explanation
        except Exception as e:
            self._logger.warning(f"Failed to extract planning rationale: {str(e)}")
            planning_rationale = "No planning rationale available"
            
        # Get relevant memories
        relevant_memories_text = "No relevant memories available"
        if memory_context and self.parent and hasattr(self.parent, "_memory") and self.parent._memory:
            try:
                # Use the typed memory interface 
                from ..memory.interfaces import MemoryInterface
                memory: MemoryInterface = self.parent._memory
                
                relevant_memories = await memory.retrieve_relevant(
                    query=state.task_description,
                    context=memory_context,
                    limit=5
                )
                
                if relevant_memories:
                    memory_items = []
                    for i, memory in enumerate(relevant_memories):
                        memory_items.append(f"{i+1}. {memory}")
                    relevant_memories_text = "\n".join(memory_items)
            except Exception as e:
                self._logger.warning(f"Error retrieving relevant memories: {str(e)}")
        
        # Prepare variables for prompt
        prompt_variables = {
            "task_description": state.task_description,
            "flow_name": flow_name,
            "flow_description": flow_description,
            "input_schema": input_schema,
            "planning_rationale": planning_rationale,
            "execution_history_text": execution_history_text,
            "relevant_memories_text": relevant_memories_text
        }
        
        # Generate structured input
        inputs = await self.llm_provider.generate_structured(
            prompt=self._input_generation_template,
            output_type=input_model,
            model_name=self.config.model_name,
            prompt_variables=prompt_variables
        )
        
        self._logger.info(f"Generated inputs for flow '{flow_name}': {str(inputs.model_dump())[:100]}...")
        return inputs
