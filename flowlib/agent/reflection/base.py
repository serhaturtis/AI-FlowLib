"""
Base reflection implementation for the agent system.

This module provides the reflection component for the agent system,
which is responsible for analyzing execution results and determining
task progress.
"""

import logging
from typing import Any, Dict, Optional

# Import prompts to ensure registration
from .prompts import (
    DefaultReflectionPrompt,
    ConversationalReflectionPrompt,
    TaskCompletionReflectionPrompt
)

from ..core.base import BaseComponent
from ..core.errors import ReflectionError, NotInitializedError
from ..models.config import ReflectionConfig
from ..models.state import AgentState
from ...providers import ProviderType
from ...providers.registry import provider_registry
from ...flows.results import FlowResult
from .models import ReflectionResult, ReflectionInput
from .interfaces import ReflectionInterface
from ...providers.llm.base import LLMProvider
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentReflection(BaseComponent, ReflectionInterface):
    """Reflection component for the agent system.
    
    Responsibilities:
    1. Analyzing execution results
    2. Determining task progress and completion
    3. Extracting insights from execution results
    
    This class implements the ReflectionInterface protocol.
    """
    
    def __init__(
        self,
        config: Optional[ReflectionConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
        name: str = "reflection"
    ):
        """Initialize the agent reflection.
        
        Args:
            config: Reflection configuration
            llm_provider: LLM provider component
            name: Component name
        """
        super().__init__(name)
        
        # Configuration
        self._config = config or ReflectionConfig(model_name="default")
        
        # Components
        self._llm_provider = llm_provider
        
        # Execution state
        self._reflection_template = None
    
    def _check_initialized(self) -> None:
        """Check if the reflection component is initialized.
        
        Raises:
            NotInitializedError: If the reflection component is not initialized
        """
        if not self._initialized:
            raise NotInitializedError(
                component_name=self._name,
                operation="reflect"
            )
    
    async def _initialize_impl(self) -> None:
        """
        Initialize the reflection component.
        
        Raises:
            ReflectionError: If initialization fails
        """
        try:
            # First, check if we have a model provider already
            if not self._llm_provider:
                # Create our own model provider like the planner does
                provider_name = self._config.provider_name or "llamacpp"
                
                # Get model provider directly from provider registry
                try:
                    self._llm_provider = await provider_registry.get(
                        ProviderType.LLM,
                        provider_name
                    )
                    logger.info(f"Created model provider '{provider_name}' for reflection")
                except Exception as e:
                    raise ReflectionError(
                        message=f"Failed to create model provider '{provider_name}' for reflection: {str(e)}",
                        agent=self.parent.name if self.parent else self.name,
                        cause=e
                    )
            
            # Ensure we have a model provider
            if not self._llm_provider:
                raise ReflectionError(
                    message="No model provider available for reflection",
                    agent=self.parent.name if self.parent else self.name
                )
            
            # Load template
            self._reflection_template = await self._load_reflection_template()
            
            logger.info(f"Initialized agent reflection with model {self._config.model_name}")
            
        except Exception as e:
            raise ReflectionError(
                message=f"Failed to initialize reflection component: {str(e)}",
                agent=self.parent.name if self.parent else self.name,
                cause=e
            ) from e
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the reflection component."""
        logger.info("Shutting down agent reflection")
    
    async def reflect(
        self,
        state: AgentState,
        flow_name: str,
        flow_inputs: Dict[str, Any],
        flow_result: FlowResult,
        memory_context: Optional[str] = None
    ) -> ReflectionResult:
        """
        Analyze execution results and update state.
        
        Args:
            state: Current agent state
            flow_name: Name of the flow that was executed
            flow_inputs: Inputs that were used for the flow
            flow_result: Result from the flow execution
            memory_context: Optional memory context for reflection
            
        Returns:
            ReflectionResult with analysis, progress and completion status
            
        Raises:
            NotInitializedError: If reflection is not initialized
            ReflectionError: If reflection fails or flow_result is not a valid FlowResult
        """
        self._check_initialized()
        
        # Ensure flow_result is a valid FlowResult
        if not isinstance(flow_result, FlowResult):
            raise ReflectionError(
                message=f"Expected FlowResult instance, got {type(flow_result).__name__}",
                agent=self.parent.name if self.parent else self.name
            )
        
        try:
            # If flow_inputs is not a BaseModel, convert it to one
            from pydantic import create_model
            if not isinstance(flow_inputs, BaseModel):
                # Create a dynamic model from the dictionary
                dynamic_inputs = create_model(
                    f"{flow_name}Inputs",
                    **{k: (type(v), ...) for k, v in flow_inputs.items()}
                )
                # Instantiate the model with the input values
                flow_inputs_model = dynamic_inputs(**flow_inputs)
            else:
                flow_inputs_model = flow_inputs
            
            # Prepare reflection input
            reflection_input = await self._prepare_reflection_input(
                state=state,
                flow_name=flow_name,
                flow_inputs=flow_inputs_model,
                flow_result=flow_result,
                memory_context=memory_context
            )
            
            # Execute reflection and return the result directly
            return await self._execute_reflection(reflection_input)
            
        except Exception as e:
            raise ReflectionError(
                message=f"Reflection failed: {str(e)}",
                agent=self.parent.name if self.parent else self.name,
                cause=e
            ) from e
    
    async def _load_reflection_template(self) -> object:
        """Load the reflection prompt template.
        
        This method attempts to load the template in the following order:
        1. From the parent component if it has a template getter
        2. From the resource registry
        3. Default built-in template
        
        Returns:
            Reflection prompt template
        """
        # Try to get the template from parent
        if self.parent and hasattr(self.parent, "get_template"):
            try:
                template = self.parent.get_template("reflection")
                if template:
                    logger.info("Using reflection template from parent")
                    return template
            except Exception as e:
                logger.warning(f"Failed to load reflection template from parent: {str(e)}")
        
        # Try to get the standard template from registry
        from ...resources.registry import resource_registry
        from ...resources.constants import ResourceType
        
        try:
            if resource_registry.contains("reflection_default", ResourceType.PROMPT):
                template = resource_registry.get_sync("reflection_default", ResourceType.PROMPT)
                logger.info("Using reflection_default template from registry")
                return template
        except Exception as e:
            logger.warning(f"Failed to get reflection template from registry: {str(e)}")
        
        # If we reach here, use the default implementation directly
        logger.info("Using built-in DefaultReflectionPrompt")
        return DefaultReflectionPrompt()
    
    async def _prepare_reflection_input(
        self,
        state: AgentState,
        flow_name: str,
        flow_inputs: BaseModel,
        flow_result: FlowResult,
        memory_context: Optional[str] = None
    ) -> ReflectionInput:
        """
        Prepare a standardized pydantic model for the reflection process.
        
        Args:
            state: Agent state
            flow_name: Name of the executed flow
            flow_inputs: Inputs provided to the flow (a Pydantic model)
            flow_result: Result from flow execution (a FlowResult)
            memory_context: Optional memory context for this reflection
            
        Returns:
            ReflectionInput pydantic model with all required data
            
        Raises:
            ReflectionError: If required data is missing or invalid
        """
        # Ensure required data is present
        if not state:
            raise ReflectionError(
                message="State cannot be None",
                agent=self.parent.name if self.parent else self.name
            )
            
        if not state.task_description:
            raise ReflectionError(
                message="Task description cannot be empty",
                agent=self.parent.name if self.parent else self.name
            )
            
        # Generate summary of state for reflection
        state_summary = f"Task: {state.task_description}\nProgress: {state.progress}%\nCycle: {state.cycles}\nComplete: {state.is_complete}"
        if state.completion_reason:
            state_summary += f"\nCompletion reason: {state.completion_reason}"
            
        # Format execution history
        execution_history_text = self._format_execution_history(state)
            
        # Extract planning rationale
        planning_rationale = self._extract_planning_rationale(state)
        
        # Create a new ReflectionInput model
        return ReflectionInput(
            task_description=state.task_description,
            flow_name=flow_name,
            flow_status=str(flow_result.status),
            flow_result=flow_result,
            flow_inputs=flow_inputs,
            state_summary=state_summary,
            execution_history_text=execution_history_text,
            planning_rationale=planning_rationale,
            cycle=state.cycles,
            progress=state.progress,
            memory_context=memory_context
        )
    
    def _format_execution_history(self, state: AgentState) -> str:
        """
        Format execution history as readable text.
        
        Args:
            state: Agent state
            
        Returns:
            Formatted execution history text
        """
        if not state.execution_history or len(state.execution_history) == 0:
            return "No execution history available"
            
        history_items = []
        for i, entry in enumerate(state.execution_history):
            if isinstance(entry, dict):
                flow = entry.get('flow_name', 'unknown')
                status = entry.get('status', 'unknown')
                cycle = entry.get('cycle', 0)
                history_items.append(f"{i+1}. Cycle {cycle}: Executed {flow} with status {status}")
            else:
                history_items.append(f"{i+1}. {str(entry)}")
                
        return "\n".join(history_items)
    
    def _extract_planning_rationale(self, state: AgentState) -> str:
        """
        Extract planning rationale from recent execution history.
        
        Args:
            state: Agent state
            
        Returns:
            Planning rationale text
        """
        # Try to extract planning rationale from recent history
        if state.execution_history:
            for entry in reversed(state.execution_history):
                if isinstance(entry, dict) and "planning" in entry.get("flow_name", "").lower():
                    rationale = entry.get("planning_rationale", "")
                    if rationale:
                        return rationale
        
        return "No planning rationale available"
    
    async def _execute_reflection(self, reflection_input: ReflectionInput) -> ReflectionResult:
        """
        Execute reflection using the LLM provider.
        
        Args:
            reflection_input: Reflection input with all required data
            
        Returns:
            ReflectionResult with analysis, progress and completion status
            
        Raises:
            ReflectionError: If reflection execution fails
        """
        try:
            # Check if we have the required components
            if not self._llm_provider:
                raise ReflectionError(
                    message="No LLM provider available for reflection",
                    agent=self.parent.name if self.parent else self.name
                )
                
            if not self._reflection_template:
                raise ReflectionError(
                    message="No reflection template available",
                    agent=self.parent.name if self.parent else self.name
                )
                
            # Format flow result for template
            flow_result_formatted = self._format_flow_result(reflection_input.flow_result)
            
            # Format flow inputs for template
            flow_inputs_formatted = self._format_flow_inputs(reflection_input.flow_inputs)
                
            # Prepare variables for the template
            template_variables = {
                "task_description": reflection_input.task_description,
                "cycle": reflection_input.cycle,
                "flow_name": reflection_input.flow_name,
                "flow_status": reflection_input.flow_status,
                "flow_inputs": flow_inputs_formatted,
                "flow_result": flow_result_formatted,
                "execution_history": reflection_input.execution_history_text,
                "planning_rationale": reflection_input.planning_rationale,
                "state_summary": reflection_input.state_summary,
                "current_progress": reflection_input.progress
            }
            
            # Generate reflection using the structured generation interface
            result = await self._llm_provider.generate_structured(
                prompt=self._reflection_template,
                output_type=ReflectionResult,
                model_name=self._config.model_name,
                prompt_variables=template_variables
            )
            
            # Ensure progress is between 0 and 100
            result.progress = max(0, min(100, result.progress))
            
            return result
            
        except Exception as e:
            raise ReflectionError(
                message=f"Failed to execute reflection: {str(e)}",
                agent=self.parent.name if self.parent else self.name,
                cause=e
            ) from e
            
    def _format_flow_result(self, flow_result: FlowResult) -> str:
        """
        Format a FlowResult as human-readable text for the template.
        
        Args:
            flow_result: FlowResult to format
            
        Returns:
            Formatted string representation
        """
        # Start with basic info
        lines = [
            f"Status: {flow_result.status}",
            f"Timestamp: {flow_result.timestamp.isoformat()}",
        ]
        
        # Add duration if available
        if flow_result.duration is not None:
            lines.append(f"Duration: {flow_result.duration:.3f} seconds")
        
        # Add error info if present
        if flow_result.error:
            lines.append(f"Error: {flow_result.error}")
        
        # Format the data section
        lines.append("Data:")
        if hasattr(flow_result.data, "model_dump") and callable(flow_result.data.model_dump):
            data_dict = flow_result.data.model_dump()
            for key, value in data_dict.items():
                lines.append(f"  {key}: {value}")
        elif isinstance(flow_result.data, dict):
            for key, value in flow_result.data.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {flow_result.data}")
        
        # Join all lines with newlines
        return "\n".join(lines)
    
    def _format_flow_inputs(self, flow_inputs: BaseModel) -> str:
        """
        Format flow inputs as human-readable text for the template.
        
        Args:
            flow_inputs: Flow inputs (Pydantic model)
            
        Returns:
            Formatted string representation
        """
        # Get model data as dictionary
        if hasattr(flow_inputs, "model_dump") and callable(flow_inputs.model_dump):
            data_dict = flow_inputs.model_dump()
        else:
            data_dict = flow_inputs.__dict__
        
        # Format each field
        lines = [f"Model type: {flow_inputs.__class__.__name__}"]
        
        for key, value in data_dict.items():
            if key.startswith("_"):
                continue
                
            # Format the value based on its type
            if isinstance(value, dict):
                value_str = ", ".join(f"{k}: {v}" for k, v in value.items())
                lines.append(f"{key}: {{{value_str}}}")
            elif isinstance(value, list):
                if len(value) <= 3:
                    value_str = ", ".join(str(item) for item in value)
                    lines.append(f"{key}: [{value_str}]")
                else:
                    lines.append(f"{key}: [List with {len(value)} items]")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
