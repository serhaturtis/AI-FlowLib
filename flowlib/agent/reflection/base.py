"""
Base reflection implementation for the agent system.

This module provides the reflection component for the agent system,
which is responsible for analyzing execution results and determining
task progress.
"""

import logging
from typing import Any, Dict, Optional, List, Union, Type

# Import prompts to ensure registration
from .prompts import (
    DefaultReflectionPrompt,
    TaskCompletionReflectionPrompt
)
from .prompts.step_reflection import (
    DefaultStepReflectionPrompt
)

from ..core.base import BaseComponent
from ..core.errors import ReflectionError, NotInitializedError
from ..models.config import ReflectionConfig
from ..models.state import AgentState
from ...providers import ProviderType
from ...providers.registry import provider_registry
from ...flows.results import FlowResult
from .models import ReflectionResult, ReflectionInput, StepReflectionResult, StepReflectionInput
from .interfaces import ReflectionInterface
from ...providers.llm.base import LLMProvider
from pydantic import BaseModel, Field
from ...utils.formatting.conversation import format_execution_history
from ..models.plan import PlanExecutionOutcome # Assuming rename
from .models import PlanReflectionContext # Assuming rename

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
        self._step_reflection_template = None # Add template for step reflection
    
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
            self._step_reflection_template = await self._load_step_reflection_template() # Load step template
            
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
        plan_context: PlanReflectionContext # Use the new context model
    ) -> ReflectionResult:
        """
        Analyze the outcome of a full plan execution based on collected step reflections.
        
        Args:
            plan_context: Context containing overall plan status and step reflections.
            
        Returns:
            ReflectionResult with analysis, progress and completion status
            
        Raises:
            NotInitializedError: If reflection is not initialized
            ReflectionError: If reflection fails or plan_outcome is malformed
        """
        self._check_initialized()
        
        try:
            if not self._reflection_template:
                raise NotInitializedError(self._name, "reflect", "Overall reflection template not loaded")
            
            # Format step reflections for the prompt
            step_reflections_formatted = self._format_step_reflections(plan_context.step_reflections)
            
            # Prepare variables for the overall reflection template
            template_variables = {
                "task_description": plan_context.task_description,
                "plan_status": plan_context.plan_status,
                "plan_error": plan_context.plan_error or "No overall plan error reported.",
                "step_reflections_summary": step_reflections_formatted,
                "execution_history_text": plan_context.execution_history_text,
                "state_summary": plan_context.state_summary,
                "current_progress": plan_context.current_progress
            }
            
            # Generate overall reflection using structured generation
            result = await self._llm_provider.generate_structured(
                prompt=self._reflection_template,
                output_type=ReflectionResult, # The final output is still ReflectionResult
                model_name=self._config.model_name,
                prompt_variables=template_variables
            )

            # Ensure progress is clamped
            result.progress = max(0, min(100, result.progress))
            
            logger.info("Overall plan reflection complete.")
            return result
            
        except Exception as e:
            logger.error(f"Overall reflection failed: {str(e)}", exc_info=True)
            raise ReflectionError(
                message=f"Reflection failed: {str(e)}",
                agent=self.parent.name if self.parent else self.name,
                cause=e
            ) from e
    
    def _format_step_reflections(self, step_reflections: List[StepReflectionResult]) -> str:
        """Formats a list of StepReflectionResult into a string for the prompt."""
        if not step_reflections:
            return "No step reflections were recorded for this plan execution."
        
        lines = ["Summary of Plan Step Reflections:"]
        for i, sr in enumerate(step_reflections):
            lines.append(f"  Step {i+1} (ID: {sr.step_id}):")
            lines.append(f"    - Success: {sr.step_success}")
            lines.append(f"    - Reflection: {sr.reflection}")
            if sr.key_observation:
                lines.append(f"    - Key Observation: {sr.key_observation}")
        return "\n".join(lines)
    
    async def _load_reflection_template(self) -> object:
        """Load the overall reflection prompt template.

        Prioritizes loading from resource_registry["reflection_default"],
        then falls back to instantiating DefaultReflectionPrompt.
        
        Returns:
            Reflection prompt template
        """
        # Try to get the standard template from registry
        from ...resources.registry import resource_registry
        from ...resources.constants import ResourceType
        
        template_name = "reflection_default"
        try:
            if resource_registry.contains(template_name, ResourceType.PROMPT):
                template = resource_registry.get_sync(template_name, ResourceType.PROMPT)
                logger.info(f"Using '{template_name}' template from resource registry for overall reflection.")
                return template
        except Exception as e:
            logger.warning(f"Failed to get reflection template from registry: {str(e)}")
        
        # If we reach here, use the default implementation directly
        logger.info(f"'{template_name}' not found in registry. Using built-in DefaultReflectionPrompt for overall reflection.")
        return DefaultReflectionPrompt()
    
    async def _load_step_reflection_template(self) -> object:
        """Load the step reflection prompt template.

        Prioritizes loading from resource_registry["step_reflection_default"],
        then falls back to instantiating DefaultStepReflectionPrompt.

        Returns:
            Step reflection prompt template
        """
        # Try to get the standard template from registry
        from ...resources.registry import resource_registry
        from ...resources.constants import ResourceType
        template_name = "step_reflection_default"
        try:
            if resource_registry.contains(template_name, ResourceType.PROMPT):
                template = resource_registry.get_sync(template_name, ResourceType.PROMPT)
                logger.info(f"Using '{template_name}' template from resource registry for step reflection.")
                return template
        except Exception as e:
            logger.warning(f"Failed to get step reflection template from registry: {str(e)}")

        # If we reach here, use the default implementation directly
        logger.info(f"'{template_name}' not found in registry. Using built-in DefaultStepReflectionPrompt for step reflection.")
        return DefaultStepReflectionPrompt()
    
    async def _prepare_reflection_input(
        self,
        state: AgentState,
        plan_outcome: PlanExecutionOutcome,
        memory_context: Optional[str] = None
    ) -> ReflectionInput:
        """
        Prepare a standardized pydantic model for the reflection process.
        
        Args:
            state: Agent state
            plan_outcome: The outcome object from executing a plan.
            memory_context: Optional memory context for this reflection
            
        Returns:
            ReflectionInput pydantic model with all required data
            
        Raises:
            ReflectionError: If required data is missing or invalid
        """
        # Extract information from plan_outcome
        plan_status = plan_outcome.status
        final_flow_result = plan_outcome.result # This might be None or a FlowResult
        plan_error = plan_outcome.error # String error message
        
        # Synthesize information for the reflection prompt
        # Use a generic name or indicate plan status
        reflection_flow_name = "PlanExecution"
        if plan_status == "ERROR":
            reflection_flow_name = f"PlanExecution_Failed"
        elif plan_status == "NO_ACTION_NEEDED":
            reflection_flow_name = "PlanExecution_NoAction"
        
        # Prepare a FlowResult-like structure for the prompt, even on failure
        # Ensure final_flow_result is always a FlowResult instance
        if not isinstance(final_flow_result, FlowResult):
            logger.debug(f"plan_outcome did not contain a FlowResult (Status: {plan_status}). Creating dummy result.")
            final_flow_result = FlowResult(
                flow_name=reflection_flow_name,
                status=plan_status,
                message=plan_error or "No result available",
                data={}
            )
        
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
        
        # Extract planning rationale from the *current plan* if available, otherwise fallback
        planning_rationale = "No specific plan rationale available"
        # Check if current_plan exists in the state's underlying model data
        current_plan_data = state._data.get("current_plan")
        if current_plan_data and isinstance(current_plan_data, dict):
            if current_plan_data.get("overall_rationale"):
                planning_rationale = current_plan_data["overall_rationale"]
            elif current_plan_data.get("steps") and len(current_plan_data["steps"]) > 0:
                # Fallback to first step rationale if no overall rationale
                first_step = current_plan_data["steps"][0]
                if isinstance(first_step, dict) and first_step.get("rationale"):
                    planning_rationale = first_step["rationale"]
        else:
            # Fallback to history extraction if no current plan info in state
            planning_rationale = self._extract_planning_rationale_from_history(state)
        
        # Create a new ReflectionInput model
        # Note: flow_inputs is set to None as inputs are now embedded in history or not relevant in aggregate
        return ReflectionInput(
            task_description=state.task_description,
            flow_name=reflection_flow_name, # Use synthesized name
            flow_status=plan_status, # Use overall plan status
            flow_result=final_flow_result, # Pass the FlowResult (real or dummy)
            flow_inputs=None, 
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
        return format_execution_history(state.execution_history)
    
    def _extract_planning_rationale_from_history(self, state: AgentState) -> str:
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
            # Handle case where flow_inputs might be None for plan-level reflection
            if reflection_input.flow_inputs is not None:
                flow_inputs_formatted = self._format_flow_inputs(reflection_input.flow_inputs)
            else:
                flow_inputs_formatted = "N/A (Plan Level Reflection)"
                
            # Prepare variables for the template
            template_variables = {
                "task_description": reflection_input.task_description,
                "cycle": reflection_input.cycle,
                "flow_name": reflection_input.flow_name,
                "flow_status": reflection_input.flow_status,
                "flow_inputs": flow_inputs_formatted,
                "flow_result": flow_result_formatted,
                "execution_history_text": reflection_input.execution_history_text,
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

    async def step_reflect(
        self,
        step_input: StepReflectionInput # Use the dedicated input model
    ) -> StepReflectionResult:
        """Analyze the outcome of a single plan step."""
        self._check_initialized()

        if not self._step_reflection_template:
            raise NotInitializedError(self._name, "step_reflect", "Step reflection template not loaded")

        try:
            # Format inputs and results using existing helper methods
            flow_inputs_formatted = self._format_flow_inputs(step_input.flow_inputs)
            flow_result_formatted = self._format_flow_result(step_input.flow_result)

            # Prepare variables for the template
            template_variables = {
                "task_description": step_input.task_description,
                "step_id": step_input.step_id,
                "step_intent": step_input.step_intent,
                "step_rationale": step_input.step_rationale,
                "flow_name": step_input.flow_name,
                "flow_inputs_formatted": flow_inputs_formatted,
                "flow_result_formatted": flow_result_formatted,
                "current_progress": step_input.current_progress
            }

            # Generate reflection using structured generation
            result = await self._llm_provider.generate_structured(
                prompt=self._step_reflection_template,
                output_type=StepReflectionResult,
                model_name=self._config.model_name,
                prompt_variables=template_variables
            )

            # Ensure step_id is correctly carried over if LLM misses it
            if not result.step_id:
                result.step_id = step_input.step_id

            logger.info(f"Step reflection complete for step ID: {result.step_id}")
            return result

        except Exception as e:
            logger.error(f"Step reflection failed for step ID {step_input.step_id}: {str(e)}", exc_info=True)
            # Return a default failure reflection result?
            return StepReflectionResult(
                step_id=step_input.step_id,
                reflection=f"Step reflection failed: {str(e)}",
                step_success=False, # Assume failure if reflection errors
                key_observation="Reflection process encountered an error."
            )
