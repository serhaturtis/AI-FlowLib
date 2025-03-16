"""LLM-based agent implementation.

This module provides a concrete implementation of the Agent class
that uses LLM calls for planning, generating inputs, and reflection.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel

from ..core.errors import ValidationError, ExecutionError, ErrorContext
from ..core.models import Context
from ..flows import Flow

from .base import Agent
from .models import PlanningResponse, ReflectionResponse, FlowDescription

logger = logging.getLogger(__name__)

class LLMAgent(Agent):
    """LLM-based agent that uses language models for reasoning.
    
    This agent uses:
    1. A planner model to decide which flows to execute
    2. An input generator model to create appropriate inputs for flows
    3. A reflection model to process results and update memory
    """
    
    async def _plan_next_action(self) -> PlanningResponse:
        """Plan the next action based on current state using LLM.
        
        Returns:
            Planning response with next action decision
        """
        # Ensure provider is initialized
        await self.initialize_provider()
        
        # Create prompt for planning
        system_prompt = self._create_planning_system_prompt()
        user_prompt = self._create_planning_user_prompt()
        
        try:
            # Get planning response from LLM
            response = await self.provider.generate_structured_output(
                model=self.config.planner_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=PlanningResponse,
                max_retries=self.config.max_retries
            )
            
            return response
            
        except Exception as e:
            # Handle LLM errors
            error_msg = f"Error during planning: {str(e)}"
            logger.error(error_msg)
            
            # Raise as execution error
            raise ExecutionError(
                error_msg,
                ErrorContext.create(
                    error_type="planning_error",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                ),
                cause=e
            )
    
    async def _generate_flow_inputs(
        self, 
        flow_desc: FlowDescription,
        planning: PlanningResponse
    ) -> Any:
        """Generate inputs for the selected flow using LLM.
        
        Args:
            flow_desc: Description of the selected flow
            planning: Planning response with context
            
        Returns:
            Generated inputs for the flow
        """
        # Ensure provider is initialized
        await self.initialize_provider()
        
        # Get input schema
        input_schema = flow_desc.input_schema
        
        if not input_schema:
            logger.warning(f"Flow {flow_desc.name} does not have an input schema")
            return {}
        
        # Create prompts for input generation
        system_prompt = self._create_input_gen_system_prompt(flow_desc)
        user_prompt = self._create_input_gen_user_prompt(flow_desc)
        
        try:
            # If the input schema is a Pydantic model, generate structured output
            if isinstance(input_schema, type) and issubclass(input_schema, BaseModel):
                return await self.provider.generate_structured_output(
                    model=self.config.input_generator_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_type=input_schema,
                    max_retries=self.config.max_retries
                )
            
            # Otherwise generate a JSON response
            json_str = await self.provider.generate_json(
                model=self.config.input_generator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=self.config.max_retries
            )
            
            # Parse JSON and validate against schema if possible
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValidationError(
                    f"Generated inputs are not valid JSON: {json_str}",
                    ErrorContext.create(
                        error_type="json_decode_error",
                        json_str=json_str
                    ),
                    cause=e
                )
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating inputs for flow {flow_desc.name}: {str(e)}"
            logger.error(error_msg)
            
            # Raise as execution error
            raise ExecutionError(
                error_msg,
                ErrorContext.create(
                    error_type="input_generation_error",
                    flow_name=flow_desc.name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                ),
                cause=e
            )
    
    async def _reflect_on_result(
        self,
        flow_desc: FlowDescription,
        inputs: Any,
        result: Context
    ) -> ReflectionResponse:
        """Reflect on flow execution results using LLM.
        
        Args:
            flow_desc: Description of the executed flow
            inputs: Inputs provided to the flow
            result: Result context from flow execution
            
        Returns:
            Reflection on the results
        """
        # Ensure provider is initialized
        await self.initialize_provider()
        
        # Create prompts for reflection
        system_prompt = self._create_reflection_system_prompt()
        user_prompt = self._create_reflection_user_prompt(flow_desc, inputs, result)
        
        try:
            # Get reflection from LLM
            reflection = await self.provider.generate_structured_output(
                model=self.config.reflection_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=ReflectionResponse,
                max_retries=self.config.max_retries
            )
            
            return reflection
            
        except Exception as e:
            # Handle errors
            error_msg = f"Error reflecting on results: {str(e)}"
            logger.error(error_msg)
            
            # Return minimal reflection to continue execution
            logger.info("Using minimal reflection due to error")
            return ReflectionResponse(
                reflection="Error generating reflection",
                progress=self.state.progress,
                is_complete=False,
                completion_reason=None,
                new_information=[]
            )
    
    def _create_planning_system_prompt(self) -> str:
        """Create system prompt for planning.
        
        Returns:
            System prompt for planning
        """
        # Get flow descriptions
        flows = self.get_flow_descriptions()
        
        # Create flow descriptions text
        flows_text = "\n\n".join([
            f"Flow: {flow['name']}\n"
            f"Input: {flow['input_schema']}\n"
            f"Output: {flow['output_schema']}"
            for flow in flows
        ])
        
        # Create system prompt
        return f"""You are a planning system for an AI agent that completes tasks by selecting appropriate flows to execute.

Available flows:
{flows_text}

Your task is to analyze the current state and decide which flow to execute next, or determine if the task is complete.

Respond with:
- selected_flow: The name of the flow to execute next, or "COMPLETE" if the task is done
- reasoning: Detailed explanation of your decision
- is_complete: true/false indicating if the task is complete
- completion_reason: Reason for completion if is_complete is true, otherwise null

{self.config.default_system_prompt}
"""

    def _create_planning_user_prompt(self) -> str:
        """Create user prompt for planning.
        
        Returns:
            User prompt for planning
        """
        # Get state summary
        summary = self.get_state_summary()
        
        # Format execution history
        history_text = ""
        for step in self.state.execution_history[-5:]:  # Show last 5 steps
            step_type = step.get("action", "unknown")
            
            if step_type == "execute_flow":
                history_text += f"Step {step['step']}: Executed flow '{step['flow']}'\n"
                history_text += f"Reasoning: {step['reasoning']}\n"
                history_text += f"Inputs: {json.dumps(step['inputs'], default=str)}\n"
                history_text += f"Outputs: {json.dumps(step['outputs'], default=str)}\n"
                history_text += f"Reflection: {step['reflection']}\n\n"
            elif step_type == "error":
                history_text += f"Step {step['step']}: Error occurred\n"
                history_text += f"Error: {step['error']}\n\n"
        
        # Format memory
        memory_text = "\n".join([f"- {item}" for item in self.state.memory])
        
        # Create user prompt
        return f"""Task: {self.state.task_description}

Current state:
- Progress: {summary['progress']}
- Steps executed: {summary['steps_executed']}
- Errors: {summary['errors']}

Recent execution history:
{history_text}

Agent memory:
{memory_text or "No memory items yet"}

What is the next flow that should be executed to make progress on the task?
"""

    def _create_input_gen_system_prompt(self, flow_desc: FlowDescription) -> str:
        """Create system prompt for input generation.
        
        Args:
            flow_desc: Description of the flow
            
        Returns:
            System prompt for input generation
        """
        # Get information about the flow
        flow_name = flow_desc.name
        input_schema = flow_desc.input_schema
        
        # Create schema description
        if hasattr(input_schema, 'schema'):
            try:
                schema_json = json.dumps(input_schema.schema(), indent=2)
            except:
                schema_json = "Schema not available"
        else:
            schema_json = "Schema not available"
        
        # Create system prompt
        return f"""You are an input generator for a flow execution system.

Flow: {flow_name}
Input Schema: {input_schema.__name__ if input_schema else "None"}

Schema Details:
{schema_json}

Your task is to generate valid inputs for this flow based on the current context and state.
The inputs must be valid according to the schema definition provided.

{self.config.default_system_prompt}
"""

    def _create_input_gen_user_prompt(self, flow_desc: FlowDescription) -> str:
        """Create user prompt for input generation.
        
        Args:
            flow_desc: Description of the flow
            
        Returns:
            User prompt for input generation
        """
        # Get state summary
        summary = self.get_state_summary()
        
        # Format memory for context
        memory_text = "\n".join([f"- {item}" for item in self.state.memory])
        
        # Format execution history
        history_text = ""
        for step in self.state.execution_history[-3:]:  # Show last 3 steps
            step_type = step.get("action", "unknown")
            
            if step_type == "execute_flow":
                history_text += f"Step {step['step']}: Executed flow '{step['flow']}'\n"
                history_text += f"Inputs: {json.dumps(step['inputs'], default=str)}\n"
                history_text += f"Outputs: {json.dumps(step['outputs'], default=str)}\n\n"
        
        # Create user prompt
        return f"""Task: {self.state.task_description}

Flow to execute: {flow_desc.name}

Current state:
- Progress: {summary['progress']}
- Steps executed: {summary['steps_executed']}

Recent execution history:
{history_text or "No execution history yet"}

Agent memory:
{memory_text or "No memory items yet"}

Generate valid inputs for the flow.
"""

    def _create_reflection_system_prompt(self) -> str:
        """Create system prompt for reflection.
        
        Returns:
            System prompt for reflection
        """
        return f"""You are a reflection system for an AI agent that learns from flow executions.

Your task is to analyze the results of a flow execution and:
1. Summarize what was learned
2. Extract important information to remember
3. Evaluate progress toward the overall task
4. Determine if the task is complete

Respond with:
- reflection: A detailed reflection on the flow execution
- progress: Estimated progress toward task completion (0-100)
- is_complete: true/false indicating if the task is complete
- completion_reason: Reason for completion if is_complete is true, otherwise null
- new_information: List of important information to remember

{self.config.default_system_prompt}
"""

    def _create_reflection_user_prompt(
        self, 
        flow_desc: FlowDescription,
        inputs: Any,
        result: Context
    ) -> str:
        """Create user prompt for reflection.
        
        Args:
            flow_desc: Description of the executed flow
            inputs: Inputs provided to the flow
            result: Result of flow execution
            
        Returns:
            User prompt for reflection
        """
        # Format inputs and outputs for display
        inputs_text = json.dumps(inputs, default=str, indent=2)
        
        outputs = result.data
        outputs_text = json.dumps(outputs, default=str, indent=2)
        
        # Check for errors
        error = None
        if hasattr(result, 'error') and result.error:
            error = str(result.error)
        
        # Create user prompt
        prompt = f"""Task: {self.state.task_description}

Executed flow: {flow_desc.name}

Inputs:
{inputs_text}

Outputs:
{outputs_text}
"""

        if error:
            prompt += f"\nError: {error}\n"
            
        prompt += "\nReflect on this flow execution and its results in relation to the overall task."
        
        return prompt 