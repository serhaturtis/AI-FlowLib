"""Conversation flow implementations for agent operations.

This module provides flow implementations for conversational agent functionality,
including message handling, planning, input generation, and reflection.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Type, Union

from pydantic import BaseModel, Field, RootModel

import flowlib as fl
from flowlib.core.models import Context
from flowlib.core.errors import ValidationError, ExecutionError, ErrorContext
from flowlib.core.registry.constants import ProviderType, ResourceType

from ..models import PlanningResponse, ReflectionResponse, MemoryItem

logger = logging.getLogger(__name__)

class MessageInput(BaseModel):
    """Input for conversation flow containing a single message."""
    message: str = Field(..., description="User message text")
    model_name: str = Field("default", description="Model name to use for conversation")

class ConversationOutput(BaseModel):
    """Output from conversation flow."""
    response: str = Field(..., description="Agent response text")
    detected_intent: Optional[str] = Field(None, description="Detected user intent")
    suggested_flow: Optional[str] = Field(None, description="Suggested flow to execute")
    requires_task_execution: bool = Field(False, description="Whether this requires executing a task")

class PlanningInput(BaseModel):
    """Input for planning flow."""
    task_description: str = Field(..., description="Description of the agent's task")
    available_flows: List[Dict[str, Any]] = Field(..., description="List of available flows")
    current_state: Dict[str, Any] = Field(..., description="Current agent state")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent execution history")
    model_name: str = Field("default", description="Model name to use for planning")

class InputGenerationInput(BaseModel):
    """Input for input generation flow."""
    flow_name: str = Field(..., description="Name of the selected flow")
    input_schema: Dict[str, Any] = Field(..., description="Schema of the flow's input")
    task_description: str = Field(..., description="Description of the agent's task")
    current_state: Dict[str, Any] = Field(..., description="Current agent state")
    planning_reasoning: str = Field("", description="Reasoning from planning stage")
    model_name: str = Field("default", description="Model name to use for input generation")
    output_type: str = Field(..., description="Name of the output type for validation")

# Define a proper root model for dynamically generated flow inputs
class GeneratedFlowInputs(RootModel):
    """Output from input generation flow containing dynamically generated inputs for a flow."""
    root: Dict[str, Any]
    
    # We provide dictionary-like access methods for backward compatibility
    def __getitem__(self, key):
        return self.root[key]
    
    def __iter__(self):
        return iter(self.root)
    
    def items(self):
        return self.root.items()
    
    def keys(self):
        return self.root.keys()
    
    def values(self):
        return self.root.values()
    
    def get(self, key, default=None):
        return self.root.get(key, default)
    
    def __str__(self):
        """Custom string representation for clearer flow schema display."""
        return "GeneratedFlowInputs (Dictionary of dynamically generated inputs)"

class ReflectionInput(BaseModel):
    """Input for reflection flow."""
    flow_name: str = Field(..., description="Name of the executed flow")
    flow_inputs: str = Field(..., description="Inputs provided to the flow")
    flow_outputs: str = Field(..., description="Outputs from the flow execution")
    task_description: str = Field(..., description="Description of the agent's task")
    current_state: Dict[str, Any] = Field(..., description="Current agent state")
    model_name: str = Field("default", description="Model name to use for reflection")


@fl.flow(name="agent-planning-flow", is_infrastructure=True)
class AgentPlanningFlow:
    """Flow for agent planning decisions."""
    
    # Add schema definitions directly to flow class
    input_schema = PlanningInput
    output_schema = PlanningResponse
    description = "A flow that determines the next action for the agent to take based on task description and available flows."
    
    @fl.stage(input_model=PlanningInput, output_model=PlanningResponse)
    async def plan(self, context: fl.Context) -> PlanningResponse:
        """Generate a planning decision using LLM."""
        # Get input data
        input_data = context.data
        
        try:
            # Get planning prompt template
            planning_prompt = await fl.resource_registry.get(
                "agent-planning", 
                resource_type=ResourceType.PROMPT
            )
            
            # Get LLM provider
            provider_name = "llamacpp"  # Could be configurable
            llm = await fl.provider_registry.get(ProviderType.LLM, provider_name)
            
            # Format prompt with input data
            flows_text = self._format_flows(input_data.available_flows)
            state_text = self._format_state(input_data.current_state)
            history_text = self._format_history(input_data.execution_history)
            
            # Debug: Log the available flow names
            flow_names = [f["name"] for f in input_data.available_flows]
            logger.info(f"Planning flow received these flow names: {flow_names}")
            
            formatted_prompt = planning_prompt.template.format(
                available_flows=flows_text,
                task_description=input_data.task_description,
                current_state=state_text,
                execution_history=history_text
            )
            
            # DEBUG: Print the formatted prompt
            print("\n===== PLANNING PROMPT =====")
            print(formatted_prompt)
            print("===== END PLANNING PROMPT =====\n")
            
            # Generate planning response
            result = await llm.generate_structured(
                formatted_prompt,
                PlanningResponse,
                input_data.model_name,
                **planning_prompt.config
            )
            
            # Log the planning decision
            logger.info(f"Planning flow decided to use flow: '{result.selected_flow}' based on task: '{input_data.task_description}'")
            logger.info(f"Planning reasoning: {result.reasoning[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in planning flow: {str(e)}")
            raise ExecutionError(
                message=f"Planning flow failed: {str(e)}",
                context=ErrorContext.create(
                    task=input_data.task_description,
                    available_flows=[f["name"] for f in input_data.available_flows]
                ),
                cause=e
            )
    
    def _format_flows(self, flows: List[Dict[str, Any]]) -> str:
        """Format flows for prompt."""
        result = []
        print("\n===== FORMATTING FLOWS FOR PROMPT =====")
        print(f"Received {len(flows)} flows to format")
        
        for flow in flows:
            # Format the flow name to stand out clearly
            flow_name = flow.get('name', 'unknown')
            print(f"Processing flow: {flow_name}")
            print(f"  Full flow data: {flow}")
            
            flow_text = f"Flow name: {flow_name}\n"  # Remove quotes to avoid confusion
            
            if flow.get('description'):
                flow_text += f"Description: {flow.get('description')}\n"
                print(f"  Added description: {flow.get('description')}")
            if flow.get('input_schema'):
                flow_text += f"Input: {flow.get('input_schema')}\n"
                print(f"  Added input schema: {flow.get('input_schema')}")
            else:
                print("  No input_schema found in flow data")
            if flow.get('output_schema'):
                flow_text += f"Output: {flow.get('output_schema')}\n"
                print(f"  Added output schema: {flow.get('output_schema')}")
            else:
                print("  No output_schema found in flow data")
            result.append(flow_text)
        
        # Add a note about flow names
        if result:
            result.append("\nNOTE: When selecting a flow, use the exact flow name without any quotes.")
        
        print("===== END FORMATTING FLOWS =====\n")
        return "\n".join(result) if result else "No flows available."
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state for prompt."""
        result = []
        for key, value in state.items():
            result.append(f"{key}: {value}")
        
        return "\n".join(result) if result else "No state information."
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format execution history for prompt."""
        result = []
        for i, step in enumerate(history, 1):
            if step.get("action") == "execute_flow":
                result.append(f"""
Step {i}:
  Flow: {step.get('flow')}
  Reasoning: {step.get('reasoning')}
  Reflection: {step.get('reflection', 'None')}
""")
            elif step.get("action") == "error":
                result.append(f"""
Step {i}:
  Error: {step.get('error')}
""")
        
        return "\n".join(result) if result else "No execution history yet."
    
    @fl.pipeline(input_model=PlanningInput, output_model=PlanningResponse)
    async def execute_planning(self, input_data: PlanningInput) -> PlanningResponse:
        """Execute the planning pipeline."""
        context = fl.Context(data=input_data)
        plan_stage = self.get_stage("plan")
        result = await plan_stage.execute(context)
        return result.data


@fl.flow(name="agent-input-generation-flow", is_infrastructure=True)
class AgentInputGenerationFlow:
    """Flow for generating inputs for a selected flow."""
    
    # Add schema definitions directly to flow class
    input_schema = InputGenerationInput
    output_schema = GeneratedFlowInputs
    description = "A flow that generates structured input data for the selected flow based on the flow's input schema and task context."
    
    @fl.stage(input_model=InputGenerationInput, output_model=GeneratedFlowInputs)
    async def generate_inputs(self, context: fl.Context) -> GeneratedFlowInputs:
        """Generate inputs for the selected flow."""
        # Get input data
        input_data = context.data
        
        try:
            # Get input generation prompt template
            input_gen_prompt = await fl.resource_registry.get(
                "agent-input-generation", 
                resource_type=ResourceType.PROMPT
            )
            
            # Get LLM provider
            provider_name = "llamacpp"  # Could be configurable
            llm = await fl.provider_registry.get(ProviderType.LLM, provider_name)
            
            # Format prompt
            formatted_prompt = input_gen_prompt.template.format(
                flow_name=input_data.flow_name,
                input_schema=json.dumps(input_data.input_schema, indent=2),
                task_description=input_data.task_description,
                current_state=self._format_state(input_data.current_state),
                planning_reasoning=input_data.planning_reasoning
            )
            
            # DEBUG: Print the formatted prompt
            print("\n===== INPUT GENERATION PROMPT =====")
            print(formatted_prompt)
            print("===== END INPUT GENERATION PROMPT =====\n")
            
            # Generate inputs as JSON
            json_response = await llm.generate_json(
                formatted_prompt,
                input_data.model_name,
                temperature=0.3
            )
            
            # Parse the JSON response
            return GeneratedFlowInputs(root=json.loads(json_response))
            
        except Exception as e:
            logger.error(f"Error in input generation flow: {str(e)}")
            raise ExecutionError(
                message=f"Input generation flow failed: {str(e)}",
                context=ErrorContext.create(
                    flow_name=input_data.flow_name,
                    task=input_data.task_description
                ),
                cause=e
            )
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state for prompt."""
        result = []
        for key, value in state.items():
            result.append(f"{key}: {value}")
        
        return "\n".join(result) if result else "No state information."
    
    @fl.pipeline(input_model=InputGenerationInput, output_model=GeneratedFlowInputs)
    async def execute_input_generation(self, input_data: InputGenerationInput) -> GeneratedFlowInputs:
        """Execute the input generation pipeline."""
        context = fl.Context(data=input_data)
        generate_stage = self.get_stage("generate_inputs")
        result = await generate_stage.execute(context)
        return result.data


@fl.flow(name="agent-reflection-flow", is_infrastructure=True)
class AgentReflectionFlow:
    """Flow for agent reflection on executed flows."""
    
    # Add schema definitions directly to flow class
    input_schema = ReflectionInput
    output_schema = ReflectionResponse
    description = "A flow that reflects on the outputs of executed flows and extracts insights and learnings."
    
    @fl.stage(input_model=ReflectionInput, output_model=ReflectionResponse)
    async def reflect(self, context: fl.Context) -> ReflectionResponse:
        """Reflect on flow execution results."""
        # Get input data
        input_data = context.data
        
        try:
            # Get reflection prompt template
            reflection_prompt = await fl.resource_registry.get(
                "agent-reflection", 
                resource_type=ResourceType.PROMPT
            )
            
            # Get LLM provider
            provider_name = "llamacpp"  # Could be configurable
            llm = await fl.provider_registry.get(ProviderType.LLM, provider_name)
            
            # Format prompt
            formatted_prompt = reflection_prompt.template.format(
                flow_name=input_data.flow_name,
                inputs=input_data.flow_inputs,
                outputs=input_data.flow_outputs,
                task_description=input_data.task_description,
                current_state=self._format_state(input_data.current_state)
            )
            
            # DEBUG: Print the formatted prompt
            print("\n===== REFLECTION PROMPT =====")
            print(formatted_prompt)
            print("===== END REFLECTION PROMPT =====\n")
            
            # DEBUG: Print the model schema
            print("\n===== EXPECTED MODEL SCHEMA =====")
            if hasattr(ReflectionResponse, "model_json_schema"):
                print(json.dumps(ReflectionResponse.model_json_schema(), indent=2))
            else:
                print(json.dumps(ReflectionResponse.schema(), indent=2))
            print("===== END MODEL SCHEMA =====\n")
            
            # Generate reflection
            try:
                print("\n===== GENERATING REFLECTION =====")
                print(f"Input model: {ReflectionResponse.__name__}")
                print(f"Model expected fields: {list(ReflectionResponse.__fields__.keys()) if hasattr(ReflectionResponse, '__fields__') else 'Unknown'}")
                
                result = await llm.generate_structured(
                    formatted_prompt,
                    ReflectionResponse,
                    input_data.model_name,
                    **reflection_prompt.config
                )
                print("===== GENERATION SUCCESSFUL =====\n")
                
            except Exception as gen_error:
                print("\n===== REFLECTION GENERATION ERROR =====")
                print(f"Error type: {type(gen_error).__name__}")
                print(f"Error message: {str(gen_error)}")
                if hasattr(gen_error, '__cause__') and gen_error.__cause__:
                    print(f"Cause: {gen_error.__cause__}")
                    print(f"Cause type: {type(gen_error.__cause__).__name__}")
                
                # Try to get the raw response if available in the error context
                if hasattr(gen_error, 'context') and gen_error.context and hasattr(gen_error.context, 'response'):
                    print(f"Raw response: {gen_error.context.response}")
                print("===== END ERROR DETAILS =====\n")
                raise
            
            # Special handling for conversation tasks - if this is just a basic greeting or question response,
            # mark it as complete after one execution
            if (input_data.flow_name == "conversation-flow" and
                ("Understand and respond to user message:" in input_data.task_description or
                 "respond to user message:" in input_data.task_description)):
                # Check if we have already generated a response
                if not result.is_complete and hasattr(result, "progress") and result.progress < 100:
                    logger.info("Basic conversation task detected - marking as complete after generating response")
                    # Update the reflection result
                    result.is_complete = True
                    result.progress = 100
                    result.completion_reason = "User message has been processed and a response was generated."
            
            # Log the decision about task completion
            logger.info(f"Reflection on flow '{input_data.flow_name}' - Is task complete: {result.is_complete}")
            if result.is_complete:
                logger.info(f"Task marked as complete. Reason: {result.completion_reason}")
            else:
                logger.info(f"Task continuing with progress: {result.progress}%")
            
            # Debug print of the final result
            print("\n===== FINAL REFLECTION RESULT =====")
            print(f"Reflection: {result.reflection[:100]}...")
            print(f"Progress: {result.progress}")
            print(f"Is complete: {result.is_complete}")
            print(f"Completion reason: {result.completion_reason}")
            print("===== END REFLECTION RESULT =====\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in reflection flow: {str(e)}")
            print(f"\n===== REFLECTION FLOW ERROR =====\nType: {type(e).__name__}\nError: {str(e)}")
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"Cause: {str(e.__cause__)}")
                print(f"Cause type: {type(e.__cause__).__name__}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            print("===== END ERROR DETAILS =====\n")
            
            # Return a basic reflection to continue execution
            return ReflectionResponse(
                reflection=f"Error generating reflection: {str(e)}",
                progress=input_data.current_state.get("progress", 0),
                is_complete=False,
                completion_reason=None
            )
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state for prompt."""
        result = []
        for key, value in state.items():
            result.append(f"{key}: {value}")
        
        return "\n".join(result) if result else "No state information."
    
    @fl.pipeline(input_model=ReflectionInput, output_model=ReflectionResponse)
    async def execute_reflection(self, input_data: ReflectionInput) -> ReflectionResponse:
        """Execute the reflection pipeline."""
        context = fl.Context(data=input_data)
        reflect_stage = self.get_stage("reflect")
        result = await reflect_stage.execute(context)
        return result.data


@fl.flow(name="conversation-flow")
class ConversationFlow:
    """Flow for handling user conversation."""
    
    # Add schema definitions directly to flow class
    input_schema = MessageInput
    output_schema = ConversationOutput
    description = "A flow that processes a user message, considers conversation history, and generates a contextually appropriate response."
    
    @fl.stage(input_model=MessageInput, output_model=ConversationOutput)
    async def process_message(self, context: fl.Context) -> ConversationOutput:
        """Process a user message and generate a response."""
        # Get input data
        input_data = context.data
        logger.info(f"Processing user message: '{input_data.message}'")
        
        try:
            # Get conversation prompt template
            conversation_prompt = await fl.resource_registry.get(
                "agent-conversation", 
                resource_type=ResourceType.PROMPT
            )
            
            # Get LLM provider
            provider_name = "llamacpp"  # Could be configurable
            llm = await fl.provider_registry.get(ProviderType.LLM, provider_name)
            
            # Retrieve conversation history from memory
            try:
                memory_provider = await fl.provider_registry.get(ProviderType.CACHE, "memory-cache")
                conversation_history = await memory_provider.get("conversation_history") or []
                logger.info(f"Retrieved conversation history with {len(conversation_history)} messages")
                
                # Debug: show conversation history content
                if conversation_history:
                    logger.info("Conversation history summary:")
                    for i, msg in enumerate(conversation_history[-3:]):  # Show last 3 messages
                        logger.info(f"  Message {i+1}: {msg.get('role', 'unknown')}: {msg.get('content', '')[:30]}...")
                else:
                    logger.info("No previous conversation history found")
                    
            except Exception as e:
                logger.warning(f"Error retrieving conversation history: {str(e)}")
                conversation_history = []
            
            # Format conversation history
            history_text = self._format_history(conversation_history)
            
            # Format prompt
            formatted_prompt = conversation_prompt.template.format(
                message=input_data.message,
                conversation_history=history_text,
                agent_state="No agent state available."  # Add default value for agent_state
            )
            
            # Generate response
            result = await llm.generate_structured(
                formatted_prompt,
                ConversationOutput,
                input_data.model_name,
                **conversation_prompt.config
            )
            
            logger.info(f"Generated response: '{result.response[:50]}...'")
            
            # Update conversation history in memory
            try:
                # Check if the current user message is already in history
                if conversation_history and len(conversation_history) >= 2:
                    last_user_msg = next((msg for msg in reversed(conversation_history) 
                                         if msg.get('role') == 'user'), None)
                

                # Add to conversation history
                conversation_history.append({"role": "user", "content": input_data.message})
                conversation_history.append({"role": "assistant", "content": result.response})
                logger.info(f"Added new conversation turn to history, now {len(conversation_history)} messages")
                
                # Keep only the last 20 messages
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                    logger.info(f"Trimmed history to last 20 messages")
                    
                # Store updated history in memory with longer TTL
                await memory_provider.set("conversation_history", conversation_history, ttl=86400)  # 24 hours
                logger.info("Successfully saved updated conversation history to memory")
                
            except Exception as e:
                logger.warning(f"Error updating conversation history: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in conversation flow: {str(e)}")
            
            # Return a basic response
            return ConversationOutput(
                response=f"I'm sorry, I encountered an error processing your message. {str(e)}",
                detected_intent="error",
                suggested_flow=None,
                requires_task_execution=False
            )
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return "No previous conversation."
            
        result = []
        # Format all messages, not just the last 10
        for message in history:
            if message.get("role") == "user":
                result.append(f"User: {message.get('content', '')}")
            else:
                result.append(f"FlowBot: {message.get('content', '')}")
        
        # Add an indication of the conversation length
        if len(history) > 0:
            result.insert(0, f"--- Conversation history ({len(history)} messages) ---\n")
            
        return "\n".join(result)
    
    @fl.pipeline(input_model=MessageInput, output_model=ConversationOutput)
    async def execute_conversation(self, input_data: MessageInput) -> ConversationOutput:
        """Execute the conversation pipeline."""
        context = fl.Context(data=input_data)
        process_stage = self.get_stage("process_message")
        result = await process_stage.execute(context)
        return result.data 