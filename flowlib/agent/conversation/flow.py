"""
Conversation Flow for Agent System

This module provides a basic conversation flow that allows the LLM to interact
with users directly, generating responses to messages without requiring
specialized flows for every type of interaction.
"""

import logging
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

from ...flows.base import Flow
from ...flows.decorators import pipeline
from ...providers import ProviderType
from ...providers.registry import provider_registry
from ...resources.constants import ResourceType
from ...resources.registry import resource_registry
from ...resources.decorators import prompt
from ..decorators.base import agent_flow

logger = logging.getLogger(__name__)


class ConversationInput(BaseModel):
    """Input model for conversation flow."""
    message: str = Field(..., description="The user message to respond to")
    language: Optional[str] = Field("English", description="Language to use for the response")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="List of conversation history entries (e.g., {'role': 'user'/'assistant', 'content': 'message'}) ")
    memory_context_summary: Optional[str] = Field(None, description="Summary of relevant memory context")
    task_result_summary: Optional[str] = Field(None, description="Summary of the result from a previously executed task")


class ConversationOutput(BaseModel):
    """Output model for conversation flow."""
    response: str = Field(..., description="The LLM's response to the user message")
    sentiment: Optional[str] = Field(None, description="Optional sentiment analysis of the response")


# Define a structured output model for the conversation
class ConversationResponse(BaseModel):
    """Structured model for conversation responses."""
    response: str = Field(..., description="The response to the user's message")


class ConversationExecuteInput(BaseModel):
    """Complete input model for the execute method including metadata fields."""
    input_data: Optional[ConversationInput] = None
    message: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = Field(None, description="Reasoning for executing this flow (for logging only)")
    flow_context: Optional[Dict[str, Any]] = Field(None, description="Flow execution context")
    
    class Config:
        extra = "allow"  # Allow extra fields for forward compatibility


@prompt("conversation-response-prompt")
class ConversationPrompt:
    """Prompt template for conversation responses."""
    
    template = """You are an AI assistant. Your persona is: {{persona}}

## Instructions
- Use the provided Conversation History, Memory Context, and Task Result (if any) to generate a relevant, helpful, and on-topic response to the latest User Message.
- If a Task Result is provided, present the key information from it clearly to the user (e.g., summarize the findings, state the outcome of the command).
- Maintain a consistent and friendly conversational tone.
- Respond in {{language}}.

## Conversation History
{{conversation_history}}

## Memory Context
{{memory_context_summary}}

## Task Result
{{task_result_summary}}

## Current User Message
{{message}}

YOUR RESPONSE:"""
    
    config = {
        "max_tokens": 500,
        "temperature": 0.7
    }


@agent_flow(
    name="ConversationFlow",
    description="Generate natural language responses to user messages",
    is_infrastructure=True
)
class ConversationFlow(Flow):
    """Flow that handles natural language conversations and generates responses."""
    
    def __init__(self):
        """Initialize the conversation flow."""
        super().__init__("ConversationFlow")
    
    def get_description(self) -> str:
        """Return the description of this flow."""
        return "Generate natural language responses to user messages"
    
    @pipeline(input_model=ConversationInput, output_model=ConversationOutput)
    async def process_conversation(self, input_data: ConversationInput) -> ConversationOutput:
        """Pipeline for conversation processing and response generation.
        
        Args:
            input_data: Conversation input data including message, history, and context summaries.
            
        Returns:
            Output containing the response
        """
        try:
            # Extract the message and structured context
            message = input_data.message
            language = input_data.language
            conversation_history = input_data.conversation_history
            memory_context_summary = input_data.memory_context_summary or "No specific memory context available."
            task_result_summary = input_data.task_result_summary or "No task result available."
            
            # Get persona from the parent agent
            agent_persona = "Default helpful assistant" # Fallback
            if hasattr(self, 'parent') and self.parent and hasattr(self.parent, 'persona'):
                agent_persona = self.parent.persona
            else:
                logger.warning("ConversationFlow could not access parent agent's persona. Using default.")
            
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history)
            
            # Create prompt variables
            prompt_vars = {
                "message": message,
                "language": language,
                "persona": agent_persona, # Pass the agent's persona
                "conversation_history": history_text,
                "memory_context_summary": memory_context_summary,
                "task_result_summary": task_result_summary,
            }
            
            # Log the conversation input
            logger.debug(f"Conversation input: message='{message}'")
            
            # Create the conversation prompt
            conversation_prompt = ConversationPrompt()
            llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
            
            # Generate response using LLM with structured output
            result = await llm.generate_structured(
                prompt=conversation_prompt,
                output_type=ConversationResponse,
                model_name="default",
                prompt_variables=prompt_vars,
            )
            
            # Get the response
            response = result.response
            
            # Add the response to the agent state as a system message if we have access to the state
            # Note: This logic might be better placed in the agent core after the flow result is received
            # if hasattr(self, 'context') and self.context and hasattr(self.context, 'state'):
            #     state = self.context.state
            #     if hasattr(state, 'add_system_message') and callable(state.add_system_message):
            #         state.add_system_message(response)
            #         logger.debug("Added system message to agent state")
            
            # Create and return the output
            return ConversationOutput(
                response=response
            )
            
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in conversation pipeline: {str(e)}", exc_info=True)
            
            # Re-raise the exception for proper error handling
            raise
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history as text.
        
        Args:
            conversation_history: List of conversation history entries (e.g., {'role': 'user', 'content': 'message'})
            
        Returns:
            Formatted conversation history text
        """
        if not conversation_history:
            return "No conversation history available."
            
        formatted = []
        for entry in conversation_history:
            role = entry.get("role", "unknown").capitalize()
            content = entry.get("content", "")
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)
    
    @classmethod
    def create(cls) -> Flow:
        """Create an instance of this flow.
        
        Returns:
            Flow instance
        """
        return cls() 