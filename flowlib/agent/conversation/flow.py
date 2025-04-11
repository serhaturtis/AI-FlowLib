"""
Conversation Flow for Agent System

This module provides a basic conversation flow that allows the LLM to interact
with users directly, generating responses to messages without requiring
specialized flows for every type of interaction.
"""

import logging
from typing import Any, Dict, Optional
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
    language: Optional[str] = Field(None, description="Language to use for the response")
    persona: Optional[str] = Field(None, description="Persona to use for the response")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context for the conversation"
    )


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
    
    template = """You are a helpful AI assistant. Respond to the following message from the user:

User message: {{message}}

{{context_text}}

RESPONSE GUIDELINES:
1. Provide a helpful, concise, and friendly response appropriate to the message.
2. Your response should be in plain conversational text without any formatting, markdown, or code blocks.
3. DO NOT include any metadata, JSON, function definitions, or system messages in your response.
4. DO NOT prefix your response with labels like "Response:" or "Assistant:".
5. DO NOT use technical formatting or delimiters like ```, <|im_start|>, or <|im_end|>.
6. DO write in a natural, conversational style as if you're having a direct conversation.
7. Keep your response focused on answering the user's question or responding to their message.

Respond directly:
"""
    config = {
        "max_tokens": 500,
        "temperature": 0.7
    }


@agent_flow(
    name="ConversationFlow",
    description="Generate natural language responses to user messages"
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
            input_data: Conversation input data
            
        Returns:
            Output containing the response
        """
        try:
            # Extract the message and any config
            message = input_data.message
            language = input_data.language
            persona = input_data.persona
            context = input_data.context
            
            # Create prompt variables
            prompt_vars = {
                "message": message,
                "language": language or "English",
                "persona": persona or "helpful assistant",
                "context_text": self._format_context(context) if context else "",
            }
            
            # Log the conversation input
            logger.debug(f"Conversation input: message='{message}'")
            
            try:
                # Create the conversation prompt
                conversation_prompt = ConversationPrompt()
                llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
                # Generate response using LLM
                result = await llm.generate_structured(
                    prompt=conversation_prompt,
                    output_type=ConversationResponse,
                    model_name="default",
                    prompt_variables=prompt_vars,
                )
                
                # Extract the response text from the structured output
                response = result.response
                
                # Clean up the response to remove any code blocks or excessive formatting
                response = self._clean_response(response)
                
                # Add the response to the agent state as a system message if we have access to the state
                if hasattr(self, 'context') and self.context and hasattr(self.context, 'state'):
                    state = self.context.state
                    if hasattr(state, 'add_system_message') and callable(state.add_system_message):
                        state.add_system_message(response)
                        logger.debug("Added system message to agent state")
                
            except Exception as e:
                error_details = f"Error: {type(e).__name__}: {str(e)}"
                logger.error(f"Failed to generate response: {error_details}", exc_info=True)
                
                # Create error response
                response = f"I'm sorry, but I encountered an error while processing your message. {error_details}"
                
                # Add error response to state
                if hasattr(self, 'context') and self.context and hasattr(self.context, 'state'):
                    state = self.context.state
                    if hasattr(state, 'add_system_message') and callable(state.add_system_message):
                        state.add_system_message(response)
                        logger.debug("Added error system message to agent state")
                
                # Return a more detailed error message directly
                return ConversationOutput(
                    response=response
                )
            
            # Create and return the output directly
            return ConversationOutput(
                response=response
            )
        
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in conversation pipeline: {str(e)}", exc_info=True)
            
            # Create error response
            response = f"I apologize, but I encountered an unexpected error: {str(e)}"
            
            # Add error response to state
            if hasattr(self, 'context') and self.context and hasattr(self.context, 'state'):
                state = self.context.state
                if hasattr(state, 'add_system_message') and callable(state.add_system_message):
                    state.add_system_message(response)
                    logger.debug("Added unexpected error system message to agent state")
            
            # Return the error output directly
            return ConversationOutput(
                response=response
            )

    def _clean_response(self, response: str) -> str:
        """Clean up the response text.
        
        Args:
            response: Response text to clean
            
        Returns:
            Cleaned response text
        """
        # Strip any code block markers
        response = response.replace("```", "")
        
        # Strip any markdown formatting but keep the content
        for marker in ["**", "__", "*", "_", "#", "##", "###"]:
            response = response.replace(marker, "")
            
        # Strip any common response prefixes
        for prefix in ["Response:", "Assistant:", "AI:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                
        return response.strip()

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into a string for the prompt.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        if not context:
            return ""
            
        # Format the context as a string with key-value pairs
        context_lines = ["Context:"]
        for key, value in context.items():
            context_lines.append(f"- {key}: {value}")
            
        return "\n".join(context_lines)

    @classmethod
    def create(cls) -> Flow:
        """Create a new instance of this flow.
        
        Returns:
            A Flow instance
        """
        return cls() 