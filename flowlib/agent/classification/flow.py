"""
Message Classification System for the Agent Architecture.

This module provides a classifier flow that determines if a user message
requires simple conversation or complex task execution.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ...flows.base import Flow
from ...flows.decorators import pipeline
from ...resources.registry import resource_registry
from ...providers.registry import provider_registry
from ...providers import ProviderType
from ..decorators.base import agent_flow
from .prompts import MessageClassifierPrompt


class MessageClassification(BaseModel):
    """Classification result for a user message"""
    execute_task: bool = Field(..., description="True if there is need to execute a task, False if conversation")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    category: str = Field(..., description="Message category (greeting, question, instruction, etc.)")
    task_description: Optional[str] = Field(None, description="Task description for execution (when execute_task=True)")


class MessageClassifierInput(BaseModel):
    """Input for message classification"""
    message: str = Field(..., description="The user message to classify")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list, 
        description="Recent conversation history"
    )
    memory_context_summary: Optional[str] = Field(None, description="Summary of relevant memory context")


@agent_flow(name="MessageClassifierFlow", description="Classify user messages into conversation or task", is_infrastructure=True)
class MessageClassifierFlow(Flow):
    """Flow that determines if a message requires simple conversation or complex task execution"""
    
    def __init__(self):
        """Initialize the message classifier flow."""
        super().__init__("MessageClassifierFlow")
        
    def get_description(self) -> str:
        """Return the description of this flow."""
        return "Classify user messages into conversation or task categories"
    
    @pipeline(input_model=MessageClassifierInput, output_model=MessageClassification)
    async def classify_message(self, input_data: MessageClassifierInput) -> MessageClassification:
        """Classify a user message into conversation or task categories.
        
        Args:
            input_data: Input containing message and conversation history
            
        Returns:
            Classification result with confidence score and category
        """
        # Get the message and conversation history
        message = input_data.message
        conversation_history = input_data.conversation_history
        memory_summary = input_data.memory_context_summary or "No specific memory context provided."
        
        try:
            # Format conversation history as text
            history_text = self._format_conversation_history(conversation_history)
            
            # Create prompt variables
            prompt_vars = {
                "message": message,
                "conversation_history": history_text,
                "memory_context_summary": memory_summary
            }
            
            # Create classification prompt
            classification_prompt = MessageClassifierPrompt()
            
            # Get LLM provider
            llm = await provider_registry.get(ProviderType.LLM, "llamacpp")
            
            # Generate classification using LLM
            result = await llm.generate_structured(
                prompt=classification_prompt,
                output_type=MessageClassification,
                model_name="default",
                prompt_variables=prompt_vars,
            )
            
            # Ensure confidence is between 0 and 1
            result.confidence = max(0.0, min(1.0, result.confidence))
            
            # Ensure task is set for non-conversation messages if missing
            if result.execute_task and not result.task_description:
                result.task_description = f"Assist the user with their request: {message}"
            
            return result
            
        except Exception as e:
            # On error, default to conversation path for safety
            return MessageClassification(
                execute_task=False,
                confidence=1.0,
                category="error_fallback",
                task=None
            )
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history as text.
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted history text
        """
        if not history:
            return "No conversation history available."
            
        formatted = []
        for item in history:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
            
        return "\n".join(formatted) 