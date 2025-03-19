"""Full conversational agent implementation.

This module provides a complete agent implementation with conversation capabilities,
short-term and long-term memory, and full planning, execution, and reflection.
"""

import logging
from typing import Optional, Dict, Any

from .base import Agent
from .models import AgentState
from .decorators import agent
from .flows import (
    ConversationInput, ConversationOutput, 
    ConversationFlow, AgentPlanningFlow, AgentInputGenerationFlow, AgentReflectionFlow
)

logger = logging.getLogger(__name__)

@agent(
    provider_name="llamacpp",
    planner_model="default",
    input_generator_model="default",
    reflection_model="default",
    working_memory="memory-cache",
    short_term_memory="memory-cache",
    long_term_memory="chroma"
)
class FullConversationalAgent(Agent):
    """Agent that converses with users and performs tasks with full planning capabilities."""
    
    async def handle_message(self, message: str) -> str:
        """Handle a user message and return a response.
        
        This method:
        1. Creates conversation context if needed
        2. Stores user message in memory
        3. Sets task description for the agent
        4. Resets agent execution state
        5. Executes the agent's planning-execution-reflection cycle
        6. Extracts and returns the response
        
        Args:
            message: The user's message
            
        Returns:
            The agent's response to the message
        """
        try:
            # Create a conversation context if not already exists
            if not hasattr(self, "conversation_context"):
                self.conversation_context = self.memory.create_context("conversation", self.base_context)
            
            # Store the user message
            await self.memory.store(
                "user_message", 
                message, 
                self.conversation_context,
                ttl=3600,
                importance=0.7
            )
            
            # Store conversation input for conversation flow
            conversation_input = ConversationInput(message=message, model_name="default")
            await self.memory.store(
                "conversation_input", 
                conversation_input, 
                self.conversation_context,
                ttl=3600
            )
            
            # Set the task description for this message
            self.state.task_description = f"Understand and respond to user message: '{message}'"
            
            # Reset the agent's execution state
            self.state.is_complete = False
            self.state.completion_reason = None
            self.state.progress = 0
            
            # Run the agent execution
            await self.execute()
            
            # Get the response
            response = "I processed your request, but I'm not sure how to respond."
            
            # Try to get response from last_result
            if self.last_result and hasattr(self.last_result.data, "response"):
                response = self.last_result.data.response
                
                # Store the response
                await self.memory.store(
                    "agent_response", 
                    response, 
                    self.conversation_context,
                    ttl=3600
                )
            
            # Log the completion of agent execution
            logger.info(f"Agent execution completed with progress: {self.state.progress}%")
            
            return response
            
        except Exception as e:
            logger.error(f"Error during agent execution: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}" 