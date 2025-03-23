"""Full conversational agent implementation.

This module provides a complete agent implementation with conversation capabilities,
short-term and long-term memory, and full planning, execution, and reflection.
"""

import logging
from typing import Optional, Dict, Any, List
import asyncio
import typing

from .base import Agent
from .decorators import agent
from flowlib.core.models.context import Context
from flowlib.flows.base import Flow 
from flowlib.flows.decorators import pipeline, flow
from .models import AgentState

# Import all flows from the flows package
from .flows import (
    # Memory flows
    MemoryExtractionFlow,
    MemoryRetrievalFlow,
    MemorySearchInput,
    ExtractedEntities,
    RetrievedMemories,
    ConversationInput as MemoryConversationInput,
    
    # Conversation flows
    MessageInput, 
    ConversationOutput,
    ConversationFlow, 
    AgentPlanningFlow, 
    AgentInputGenerationFlow, 
    AgentReflectionFlow
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
    """Agent that converses with users and performs tasks with full planning capabilities.
    
    This agent supports:
    1. Natural conversation with users
    2. Task planning, execution, and reflection
    3. Entity-centric memory for contextual responses
    4. Automatic extraction of entities from conversations
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the agent with hybrid memory enabled by default."""
        # Enable hybrid memory by default
        kwargs["use_hybrid_memory"] = kwargs.get("use_hybrid_memory", True)
        super().__init__(*args, **kwargs)
    
    async def handle_message(self, message: str) -> str:
        """Handle a user message and return a response.
        
        This method:
        1. Creates conversation context if needed
        2. Stores user message in memory
        3. Retrieves relevant memories for context
        4. Sets task description for the agent
        5. Resets agent execution state
        6. Executes the agent's planning-execution-reflection cycle
        7. Extracts entities from the conversation and stores them in memory
        8. Returns the response
        
        Args:
            message: The user's message
            
        Returns:
            The agent's response to the message
        """
        try:
            print("\n===== DEBUGGING HANDLE_MESSAGE =====")
            print(f"Received message: '{message}'")
            print(f"Agent instance ID: {id(self)}")
            
            # Create a conversation context if not already exists
            if not hasattr(self, "conversation_context"):
                print("Creating new conversation context")
                self.conversation_context = self.memory.create_context("conversation", self.base_context)
                
                # Also initialize conversation history for memory operations
                self.conversation_history = []
                print("Initialized conversation history")
            else:
                print("Using existing conversation context")
            
            # Create the current message dictionary
            current_message = {"speaker": "user", "content": message}
            print(f"Created message dict: {current_message}")
            
            # Add to conversation history
            self.conversation_history.append(current_message)
            print(f"Added to history, now {len(self.conversation_history)} messages")
            
            # Store the user message
            print("Storing user message in memory")
            await self.memory.store(
                "user_message", 
                message, 
                self.conversation_context,
                ttl=3600,
                importance=0.7
            )
            print("Successfully stored user message")
            
            # Store conversation input for conversation flow
            print("Creating MessageInput instance")
            conversation_input = MessageInput(message=message, model_name="default")
            print(f"MessageInput instance: {conversation_input}")
            print(f"MessageInput type: {type(conversation_input)}")
            
            print("Storing conversation input in memory")
            await self.memory.store(
                "conversation_input", 
                conversation_input, 
                self.conversation_context,
                ttl=3600
            )
            print("Successfully stored conversation input")
            
            # Create input for memory operations
            print("Creating MemoryConversationInput instance")
            memory_conversation_input = MemoryConversationInput(
                conversation_history=self.conversation_history,
                latest_message=current_message,
                source="user_conversation"
            )
            print(f"Memory instance type: {type(memory_conversation_input)}")
            
            # Retrieve relevant memories if hybrid memory is enabled
            memory_context = ""
            if self.use_hybrid_memory:
                print("Hybrid memory enabled, retrieving memories")
                try:
                    # Retrieve relevant memories
                    print("Calling retrieve_memories")
                    retrieved_memories = await self.retrieve_memories(
                        conversation=memory_conversation_input
                    )
                    print(f"Retrieved memories: {retrieved_memories}")
                    
                    # If we have relevant memories, format them for injection into context
                    if retrieved_memories and retrieved_memories.context:
                        memory_context = retrieved_memories.context
                        logger.info(f"Retrieved {len(retrieved_memories.entities)} memories for context")
                        
                        # Store memory context for flows to use
                        await self.memory.store(
                            "memory_context", 
                            memory_context, 
                            self.conversation_context,
                            ttl=3600
                        )
                except Exception as mem_err:
                    print(f"Memory retrieval error: {str(mem_err)}")
                    print(f"Error type: {type(mem_err).__name__}")
                    import traceback
                    traceback.print_exc()
                    logger.warning(f"Memory retrieval failed: {str(mem_err)}")
            else:
                print("Hybrid memory not enabled, skipping memory retrieval")
            
            # Set the task description for this message (with memory context if available)
            task_description = f"Understand and respond to user message: '{message}'"
            if memory_context:
                task_description += f"\n\nRelevant memory context:\n{memory_context}"
            print(f"Set task description: {task_description[:100]}...")
                
            self.state.task_description = task_description
            
            # Reset the agent's execution state
            self.state.is_complete = False
            self.state.completion_reason = None
            self.state.progress = 0
            print("Reset agent execution state")
            
            # Run the agent execution
            print("Calling agent.execute()")
            await self.execute()
            print("Completed agent.execute()")
            
            # Get the response
            response = "I processed your request, but I'm not sure how to respond."
            
            # Try to get response from last_result
            if self.last_result and hasattr(self.last_result.data, "response"):
                print("Found response in last_result")
                response = self.last_result.data.response
                print(f"Response: {response[:100]}...")
                
                # Store the response
                await self.memory.store(
                    "agent_response", 
                    response, 
                    self.conversation_context,
                    ttl=3600
                )
                
                # Add agent response to conversation history
                self.conversation_history.append({"speaker": "agent", "content": response})
                
                # Extract entities from the conversation and store them in memory
                if self.use_hybrid_memory:
                    try:
                        # Update conversation input with the full history including agent response
                        memory_conversation_input = MemoryConversationInput(
                            conversation_history=self.conversation_history,
                            source="user_conversation"
                        )
                        
                        # Extract and store entities
                        extracted_entities = await self.extract_and_store_memories(memory_conversation_input)
                        if extracted_entities and extracted_entities.entities:
                            logger.info(f"Extracted and stored {len(extracted_entities.entities)} entities from conversation")
                    except Exception as mem_err:
                        logger.warning(f"Memory extraction failed: {str(mem_err)}")
            else:
                print("No response found in last_result")
                if self.last_result:
                    print(f"last_result type: {type(self.last_result)}")
                    print(f"last_result attributes: {dir(self.last_result)}")
                    if hasattr(self.last_result, 'data'):
                        print(f"last_result.data type: {type(self.last_result.data)}")
                        print(f"last_result.data attributes: {dir(self.last_result.data)}")
            
            # Log the completion of agent execution
            logger.info(f"Agent execution completed with progress: {self.state.progress}%")
            print("===== END DEBUGGING =====\n")
            
            return response
            
        except Exception as e:
            print(f"\n===== CRITICAL ERROR =====")
            print(f"Error in handle_message: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("===== END CRITICAL ERROR =====\n")
            logger.error(f"Error during agent execution: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}" 