"""
Agent system for flowlib.

This module provides an enhanced agent framework that uses LLMs for planning,
input generation, and reflection on flow execution.
"""

# Import memory components
from .memory.manager import HybridMemoryManager
from .memory.models import Entity, EntityAttribute, EntityRelationship

# Import from flows package - now all in one place
from .flows import (
    MessageInput, 
    ConversationOutput,
    ConversationFlow, 
    AgentPlanningFlow, 
    AgentInputGenerationFlow, 
    AgentReflectionFlow,
    MemoryExtractionFlow, 
    MemoryRetrievalFlow, 
    ConversationInput, 
    MemorySearchInput
)

# Import base agent components
from .base import Agent, AgentState
from .decorators import agent
from .memory_manager import MemoryManager, MemoryContext, MemoryItem
from .discovery import FlowDiscovery

# Import full agent implementation 
from .full import FullConversationalAgent

# Import prompts to ensure that decorated prompts are registered
from . import prompts  # Keep this temporarily until we remove prompts.py

# Import the new prompts package
from flowlib.agents import prompts as agent_prompts

__all__ = [
    # Base agent components
    "Agent",
    "AgentState",
    "agent",
    "MemoryManager",
    "MemoryContext",
    "MemoryItem",
    "FlowDiscovery",
    "FullConversationalAgent",
    
    # Message handling
    "MessageInput",
    "ConversationOutput",
    "ConversationFlow", 
    "AgentPlanningFlow", 
    "AgentInputGenerationFlow", 
    "AgentReflectionFlow",
    
    # Entity-centric memory components
    "HybridMemoryManager",
    "Entity",
    "EntityAttribute",
    "EntityRelationship",
    "MemoryExtractionFlow",
    "MemoryRetrievalFlow",
    "ConversationInput",
    "MemorySearchInput"
]