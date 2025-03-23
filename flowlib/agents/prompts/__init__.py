"""
Prompt templates for agent operations.

This module provides all prompt templates used by the agent system.
"""

# Import agent prompts
from .agent_prompts import (
    AgentPlanningPrompt,
    AgentInputGenerationPrompt,
    AgentReflectionPrompt,
    AgentConversationPrompt
)

# Import memory prompts
from .memory_prompts import (
    EntityExtractionPrompt,
    MemoryRetrievalPrompt,
    MemoryUpdatePrompt,
    MemoryReflectionPrompt
)

__all__ = [
    # Agent prompts
    "AgentPlanningPrompt",
    "AgentInputGenerationPrompt",
    "AgentReflectionPrompt",
    "AgentConversationPrompt",
    
    # Memory prompts
    "EntityExtractionPrompt",
    "MemoryRetrievalPrompt",
    "MemoryUpdatePrompt",
    "MemoryReflectionPrompt"
] 