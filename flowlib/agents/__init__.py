"""
Agent system for flowlib.

This module provides an enhanced agent framework that uses LLMs for planning,
input generation, and reflection on flow execution.
"""

from .base import Agent, AgentState
from .decorators import agent
from .memory_manager import MemoryManager, MemoryContext, MemoryItem
from .discovery import FlowDiscovery
from .full import FullConversationalAgent
# Import prompts to ensure that decorated prompts are registered
from . import prompts

__all__ = [
    "Agent",
    "AgentState",
    "agent",
    "MemoryManager",
    "MemoryContext",
    "MemoryItem",
    "FlowDiscovery",
    "FullConversationalAgent"
]