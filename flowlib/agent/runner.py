"""
Simple Agent Runner for the FlowLib Agent System.

This module provides a minimal AgentRunner class for running agents.
"""

import logging
from typing import Any, Type

from ..resources.registry import resource_registry
from ..resources.constants import ResourceType
from .models.config import AgentConfig
from .planning import DefaultPlanningPrompt
from .reflection import DefaultReflectionPrompt

logger = logging.getLogger(__name__)


class AgentRunner:
    """Minimal runner for FlowLib agents.
    
    This class provides a simple way to initialize and run agents.
    """
    
    async def run_agent(self, agent_class: Type) -> Any:
        """Run an agent.
        
        Args:
            agent_class: Agent class to run
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If the agent class is invalid
        """
        try:
            # Check if agent class is valid
            if not hasattr(agent_class, "create"):
                raise ValueError(
                    f"Invalid agent class {agent_class.__name__}. "
                    "Make sure it's decorated with @agent or @conversation_handler."
                )
            
            # Create agent instance
            agent_instance = agent_class.create()
            
            # Apply basic configuration
            agent_instance.config = AgentConfig(
                name=getattr(agent_class, "__name__", "Agent"),
                description="Agent created with simplified runner"
            )
            
            # Register default prompts if not already registered
            if not resource_registry.contains("planning_default", ResourceType.PROMPT):
                resource_registry.register_sync(DefaultPlanningPrompt, "planning_default", ResourceType.PROMPT)
            
            if not resource_registry.contains("reflection_default", ResourceType.PROMPT):
                resource_registry.register_sync(DefaultReflectionPrompt, "reflection_default", ResourceType.PROMPT)
            
            # Initialize agent
            await agent_instance.initialize()
            
            # Run setup if available
            if hasattr(agent_instance, "setup") and callable(agent_instance.setup):
                await agent_instance.setup()
            
            return agent_instance
        
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}", exc_info=True)
            raise 