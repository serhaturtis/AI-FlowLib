"""Agent registry for tracking and accessing agent classes."""

import logging
from typing import Dict, Type, Any, Optional

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for agent classes."""

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, agent_class: Type, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register an agent class.

        Args:
            name: The unique name for the agent.
            agent_class: The agent class (or wrapper class) to register.
            metadata: Optional dictionary of metadata associated with the agent.

        Raises:
            ValueError: If an agent with the same name is already registered.
        """
        if name in self._agents:
            # Allow re-registration for development/hot-reloading, but log a warning
            logger.warning(f"Agent '{name}' is already registered. Overwriting registration.")
            # To disallow overwriting, uncomment the following line:
            # raise ValueError(f"Agent '{name}' is already registered.")

        self._agents[name] = {
            "class": agent_class,
            "metadata": metadata or {}
        }
        logger.debug(f"Registered agent: {name} (Class: {agent_class.__name__})")

    def get_agent_class(self, name: str) -> Optional[Type]:
        """Get the registered agent class by name.

        Args:
            name: The name of the agent.

        Returns:
            The agent class, or None if not found.
        """
        agent_info = self._agents.get(name)
        return agent_info.get("class") if agent_info else None

    def get_agent_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the metadata for a registered agent.

        Args:
            name: The name of the agent.

        Returns:
            The metadata dictionary, or None if not found.
        """
        agent_info = self._agents.get(name)
        return agent_info.get("metadata") if agent_info else None
        
    def get_agent_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get all registered information for an agent.

        Args:
            name: The name of the agent.

        Returns:
            A dictionary containing 'class' and 'metadata', or None if not found.
        """
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """List the names of all registered agents.

        Returns:
            A list of agent names.
        """
        return sorted(list(self._agents.keys()))

    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents = {}
        logger.debug("Agent registry cleared.")

# Global instance of the agent registry
agent_registry = AgentRegistry() 