"""
Agent registry for tracking agent instances.

This module provides a registry for keeping track of agent instances and
their metadata, enabling global access and discovery.
"""

import logging
from typing import Any, Dict, List, Optional

from .errors import AgentError

# Configure logger
logger = logging.getLogger(__name__)


class AgentRegistryError(AgentError):
    """Error in agent registry operations."""
    
    def __init__(
        self, 
        message: str,
        operation: str,
        agent_name: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize registry error.
        
        Args:
            message: Error message
            operation: Registry operation that failed
            agent_name: Optional agent name
            cause: Original exception that caused this error
            **context: Additional context information
        """
        context["operation"] = operation
        if agent_name:
            context["agent_name"] = agent_name
            
        super().__init__(message, cause, **context)


class AgentRegistry:
    """Registry for agent instances.
    
    This registry enables:
    1. Global access to agent instances
    2. Discovery of agents by name or characteristics
    3. Tracking of agent lifecycle
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agents = {}  # Maps agent name to agent instance
        self._metadata = {}  # Maps agent name to metadata
        self._logger = logging.getLogger(f"{__name__}.AgentRegistry")
        self._logger.debug("Agent registry initialized")
    
    def register(
        self, 
        name: str, 
        agent: Any, 
        overwrite: bool = False,
        **metadata
    ) -> None:
        """Register an agent.
        
        Args:
            name: Agent name
            agent: Agent instance
            overwrite: Whether to overwrite existing entry
            **metadata: Additional metadata about the agent
            
        Raises:
            AgentRegistryError: If agent with name already registered
        """
        if name in self._agents and not overwrite:
            raise AgentRegistryError(
                message=f"Agent with name '{name}' already registered",
                operation="register",
                agent_name=name
            )
            
        # Register agent
        self._agents[name] = agent
        
        # Store metadata
        self._metadata[name] = metadata or {}
        
        self._logger.debug(f"Registered agent '{name}'")
    
    def unregister(self, name: str) -> bool:
        """Unregister an agent.
        
        Args:
            name: Agent name
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if name in self._agents:
            del self._agents[name]
            
            if name in self._metadata:
                del self._metadata[name]
                
            self._logger.debug(f"Unregistered agent '{name}'")
            return True
            
        self._logger.debug(f"Agent '{name}' not found for unregistration")
        return False
    
    def get(self, name: str) -> Optional[Any]:
        """Get an agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None if not found
        """
        agent = self._agents.get(name)
        if agent is None:
            self._logger.debug(f"Agent '{name}' not found")
        
        return agent
    
    def contains(self, name: str) -> bool:
        """Check if an agent is registered.
        
        Args:
            name: Agent name
            
        Returns:
            True if the agent is registered
        """
        return name in self._agents
    
    def list(self) -> List[str]:
        """List registered agents.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for an agent.
        
        Args:
            name: Agent name
            
        Returns:
            Agent metadata or empty dict if not found
        """
        return self._metadata.get(name, {})
    
    def find(self, **criteria) -> List[str]:
        """Find agents matching criteria.
        
        Args:
            **criteria: Key-value pairs to match in metadata
            
        Returns:
            List of matching agent names
        """
        matching = []
        
        for name, metadata in self._metadata.items():
            match = True
            for key, value in criteria.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
                    
            if match:
                matching.append(name)
                
        return matching
    
    def update_metadata(self, name: str, **metadata) -> bool:
        """Update metadata for an agent.
        
        Args:
            name: Agent name
            **metadata: Metadata to update
            
        Returns:
            True if metadata was updated, False if agent not found
        """
        if name not in self._agents:
            self._logger.debug(f"Agent '{name}' not found for metadata update")
            return False
            
        if name not in self._metadata:
            self._metadata[name] = {}
            
        self._metadata[name].update(metadata)
        self._logger.debug(f"Updated metadata for agent '{name}'")
        return True
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered agents.
        
        This will call the shutdown method on all agents that have one.
        """
        self._logger.info(f"Shutting down {len(self._agents)} agents")
        
        for name, agent in list(self._agents.items()):
            try:
                if hasattr(agent, "shutdown") and callable(agent.shutdown):
                    self._logger.debug(f"Shutting down agent '{name}'")
                    await agent.shutdown()
                    self._logger.debug(f"Agent '{name}' shut down successfully")
            except Exception as e:
                self._logger.error(f"Error shutting down agent '{name}': {str(e)}", exc_info=True)
                
        self._logger.info("All agents shutdown complete")
    
    def clear(self) -> None:
        """Clear all registered agents."""
        count = len(self._agents)
        self._agents.clear()
        self._metadata.clear()
        self._logger.info(f"Cleared {count} agents from registry")


# Global registry instance
agent_registry = AgentRegistry() 