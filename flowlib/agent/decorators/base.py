"""
Simplified decorator API for agent system.

This module provides streamlined decorators for creating agent classes and flows,
with reduced complexity and better alignment with flowlib patterns.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ...flows.decorators import flow
from ...flows.base import Flow
from ...flows.registry import stage_registry

from ..core.agent import AgentCore
from ..models.config import AgentConfig

logger = logging.getLogger(__name__)

def agent(
    name: str = None,
    provider_name: str = "llamacpp",
    model_name: str = None,
    **kwargs
) -> Callable:
    """Decorator for creating agent classes with minimal complexity.
    
    Args:
        name: Agent name
        provider_name: Name of LLM provider to use
        model_name: Model to use (for planning, reflection, etc.)
        **kwargs: Additional configuration options
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Create a wrapper class that extends AgentCore for core functionality
        class AgentWrapper(AgentCore):
            """Agent class created with @agent decorator."""
            
            def __init__(
                self,
                flows: List[Flow] = None,
                config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
                task_description: str = "",
                **init_kwargs
            ):
                """Initialize the agent."""
                # Build simplified configuration
                config_dict = {}
                
                # Set name
                config_dict["name"] = name or cls.__name__
                
                # Set provider
                if provider_name:
                    config_dict["provider_name"] = provider_name
                
                # Set model name for all components
                if model_name:
                    config_dict["model_name"] = model_name
                    
                # Add any additional kwargs
                config_dict.update(kwargs)
                
                # If config is provided, merge with our config
                if config:
                    if isinstance(config, dict):
                        # Only add keys that don't exist
                        for key, value in config.items():
                            if key not in config_dict:
                                config_dict[key] = value
                    elif isinstance(config, AgentConfig):
                        # Convert to dict and merge
                        config_data = config.model_dump()
                        for key, value in config_data.items():
                            if key not in config_dict:
                                config_dict[key] = value
                
                # Create final config
                effective_config = AgentConfig(**config_dict)
                
                # Initialize AgentCore
                super().__init__(
                    config=effective_config,
                    task_description=task_description
                )
                
                # Create implementation instance
                self._impl = cls(**init_kwargs)
                
                # Give implementation access to the agent
                if hasattr(self._impl, "set_agent"):
                    self._impl.set_agent(self)
                else:
                    self._impl.agent = self
                
                # Register flows if provided
                if flows:
                    for flow in flows:
                        self.register_flow(flow)
            
            def get_flows(self) -> Dict[str, Any]:
                """Get the dictionary of registered flows.
                
                Returns:
                    Dictionary of flow name to flow instance
                """
                return self.flows
            
            # Forward method calls to the implementation class
            def __getattr__(self, name):
                """Forward attribute access to the implementation class.
                
                This allows methods defined on the implementation class to be 
                called directly on the agent instance.
                
                Args:
                    name: Name of the attribute to retrieve
                    
                Returns:
                    Attribute from the implementation class
                    
                Raises:
                    AttributeError: If attribute not found
                """
                if hasattr(self._impl, name):
                    return getattr(self._impl, name)
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Add the create class method to the class
        @classmethod
        def create(cls, *args, **kwargs):
            """Create an agent instance."""
            return AgentWrapper(*args, **kwargs)
        
        # Set the create method on the class
        cls.create = create
        
        # Store the wrapper class
        cls.__agent_class__ = AgentWrapper
        
        return cls
    
    return decorator

def agent_flow(
    name: str = None,
    description: str = None,
    category: str = "agent",
    is_infrastructure: bool = False
) -> Callable:
    """Decorator for creating agent-specific flows.
    
    This extends the standard @flow decorator with agent-specific metadata
    and registers the flow with stage_registry for discovery.
    
    Args:
        name: Flow name
        description: Human-readable description of the flow
        category: Category for the flow (conversation, tool, etc.)
        is_infrastructure: Whether this is an internal flow
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Apply standard flow decorator first
        if description:
            flow_cls = flow(name=name, description=description, is_infrastructure=is_infrastructure)(cls)
        else:
            # If no description, check if the class already has a get_description method
            if hasattr(cls, 'get_description'):
                # Store the original method
                original_get_description = cls.get_description
                # Apply flow decorator
                flow_cls = flow(name=name, description="", is_infrastructure=is_infrastructure)(cls)
                # Restore the original get_description method
                flow_cls.get_description = original_get_description
            else:
                # No description and no get_description method, use class docstring or empty string
                doc_description = cls.__doc__ or ""
                flow_cls = flow(name=name, description=doc_description, is_infrastructure=is_infrastructure)(cls)
        
        # Add agent-specific metadata
        if not hasattr(flow_cls, "__flow_metadata__"):
            flow_cls.__flow_metadata__ = {}
            
        # Add metadata for discovery
        flow_cls.__flow_metadata__.update({
            "category": category,
            "agent_flow": True
        })
        
        # Register with stage_registry for better discovery
        try:
            if stage_registry:
                # Extract flow name from metadata or class name
                flow_name = name or flow_cls.__flow_metadata__.get("name") or flow_cls.__name__
                # Register directly without checking if it already exists
                # (since contains_flow doesn't exist, we'll let the registry handle duplicates)
                logger.debug(f"Registering agent flow '{flow_name}' with stage_registry")
                stage_registry.register_flow(flow_name, flow_cls)
        except Exception as e:
            logger.warning(f"Failed to register agent flow with stage_registry: {str(e)}")
        
        return flow_cls
        
    return decorator

def conversation_handler(
    name: str = None,
    provider_name: str = "llamacpp",
    model_name: str = "default",
    **kwargs
) -> Callable:
    """Decorator for creating conversation handler agents.
    
    This is a specialized version of @agent focused on conversation.
    
    Args:
        name: Agent name
        provider_name: Name of LLM provider to use
        model_name: Model to use for conversations
        **kwargs: Additional configuration options
        
    Returns:
        Decorator function
    """
    # Set up conversation specific config
    kwargs["conversation_handler"] = True
    
    # Use the standard agent decorator
    return agent(
        name=name,
        provider_name=provider_name,
        model_name=model_name,
        **kwargs
    ) 