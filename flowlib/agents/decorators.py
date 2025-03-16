"""Agent decorators for creating agent classes.

This module provides decorator-based utilities for defining agent classes
in a style consistent with the rest of the flowlib library.
"""

import inspect
import functools
from typing import Callable, List, Dict, Any, Optional, Type, Union, get_type_hints

from pydantic import BaseModel

from ..flows.base import Flow
from ..core.registry import provider_registry
from ..core.registry.constants import ProviderType
from .models import AgentConfig
from .base import Agent
from .llm_agent import LLMAgent

def agent(
    *,
    provider_name: str = "llamacpp",
    planner_model: Optional[str] = None,
    input_generator_model: Optional[str] = None,
    reflection_model: Optional[str] = None,
    max_retries: int = 3,
    default_system_prompt: str = ""
) -> Callable:
    """Decorator for creating an agent class.
    
    This decorator transforms a class into an Agent implementation with the LLM-based logic
    for planning, input generation, and reflection.
    
    Args:
        provider_name: Name of LLM provider to use
        planner_model: Model to use for planning (defaults to class variable if not specified)
        input_generator_model: Model to use for input generation (defaults to planner_model if not specified)
        reflection_model: Model to use for reflection (defaults to planner_model if not specified)
        max_retries: Maximum number of retries for LLM calls
        default_system_prompt: Default system prompt for the agent
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Extract docstring from the class for agent description
        agent_description = cls.__doc__ or ""
        
        # Check if the class defines the required methods
        has_plan_method = hasattr(cls, "plan_next_action")
        has_input_method = hasattr(cls, "generate_flow_inputs")
        has_reflect_method = hasattr(cls, "reflect_on_result")
        
        # Create the wrapper class that extends LLMAgent
        # Instead of using functools.wraps(cls), we'll manually copy necessary attributes
        class AgentWrapper(LLMAgent):
            """Agent class created with @agent decorator."""
            
            def __init__(
                self,
                flows: List[Flow],
                config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
                task_description: str = "",
                **kwargs
            ):
                # Merge provided config with decorator config
                if config is None:
                    config = {}
                    
                if isinstance(config, dict):
                    # Get model names from class if not provided in decorator
                    model_name = getattr(cls, "model_name", None)
                    
                    # Use provided decorator values or fall back to class variables
                    agent_config = {
                        "provider_name": provider_name,
                        "planner_model": planner_model or model_name or "default",
                        "input_generator_model": input_generator_model or planner_model or model_name or "default",
                        "reflection_model": reflection_model or planner_model or model_name or "default",
                        "max_retries": max_retries,
                        "default_system_prompt": default_system_prompt
                    }
                    
                    # Update with any config provided to constructor
                    agent_config.update(config)
                    config = AgentConfig(**agent_config)
                
                # Initialize LLMAgent
                super().__init__(flows, config, task_description)
                
                # Create instance of the decorated class
                self.agent_impl = cls(**kwargs)
                
                # Add any instance variables from kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            # Override methods if they exist in the decorated class
            async def _plan_next_action(self):
                if has_plan_method:
                    # Use the plan_next_action method from the decorated class
                    return await self.agent_impl.plan_next_action(
                        self.state,
                        self.get_flow_descriptions(),
                        self.last_result
                    )
                else:
                    # Use the default implementation from LLMAgent
                    return await super()._plan_next_action()
            
            async def _generate_flow_inputs(self, flow_desc, planning):
                if has_input_method:
                    # Use the generate_flow_inputs method from the decorated class
                    return await self.agent_impl.generate_flow_inputs(
                        flow_desc,
                        planning,
                        self.state
                    )
                else:
                    # Use the default implementation from LLMAgent
                    return await super()._generate_flow_inputs(flow_desc, planning)
            
            async def _reflect_on_result(self, flow_desc, inputs, result):
                if has_reflect_method:
                    # Use the reflect_on_result method from the decorated class
                    return await self.agent_impl.reflect_on_result(
                        flow_desc,
                        inputs,
                        result,
                        self.state
                    )
                else:
                    # Use the default implementation from LLMAgent
                    return await super()._reflect_on_result(flow_desc, inputs, result)
        
        # Add class attributes from the original class
        for attr_name, attr_value in cls.__dict__.items():
            if not attr_name.startswith('__') and not hasattr(AgentWrapper, attr_name):
                setattr(AgentWrapper, attr_name, attr_value)
        
        # Add create class method
        @classmethod
        def create(cls, flows, task_description="", **kwargs):
            """Create an instance of the agent with the specified flows.
            
            Args:
                flows: List of flows the agent can use
                task_description: Description of the task to perform
                **kwargs: Additional arguments for the agent
                
            Returns:
                Agent instance
            """
            return cls.__agent_class__(flows=flows, task_description=task_description, **kwargs)
        
        # Add the create method to the original class
        cls.create = create
        # Store the wrapper class
        cls.__agent_class__ = AgentWrapper
        
        return cls
    
    return decorator 