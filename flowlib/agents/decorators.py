"""Agent decorators for creating agent classes.

This module provides decorator-based utilities for defining agent classes
in a style consistent with the rest of the flowlib library.
"""

import inspect
import functools
from typing import Callable, List, Dict, Any, Optional, Type, Union, get_type_hints


from ..flows.base import Flow
from ..core.registry import provider_registry
from ..core.registry.constants import ProviderType
from .base import Agent
from .models import AgentConfig

def agent(
    *,
    provider_name: str = "llamacpp",
    planner_model: Optional[str] = None,
    input_generator_model: Optional[str] = None,
    reflection_model: Optional[str] = None,
    working_memory: str = "memory-cache",
    short_term_memory: str = "memory-cache",
    long_term_memory: str = "chroma",
    max_retries: int = 3,
    default_system_prompt: str = ""
) -> Callable:
    """Decorator for creating an agent class.
    
    This decorator transforms a class into an Agent implementation with the
    specified configuration for planning, input generation, and reflection.
    
    Args:
        provider_name: Name of LLM provider to use
        planner_model: Model to use for planning
        input_generator_model: Model to use for input generation
        reflection_model: Model to use for reflection
        working_memory: Provider to use for working memory
        short_term_memory: Provider to use for short-term memory
        long_term_memory: Provider to use for long-term memory
        max_retries: Maximum number of retries for LLM calls
        default_system_prompt: Default system prompt for the agent
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Create the wrapper class that extends Agent
        class AgentWrapper(Agent):
            """Agent class created with @agent decorator."""
            
            def __init__(
                self,
                flows: List[Flow] = None,
                config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
                task_description: str = "",
                flow_paths: List[str] = ["./flows"],
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
                        "working_memory": working_memory,
                        "short_term_memory": short_term_memory,
                        "long_term_memory": long_term_memory,
                        "max_retries": max_retries,
                        "default_system_prompt": default_system_prompt
                    }
                    
                    # Update with any config provided to constructor
                    agent_config.update(config)
                    config = AgentConfig(**agent_config)
                
                # Initialize base Agent
                super().__init__(flows, config, task_description, flow_paths)
                
                # Create instance of the decorated class
                self.agent_impl = cls(**kwargs)
                
                # Copy methods from decorated class
                self._copy_methods()
                
                # Add any instance variables from kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def _copy_methods(self):
                """Copy methods from the decorated class to this instance."""
                for name, method in inspect.getmembers(self.agent_impl, inspect.ismethod):
                    # Skip magic methods
                    if name.startswith('__') and name.endswith('__'):
                        continue
                    
                    # Bind the method to this instance
                    setattr(self, name, method.__get__(self, self.__class__))
            
            # Override methods if they exist in the decorated class
            async def _plan_next_action(self):
                if hasattr(self.agent_impl, 'plan_next_action'):
                    # Use the plan_next_action method from the decorated class
                    return await self.agent_impl.plan_next_action(
                        self.state,
                        self.get_flow_descriptions(),
                        self.last_result
                    )
                else:
                    # Use the default implementation from Agent
                    return await super()._plan_next_action()
            
            async def _generate_flow_inputs(self, flow_name, planning):
                if hasattr(self.agent_impl, 'generate_flow_inputs'):
                    # Use the generate_flow_inputs method from the decorated class
                    return await self.agent_impl.generate_flow_inputs(
                        flow_name,
                        planning,
                        self.state
                    )
                else:
                    # Use the default implementation from Agent
                    return await super()._generate_flow_inputs(flow_name, planning)
            
            async def _reflect_on_result(self, flow_name, inputs, result):
                if hasattr(self.agent_impl, 'reflect_on_result'):
                    # Use the reflect_on_result method from the decorated class
                    return await self.agent_impl.reflect_on_result(
                        flow_name,
                        inputs,
                        result,
                        self.state
                    )
                else:
                    # Use the default implementation from Agent
                    return await super()._reflect_on_result(flow_name, inputs, result)
        
        # Add the create class method to the original class
        @classmethod
        def create(cls, flows=None, task_description="", flow_paths=None, config=None, **kwargs):
            """Create an instance of the agent with the specified flows.
            
            Args:
                flows: List of flows the agent can use
                task_description: Description of the task to perform
                flow_paths: Paths to scan for flow definitions
                config: Additional configuration
                **kwargs: Additional arguments for the agent
                
            Returns:
                Agent instance
            """
            return AgentWrapper(
                flows=flows or [],
                task_description=task_description,
                flow_paths=flow_paths or ["./flows"],
                config=config,
                **kwargs
            )
        
        cls.create = create
        # Store the wrapper class
        cls.__agent_class__ = AgentWrapper
        
        return cls
    
    return decorator