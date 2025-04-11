"""Standalone stage implementation for reusable, importable flow components.

This module provides enhanced functionality for creating standalone stages
that can be easily imported, composed, and reused across different flows.
"""

import functools
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

from .stage import Stage
from .registry import stage_registry

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


class StandaloneStage:
    """Enhanced standalone stage wrapper for reusable flow components.
    
    This class provides a clean interface for:
    1. Converting functions to standalone, reusable stages
    2. Type inference from function signatures
    3. Compatibility with both decorator and functional usage
    4. Automatic registration with the stage registry
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize standalone stage.
        
        Args:
            func: Function to wrap as a stage
            name: Optional name for the stage (defaults to function name)
            input_model: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
            output_model: Optional Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
            metadata: Optional metadata about the stage
            
        Raises:
            ValueError: If input_model or output_model is provided but not a Pydantic BaseModel subclass.
        """
        # Validate input_model and output_model are Pydantic models if provided
        if input_model is not None and not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise ValueError(f"Standalone stage input_model must be a Pydantic BaseModel subclass, got {input_model}")
        
        if output_model is not None and not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
            raise ValueError(f"Standalone stage output_model must be a Pydantic BaseModel subclass, got {output_model}")
            
        self.func = func
        self.name = name or func.__name__
        self.input_model = input_model
        self.output_model = output_model
        self.metadata = metadata or {}
        
        # Add function metadata
        self.metadata.update({
            "function_name": func.__name__,
            "module": func.__module__,
            "doc": func.__doc__
        })
        
        # Set __module__ and __name__ so standalone stages can be imported
        functools.update_wrapper(self, func)
        
        # Register with the stage registry
        stage_registry.register_stage(
            stage_name=self.name,
            flow_name=None,  # None indicates a standalone stage
            func=self.func,
            input_model=self.input_model,
            output_model=self.output_model,
            metadata=self.metadata
        )
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the standalone stage function directly.
        
        This allows the standalone stage to be used as a regular function
        when needed, bypassing the flow execution machinery.
        """
        return self.func(*args, **kwargs)
    
    def to_stage(self) -> Stage:
        """Convert to a Stage instance.
        
        Returns:
            Stage instance configured with this standalone stage's function
        """
        return Stage(
            name=self.name,
            process_fn=self.func,
            input_schema=self.input_model,
            output_schema=self.output_model,
            metadata=self.metadata
        )
    
    @classmethod
    def create(
        cls,
        name: Optional[str] = None,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable[[F], 'StandaloneStage']:
        """Create a decorator for standalone stages.
        
        Args:
            name: Optional name for the stage (defaults to function name)
            input_model: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
            output_model: Optional Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
            metadata: Optional metadata about the stage
            
        Returns:
            Decorator function that wraps a function into a StandaloneStage
            
        Raises:
            ValueError: If input_model or output_model is provided but not a Pydantic BaseModel subclass.
        """
        # Validate input_model and output_model are Pydantic models if provided
        if input_model is not None and not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise ValueError(f"Standalone stage input_model must be a Pydantic BaseModel subclass, got {input_model}")
        
        if output_model is not None and not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
            raise ValueError(f"Standalone stage output_model must be a Pydantic BaseModel subclass, got {output_model}")
            
        def decorator(func: F) -> 'StandaloneStage':
            stage_name = name or func.__name__
            instance = cls(
                func=func,
                name=stage_name,
                input_model=input_model,
                output_model=output_model,
                metadata=metadata
            )
            return instance
        return decorator
        
    def __str__(self) -> str:
        """String representation."""
        return f"StandaloneStage(name='{self.name}', function='{self.func.__name__}')" 