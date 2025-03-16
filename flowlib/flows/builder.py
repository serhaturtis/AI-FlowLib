"""Flow builder implementation for programmatic flow creation.

This module provides an enhanced FlowBuilder class for programmatically
creating flows with a clean, fluent interface.
"""

from typing import Dict, List, Optional, Type, Any, Callable, Union
from pydantic import BaseModel

from ..core.models.context import Context
from ..core.models.result import FlowResult
from .base import Flow
from .stage import Stage
from .composite import CompositeFlow
from .standalone import StandaloneStage

class FlowBuilder:
    """Enhanced flow builder for programmatic flow creation.
    
    This class provides:
    1. A fluent interface for programmatically building flows
    2. Support for multiple stages, branch conditions, and error handlers
    3. Clean validation of inputs and outputs
    """
    
    def __init__(
        self,
        name: str,
        input_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize flow builder.
        
        Args:
            name: Name for the resulting flow
            input_schema: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
            metadata: Optional metadata about the flow
            
        Raises:
            ValueError: If input_schema is provided but not a Pydantic BaseModel subclass.
        """
        # Validate input_schema is a Pydantic model if provided
        if input_schema is not None and not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
            raise ValueError(f"Flow input_schema must be a Pydantic BaseModel subclass, got {input_schema}")
            
        self.name = name
        self.input_schema = input_schema
        self.metadata = metadata or {}
        self.stages: List[Flow] = []
        self.output_schema: Optional[Type[BaseModel]] = None
    
    def add_stage(
        self,
        stage_or_fn: Union[Flow, Callable, StandaloneStage],
        name: Optional[str] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'FlowBuilder':
        """Add a stage to the flow.
        
        Args:
            stage_or_fn: Flow instance, function, or StandaloneStage to add
            name: Optional name for the stage (required if function provided)
            input_schema: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
            output_schema: Optional Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
            metadata: Optional metadata about the stage
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If input_schema or output_schema is provided but not a Pydantic BaseModel subclass.
        """
        # Validate input_schema and output_schema are Pydantic models if provided
        if input_schema is not None and not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
            raise ValueError(f"Stage input_schema must be a Pydantic BaseModel subclass, got {input_schema}")
        
        if output_schema is not None and not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise ValueError(f"Stage output_schema must be a Pydantic BaseModel subclass, got {output_schema}")
            
        if isinstance(stage_or_fn, Flow):
            # If it's already a Flow, just add it
            self.stages.append(stage_or_fn)
        elif isinstance(stage_or_fn, StandaloneStage):
            # Convert StandaloneStage to Stage
            self.stages.append(stage_or_fn.to_stage())
        else:
            # It's a function, create a Stage
            stage_name = name or getattr(stage_or_fn, "__name__", "unnamed_stage")
            self.stages.append(Stage(
                name=stage_name,
                process_fn=stage_or_fn,
                input_schema=input_schema,
                output_schema=output_schema,
                metadata=metadata
            ))
        
        return self
    
    def with_output_schema(self, output_schema: Type[BaseModel]) -> 'FlowBuilder':
        """Set output schema for the flow.
        
        Args:
            output_schema: Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If output_schema is not a Pydantic BaseModel subclass.
        """
        # Validate output_schema is a Pydantic model
        if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise ValueError(f"Flow output_schema must be a Pydantic BaseModel subclass, got {output_schema}")
            
        self.output_schema = output_schema
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'FlowBuilder':
        """Add metadata to the flow.
        
        Args:
            metadata: Metadata to add
            
        Returns:
            Self for chaining
        """
        self.metadata.update(metadata)
        return self
    
    def build(self) -> CompositeFlow:
        """Build the flow.
        
        Returns:
            CompositeFlow instance with configured stages and options
        """
        return CompositeFlow(
            name=self.name,
            stages=self.stages,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            metadata=self.metadata
        ) 