"""
Flow metadata models and validation.

This module provides the FlowMetadata class that enforces the Flow interface contract,
particularly the requirement that all flows must implement get_description().
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Type, ClassVar

class FlowMetadata(BaseModel):
    """
    Standard metadata for all flows.
    
    This provides a single source of truth for flow descriptions,
    input/output models, and other metadata needed by agents.
    """
    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    version: str = "1.0.0"
    tags: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('input_model', 'output_model')
    @classmethod
    def validate_model_types(cls, v):
        if not (isinstance(v, type) and issubclass(v, BaseModel)):
            raise ValueError(f"Model must be a Pydantic BaseModel subclass, got {type(v).__name__}")
        return v
    
    @classmethod
    def from_flow(cls, flow: Any, name: str) -> "FlowMetadata":
        """
        Create metadata from a flow instance.
        
        Args:
            flow: The flow instance
            name: The flow name
            
        Returns:
            Flow metadata
            
        Raises:
            ValueError: If the flow's get_description() method returns an invalid value
                       or if required models are missing
        """
        # Get description using the required interface
        try:
            # Try calling get_description as an instance method
            if isinstance(flow, type):
                # If flow is a class, create an instance and call get_description
                try:
                    flow_instance = flow()
                    description = flow_instance.get_description()
                except Exception:
                    # If instantiation fails, try getting the docstring
                    description = flow.__doc__ or ""
            else:
                # If flow is already an instance, call get_description directly
                description = flow.get_description()
            
            if not isinstance(description, str):
                raise ValueError(f"get_description() must return a string, got {type(description).__name__}")
        except Exception as e:
            raise ValueError(f"Error getting description from flow {name}: {str(e)}")
            
        # Get input/output models
        input_model = None
        output_model = None
        
        try:
            # Check if we have a class or instance
            if isinstance(flow, type):
                # For class, use class methods or attributes
                if hasattr(flow, "get_pipeline_method_cls") and callable(flow.get_pipeline_method_cls):
                    pipeline = flow.get_pipeline_method_cls()
                    if pipeline:
                        # Input model
                        if hasattr(pipeline, "__input_model__") and pipeline.__input_model__:
                            input_model = pipeline.__input_model__
                        
                        # Output model
                        if hasattr(pipeline, "__output_model__") and pipeline.__output_model__:
                            output_model = pipeline.__output_model__
            else:
                # For instance, use instance methods
                if hasattr(flow, "get_pipeline_method") and callable(flow.get_pipeline_method):
                    pipeline = flow.get_pipeline_method()
                    if pipeline:
                        # Input model
                        if hasattr(pipeline, "__input_model__") and pipeline.__input_model__:
                            input_model = pipeline.__input_model__
                        
                        # Output model
                        if hasattr(pipeline, "__output_model__") and pipeline.__output_model__:
                            output_model = pipeline.__output_model__
                    
        except Exception as e:
            # Re-raise with explicit error about missing pipeline models
            raise ValueError(f"Error extracting models from flow {name}: {str(e)}")
        
        # Validate that we have both input and output models
        if not input_model:
            raise ValueError(f"Flow {name} does not have an input model defined in its pipeline")
        
        if not output_model:
            raise ValueError(f"Flow {name} does not have an output model defined in its pipeline")
            
        # Validate model types
        if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise ValueError(f"Input model must be a Pydantic BaseModel subclass, got {type(input_model).__name__}")
            
        if not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
            raise ValueError(f"Output model must be a Pydantic BaseModel subclass, got {type(output_model).__name__}")
        
        return cls(
            name=name,
            description=description,
            input_model=input_model,
            output_model=output_model
        ) 