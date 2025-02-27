"""Tool for executing flows."""

from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, create_model
import inspect

from ...flows.base import Flow
from ...core.errors.base import ValidationError, ErrorContext
from ...core.resources import ResourceRegistry
from ..core.tool import Tool

class FlowTool(Tool):
    """Tool for executing flows."""
    
    def __init__(
        self,
        flow_class: Type[Flow],
        description: Optional[str] = None,
        **flow_kwargs: Any
    ):
        """Initialize flow tool.
        
        Args:
            flow_class: Flow class to execute
            description: Optional tool description
            **flow_kwargs: Additional arguments to pass to flow constructor
        """
        self.flow_class = flow_class
        self.flow_kwargs = flow_kwargs
        
        # Get flow name and metadata
        self.flow_name = getattr(flow_class, '_flow_name', flow_class.__name__)
        self.flow_metadata = getattr(flow_class, '_flow_metadata', {})
        
        # Get pipeline method
        pipeline_method = None
        for name, attr in vars(flow_class).items():
            if getattr(attr, '_pipeline', False):
                pipeline_method = attr
                break
                
        if not pipeline_method:
            raise ValidationError(
                f"Flow class {flow_class.__name__} has no pipeline method",
                ErrorContext.create(flow_name=self.flow_name)
            )
            
        # Get input/output models from pipeline
        self.input_model = getattr(pipeline_method, 'input_model', None)
        self.output_model = getattr(pipeline_method, 'output_model', None)
        
        # Check if pipeline method requires inputs
        sig = inspect.signature(pipeline_method)
        self.requires_inputs = bool(sig.parameters)
        
        # Create empty input model if no inputs required
        if not self.requires_inputs:
            self.input_model = create_model(f"{self.flow_name}Input", __base__=BaseModel)
            
        # Ensure we have an output model
        if not self.output_model:
            raise ValidationError(
                f"Flow class {flow_class.__name__} pipeline must specify an output model",
                ErrorContext.create(flow_name=self.flow_name)
            )
            
        # Initialize base class
        super().__init__(
            name=self.flow_name,
            description=description or self.flow_metadata.get('description', f"Execute {self.flow_name} flow"),
            input_model=self.input_model,
            output_model=self.output_model
        )
        
    async def execute(self, input_data: Optional[BaseModel] = None) -> BaseModel:
        """Execute flow with given input data.
        
        Args:
            input_data: Input data for flow, must be a Pydantic model or None if no inputs required
            
        Returns:
            Flow output data as a Pydantic model
            
        Raises:
            ValidationError: If input validation fails or execution fails
        """
        try:
            # Create flow instance
            flow = self.flow_class(**self.flow_kwargs)
            
            # Execute flow
            async with flow:
                # Find pipeline method
                pipeline_method = None
                for name, attr in vars(self.flow_class).items():
                    if getattr(attr, '_pipeline', False):
                        pipeline_method = getattr(flow, name)
                        break
                
                if not pipeline_method:
                    raise ValidationError(
                        f"Flow class {self.flow_class.__name__} has no pipeline method",
                        ErrorContext.create(flow_name=self.flow_name)
                    )
                
                # Execute pipeline with or without inputs
                if self.requires_inputs:
                    if not input_data:
                        raise ValidationError(
                            f"Flow {self.flow_name} requires input data but none provided",
                            ErrorContext.create(flow_name=self.flow_name)
                        )
                    result = await pipeline_method(**input_data.model_dump())
                else:
                    result = await pipeline_method()
                
                # Ensure result is a Pydantic model
                if not isinstance(result, BaseModel):
                    raise ValidationError(
                        f"Flow {self.flow_name} must return a Pydantic model",
                        ErrorContext.create(flow_name=self.flow_name)
                    )
                
                return result
            
        except Exception as e:
            # Wrap any errors in ValidationError
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    "Flow execution failed",
                    ErrorContext.create(
                        flow_name=self.flow_name,
                        error=str(e)
                    )
                ) from e
            raise 