# src/flows/stage.py

import asyncio
from typing import Any, Dict, Optional, Type, Callable, Union
from functools import partial
from pydantic import BaseModel
from datetime import datetime

from .base import Flow
from ..core.models.context import Context
from ..core.models.results import FlowResult, FlowStatus
from ..core.errors.base import ExecutionError, ValidationError
from ..utils.error.handling import ErrorHandling
from ..utils.validation import FlowValidation
from ..utils.metadata import create_execution_metadata

class Stage(Flow):
    """
    A unified stage implementation that can be used as a decorator or instantiated directly.
    Combines functionality from previous Stage and StageFlow implementations.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        process_fn: Optional[Callable] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize stage.
        
        Args:
            name: Stage name
            process_fn: Function that implements the stage's logic
            input_schema: Optional schema for input validation
            output_schema: Optional schema for output validation
            metadata: Optional stage metadata
        """
        # Handle decorator usage
        if callable(name) and process_fn is None:
            process_fn = name
            name = process_fn.__name__
            
        super().__init__(
            name=name or "unnamed_stage",
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )
        
        self._process = process_fn
        self._is_async = process_fn and asyncio.iscoroutinefunction(process_fn)
        self._decorator_mode = process_fn is None
        
        # Validate process function if provided
        if process_fn and not callable(process_fn):
            raise ValidationError(
                message="Process function must be callable",
                context=ErrorHandling.create_error_context(
                    self.name,
                    process_type=type(process_fn).__name__
                )
            )

    def __call__(self, func: Optional[Callable] = None, **kwargs) -> Union['Stage', Callable]:
        """Handle decorator usage."""
        if self._decorator_mode:
            if func is None:
                # No function yet, return partial for when it comes
                return partial(self.__call__, **kwargs)
            # Got the function, create new Stage with it
            return Stage(
                name=self.name,
                process_fn=func,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                metadata=self.metadata
            )
        else:
            raise TypeError("Stage instance is not callable")

    async def _execute(self, context: Context) -> FlowResult:
        """
        Execute stage logic with enhanced error handling and metadata tracking.
        
        Args:
            context: Execution context
            
        Returns:
            Flow execution result
            
        Raises:
            ExecutionError: If execution fails
        """
        start_time = datetime.now()
        
        try:
            # Validate process function
            if not self._process:
                raise ExecutionError(
                    message="No process function defined",
                    context=ErrorHandling.create_error_context(self.name)
                )
            
            # Get and validate input
            input_data = self._prepare_input(context)
            
            # Execute process function
            try:
                if self._is_async:
                    output_data = await self._process(input_data)
                else:
                    output_data = self._process(input_data)
            except Exception as e:
                raise ErrorHandling.wrap_error(
                    e,
                    self.name,
                    input_data=input_data
                )
            
            # Validate and prepare output
            if self.output_schema:
                output_data = FlowValidation.validate_schema(
                    output_data,
                    self.output_schema,
                    self.name,
                    "output"
                )
            
            # Calculate duration and create result
            duration = (datetime.now() - start_time).total_seconds()
            metadata = create_execution_metadata(
                self.name,
                duration,
                list(input_data.keys()),
                list(output_data.keys() if isinstance(output_data, dict) else [])
            )
            
            return FlowResult(
                flow_name=self.name,
                status=FlowStatus.SUCCESS,
                data=output_data,
                metadata=metadata
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            raise ErrorHandling.wrap_error(
                e,
                self.name,
                duration=duration,
                input_data=context.data if 'input_data' not in locals() else input_data
            )
    
    def cleanup(self) -> None:
        """Clean up stage resources."""
        # Most stages don't need cleanup, but we provide the method
        # for consistency with the Flow interface
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"Stage(name='{self.name}', async={self._is_async})"

# Example usage:
"""
# As a simple decorator
@Stage
async def my_stage(data):
    return processed_data

# With arguments
@Stage(
    name="custom_name",
    input_schema=InputModel,
    output_schema=OutputModel
)
async def my_stage(data):
    return processed_data

# Direct instantiation
stage = Stage(
    name="my_stage",
    process_fn=process_function
)
"""