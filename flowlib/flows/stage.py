"""Enhanced stage implementation with improved error handling and metadata.

This module provides a more robust implementation of flow stages with better
error handling, metadata tracking, and execution state management.
"""

import inspect
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin
from pydantic import BaseModel

from ..core.models.context import Context  
from ..core.models.result import FlowResult, FlowStatus
from ..core.errors import ErrorManager, ExecutionError, ValidationError, ErrorContext, BaseError
from .base import Flow

T = TypeVar('T')

logger = logging.getLogger(__name__)

class Stage(Flow[T]):
    """Enhanced Stage implementation with improved process function handling.
    
    This class provides:
    1. Support for both async and sync process functions
    2. Clean validation of inputs and outputs
    3. Type-safe process function execution
    4. Debugging capabilities for flow execution
    """
    
    def __init__(
        self,
        name: str,
        process_fn: Callable,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize stage.
        
        Args:
            name: Unique name for the stage
            process_fn: Function that implements stage logic
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation
            metadata: Optional metadata about the stage
            **kwargs: Additional options passed to the base Flow
        """
        # Initialize the parent class properly
        super().__init__(
            name_or_instance=name,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
            **kwargs
        )
        
        self.process_fn = process_fn
        
        # Check if process function is async or sync
        self._is_async = inspect.iscoroutinefunction(process_fn)
        
        # Add additional metadata
        self.metadata.update({
            "function_name": process_fn.__name__,
            "is_async": self._is_async,
            "doc": process_fn.__doc__,
        })
        
        # Infer schemas from function if not provided
        self._infer_schemas()
    
    def _infer_schemas(self) -> None:
        """Infer input and output schemas from process function type hints."""
        if not (self.input_schema and self.output_schema):
            try:
                hints = get_type_hints(self.process_fn)
                
                # Try to infer input schema if not provided
                if not self.input_schema and "context" in hints:
                    context_type = hints["context"]
                    if hasattr(context_type, '__origin__') and context_type.__origin__ is Context:
                        # If type hint is Context[SomeModel], extract SomeModel
                        self.input_schema = context_type.__args__[0]
                
                # Try to infer output schema if not provided
                if not self.output_schema and "return" in hints:
                    return_type = hints["return"]
                    # Handle case where return type is specified as FlowResult[SomeModel]
                    if hasattr(return_type, '__origin__') and return_type.__origin__ is FlowResult:
                        if len(return_type.__args__) > 0:
                            self.output_schema = return_type.__args__[0]
            except Exception as e:
                logger.debug(f"Schema inference failed for {self.name}: {str(e)}")
    
    async def execute(self, context: Context) -> FlowResult:
        """Execute the stage with given context.
        
        This method is specifically for Stage objects and does not look for pipeline methods.
        Instead, it directly calls the _execute method.
        
        Args:
            context: Execution context
            
        Returns:
            FlowResult containing execution outcome with attribute-based access
            
        Raises:
            BaseError: If execution fails
        """
        # Initialize execution
        execution_start = datetime.now()
        error_context = ErrorContext.create(
            stage_name=self.name
        )

        try:
            # Create error boundary
            async with self.error_manager.async_error_boundary(context.data):
                # Validate input if schema exists
                if self.input_schema:
                    input_data = self._prepare_input(context)
                    error_context = error_context.add(input_data=input_data)
                    self._validate_input(input_data)

                # Execute stage-specific logic
                result = await self._execute(context)

                # Validate output if schema exists
                if self.output_schema and result.data:
                    self._validate_output(result.data)

                # Record successful execution
                execution_time = (datetime.now() - execution_start).total_seconds()

                # Add execution info to result metadata
                result_with_metadata = FlowResult(
                    data=result.data,
                    original_type=getattr(result, '_original_type', None),
                    flow_name=self.name,
                    status=FlowStatus.SUCCESS,
                    error=result.error,
                    error_details=result.error_details,
                    metadata={
                        **result.metadata,
                        "execution_time": execution_time,
                    },
                    timestamp=result.timestamp,
                    duration=execution_time
                )

                return result_with_metadata

        except Exception as e:
            execution_time = (datetime.now() - execution_start).total_seconds()

            # Enhance error context
            error_context = error_context.add(
                execution_time=execution_time
            )

            # Convert to appropriate error type
            if isinstance(e, ExecutionError):
                error = e
            else:
                error = ExecutionError(
                    message=str(e),
                    context=error_context,
                    cause=e
                )

            # Create error result
            error_result = FlowResult(
                data={},
                flow_name=self.name,
                status=FlowStatus.ERROR,
                error=str(error),
                error_details={"error_type": type(error).__name__},
                metadata={"execution_time": execution_time},
                duration=execution_time
            )
            
            # Attach result to error
            error.result = error_result
            raise error
    
    async def _execute(self, context: Context) -> FlowResult:
        """Execute the stage with the given context.
        
        This is the core execution method that handles both sync and async process functions.
        It also handles error handling and result formatting.
        
        Args:
            context: Execution context
            
        Returns:
            Flow result with data and metadata
            
        Raises:
            ExecutionError: If execution fails
            ValidationError: If result is not a Pydantic model matching output_schema
        """
        try:
            # Execute process function based on signature
            if self._is_async:
                result = await self._execute_async(context)
            else:
                result = self._execute_sync(context)
            
            # Validate result type against output_schema if defined
            if self.output_schema is not None and not isinstance(result, FlowResult):
                # Result must be an instance of the expected output model
                if not isinstance(result, self.output_schema):
                    raise ValidationError(
                        f"Stage '{self.name}' must return an instance of {self.output_schema.__name__}, got {type(result).__name__}"
                    )
            
            # Convert result to FlowResult if needed
            if not isinstance(result, FlowResult):
                # Create FlowResult from Pydantic model
                if isinstance(result, BaseModel):
                    return FlowResult(
                        data=result,
                        original_type=type(result),
                        flow_name=self.name,
                        status=FlowStatus.SUCCESS
                    )
                else:
                    # This should not happen due to the validation above, but just in case
                    raise ValidationError(
                        f"Stage '{self.name}' returned unexpected type: {type(result).__name__}"
                    )
            
            return result
            
        except ValidationError as e:
            # Rethrow validation errors with stage context
            if hasattr(e, 'context') and e.context:
                e.context = e.context.add(stage=self.name)
            else:
                # Create a context if none exists
                e.context = ErrorContext.create(stage=self.name)
            raise
            
        except BaseError as e:
            # Add stage context to know which stage failed
            if hasattr(e, 'context') and e.context:
                e.context = e.context.add(stage=self.name)
            raise
            
        except Exception as e:
            # Convert other exceptions to ExecutionError
            error_context = ErrorContext.create(
                stage=self.name,
                function_name=getattr(self.process_fn, "__name__", "unknown"),
                context_data=getattr(context, "data", {})
            )
            
            raise ExecutionError(
                message=f"Stage execution failed: {str(e)}",
                context=error_context,
                cause=e
            )
    
    def _execute_sync(self, context: Context) -> Any:
        """Execute a synchronous process function.
        
        Args:
            context: Execution context
            
        Returns:
            Function execution result
        """
        return self.process_fn(context)
    
    async def _execute_async(self, context: Context) -> Any:
        """Execute an asynchronous process function.
        
        Args:
            context: Execution context
            
        Returns:
            Function execution result
        """
        return await self.process_fn(context)
    
    def __str__(self) -> str:
        """String representation."""
        fn_name = getattr(self.process_fn, "__name__", "unknown")
        return f"Stage(name='{self.name}', fn='{fn_name}', status={self.state.status})" 