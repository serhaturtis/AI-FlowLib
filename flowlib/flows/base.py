"""Base flow implementation with enhanced execution model.

This module provides the foundation for all flow components with
enhanced result handling and error management.
"""

from abc import ABC, abstractmethod
from datetime import datetime
import inspect
from typing import Optional, Dict, Any, Type, Union, TypeVar, Generic, get_type_hints
from pydantic import BaseModel

from ..core.models.context import Context
from ..core.models.result import FlowResult, FlowStatus
from ..core.errors import (
    BaseError,
    ValidationError,
    ExecutionError,
    ErrorContext,
    ErrorManager,
    default_manager
)
from ..core.validation import validate_data

T = TypeVar('T')

class Flow(ABC, Generic[T]):
    """Base class for all flow components with enhanced result handling.
    
    This class provides:
    1. Consistent execution pattern with error handling
    2. Input and output validation with Pydantic models
    3. Attribute-based access to results
    """
    
    def __init__(
        self,
        name_or_instance: Union[str, object],
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_manager: Optional[ErrorManager] = None
    ):
        """Initialize flow.
        
        Args:
            name_or_instance: Either a name string or a flow class instance
            input_schema: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
            output_schema: Optional Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
            metadata: Optional metadata about the flow
            error_manager: Optional error manager instance
            
        Raises:
            ValueError: If input_schema or output_schema is provided but not a Pydantic BaseModel subclass.
        """
        # Validate input_schema and output_schema are Pydantic models if provided
        if input_schema is not None and not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
            raise ValueError(f"Flow input_schema must be a Pydantic BaseModel subclass, got {input_schema}")
        
        if output_schema is not None and not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise ValueError(f"Flow output_schema must be a Pydantic BaseModel subclass, got {output_schema}")
            
        if isinstance(name_or_instance, str):
            self.name = name_or_instance
            self.flow_instance = None
        else:
            # Handle when a flow class instance is passed in
            self.flow_instance = name_or_instance
            self.name = getattr(name_or_instance, "__flow_name__", name_or_instance.__class__.__name__)
            
            # Look for pipeline methods
            self.pipeline_method = None
            for name in dir(name_or_instance):
                method = getattr(name_or_instance, name)
                if hasattr(method, "_pipeline") and method._pipeline:
                    self.pipeline_method = method
                    
                    # Use pipeline input/output models if available
                    if hasattr(method, "input_model") and method.input_model:
                        input_schema = method.input_model
                        # Validate the pipeline's input model
                        if not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
                            raise ValueError(f"Pipeline input_model must be a Pydantic BaseModel subclass, got {input_schema}")
                            
                    if hasattr(method, "output_model") and method.output_model:
                        output_schema = method.output_model
                        # Validate the pipeline's output model
                        if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
                            raise ValueError(f"Pipeline output_model must be a Pydantic BaseModel subclass, got {output_schema}")
                    break

        self.input_schema = input_schema
        self.output_schema = output_schema
        self.metadata = metadata or {}
        self.error_manager = error_manager or default_manager
    
    async def execute(self, context: Context) -> FlowResult:
        """Execute the flow with given context.
        
        This is the ONLY method that should be called from outside the flow.
        It automatically executes the flow's pipeline method.
        
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
            flow_name=self.name
        )

        try:
            # Create error boundary
            async with self.error_manager.async_error_boundary(context.data):
                # Validate input if schema exists
                if self.input_schema:
                    input_data = self._prepare_input(context)
                    error_context = error_context.add(input_data=input_data)
                    self._validate_input(input_data)

                # Execute the flow's pipeline method
                # Determine the right method to call based on the class type
                
                if hasattr(self, 'flow_instance') and self.flow_instance and self.pipeline_method:
                    # Legacy pattern - Flow instance (not a decorated class), use self._execute
                    result = await self._execute(context)
                else:
                    # This is a decorated @flow class, call the pipeline method directly
                    pipeline_method_name = getattr(self.__class__, '__pipeline_method__', None)
                    
                    if not pipeline_method_name:
                        raise ExecutionError(
                            "No pipeline method found on flow. Each flow must have exactly one @pipeline method.",
                            error_context
                        )
                    
                    # Get the pipeline method
                    pipeline_method = getattr(self, pipeline_method_name)
                    if not pipeline_method:
                        raise ExecutionError(
                            f"Pipeline method '{pipeline_method_name}' not found in flow class.",
                            error_context
                        )
                    
                    # Add debug output
                    print("="*50)
                    print(f"DEBUG: Pipeline method name: {pipeline_method_name}")
                    print(f"DEBUG: Pipeline method: {pipeline_method}")
                    print(f"DEBUG: Pipeline method type: {type(pipeline_method)}")
                    print(f"DEBUG: Pipeline method dir: {dir(pipeline_method)}")
                    
                    # Execute the pipeline function
                    try:
                        pipeline_args = {}
                        pipeline_sig = inspect.signature(pipeline_method)
                        print(f"DEBUG: Pipeline signature: {pipeline_sig}")
                        
                        # Check if the pipeline expects a context parameter
                        if 'context' in pipeline_sig.parameters:
                            pipeline_args['context'] = context
                            print(f"DEBUG: Added context to pipeline_args")
                        elif 'ctx' in pipeline_sig.parameters:
                            pipeline_args['ctx'] = context
                            print(f"DEBUG: Added ctx to pipeline_args")
                        
                        # Handle differently based on input parameters for the pipeline
                        print(f"DEBUG: Pipeline parameters length: {len(pipeline_sig.parameters)}")
                        # Dump each parameter
                        for param_name, param in pipeline_sig.parameters.items():
                            print(f"DEBUG: Parameter '{param_name}': {param}")
                        
                        # Handle differently based on input parameters for the pipeline
                        # For a bound method, the signature doesn't include 'self',
                        # but we need to handle it as if it does when calling the method
                        # If there's at least one parameter, we should pass input_data
                        if pipeline_sig.parameters:
                            # Get input data
                            input_data = context.data
                            
                            # Debug logging
                            print(f"DEBUG: Pipeline method: {pipeline_method.__name__}")
                            print(f"DEBUG: Pipeline parameters: {pipeline_sig.parameters}")
                            print(f"DEBUG: Context data type: {type(input_data)}")
                            
                            # Validate input data against input_schema if set
                            if getattr(pipeline_method, '__input_model__', None):
                                input_model = pipeline_method.__input_model__
                                print(f"DEBUG: Expected input model: {input_model}")
                                # Input data must be an instance of the expected model
                                if not isinstance(input_data, input_model):
                                    raise ValidationError(
                                        f"Pipeline input must be an instance of {input_model.__name__}, got {type(input_data).__name__}"
                                    )
                            
                            # Get the parameter names
                            param_names = list(pipeline_sig.parameters.keys())
                            print(f"DEBUG: Parameter names: {param_names}")
                            print(f"DEBUG: Pipeline args: {pipeline_args}")
                            
                            # For methods, the first parameter is for input data
                            first_param = param_names[0] if param_names else None 
                            if first_param:
                                print(f"DEBUG: First parameter: {first_param}")
                                
                                # If the first parameter is 'context' or 'ctx', we shouldn't pass input_data as first arg
                                if first_param in ['context', 'ctx']:
                                    print("DEBUG: Calling pipeline_method with kwargs only (first param is context/ctx)")
                                    pipeline_result = await pipeline_method(**pipeline_args)
                                else:
                                    # Pass input_data as the first positional argument
                                    print("DEBUG: Calling pipeline_method with input_data as first arg")
                                    pipeline_result = await pipeline_method(input_data, **pipeline_args)
                            else:
                                # No parameters, don't pass input_data
                                print("DEBUG: Calling pipeline_method with no args")
                                pipeline_result = await pipeline_method(**pipeline_args)
                        else:
                            # No parameters at all
                            print("DEBUG: Calling pipeline_method with no args (no parameters)")
                            pipeline_result = await pipeline_method(**pipeline_args)
                        
                        # Validate result type against output_schema if defined
                        output_model = getattr(pipeline_method, '__output_model__', None)
                        if output_model is not None:
                            if not isinstance(pipeline_result, output_model):
                                raise ValidationError(
                                    f"Pipeline '{pipeline_method.__name__}' must return an instance of {output_model.__name__}, got {type(pipeline_result).__name__}"
                                )
                        
                        # Convert result to FlowResult
                        if isinstance(pipeline_result, FlowResult):
                            result = pipeline_result
                        elif isinstance(pipeline_result, BaseModel):
                            # For Pydantic models, store the model directly instead of converting to dict
                            # This ensures we maintain the Pydantic model instance throughout the pipeline
                            result = FlowResult(
                                data=pipeline_result,  # Store the model directly
                                original_type=type(pipeline_result),
                                flow_name=self.name,
                                status=FlowStatus.SUCCESS
                            )
                        else:
                            # For non-Pydantic model results, error out as this violates the strict model rule
                            raise ValidationError(
                                f"Pipeline must return a Pydantic model (found {type(pipeline_result).__name__}). Non-model return types are not allowed."
                            )
                            
                    except ValidationError:
                        # Reraise validation errors
                        raise
                    except Exception as e:
                        # Convert other errors to ExecutionError
                        raise ExecutionError(
                            message=f"Pipeline execution failed: {str(e)}",
                            context=error_context,
                            cause=e
                        )

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
            if isinstance(e, BaseError):
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

    def _validate_input(self, data: Any) -> None:
        """Validate input data against schema.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValidationError: If validation fails or if data is not a Pydantic model
        """
        if self.input_schema:
            # Input data must be an instance of the expected model
            if not isinstance(data, self.input_schema):
                raise ValidationError(
                    f"Input must be an instance of {self.input_schema.__name__}, got {type(data).__name__}",
                    validation_errors=[{
                        "location": "input",
                        "message": f"Expected {self.input_schema.__name__}, got {type(data).__name__}",
                        "type": "type_error"
                    }],
                    context=ErrorContext.create(
                        flow_name=self.name,
                        expected_type=self.input_schema.__name__,
                        actual_type=type(data).__name__
                    )
                )

    def _validate_output(self, data: Any) -> None:
        """Validate output data against schema.
        
        Args:
            data: Output data to validate
            
        Raises:
            ValidationError: If validation fails or if data is not a Pydantic model
        """
        if self.output_schema:
            # For FlowResult, we validate its data dict against the schema
            if isinstance(data, dict) and self.output_schema:
                # Should never directly pass dicts in the new strict mode
                raise ValidationError(
                    f"Output must be an instance of {self.output_schema.__name__}, got dict",
                    validation_errors=[{
                        "location": "output",
                        "message": f"Expected {self.output_schema.__name__}, got dict",
                        "type": "type_error"
                    }],
                    context=ErrorContext.create(
                        flow_name=self.name,
                        expected_type=self.output_schema.__name__,
                        actual_type="dict"
                    )
                )
            # If not a dict, must be an instance of the output schema
            elif not isinstance(data, self.output_schema):
                raise ValidationError(
                    f"Output must be an instance of {self.output_schema.__name__}, got {type(data).__name__}",
                    validation_errors=[{
                        "location": "output",
                        "message": f"Expected {self.output_schema.__name__}, got {type(data).__name__}",
                        "type": "type_error"
                    }],
                    context=ErrorContext.create(
                        flow_name=self.name,
                        expected_type=self.output_schema.__name__,
                        actual_type=type(data).__name__
                    )
                )

    def _prepare_input(self, context: Context) -> Any:
        """Prepare input data from context.
        
        Args:
            context: Execution context
            
        Returns:
            Prepared input data which must be a Pydantic model
            
        Raises:
            ExecutionError: If preparation fails
            ValidationError: If input data is not a Pydantic model when input_schema is defined
        """
        try:
            # The new Context has attribute-based access
            data = context.data
            
            # Handle the nested context structure used in CompositeFlow
            if isinstance(data, dict) and 'input' in data:
                data = data['input']
            
            # Input data must be a Pydantic model if input_schema is defined
            if self.input_schema and not isinstance(data, self.input_schema):
                raise ValidationError(
                    f"Input must be an instance of {self.input_schema.__name__}",
                    validation_errors=[{
                        "location": "input",
                        "message": f"Expected {self.input_schema.__name__}, got {type(data).__name__}",
                        "type": "type_error"
                    }],
                    context=ErrorContext.create(
                        flow_name=self.name,
                        expected_type=self.input_schema.__name__,
                        actual_type=type(data).__name__
                    )
                )
                
            # Return the data directly
            return data
        except ValidationError as e:
            # Re-raise validation errors with additional context
            raise
        except Exception as e:
            raise ExecutionError(
                message="Failed to prepare input",
                context=ErrorContext.create(
                    flow_name=self.name,
                    input_data=getattr(context, 'data', {})
                ),
                cause=e
            )

    async def _execute(self, context: Context) -> FlowResult:
        """Execute flow-specific logic.
        Must be implemented by flow subclasses unless a flow instance
        with a pipeline method is provided.
        
        Args:
            context: Execution context
            
        Returns:
            FlowResult containing execution outcome
        """
        if self.flow_instance and self.pipeline_method:
            # Execute pipeline method on flow instance
            result = await self.pipeline_method(context.data)
            
            # Convert to FlowResult if needed
            if not isinstance(result, FlowResult):
                if isinstance(result, BaseModel):
                    # For Pydantic models
                    return FlowResult(
                        data=result.dict(),
                        original_type=type(result),
                        flow_name=self.name,
                        status=FlowStatus.SUCCESS
                    )
                elif isinstance(result, dict):
                    # For dictionaries
                    return FlowResult(
                        data=result,
                        flow_name=self.name,
                        status=FlowStatus.SUCCESS
                    )
                else:
                    # For primitive types or other objects
                    return FlowResult(
                        data={"result": result},
                        flow_name=self.name,
                        status=FlowStatus.SUCCESS
                    )
            return result
        else:
            # Abstract method must be implemented by subclasses
            raise NotImplementedError(
                "Either implement _execute method in a Flow subclass or provide a flow instance with a pipeline method"
            )

    def add_error_handler(
        self,
        error_type: Type[BaseError],
        handler: Any
    ) -> None:
        """Add error handler to flow.
        
        Args:
            error_type: Type of error to handle
            handler: Handler instance
        """
        self.error_manager.register(error_type, handler)

    @property
    def full_name(self) -> str:
        """Get fully qualified flow name."""
        return f"{self.__class__.__name__}.{self.name}"

    def __str__(self) -> str:
        """String representation."""
        return f"Flow(name='{self.name}')" 