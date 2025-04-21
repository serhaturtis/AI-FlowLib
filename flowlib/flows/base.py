"""Base flow implementation with enhanced execution model.

This module provides the foundation for all flow components with
enhanced result handling and error management.
"""

from abc import ABC
from datetime import datetime
import inspect
from enum import Enum
from typing import Optional, Dict, Any, Type, Union, TypeVar, Generic, List
from pydantic import BaseModel, Field, field_validator

from ..core.context import Context
from ..core.errors import ValidationError
from ..core.errors import ErrorManager, default_manager, ErrorContext, BaseError, ExecutionError
from .results import FlowResult
from .constants import FlowStatus
from pydantic import ValidationError

#T = TypeVar('T', bound='BaseModel')
    

class FlowSettings(BaseModel):
    """Settings for configuring flow execution behavior.
    
    This class provides:
    1. Timeout configuration for flow execution
    2. Retry behavior for handling transient errors
    3. Logging and debugging options
    4. Resource management settings
    """
    
    # Execution settings
    timeout_seconds: Optional[float] = None
    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    
    # Validation settings
    validate_inputs: bool = True
    validate_outputs: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Resource settings
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[int] = None
    
    # Advanced settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("max_retries")
    def validate_max_retries(cls, v: int) -> int:
        """Validate max retries."""
        if v < 0:
            raise ValueError("Max retries must be non-negative")
        return v
    
    @field_validator("retry_delay_seconds")
    def validate_retry_delay(cls, v: float) -> float:
        """Validate retry delay."""
        if v < 0:
            raise ValueError("Retry delay must be non-negative")
        return v
    
    def merge(self, other: Union['FlowSettings', Dict[str, Any]]) -> 'FlowSettings':
        """Merge with another settings object or dictionary.
        
        Args:
            other: Settings object or dictionary to merge with
            
        Returns:
            New merged settings object
        """
        if isinstance(other, FlowSettings):
            # Convert to dictionary
            other_dict = other.model_dump()
        elif isinstance(other, dict):
            # Convert dict to settings
            other_dict = other
        else:
            raise TypeError(f"Cannot merge with {type(other)}")
        
        # Create a copy of self as dictionary
        merged_dict = self.model_dump()
        
        # Update with other dict, handling nested custom_settings specially
        for key, value in other_dict.items():
            if key == "custom_settings" and value:
                # Create a new dict that is a copy of the current custom_settings
                if "custom_settings" not in merged_dict:
                    merged_dict["custom_settings"] = {}
                merged_dict["custom_settings"].update(value)
            else:
                merged_dict[key] = value
                
        return FlowSettings.model_validate(merged_dict)
    
    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> 'FlowSettings':
        """Create settings from dictionary.
        
        Args:
            settings_dict: Dictionary of settings
            
        Returns:
            Settings object
        """
        return cls.model_validate(settings_dict)
    
    def update(self, **kwargs) -> 'FlowSettings':
        """Create a new settings object with updated values.
        
        Args:
            **kwargs: New values to set
            
        Returns:
            New settings object
        """
        settings_dict = self.model_dump()
        settings_dict.update(kwargs)
        return FlowSettings.model_validate(settings_dict)
    
    def __str__(self) -> str:
        """String representation."""
        timeout_str = f"{self.timeout_seconds}s" if self.timeout_seconds else "None"
        return f"FlowSettings(timeout={timeout_str}, retries={self.max_retries}, debug={self.debug_mode})"

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
                if hasattr(method, "__pipeline__") and method.__pipeline__:
                    self.pipeline_method = method
                    
                    # Use correct attributes set by @pipeline decorator to potentially override schemas
                    # Use double underscores as set by the decorator and used by FlowMetadata
                    if hasattr(method, "__input_model__") and method.__input_model__:
                        input_schema = method.__input_model__
                        # Validate the pipeline's input model
                        if not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
                            raise ValueError(f"Pipeline __input_model__ must be a Pydantic BaseModel subclass, got {input_schema}")
                            
                    if hasattr(method, "__output_model__") and method.__output_model__:
                        output_schema = method.__output_model__
                        # Validate the pipeline's output model
                        if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
                            raise ValueError(f"Pipeline __output_model__ must be a Pydantic BaseModel subclass, got {output_schema}")
                    break

        self.input_schema = input_schema
        self.output_schema = output_schema
        self.metadata = metadata or {}
        self.error_manager = error_manager or default_manager
    
    def get_pipeline_input_model(self) -> Optional[Type[BaseModel]]:
        """Get the input model for this flow's pipeline.
        
        This method retrieves the input schema from either:
        1. The pipeline method's __input_model__ attribute if available
        2. The flow's input_schema attribute as a fallback
        
        Returns:
            The input model class or None if not defined
        """
        # First check if we can get it from the pipeline method
        pipeline_method_name = getattr(self.__class__, '__pipeline_method__', None)
        if pipeline_method_name:
            pipeline_method = getattr(self, pipeline_method_name)
            if pipeline_method and hasattr(pipeline_method, '__input_model__'):
                return getattr(pipeline_method, '__input_model__')
        
        # Fall back to the flow's input_schema attribute
        return self.input_schema
    
    def get_pipeline_output_model(self) -> Optional[Type[BaseModel]]:
        """Get the output model for this flow's pipeline.
        
        This method retrieves the output schema from either:
        1. The pipeline method's __output_model__ attribute if available
        2. The flow's output_schema attribute as a fallback
        
        Returns:
            The output model class or None if not defined
        """
        # First check if we can get it from the pipeline method
        pipeline_method_name = getattr(self.__class__, '__pipeline_method__', None)
        if pipeline_method_name:
            pipeline_method = getattr(self, pipeline_method_name)
            if pipeline_method and hasattr(pipeline_method, '__output_model__'):
                return getattr(pipeline_method, '__output_model__')
        
        # Fall back to the flow's output_schema attribute
        return self.output_schema
    
    def get_pipeline_method(self) -> Optional[Any]:
        """Get the pipeline method for this flow.
        
        Returns:
            The pipeline method or None if not found
        """
        # Find the pipeline method
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '__pipeline__') and attr.__pipeline__:
                return attr
        return None

    @classmethod
    def get_pipeline_method_cls(cls) -> Optional[Any]:
        """Get the pipeline method for this flow class.
        
        Returns:
            The pipeline method or None if not found
        """
        # Find the pipeline method
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, '__pipeline__') and attr.__pipeline__:
                return attr
        return None
    
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
        from pydantic import ValidationError # Ensure ValidationError is in scope for the except block
        try:
            # Initialize prepared input data variable
            pipeline_input_arg = None # Renamed from prepared_input_data

            # Create error boundary
            async with self.error_manager.async_error_boundary(context.data):
                
                # === MODIFIED SECTION: Get pipeline method and its expected input model ===
                pipeline_method_name = getattr(self.__class__, '__pipeline_method__', None)
                if not pipeline_method_name:
                    raise ExecutionError(
                        "No pipeline method found on flow. Each flow must have exactly one @pipeline method.",
                        error_context
                    )
                
                pipeline_method = getattr(self, pipeline_method_name)
                if not pipeline_method:
                    raise ExecutionError(
                        f"Pipeline method '{pipeline_method_name}' not found in flow class.",
                        error_context
                    )
                
                expected_input_model = getattr(pipeline_method, '__input_model__', None)
                # ==========================================================================

                # Prepare and validate input based on the pipeline's expected model
                if expected_input_model:
                    if not context.data:
                         raise ValueError("Context data is missing for pipeline requiring input.")
                         
                    # Ensure context.data matches or can be parsed into the expected model
                    if isinstance(context.data, expected_input_model):
                        pipeline_input_arg = context.data
                    elif isinstance(context.data, dict):
                        try:
                            pipeline_input_arg = expected_input_model(**context.data)
                        except Exception as validation_err:
                            raise ValidationError(f"Input validation failed for {expected_input_model.__name__}: {validation_err}") from validation_err
                    else:
                         raise TypeError(f"Context data type {type(context.data)} cannot be used for pipeline expecting {expected_input_model.__name__}")
                    
                    # Add validated input to error context
                    error_context = error_context.add(input_data=pipeline_input_arg)
                    
                    # Optional: Re-validate using the schema if needed (might be redundant)
                    # self._validate_input(pipeline_input_arg) 
                else:
                     # Pipeline expects no input model
                     pipeline_input_arg = None 

                # === ORIGINAL VALIDATION (can be removed or kept if Flow has separate schema) ===
                # if self.input_schema:
                #     prepared_input_data = self._prepare_input(context)
                #     error_context = error_context.add(input_data=prepared_input_data)
                #     self._validate_input(prepared_input_data)
                # ================================================================================

                # Add debug output
                print("="*50)
                print(f"DEBUG: Pipeline method name: {pipeline_method_name}")
                print(f"DEBUG: Pipeline method: {pipeline_method}")
                print(f"DEBUG: Pipeline method type: {type(pipeline_method)}")
                # print(f"DEBUG: Pipeline method dir: {dir(pipeline_method)}") # Less useful
                
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
                    
                    # Dump parameters for debugging
                    print(f"DEBUG: Pipeline parameters length: {len(pipeline_sig.parameters)}")
                    for param_name, param in pipeline_sig.parameters.items():
                        print(f"DEBUG: Parameter '{param_name}': {param}")
                    
                    # Handle passing the main input argument
                    if pipeline_sig.parameters:
                        param_names = list(pipeline_sig.parameters.keys())
                        first_param_name = param_names[0]
                        print(f"DEBUG: First parameter name: {first_param_name}")
                        
                        # If the first param isn't context/ctx, it expects the main input
                        if first_param_name not in ['context', 'ctx']:
                            print(f"DEBUG: Passing main input argument: {type(pipeline_input_arg)}")
                            pipeline_result = await pipeline_method(pipeline_input_arg, **pipeline_args)
                        else:
                            # First param is context/ctx, only pass kwargs
                            print("DEBUG: Calling pipeline_method with kwargs only (first param is context/ctx)")
                            pipeline_result = await pipeline_method(**pipeline_args)
                    else:
                        # No parameters, call with kwargs only (which might be empty)
                        print("DEBUG: Calling pipeline_method with kwargs only (no parameters)")
                        pipeline_result = await pipeline_method(**pipeline_args)
                    
                    # Validate result type against output_schema if defined
                    output_model = getattr(pipeline_method, '__output_model__', None)
                    if output_model is not None:
                        if not isinstance(pipeline_result, output_model):
                            # Import validation error if needed
                            from pydantic import ValidationError
                            raise ValidationError(
                                f"Pipeline '{pipeline_method.__name__}' must return an instance of {output_model.__name__}, got {type(pipeline_result).__name__}"
                            )
                    
                    result = FlowResult(
                        data=pipeline_result,  # Store the model directly
                        original_type=type(pipeline_result),
                        flow_name=self.name,
                        status=FlowStatus.SUCCESS
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
        
        If input_schema is defined, retrieves the validated Pydantic model instance
        from the context using `context.as_model()`. Otherwise, returns raw context data.

        Args:
            context: Execution context
            
        Returns:
            Prepared input data (Pydantic model instance if schema defined, else dict)
            
        Raises:
            ExecutionError: If preparation fails
            ValidationError: If `context.as_model()` fails or returns wrong type
        """
        try:
            if self.input_schema:
                # Context stores model as dict; reconstruct instance using as_model()
                print(f"DEBUG: _prepare_input calling context.as_model() for {self.input_schema.__name__}")
                model_instance = context.as_model()
                
                if model_instance is None:
                     # This happens if context had no model_type initially
                     raise ValidationError(
                         f"Context does not contain a model of type {self.input_schema.__name__} needed by flow '{self.name}'",
                         context=ErrorContext.create(flow_name=self.name, expected_type=self.input_schema.__name__)
                     )

                # Validate the reconstructed model instance type
                if not isinstance(model_instance, self.input_schema):
                    raise ValidationError(
                        f"Expected input model of type {self.input_schema.__name__}, but context provided {type(model_instance).__name__}",
                        validation_errors=[{
                            "location": "input",
                            "message": f"Expected {self.input_schema.__name__}, got {type(model_instance).__name__}",
                            "type": "type_error"
                        }],
                        context=ErrorContext.create(
                            flow_name=self.name,
                            expected_type=self.input_schema.__name__,
                            actual_type=type(model_instance).__name__
                        )
                    )
                
                print(f"DEBUG: _prepare_input successfully got model instance: {type(model_instance)}")
                # Add explicit logging of the instance being returned
                print(f"DEBUG: _prepare_input returning instance: {repr(model_instance)}") 
                return model_instance
            else:
                # No input schema defined for this flow, return raw data dict
                print("DEBUG: _prepare_input returning context.data (no input schema)")
                return context.data

        except ValidationError:
            # Re-raise validation errors
            raise
        except ValueError as ve:
            # Catch specific errors from as_model() if internal data is inconsistent
             raise ValidationError(
                f"Failed to reconstruct input model {self.input_schema.__name__} from context: {str(ve)}",
                context=ErrorContext.create(flow_name=self.name),
                cause=ve
             )
        except Exception as e:
            # Wrap other exceptions
            raise ExecutionError(
                message="Failed to prepare input",
                context=ErrorContext.create(
                    flow_name=self.name,
                    input_data=getattr(context, 'data', {})
                ),
                cause=e
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
    
    def get_description(self) -> str:
        """
        Get the flow description.
        
        This method is automatically added by the @flow decorator.
        
        Returns:
            Flow description
        """
        # This method is always implemented by the decorator
        # No need to raise NotImplementedError
        return ""

    