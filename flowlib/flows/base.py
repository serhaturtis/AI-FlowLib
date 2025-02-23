# src/core/base/flow.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel

from ..core.models.context import Context
from ..core.models.results import FlowResult, FlowStatus
from ..core.models.state import FlowState
from ..core.errors.base import BaseError, ValidationError, ExecutionError, ErrorContext
from ..core.errors.manager import ErrorManager, ErrorHandler

class Flow(ABC):
    """
    Base class for all flow components.
    Enhanced with improved error handling.
    """
    
    def __init__(
        self,
        name: str,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_manager: Optional[ErrorManager] = None
    ):
        """
        Initialize flow.
        
        Args:
            name: Unique name for the flow
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation 
            metadata: Optional metadata about the flow
            error_manager: Optional error manager instance
        """
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.state = FlowState()
        self.metadata = metadata or {}
        self.error_manager = error_manager or ErrorManager()

    async def execute(self, context: Context) -> FlowResult:
        """
        Execute the flow with given context.
        
        Args:
            context: Execution context
            
        Returns:
            FlowResult containing execution outcome
            
        Raises:
            BaseError: If execution fails
        """
        # Initialize execution
        self.state.start_execution()
        execution_start = datetime.now()
        error_context = ErrorContext.create(
            flow_name=self.name,
            execution_id=context.id
        )

        try:
            # Create error boundary
            async with self.error_manager.error_boundary(context.data):
                # Validate input if schema exists
                if self.input_schema:
                    input_data = self._prepare_input(context)
                    error_context = error_context.add(input_data=input_data)
                    self._validate_input(input_data)

                # Execute flow-specific logic
                result = await self._execute(context)

                # Validate output if schema exists
                if self.output_schema and result.data:
                    self._validate_output(result.data)

                # Record successful execution
                self.state.mark_success()
                execution_time = (datetime.now() - execution_start).total_seconds()

                # Add execution info to result
                result.metadata.update({
                    "execution_time": execution_time,
                    "flow_name": self.name,
                    "status": FlowStatus.SUCCESS
                })

                return result

        except Exception as e:
            # Record failure
            self.state.mark_failure(error=e)
            execution_time = (datetime.now() - execution_start).total_seconds()

            # Enhance error context
            error_context = error_context.add(
                execution_time=execution_time,
                flow_status=self.state.status.value
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

            raise error

    def _validate_input(self, data: Dict[str, Any]) -> None:
        """
        Validate input data against schema.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            self.input_schema.model_validate(data)
        except Exception as e:
            raise ValidationError(
                message="Input validation failed",
                validation_errors=[{
                    "location": "input",
                    "message": str(e)
                }],
                context=ErrorContext.create(
                    flow_name=self.name,
                    input_data=data
                ),
                cause=e
            )

    def _validate_output(self, data: Dict[str, Any]) -> None:
        """
        Validate output data against schema.
        
        Args:
            data: Output data to validate
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            self.output_schema.model_validate(data)
        except Exception as e:
            raise ValidationError(
                message="Output validation failed",
                validation_errors=[{
                    "location": "output",
                    "message": str(e)
                }],
                context=ErrorContext.create(
                    flow_name=self.name,
                    output_data=data
                ),
                cause=e
            )

    def _prepare_input(self, context: Context) -> Dict[str, Any]:
        """
        Prepare input data from context.
        
        Args:
            context: Execution context
            
        Returns:
            Prepared input data
            
        Raises:
            ExecutionError: If preparation fails
        """
        try:
            return context.data.copy()
        except Exception as e:
            raise ExecutionError(
                message="Failed to prepare input",
                context=ErrorContext.create(
                    flow_name=self.name,
                    input_data=getattr(context, 'data', {})
                ),
                cause=e
            )

    def _prepare_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare output data for flow result.
        
        This method handles:
        1. Output data validation if schema exists
        2. Data transformation/cleanup if needed
        3. Ensuring output matches expected format
        4. Proper handling of nested models and composite results
        
        Args:
            data: Raw output data to prepare
            
        Returns:
            Prepared output data
            
        Raises:
            ValidationError: If output validation fails
            ExecutionError: If output preparation fails
        """
        try:
            # If no output schema, return data as is
            if not self.output_schema:
                return data

            # For composite flows, structure the results based on schema
            if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                # Get the expected fields from the output schema
                expected_fields = self.output_schema.model_fields
                structured_data = {}

                # Map stage results to schema fields
                for field_name, field_info in expected_fields.items():
                    field_type = field_info.annotation
                    
                    # Find the stage result that matches this field's type
                    for stage_name, stage_data in data.items():
                        try:
                            # Try to validate the stage data against this field's type
                            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                                validated = field_type.model_validate(stage_data)
                                structured_data[field_name] = validated.model_dump()
                                break
                        except Exception:
                            continue
                
                # If we couldn't map all required fields, try direct validation
                if len(structured_data) < len([f for f in expected_fields.values() if f.is_required]):
                    # Try to validate the merged data
                    merged = {}
                    for stage_data in data.values():
                        merged.update(stage_data)
                    return self._validate_and_convert(merged)
                
                return structured_data

            # Handle single result validation
            return self._validate_and_convert(data)

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ExecutionError(
                message="Failed to prepare output",
                context=ErrorContext.create(
                    flow_name=self.name,
                    output_data=data,
                    error_type=type(e).__name__
                ),
                cause=e
            )

    def _validate_and_convert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against schema and convert to dictionary.
        
        Args:
            data: Data to validate and convert
            
        Returns:
            Validated and converted data
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(data, self.output_schema):
                return data.model_dump()
            else:
                validated = self.output_schema.model_validate(data)
                return validated.model_dump()
        except Exception as e:
            raise ValidationError(
                message="Output validation failed",
                validation_errors=[{
                    "location": "output",
                    "message": str(e)
                }],
                context=ErrorContext.create(
                    flow_name=self.name,
                    output_data=data,
                    schema=self.output_schema.model_json_schema()
                ),
                cause=e
            )

    @abstractmethod
    async def _execute(self, context: Context) -> FlowResult:
        """
        Execute flow-specific logic.
        Must be implemented by flow subclasses.
        
        Args:
            context: Execution context
            
        Returns:
            FlowResult containing execution outcome
        """
        pass

    def add_error_handler(
        self,
        error_type: Type[BaseError],
        handler: ErrorHandler
    ) -> None:
        """
        Add error handler to flow.
        
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
        return f"Flow(name='{self.name}', status={self.state.status})"