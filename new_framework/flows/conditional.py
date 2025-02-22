# src/flows/conditional.py

from typing import Any, Dict, Optional, Type, Callable
from pydantic import BaseModel

from .base import Flow
from ..core.models.context import Context
from ..core.models.results import FlowResult, FlowStatus
from ..core.errors.base import ValidationError, ExecutionError, ErrorContext

class ConditionalFlow(Flow):
    """A flow that conditionally executes one of two flows based on a condition.
    
    This flow evaluates a condition function on the input data and then either:
    1. Executes the success_flow if the condition returns True
    2. Executes the failure_flow if the condition returns False
    3. Passes through the input data if no flow is specified for the condition result
    
    All execution results include metadata about the condition evaluation and path taken.
    """
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        success_flow: Optional[Flow] = None,
        failure_flow: Optional[Flow] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize ConditionalFlow.
        
        Args:
            name: Flow name
            condition: Function that takes flow data and returns bool
            success_flow: Flow to execute if condition is True
            failure_flow: Flow to execute if condition is False
            input_schema: Optional schema for input validation
            output_schema: Optional schema for output validation
            metadata: Optional flow metadata
            
        Raises:
            ValidationError: If condition is not callable
        """
        super().__init__(
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )
        
        if not callable(condition):
            raise ValidationError(
                "Condition must be callable",
                ErrorContext(details={
                    "flow_name": name,
                    "condition_type": type(condition)
                })
            )
            
        self.condition = condition
        self.success_flow = success_flow
        self.failure_flow = failure_flow
        
    async def _execute(self, context: Context) -> FlowResult:
        """Execute conditional flow.
        
        The execution follows these steps:
        1. Validate and prepare input data
        2. Evaluate the condition
        3. Select appropriate flow based on condition result
        4. Execute selected flow or pass through input
        5. Return result with condition metadata
        
        Args:
            context: Execution context containing input data and metadata
            
        Returns:
            FlowResult containing execution results and metadata
            
        Raises:
            ExecutionError: If condition evaluation or flow execution fails
            ValidationError: If input validation fails
        """
        try:
            # Get input data
            input_data = self._prepare_input(context)
            
            # Evaluate condition
            try:
                condition_result = self.condition(input_data)
            except Exception as e:
                raise ExecutionError(
                    message="Condition evaluation failed",
                    context=ErrorContext(details={
                        "flow_name": self.name,
                        "error": str(e),
                        "input_keys": list(input_data.keys())
                    })
                )
            
            # Select flow based on condition
            selected_flow = self.success_flow if condition_result else self.failure_flow
            selected_path = "success" if condition_result else "failure"
            
            # If no flow selected, just pass through the input
            if selected_flow is None:
                return FlowResult(
                    flow_name=self.name,
                    status=FlowStatus.SUCCESS,
                    data=input_data,
                    metadata={
                        "condition_result": condition_result,
                        "selected_path": selected_path,
                        "passthrough": True
                    }
                )
            
            # Execute selected flow
            result = await selected_flow._execute(context)
            
            # Add conditional metadata
            result.metadata.update({
                "condition_result": condition_result,
                "selected_path": selected_path,
                "passthrough": False
            })
            
            return result
            
        except Exception as e:
            if not isinstance(e, (ValidationError, ExecutionError)):
                e = ExecutionError(
                    message=f"Conditional flow execution failed: {str(e)}",
                    context=ErrorContext(details={
                        "flow_name": self.name,
                        "input_keys": list(input_data.keys()) if 'input_data' in locals() else None
                    })
                )
            raise e
    
    def cleanup(self) -> None:
        """Clean up resources used by success and failure flows."""
        if self.success_flow:
            self.success_flow.cleanup()
        if self.failure_flow:
            self.failure_flow.cleanup()
            
    def __str__(self) -> str:
        """String representation showing flow name and available paths."""
        paths = []
        if self.success_flow:
            paths.append(f"success->{self.success_flow.name}")
        if self.failure_flow:
            paths.append(f"failure->{self.failure_flow.name}")
        return f"ConditionalFlow(name='{self.name}', paths=[{', '.join(paths)}])"