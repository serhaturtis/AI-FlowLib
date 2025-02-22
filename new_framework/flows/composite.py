# src/flows/composite.py

from typing import Dict, Any, Optional, List, Type, Callable, Awaitable
from pydantic import BaseModel
from datetime import datetime

from .base import Flow
from ..core.models.context import Context
from ..core.models.results import FlowResult, FlowStatus
from ..core.errors.base import ValidationError, ExecutionError, ErrorContext
from ..core.errors.handlers import ErrorHandler

class CompositeError(ExecutionError):
    """Error specific to composite flow operations."""
    pass

class FlowExecutionError(CompositeError):
    """Error during individual flow execution."""
    
    def __init__(
        self,
        message: str,
        flow_name: str,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.failed_flow = flow_name
        if self.context:
            self.context.add_metadata(failed_flow=flow_name)

class CompositeFlow(Flow):
    """A flow that combines multiple flows into a single workflow.
    
    This flow can operate in two modes:
    1. Sequential mode: Flows are executed in sequence, with each flow's output available to the next
    2. Connected mode: Flows are connected explicitly via a connection map
    
    In both modes, error handlers can be registered to handle specific error types.
    """
    
    def __init__(
        self,
        name: str,
        flows: List[Flow],
        connections: Optional[Dict[str, str]] = None,
        error_handlers: Optional[Dict[str, Flow]] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize CompositeFlow.
        
        Args:
            name: Flow name
            flows: List of component flows to execute
            connections: Optional flow connection mapping. If None, flows are executed sequentially
            error_handlers: Optional error handler mapping
            input_schema: Optional schema for input validation
            output_schema: Optional schema for output validation
            metadata: Optional flow metadata
            
        Raises:
            ValidationError: If flow configuration is invalid
        """
        super().__init__(
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )
        
        if not flows:
            raise ValidationError(
                "CompositeFlow requires at least one flow",
                ErrorContext.create(
                    flow_name=name
                )
            )
            
        self.flows = flows
        self.connections = connections
        self.error_handlers = error_handlers or {}
        self._flow_map = {f.name: f for f in flows}
        
        # Validate connections if provided
        if connections:
            self._validate_connections()
            
        # Validate output schemas
        if output_schema:
            self._validate_output_schemas()
    
    def _validate_connections(self) -> None:
        """Validate flow connections.
        
        Raises:
            ValidationError: If connections are invalid
        """
        flow_names = set(self._flow_map.keys())
        
        for source, target in self.connections.items():
            if source not in flow_names:
                raise ValidationError(
                    f"Unknown source flow: {source}",
                    ErrorContext.create(
                        flow_name=self.name,
                        available_flows=list(flow_names)
                    )
                )
            if target not in flow_names:
                raise ValidationError(
                    f"Unknown target flow: {target}",
                    ErrorContext.create(
                        flow_name=self.name,
                        available_flows=list(flow_names)
                    )
                )
    
    def _validate_output_schemas(self) -> None:
        """
        Validate that flow output schemas are compatible with composite output schema.
        
        This ensures that flow outputs can be properly structured into the expected format.
        Handles both direct schema matches and nested composite flows.
        
        Raises:
            ValidationError: If schemas are incompatible
        """
        if not self.output_schema:
            return
            
        output_fields = self.output_schema.model_fields
        
        # Check each flow's output schema
        for flow in self.flows:
            if not flow.output_schema:
                continue
                
            # Case 1: Direct schema match
            if flow.output_schema == self.output_schema:
                continue
                
            # Case 2: Flow output schema matches a field in our output schema
            field_match = False
            for field_name, field_info in output_fields.items():
                if (isinstance(field_info.annotation, type) and 
                    issubclass(flow.output_schema, field_info.annotation)):
                    field_match = True
                    break
            
            if field_match:
                continue
                
            # Case 3: Flow is composite and its fields match our schema fields
            if isinstance(flow, CompositeFlow) and flow.output_schema:
                flow_fields = flow.output_schema.model_fields
                fields_match = True
                
                # Check if all required fields in our schema can be filled by the flow
                for field_name, field_info in output_fields.items():
                    if not field_info.is_required:
                        continue
                        
                    field_match = False
                    if field_name in flow_fields:
                        field_match = True
                    
                    if not field_match:
                        fields_match = False
                        break
                
                if fields_match:
                    continue
            
            # If we get here, schemas are not compatible
            raise ValidationError(
                f"Flow '{flow.name}' output schema is not compatible with composite output schema",
                ErrorContext.create(
                    flow_name=self.name,
                    flow_schema=flow.output_schema.model_json_schema(),
                    output_schema=self.output_schema.model_json_schema(),
                    flow_type=type(flow).__name__
                )
            )
    
    def _structure_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Structure flow results according to output schema.
        
        Args:
            results: Dictionary of flow results
            
        Returns:
            Structured data matching output schema
            
        Raises:
            ValidationError: If results cannot be structured properly
        """
        if not self.output_schema:
            return results
            
        output_fields = self.output_schema.model_fields
        structured = {}
        
        # Case 1: Single flow produced complete output
        for flow_name, flow_data in results.items():
            flow = self._flow_map[flow_name]
            if flow.output_schema == self.output_schema:
                try:
                    return self.output_schema(**flow_data).model_dump()
                except Exception:
                    # If validation fails, try other structuring methods
                    pass
        
        # Case 2: Multiple flows contribute to different fields
        for flow_name, flow_data in results.items():
            flow = self._flow_map[flow_name]
            if not flow.output_schema:
                continue
                
            # Handle composite flow results
            if isinstance(flow, CompositeFlow):
                for field_name, field_info in output_fields.items():
                    if field_name in flow_data:
                        structured[field_name] = flow_data[field_name]
                continue
            
            # Handle individual flow results
            for field_name, field_info in output_fields.items():
                if (isinstance(field_info.annotation, type) and 
                    issubclass(flow.output_schema, field_info.annotation)):
                    structured[field_name] = flow_data
                    break
        
        # Add default values for optional fields
        for field_name, field_info in output_fields.items():
            if field_name not in structured and field_info.default is not None:
                structured[field_name] = field_info.default
        
        # Validate final structure
        try:
            return self.output_schema(**structured).model_dump()
        except Exception as e:
            raise ValidationError(
                "Failed to create valid output structure",
                ErrorContext.create(
                    flow_name=self.name,
                    error=str(e),
                    structured_data=structured,
                    required_fields=[
                        name for name, field in output_fields.items() 
                        if field.is_required
                    ]
                )
            )
    
    async def _execute(self, context: Context) -> FlowResult:
        """Execute composite flow logic.
        
        In sequential mode, flows are executed in order with results passed forward.
        In connected mode, flows are executed according to the connection map.
        
        Args:
            context: Execution context containing input data and metadata
            
        Returns:
            FlowResult containing execution results and metadata
            
        Raises:
            ExecutionError: If flow execution fails
            ValidationError: If input validation fails
        """
        results = {}
        executed_flows = []
        current_context = context.copy()
        start_time = datetime.now()
        
        try:
            if self.connections:
                # Connected mode execution
                current_flow = self.flows[0]
                while current_flow:
                    result = await self._execute_flow(current_flow, current_context)
                    results[current_flow.name] = result.data
                    executed_flows.append(current_flow.name)
                    
                    # Update context for next flow
                    current_context.data.update(results)
                    
                    # Get next flow from connections
                    next_name = self.connections.get(current_flow.name)
                    current_flow = self._flow_map.get(next_name)
            else:
                # Sequential mode execution
                for flow in self.flows:
                    result = await self._execute_flow(flow, current_context)
                    results[flow.name] = result.data
                    executed_flows.append(flow.name)
                    
                    # Update context for next flow
                    current_context.data.update(results)
            
            # Structure and validate final output
            try:
                output_data = self._structure_results(results)
            except ValidationError as e:
                return FlowResult(
                    flow_name=self.name,
                    status=FlowStatus.ERROR,
                    data=results,  # Include raw results
                    error="Failed to structure flow results",
                    error_details={
                        "error": str(e),
                        "flow_results": results
                    },
                    metadata={
                        "executed_flows": executed_flows,
                        "duration": (datetime.now() - start_time).total_seconds()
                    }
                )
            
            return FlowResult(
                flow_name=self.name,
                status=FlowStatus.SUCCESS,
                data=output_data,
                metadata={
                    "executed_flows": executed_flows,
                    "flow_count": len(executed_flows),
                    "duration": (datetime.now() - start_time).total_seconds()
                }
            )
            
        except Exception as e:
            if not isinstance(e, (ValidationError, ExecutionError)):
                e = ExecutionError(
                    message=f"Composite flow execution failed: {str(e)}",
                    context=ErrorContext.create(
                        flow_name=self.name,
                        executed_flows=executed_flows,
                        current_flow=current_flow.name if 'current_flow' in locals() else None
                    )
                )
            
            # Try error handler if available
            error_type = type(e).__name__
            if error_type in self.error_handlers:
                handler = self.error_handlers[error_type]
                try:
                    error_context = current_context.copy()
                    error_context.data["error"] = str(e)
                    return await handler._execute(error_context)
                except Exception as handler_error:
                    # If handler fails, raise original error
                    raise e
            raise e
    
    async def _execute_flow(self, flow: Flow, context: Context) -> FlowResult:
        """Execute a single flow with error handling.
        
        Args:
            flow: Flow to execute
            context: Current execution context
            
        Returns:
            Flow execution result
            
        Raises:
            ExecutionError: If flow execution fails
        """
        try:
            return await flow._execute(context)
        except Exception as e:
            if not isinstance(e, (ValidationError, ExecutionError)):
                e = ExecutionError(
                    message=f"Flow execution failed: {str(e)}",
                    context=ErrorContext.create(
                        flow_name=flow.name,
                        parent_flow=self.name
                    )
                )
            raise e
    
    def cleanup(self) -> None:
        """Clean up resources used by all flows."""
        for flow in self.flows:
            flow.cleanup()
        for handler in self.error_handlers.values():
            handler.cleanup()
    
    def __str__(self) -> str:
        """String representation showing flow structure."""
        if self.connections:
            structure = [f"{f.name} -> {self.connections.get(f.name, 'END')}" for f in self.flows]
        else:
            structure = [f.name for f in self.flows]
        return f"CompositeFlow(name='{self.name}', structure=[{' -> '.join(structure)}])"

    def create_processor(self) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        """Create a processor function that can be used as a stage process_fn.
        
        This creates a standardized wrapper that handles:
        1. Context creation from input data
        2. Flow execution
        3. Error status handling
        4. Result data extraction
        
        Returns:
            Async callable that processes input data and returns output data
        """
        async def processor(data: Dict[str, Any]) -> Dict[str, Any]:
            context = Context(data=data)
            result = await self._execute(context)
            if result.status != FlowStatus.SUCCESS:
                raise ExecutionError(
                    f"Composite flow execution failed: {self.name}",
                    ErrorContext.create(
                        flow_name=self.name,
                        error=result.error,
                        metadata=result.metadata
                    )
                )
            return result.data
        
        return processor

class CompositeErrorHandler(ErrorHandler):
    """Handles errors in composite flows."""
    
    def __init__(self, error_handlers: Dict[str, Flow]):
        self.error_handlers = error_handlers

    async def handle(
        self,
        error: ExecutionError,
        context: Dict[str, Any]
    ) -> Optional[FlowResult]:
        """
        Handle flow execution error.
        
        Args:
            error: Error to handle
            context: Execution context
            
        Returns:
            Handler result if successful, None otherwise
        """
        if not isinstance(error, FlowExecutionError):
            return None
            
        # Find appropriate error handler
        handler = None
        error_type = type(error.cause).__name__
        
        if error_type in self.error_handlers:
            handler = self.error_handlers[error_type]
        elif "default" in self.error_handlers:
            handler = self.error_handlers["default"]
            
        if not handler:
            return None
            
        try:
            # Execute error handler
            error_context = Context(data=context)
            error_context.set("error", str(error), temporary=True)
            error_context.set("error_flow", error.failed_flow, temporary=True)
            
            result = await handler._execute(error_context)
            
            # Add error handling metadata
            result.metadata.update({
                "handled_error": str(error),
                "handler_flow": handler.name,
                "original_flow": error.failed_flow
            })
            
            return result
            
        except Exception as e:
            # Chain error handler failure
            if not isinstance(e, ExecutionError):
                e = CompositeError(
                    message=f"Error handler failed: {str(e)}",
                    context=error.context.add_metadata(
                        handler_flow=handler.name
                    ),
                    cause=e
                )
            error.chain(e)
            return None

    def __str__(self) -> str:
        """String representation."""
        return (
            f"CompositeFlow(name='{self.name}', "
            f"flows={len(self.flows)}, "
            f"connections={len(self.connections)})"
        )