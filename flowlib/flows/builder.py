"""Flow builder for creating and composing flows.

This module provides a builder pattern for constructing flow pipelines.
The builder supports:
- Linear flow composition
- Conditional branching
- Output schema validation
- Metadata attachment
"""

from typing import Dict, Any, List, Optional, Callable, Type
from pydantic import BaseModel

from .base import Flow
from .stage import Stage
from .composite import CompositeFlow
from .conditional import ConditionalFlow
from ..core.errors.base import ValidationError, ErrorContext

class FlowBuilder:
    """Builder pattern for creating and composing flows.
    
    This class provides a fluent interface for constructing flow pipelines
    with support for linear composition, conditional branching, validation,
    and metadata attachment.
    
    Example:
        ```python
        flow = (FlowBuilder("my_pipeline")
                .add_stage("stage1", process_fn1)
                .add_stage("stage2", process_fn2, 
                          condition=lambda x: x["value"] > 0,
                          alternative=fallback_flow)
                .set_output_schema(OutputModel)
                .build())
        ```
    """
    
    def __init__(self, name: str):
        """Initialize flow builder.
        
        Args:
            name: Name of the flow pipeline
        """
        self.name = name
        self._stages: List[Flow] = []
        self._conditions: Dict[int, Callable[[Dict[str, Any]], bool]] = {}
        self._alternative_paths: Dict[int, Flow] = {}
        self._output_schema: Optional[Type[BaseModel]] = None
        self._metadata: Dict[str, Any] = {}
    
    def add_stage(
        self,
        name: str,
        process_fn: Callable,
        output_schema: Optional[Type[BaseModel]] = None,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        alternative: Optional[Flow] = None
    ) -> 'FlowBuilder':
        """Add a processing stage to the pipeline.
        
        Args:
            name: Stage name
            process_fn: Processing function (can be async)
            output_schema: Optional output validation schema
            condition: Optional condition for conditional execution
            alternative: Optional alternative flow if condition fails
            
        Returns:
            Self for chaining
        """
        stage = Stage(
            name=name,
            process_fn=process_fn,
            output_schema=output_schema
        )
        
        stage_index = len(self._stages)
        self._stages.append(stage)
        
        if condition:
            self._conditions[stage_index] = condition
        if alternative:
            self._alternative_paths[stage_index] = alternative
            
        return self
    
    def set_output_schema(self, schema: Type[BaseModel]) -> 'FlowBuilder':
        """Set output schema for final validation.
        
        Args:
            schema: Pydantic model for output validation
            
        Returns:
            Self for chaining
        """
        self._output_schema = schema
        return self
    
    def add_metadata(self, **metadata: Any) -> 'FlowBuilder':
        """Add metadata to the flow pipeline.
        
        Args:
            **metadata: Key-value metadata pairs
            
        Returns:
            Self for chaining
        """
        self._metadata.update(metadata)
        return self
    
    def build(self) -> Flow:
        """Build the flow pipeline.
        
        Returns:
            Constructed flow pipeline
            
        Raises:
            ValidationError: If pipeline configuration is invalid
        """
        if not self._stages:
            raise ValidationError(
                "Flow pipeline must have at least one stage",
                ErrorContext.create(flow_name=self.name)
            )
        
        # For single stage without conditions
        if len(self._stages) == 1 and not self._conditions:
            flow = self._stages[0]
            if self._output_schema:
                flow.output_schema = self._output_schema
            flow.metadata.update(self._metadata)
            return flow
        
        # Build pipeline with conditions and alternatives
        current_flow = None
        for i, stage in enumerate(reversed(self._stages)):
            index = len(self._stages) - i - 1
            
            if index in self._conditions:
                # Create conditional flow
                condition = self._conditions[index]
                alternative = self._alternative_paths.get(index)
                
                if current_flow:
                    # Connect to next stage
                    flow = ConditionalFlow(
                        name=f"{stage.name}_conditional",
                        condition=condition,
                        success_flow=stage,
                        failure_flow=alternative or current_flow
                    )
                else:
                    # Last stage
                    flow = ConditionalFlow(
                        name=f"{stage.name}_conditional",
                        condition=condition,
                        success_flow=stage,
                        failure_flow=alternative or stage
                    )
            else:
                # Create sequential flow
                if current_flow:
                    flow = CompositeFlow(
                        name=f"{stage.name}_composite",
                        flows=[stage, current_flow]
                    )
                else:
                    flow = stage
            
            current_flow = flow
        
        # Set output schema and metadata on final flow
        if self._output_schema:
            current_flow.output_schema = self._output_schema
        current_flow.metadata.update(self._metadata)
        
        return current_flow
    
    @staticmethod
    def linear(*stages: Flow) -> Flow:
        """Create a simple linear pipeline from stages.
        
        Args:
            *stages: Flow stages in order
            
        Returns:
            Composite flow of stages
            
        Raises:
            ValidationError: If no stages provided
        """
        if not stages:
            raise ValidationError(
                "Linear pipeline requires at least one stage",
                ErrorContext.create()
            )
        
        if len(stages) == 1:
            return stages[0]
            
        return CompositeFlow(
            name="linear_pipeline",
            flows=list(stages)
        )
    
    @staticmethod
    def conditional(
        condition: Callable[[Dict[str, Any]], bool],
        success_flow: Flow,
        failure_flow: Optional[Flow] = None,
        name: str = "conditional"
    ) -> Flow:
        """Create a simple conditional flow.
        
        Args:
            condition: Condition function
            success_flow: Flow to execute on success
            failure_flow: Optional flow to execute on failure
            name: Optional flow name
            
        Returns:
            Conditional flow
        """
        return ConditionalFlow(
            name=name,
            condition=condition,
            success_flow=success_flow,
            failure_flow=failure_flow
        )