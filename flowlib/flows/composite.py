"""Composite flow implementation for multi-stage pipelines.

This module provides an enhanced CompositeFlow class for organizing
multiple stages into pipelines with improved flow control, error handling,
and result propagation.
"""

from typing import Dict, List, Optional, Type, Any, Iterable, Mapping, Sequence, Union
from pydantic import BaseModel
import logging

from ..core.models.context import Context
from ..core.models.result import FlowResult, FlowStatus
from ..core.errors import ExecutionError, ErrorContext
from .base import Flow
from .stage import Stage
from .registry import stage_registry

logger = logging.getLogger(__name__)

class CompositeFlow(Flow):
    """Enhanced composite flow for multi-stage pipelines.
    
    This class provides:
    1. Clean organization of multiple stages into a pipeline
    2. Improved flow control with success/error handling
    3. Rich metadata for debugging and monitoring
    4. Access to individual stage results
    5. Automatic access to registered standalone stages
    """
    
    def __init__(
        self,
        name: str,
        stages: Optional[List[Flow]] = None,
        stage_names: Optional[List[str]] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_load_stages: bool = True,
        **kwargs
    ):
        """Initialize composite flow.
        
        Args:
            name: Unique name for the composite flow
            stages: Optional list of stages to execute
            stage_names: Optional list of stage names to load from registry
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation
            metadata: Optional metadata about the flow
            auto_load_stages: Whether to automatically load standalone stages from registry
            **kwargs: Additional options passed to the base Flow
        """
        super().__init__(
            name_or_instance=name,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
            **kwargs
        )
        self.stages = stages or []
        self._stage_instances = {}
        
        # Set the flow_instance and pipeline_method attributes
        # to ensure the Flow.execute method calls our _execute method
        self.flow_instance = self
        self.pipeline_method = self.run_pipeline
        
        # Load stages from registry if specified
        if stage_names:
            self._load_stages_from_registry(stage_names)
        
        # If auto_load_stages is True, load all standalone stages
        if auto_load_stages:
            self._load_all_standalone_stages()
        
        # Update metadata with stage info
        self._update_metadata()
    
    def _load_stages_from_registry(self, stage_names: List[str]) -> None:
        """Load specified stages from the registry.
        
        Args:
            stage_names: List of stage names to load
        """
        for stage_name in stage_names:
            try:
                stage_info = stage_registry.get_stage(stage_name)
                if stage_info.get('is_standalone', False):
                    stage = Stage(
                        name=stage_name,
                        process_fn=stage_info.get('func'),
                        input_schema=stage_info.get('input_model'),
                        output_schema=stage_info.get('output_model'),
                        metadata=stage_info.get('metadata', {})
                    )
                    self.stages.append(stage)
                    self._stage_instances[stage_name] = stage
                else:
                    logger.warning(f"Stage '{stage_name}' is not a standalone stage and cannot be loaded directly")
            except KeyError:
                logger.warning(f"Stage '{stage_name}' not found in registry")
    
    def _load_all_standalone_stages(self) -> None:
        """Load all standalone stages from the registry."""
        standalone_stage_names = stage_registry.get_stages()
        self._load_stages_from_registry(standalone_stage_names)
    
    def _update_metadata(self) -> None:
        """Update metadata with stage information."""
        stage_info = []
        for stage in self.stages:
            stage_info.append({
                "name": stage.name,
                "type": stage.__class__.__name__
            })
        
        self.metadata["stages"] = stage_info
        self.metadata["stage_count"] = len(self.stages)
    
    def add_stage(self, stage: Flow) -> 'CompositeFlow':
        """Add a stage to the composite flow.
        
        Args:
            stage: Flow instance to add as a stage
            
        Returns:
            Self for chaining
        """
        self.stages.append(stage)
        self._stage_instances[stage.name] = stage
        self._update_metadata()
        return self
    
    def add_stages(self, stages: Iterable[Flow]) -> 'CompositeFlow':
        """Add multiple stages to the composite flow.
        
        Args:
            stages: Iterable of Flow instances to add as stages
            
        Returns:
            Self for chaining
        """
        for stage in stages:
            self.stages.append(stage)
            self._stage_instances[stage.name] = stage
        self._update_metadata()
        return self
    
    def get_stage(self, stage_name: str, required: bool = True) -> Optional[Flow]:
        """Get a stage by name.
        
        Args:
            stage_name: Name of the stage to retrieve
            required: If True, raise an error when stage not found
            
        Returns:
            Stage instance or None if not found and required=False
            
        Raises:
            ValueError: If stage not found and required=True
        """
        # Check if the stage is already loaded
        if stage_name in self._stage_instances:
            return self._stage_instances[stage_name]
        
        # Try to load the stage from the registry if it's a standalone stage
        try:
            stage_info = stage_registry.get_stage(stage_name)
            if stage_info.is_standalone:
                stage = Stage(
                    name=stage_name,
                    process_fn=stage_info.standalone_func,
                    input_schema=stage_info.input_model,
                    output_schema=stage_info.output_model,
                    metadata=stage_info.metadata
                )
                self.stages.append(stage)
                self._stage_instances[stage_name] = stage
                self._update_metadata()
                return stage
        except KeyError:
            pass
        
        if required:
            raise ValueError(f"Stage '{stage_name}' not found in flow {self.name}")
        return None
    
    async def _execute(self, context: Context) -> FlowResult:
        """Execute all stages in sequence.
        
        Args:
            context: Execution context
            
        Returns:
            FlowResult containing execution outcome
            
        Raises:
            ExecutionError: If execution fails
        """
        if not self.stages:
            logger.warning(f"CompositeFlow '{self.name}' has no stages to execute")
            return FlowResult(
                data={},
                flow_name=self.name,
                status=FlowStatus.SUCCESS,
                metadata={"message": "No stages to execute"}
            )
        
        # Track individual stage results
        stage_results: Dict[str, Any] = {}
        final_result = {}
        
        try:
            # Create a new Context with the input data
            # This ensures we have a proper Context object with all needed methods
            input_data = context.data
            current_context = Context(data={"input": input_data})
            
            # Execute each stage in sequence
            for i, stage in enumerate(self.stages):
                logger.debug(f"Executing stage {i+1}/{len(self.stages)}: {stage.name}")
                
                # Execute the stage
                result = await stage.execute(current_context)
                
                # Store the stage result
                stage_results[stage.name] = result
                
                if result.status == FlowStatus.ERROR:
                    logger.error(f"Stage {stage.name} failed: {result.error}")
                    # Create error result with stage results metadata
                    return FlowResult(
                        data={},
                        flow_name=self.name,
                        status=FlowStatus.ERROR,
                        error=f"Stage '{stage.name}' failed: {result.error}",
                        error_details=result.error_details,
                        metadata={"stage_results": stage_results}
                    )
                
                # Store the stage result in the context
                # Use the stage name as the key
                current_context.set(stage.name, result.data)
                
                # If this is the last stage, use its result data as the final result
                if i == len(self.stages) - 1:
                    final_result = result.data
            
            # Create success result with stage results metadata
            return FlowResult(
                data=final_result,
                flow_name=self.name,
                status=FlowStatus.SUCCESS,
                metadata={"stage_results": stage_results}
            )
            
        except Exception as e:
            error_context = ErrorContext.create(
                flow_name=self.name,
                stage_results=stage_results
            )
            
            raise ExecutionError(
                message=f"Composite flow execution failed: {str(e)}",
                context=error_context,
                cause=e
            )
    
    def __str__(self) -> str:
        """String representation."""
        return f"CompositeFlow(name='{self.name}', stages={len(self.stages)})"

    def get_stages(self) -> List[str]:
        """Get all available stages for this flow.
        
        Returns:
            List of stage names
        """
        # Return stages already instantiated
        stages = list(self._stage_instances.keys())
        
        # Include any stages in the stages list that aren't in _stage_instances
        for stage in self.stages:
            if stage.name not in stages:
                stages.append(stage.name)
        
        # Add standalone stages from registry
        standalone_stages = stage_registry.get_stages()
        for stage_name in standalone_stages:
            if stage_name not in stages:
                stages.append(stage_name)
                
        return stages

    # Add a pipeline method that will be recognized by the Flow.execute method
    async def run_pipeline(self, context: Context) -> Any:
        """Execute the composite flow pipeline.
        
        This pipeline method coordinates the execution of all stages in sequence.
        It delegates to the _execute method for the actual implementation.
        
        Args:
            context: Execution context
            
        Returns:
            Flow execution result
        """
        return await self._execute(context) 