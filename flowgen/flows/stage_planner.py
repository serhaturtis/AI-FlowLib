"""Stage Planner stage implementation."""

from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, PipelineModels, StageSequence,
    StageDefinitionList, StageOrderList, ParallelGroupList
)

logger = logging.getLogger(__name__)

@flow("stage_planner")
class StagePlanner:
    """Plans the stages of a flow and their execution sequence."""
    
    def __init__(self):
        """Initialize the planner."""
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
    
    async def __aenter__(self) -> 'StagePlanner':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        prompt_path = self.prompts_dir / filename
        with open(prompt_path) as f:
            return f.read()
    
    def _format_schema(self, schema: dict) -> str:
        """Format schema for LLM prompt."""
        properties = schema.get("properties", {})
        formatted_props = {}
        
        for prop, details in properties.items():
            formatted_props[prop] = {
                "type": details.get("type"),
                "description": details.get("description")
            }
        
        return str({
            "type": "object",
            "properties": formatted_props,
            "required": schema.get("required", [])
        })
    
    @stage(output_model=StageDefinitionList)
    async def identify_stages(
        self,
        flow_description: FlowDescription,
        pipeline_models: PipelineModels
    ) -> StageDefinitionList:
        """Identify the stages needed for the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("identify_stages.txt").format(
                overview=flow_description.overview,
                components="\n".join(f"- {c.name}: {c.purpose}" for c in flow_description.components),
                transformations="\n".join(
                    f"- {t.input_data} → {t.output_data}" for t in flow_description.data_transformations
                ),
                input_model=self._format_schema(pipeline_models.input_model.model_dump()),
                output_model=self._format_schema(pipeline_models.output_model.model_dump()),
                constraints="\n".join(f"- {c}" for c in flow_description.constraints),
                schema=self._format_schema(StageDefinitionList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=StageDefinitionList,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=StageOrderList)
    async def determine_execution_order(
        self,
        stages: StageDefinitionList
    ) -> StageOrderList:
        """Determine the optimal execution order for stages."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("determine_execution_order.txt").format(
                stages="\n".join(
                    f"- {s.name}: {s.purpose}\n  Dependencies: {[d.required_stage for d in s.dependencies]}"
                    for s in stages.items
                ),
                schema=self._format_schema(StageOrderList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=StageOrderList,
            max_tokens=1024,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ParallelGroupList)
    async def identify_parallel_groups(
        self,
        stages: StageDefinitionList,
        execution_order: StageOrderList
    ) -> ParallelGroupList:
        """Identify groups of stages that can be executed in parallel."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("identify_parallel_groups.txt").format(
                stages="\n".join(
                    f"- {s.name}: {s.purpose}\n  Dependencies: {[d.required_stage for d in s.dependencies]}"
                    for s in stages.items
                ),
                execution_order="\n".join(f"- {s}" for s in execution_order.items),
                schema=self._format_schema(ParallelGroupList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ParallelGroupList,
            max_tokens=1024,
            temperature=0.7
        )
        return result
    
    @pipeline(
        input_model=tuple([FlowDescription, PipelineModels]),
        output_model=StageSequence
    )
    async def plan_stages(
        self,
        flow_description: FlowDescription,
        pipeline_models: PipelineModels
    ) -> StageSequence:
        """Plan the complete sequence of stages.
        
        Args:
            flow_description: Description of the flow
            pipeline_models: Input/output model definitions
            
        Returns:
            Complete stage sequence with execution order
        """
        # Identify required stages
        stages = await self.identify_stages(flow_description, pipeline_models)
        logger.info(f"Identified {len(stages.items)} stages")
        
        # Determine execution order
        execution_order = await self.determine_execution_order(stages)
        logger.info(f"Determined execution order: {' -> '.join(execution_order.items)}")
        
        # Identify parallel execution opportunities
        parallel_groups = await self.identify_parallel_groups(stages, execution_order)
        logger.info(f"Identified {len(parallel_groups.items)} parallel groups")
        
        # Identify validation points (after parallel groups and critical stages)
        validation_points = []
        stage_map = {s.name: s for s in stages.items}
        
        # Add validation after parallel groups
        for group in parallel_groups.items:
            # Find the last stage in execution order that's in this group
            last_stage = max(group, key=lambda x: execution_order.items.index(x))
            validation_points.append(last_stage)
        
        # Add validation after stages with critical outputs
        for stage in stages.items:
            if any(r.priority == "must_have" for r in stage.requirements):
                validation_points.append(stage.name)
        
        # Remove duplicates while preserving order
        validation_points = list(dict.fromkeys(validation_points))
        
        # Create error recovery mapping
        error_recovery = {}
        for stage in stages.items:
            # For each stage, map what to do if it fails
            recovery_stages = []
            
            # If stage has retryable requirements, add self to recovery
            if any(r.type == "processing" for r in stage.requirements):
                recovery_stages.append(stage.name)
            
            # If stage has dependencies, add them to recovery
            for dep in stage.dependencies:
                if dep.dependency_type == "strict":
                    recovery_stages.append(dep.required_stage)
            
            if recovery_stages:
                error_recovery[stage.name] = recovery_stages
        
        return StageSequence(
            stages=stages.items,
            execution_order=execution_order.items,
            parallel_groups=parallel_groups.items,
            validation_points=validation_points,
            error_recovery=error_recovery
        ) 