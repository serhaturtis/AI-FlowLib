"""Stage Model Generator stage implementation."""

from typing import Dict
from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, PipelineModels, StageSequence, StageModels,
    StageInterface, DataFlow, ModelDefinition, StageInterfaceDict, ModelDefinitionDict
)

logger = logging.getLogger(__name__)

@flow("stage_model_generator")
class StageModelGenerator:
    """Generates input/output models for each stage in the flow."""
    
    def __init__(self):
        """Initialize the generator."""
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
    
    async def __aenter__(self) -> 'StageModelGenerator':
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
    
    @stage(output_model=StageInterfaceDict)
    async def design_stage_interfaces(
        self,
        flow_description: FlowDescription,
        pipeline_models: PipelineModels,
        stage_sequence: StageSequence
    ) -> StageInterfaceDict:
        """Design the interface models for each stage."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("design_stage_interfaces.txt").format(
                overview=flow_description.overview,
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}\n  Requirements: {s.requirements}"
                    for s in stage_sequence.stages
                ),
                execution_order="\n".join(f"- {s}" for s in stage_sequence.execution_order),
                input_model=self._format_schema(pipeline_models.input_model.model_dump()),
                output_model=self._format_schema(pipeline_models.output_model.model_dump()),
                schema=self._format_schema(StageInterfaceDict.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=StageInterfaceDict,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @stage(output_model=DataFlow)
    async def analyze_data_flow(
        self,
        stage_sequence: StageSequence,
        interfaces: Dict[str, StageInterface]
    ) -> DataFlow:
        """Analyze the data flow between stages."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("analyze_data_flow.txt").format(
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}\n  Dependencies: {s.dependencies}"
                    for s in stage_sequence.stages
                ),
                interfaces="\n".join(
                    f"- {name}:\n  Input: {interface.input_model.name}\n  Output: {interface.output_model.name}"
                    for name, interface in interfaces.items()
                ),
                execution_order="\n".join(f"- {s}" for s in stage_sequence.execution_order),
                schema=DataFlow.model_json_schema()
            ),
            model_name=self.model_name,
            response_model=DataFlow,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ModelDefinitionDict)
    async def identify_common_types(
        self,
        interfaces: StageInterfaceDict,
        data_flow: DataFlow
    ) -> ModelDefinitionDict:
        """Identify common data types used across stages."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("identify_common_types.txt").format(
                interfaces="\n".join(
                    f"- {name}:\n  Models: {list(interface.internal_models.keys())}"
                    for name, interface in interfaces.items.items()
                ),
                data_flow="\n".join(
                    f"- {dep.source_stage} → {dep.target_stage}: {dep.data_fields}"
                    for dep in data_flow.dependencies
                ),
                shared_models="\n".join(
                    f"- {name}: {model.description}"
                    for name, model in data_flow.shared_models.items()
                ),
                schema=self._format_schema(ModelDefinitionDict.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ModelDefinitionDict,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @pipeline(
        input_model=tuple([FlowDescription, PipelineModels, StageSequence]),
        output_model=StageModels
    )
    async def generate_stage_models(
        self,
        flow_description: FlowDescription,
        pipeline_models: PipelineModels,
        stage_sequence: StageSequence
    ) -> StageModels:
        """Generate the complete set of models for all stages.
        
        Args:
            flow_description: Description of the flow
            pipeline_models: Pipeline input/output models
            stage_sequence: Sequence of stages
            
        Returns:
            Complete set of models for all stages
        """
        # Design interfaces for each stage
        interfaces = await self.design_stage_interfaces(
            flow_description,
            pipeline_models,
            stage_sequence
        )
        logger.info(f"Designed interfaces for {len(interfaces.items)} stages")
        
        # Analyze data flow between stages
        data_flow = await self.analyze_data_flow(stage_sequence, interfaces.items)
        logger.info(f"Analyzed data flow with {len(data_flow.dependencies)} dependencies")
        
        # Identify common types
        common_types = await self.identify_common_types(interfaces, data_flow)
        logger.info(f"Identified {len(common_types.items)} common types")
        
        # Create LLM interfaces for stages that need them
        llm_interfaces = {}
        for stage in stage_sequence.stages:
            if stage.uses_llm:
                stage_interface = interfaces.items[stage.name]
                llm_interfaces[f"{stage.name}_prompt"] = ModelDefinition(
                    name=f"{stage.name}PromptInput",
                    description=f"Input model for {stage.name} LLM prompt",
                    fields=[],  # Will be filled by prompt template generator
                    base_classes=["BaseModel"]
                )
                llm_interfaces[f"{stage.name}_response"] = ModelDefinition(
                    name=f"{stage.name}PromptResponse",
                    description=f"Response model for {stage.name} LLM prompt",
                    fields=[],  # Will be filled by prompt template generator
                    base_classes=["BaseModel"]
                )
        
        # Create validation models for validation points
        validation_models = {}
        for point in stage_sequence.validation_points:
            stage_interface = interfaces.items[point]
            validation_models[f"{point}_validation"] = ModelDefinition(
                name=f"{point}ValidationResult",
                description=f"Validation result model for {point}",
                fields=[],  # Will be filled by validation generator
                base_classes=["BaseModel"]
            )
        
        return StageModels(
            interfaces=interfaces.items,
            data_flow=data_flow,
            common_types=common_types.items,
            llm_interfaces=llm_interfaces,
            validation_models=validation_models
        ) 