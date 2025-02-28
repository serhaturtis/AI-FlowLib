"""Pipeline Model Designer stage implementation."""

from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, PipelineModels,
    ModelDefinition, ModelDefinitionDict
)

logger = logging.getLogger(__name__)

@flow("pipeline_model_designer")
class PipelineModelDesigner:
    """Designs input and output models for a flow based on its description."""
    
    def __init__(self):
        """Initialize the designer."""
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
    
    async def __aenter__(self) -> 'PipelineModelDesigner':
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
    
    @stage(output_model=ModelDefinition)
    async def design_input_model(self, flow_description: FlowDescription) -> ModelDefinition:
        """Design the input model for the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("design_input_model.txt").format(
                overview=flow_description.overview,
                components="\n".join(f"- {c.name}: {c.purpose}" for c in flow_description.components),
                transformations="\n".join(
                    f"- {t.input_data} → {t.output_data}" for t in flow_description.data_transformations
                ),
                constraints="\n".join(f"- {c}" for c in flow_description.constraints),
                schema=self._format_schema(ModelDefinition.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ModelDefinition,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ModelDefinition)
    async def design_output_model(
        self,
        flow_description: FlowDescription,
        input_model: ModelDefinition
    ) -> ModelDefinition:
        """Design the output model for the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("design_output_model.txt").format(
                overview=flow_description.overview,
                components="\n".join(f"- {c.name}: {c.purpose}" for c in flow_description.components),
                transformations="\n".join(
                    f"- {t.input_data} → {t.output_data}" for t in flow_description.data_transformations
                ),
                input_model=self._format_schema(input_model.model_dump()),
                constraints="\n".join(f"- {c}" for c in flow_description.constraints),
                schema=self._format_schema(ModelDefinition.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ModelDefinition,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ModelDefinitionDict)
    async def identify_shared_models(
        self,
        flow_description: FlowDescription,
        input_model: ModelDefinition,
        output_model: ModelDefinition
    ) -> ModelDefinitionDict:
        """Identify models that are shared between input and output."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("identify_shared_models.txt").format(
                overview=flow_description.overview,
                components="\n".join(f"- {c.name}: {c.purpose}" for c in flow_description.components),
                transformations="\n".join(
                    f"- {t.input_data} → {t.output_data}" for t in flow_description.data_transformations
                ),
                input_model=self._format_schema(input_model.model_dump()),
                output_model=self._format_schema(output_model.model_dump()),
                schema=self._format_schema(ModelDefinitionDict.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ModelDefinitionDict,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @pipeline(input_model=FlowDescription, output_model=PipelineModels)
    async def design_models(self, flow_description: FlowDescription) -> PipelineModels:
        """Design the complete set of pipeline models.
        
        Args:
            flow_description: Description of the flow to design models for
            
        Returns:
            Complete set of pipeline models
        """
        # Design input model
        input_model = await self.design_input_model(flow_description)
        logger.info(f"Designed input model: {input_model.name}")
        
        # Design output model
        output_model = await self.design_output_model(flow_description, input_model)
        logger.info(f"Designed output model: {output_model.name}")
        
        # Identify shared models
        shared_models = await self.identify_shared_models(flow_description, input_model, output_model)
        logger.info(f"Identified {len(shared_models.items)} shared models")
        
        # Collect validation notes
        validation_notes = []
        
        # Check field coverage
        input_fields = {f.name for f in input_model.fields}
        output_fields = {f.name for f in output_model.fields}
        
        # Check for unused input fields
        unused_inputs = input_fields - output_fields
        if unused_inputs:
            validation_notes.append(
                f"Input fields not reflected in output: {', '.join(unused_inputs)}"
            )
        
        # Check for undefined output fields
        undefined_outputs = output_fields - input_fields
        if undefined_outputs:
            validation_notes.append(
                f"Output fields without input source: {', '.join(undefined_outputs)}"
            )
        
        # Return complete model definitions
        return PipelineModels(
            input_model=input_model,
            output_model=output_model,
            shared_models=shared_models.items,
            validation_notes=validation_notes
        ) 