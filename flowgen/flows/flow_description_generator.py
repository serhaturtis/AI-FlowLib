"""Flow Description Generator stage implementation."""

from typing import List
from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, ComponentDescription, ComponentDescriptionList,
    DataTransformation, DataTransformationList,
    ArchitecturalDecisionList, FlowGeneratorInput
)

logger = logging.getLogger(__name__)

@flow("flow_description_generator")
class FlowDescriptionGenerator:
    """Generates a detailed, structured description of a flow based on input requirements."""
    
    def __init__(self):
        """Initialize the generator."""
        logger.debug("Initializing FlowDescriptionGenerator")
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        logger.debug(f"Got LLM provider: {self.provider}")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        logger.debug(f"Using model name: {self.model_name}")
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
        logger.debug(f"Initialized with prompts_dir: {self.prompts_dir}")
    
    async def __aenter__(self) -> 'FlowDescriptionGenerator':
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
    
    @stage(output_model=ComponentDescriptionList)
    async def identify_components(self, input_data: FlowGeneratorInput) -> ComponentDescriptionList:
        """Identify the main components of the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("identify_components.txt").format(
                task=input_data.task_description,
                input_reqs=input_data.input_requirements,
                output_reqs=input_data.output_requirements,
                schema=self._format_schema(ComponentDescriptionList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ComponentDescriptionList,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=DataTransformationList)
    async def analyze_transformations(
        self,
        input_data: FlowGeneratorInput,
        components: List[ComponentDescription]
    ) -> DataTransformationList:
        """Analyze the data transformations in the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("analyze_transformations.txt").format(
                task=input_data.task_description,
                components="\n".join(f"- {c.name}: {c.purpose}" for c in components),
                input_reqs=input_data.input_requirements,
                output_reqs=input_data.output_requirements,
                schema=self._format_schema(DataTransformationList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=DataTransformationList,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ArchitecturalDecisionList)
    async def make_architectural_decisions(
        self,
        input_data: FlowGeneratorInput,
        components: List[ComponentDescription],
        transformations: List[DataTransformation]
    ) -> ArchitecturalDecisionList:
        """Make key architectural decisions for the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("architectural_decisions.txt").format(
                task=input_data.task_description,
                components="\n".join(f"- {c.name}: {c.purpose}" for c in components),
                transformations="\n".join(f"- {t.input_data} → {t.output_data}" for t in transformations),
                performance_reqs=input_data.performance_requirements,
                resource_reqs=input_data.resource_requirements,
                integration_reqs=input_data.integration_requirements,
                schema=self._format_schema(ArchitecturalDecisionList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ArchitecturalDecisionList,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @pipeline(input_model=FlowGeneratorInput, output_model=FlowDescription)
    async def generate_description(self, input_data: FlowGeneratorInput) -> FlowDescription:
        """Generate a complete flow description.
        
        Args:
            input_data: Input requirements for flow generation
            
        Returns:
            Complete flow description
        """
        # Identify main components
        components = await self.identify_components(input_data)
        logger.info(f"Identified {len(components.items)} components")
        
        # Analyze data transformations
        transformations = await self.analyze_transformations(input_data, components.items)
        logger.info(f"Identified {len(transformations.items)} data transformations")
        
        # Make architectural decisions
        decisions = await self.make_architectural_decisions(input_data, components.items, transformations.items)
        logger.info(f"Made {len(decisions.items)} architectural decisions")
        
        # Create overview by summarizing components and decisions
        overview = f"A flow that {input_data.task_description}. "
        overview += f"It consists of {len(components.items)} main components "
        overview += f"performing {len(transformations.items)} data transformations. "
        overview += "Key architectural decisions include: "
        overview += ", ".join(d.decision for d in decisions.items[:3]) + "."
        
        # Collect constraints from requirements
        constraints = []
        if input_data.performance_requirements:
            constraints.extend(input_data.performance_requirements)
        if input_data.resource_requirements:
            constraints.extend(f"Requires {r.type}: {r.description}" 
                            for r in input_data.resource_requirements)
        if input_data.integration_requirements:
            constraints.extend(input_data.integration_requirements)
        
        # Return complete description
        return FlowDescription(
            overview=overview,
            components=components.items,
            data_transformations=transformations.items,
            architectural_decisions=decisions.items,
            constraints=constraints
        ) 