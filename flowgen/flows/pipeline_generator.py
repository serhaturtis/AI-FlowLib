"""Pipeline Generator stage implementation."""

from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, StageSequence, StageModels, StageImplementations,
    FlowImplementation, GeneratedPipeline, FlowTestSuite
)

logger = logging.getLogger(__name__)

@flow("pipeline_generator")
class PipelineGenerator:
    """Generates the main flow implementation that orchestrates all stages."""
    
    def __init__(self):
        """Initialize the generator."""
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
    
    async def __aenter__(self) -> 'PipelineGenerator':
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
    
    @stage(output_model=FlowImplementation)
    async def generate_flow_implementation(
        self,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels,
        stage_implementations: StageImplementations
    ) -> FlowImplementation:
        """Generate the main flow implementation."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_flow_implementation.txt").format(
                overview=flow_description.overview,
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}\n  Description: {s.description}"
                    for s in stage_sequence.stages
                ),
                execution_order="\n".join(f"- {s}" for s in stage_sequence.execution_order),
                parallel_groups="\n".join(
                    f"- Group {i}: {', '.join(group)}"
                    for i, group in enumerate(stage_sequence.parallel_groups, 1)
                ),
                validation_points="\n".join(f"- {p}" for p in stage_sequence.validation_points),
                error_recovery="\n".join(
                    f"- {stage}: {' -> '.join(recovery)}"
                    for stage, recovery in stage_sequence.error_recovery.items()
                ),
                schema=self._format_schema(FlowImplementation.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=FlowImplementation,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @stage(output_model=FlowTestSuite)
    async def generate_test_suite(
        self,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels,
        flow_implementation: FlowImplementation
    ) -> FlowTestSuite:
        """Generate the test suite for the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_flow_tests.txt").format(
                overview=flow_description.overview,
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}\n  Description: {s.description}"
                    for s in stage_sequence.stages
                ),
                execution_order="\n".join(f"- {s}" for s in stage_sequence.execution_order),
                pipeline_method=flow_implementation.pipeline_method.code,
                schema=self._format_schema(FlowTestSuite.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=FlowTestSuite,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @pipeline(
        input_model=tuple([FlowDescription, StageSequence, StageModels, StageImplementations]),
        output_model=GeneratedPipeline
    )
    async def generate_pipeline(
        self,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels,
        stage_implementations: StageImplementations
    ) -> GeneratedPipeline:
        """Generate the complete pipeline implementation.
        
        Args:
            flow_description: Description of the flow
            stage_sequence: Sequence of stages
            stage_models: Stage interface models
            stage_implementations: Stage implementations
            
        Returns:
            Complete generated pipeline package
        """
        # Generate main flow implementation
        implementation = await self.generate_flow_implementation(
            flow_description,
            stage_sequence,
            stage_models,
            stage_implementations
        )
        logger.info("Generated main flow implementation")
        
        # Generate test suite
        tests = await self.generate_test_suite(
            flow_description,
            stage_sequence,
            stage_models,
            implementation
        )
        logger.info("Generated test suite")
        
        # Create configuration
        configuration = {
            "flow_name": implementation.flow_name,
            "resources": {
                name: resource.configuration
                for name, resource in implementation.resources.items()
            },
            "monitoring": {
                "metrics": implementation.monitoring.metrics,
                "logging_points": implementation.monitoring.logging_points,
                "alert_conditions": implementation.monitoring.alert_conditions
            }
        }
        
        # Generate documentation
        documentation = f"# {implementation.flow_name}\n\n"
        documentation += f"{flow_description.overview}\n\n"
        documentation += "## Components\n\n"
        for component in flow_description.components:
            documentation += f"### {component.name}\n"
            documentation += f"{component.purpose}\n\n"
            documentation += "Responsibilities:\n"
            for resp in component.responsibilities:
                documentation += f"- {resp}\n"
            documentation += "\n"
        
        # Create deployment configuration
        deployment = {
            "requirements": "\n".join(stage_implementations.common_requirements),
            "environment_setup": str({
                "python_version": ">=3.8",
                "dependencies": stage_implementations.common_requirements
            }),
            "resource_setup": str({
                name: resource.initialization_code
                for name, resource in implementation.resources.items()
            })
        }
        
        # Create monitoring configuration
        monitoring_config = {
            "metrics": implementation.monitoring.metrics,
            "logging": {
                "points": implementation.monitoring.logging_points,
                "levels": {
                    "default": "INFO",
                    "performance": "DEBUG",
                    "validation": "INFO",
                    "error": "ERROR"
                }
            },
            "alerts": implementation.monitoring.alert_conditions
        }
        
        # Create usage examples
        examples = [
            {
                "name": "Basic Usage",
                "description": "Simple example of using the flow",
                "code": f"""
from {implementation.flow_name.lower()} import {implementation.flow_name}

async def main():
    flow = {implementation.flow_name}()
    async with flow:
        input_data = {{}}  # Add your input data here
        output = await flow.{implementation.pipeline_method.name}(input_data)
        print(f"Output: {{output}}")
"""
            }
        ]
        
        return GeneratedPipeline(
            implementation=implementation,
            tests=tests,
            configuration=configuration,
            documentation=documentation,
            deployment=deployment,
            monitoring_config=monitoring_config,
            examples=examples
        )