"""Flow Generator flow implementation."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import os

from flowlib import flow, pipeline
from flowlib.core.resources import ResourceRegistry
from flowlib.providers.llm import LLMProvider
from pydantic import BaseModel, Field

from ..models.flowgen_models import (
    FlowDescription, GeneratedPipeline,
    StageSequence, StageModels, StageImplementations, SupportFiles,
    FlowGeneratorInput, GeneratedFile
)
from ..config.models import FlowGeneratorModel  # Import the model config

from .flow_description_generator import FlowDescriptionGenerator
from .pipeline_model_designer import PipelineModelDesigner
from .stage_planner import StagePlanner
from .stage_model_generator import StageModelGenerator
from .flow_validator import FlowValidator
from .stage_implementation_generator import StageImplementationGenerator
from .support_file_generator import SupportFileGenerator
from .pipeline_generator import PipelineGenerator
from .file_output_generator import FileOutputGenerator

logger = logging.getLogger(__name__)

class FlowGeneratorOutput(BaseModel):
    """Output from the flow generator."""
    status: str = Field(..., description="Status of flow generation (success/failed)")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    errors: Optional[List[str]] = Field(None, description="List of critical issues if validation failed")
    flow_description: Optional[FlowDescription] = Field(None, description="Generated flow description")
    pipeline_models: Optional[StageModels] = Field(None, description="Generated pipeline models")
    stage_sequence: Optional[StageSequence] = Field(None, description="Generated stage sequence")
    stage_models: Optional[StageModels] = Field(None, description="Generated stage models")
    validation_result: Optional[Any] = Field(None, description="Flow validation results")
    implementations: Optional[StageImplementations] = Field(None, description="Generated stage implementations")
    pipeline: Optional[GeneratedPipeline] = Field(None, description="Generated pipeline")
    support_files: Optional[SupportFiles] = Field(None, description="Generated support files")
    generated_files: Optional[Dict[str, GeneratedFile]] = Field(None, description="Generated output files")

@flow("flow_generator")
class FlowGenerator:
    """A flow that orchestrates the entire flow generation process."""

    def __init__(self):
        """Initialize the flow generator."""
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
        self.output_dir = self.workspace_root / "output"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    async def __aenter__(self) -> 'FlowGenerator':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass

    @pipeline(input_model=FlowGeneratorInput, output_model=FlowGeneratorOutput)
    async def generate_flow(self, input_data: FlowGeneratorInput) -> FlowGeneratorOutput:
        """Generate a complete flow based on input requirements.
        
        Args:
            input_data: Input requirements for flow generation
            
        Returns:
            FlowGeneratorOutput containing all generated artifacts
        """
        # Create and initialize LLM provider
        provider = LLMProvider(name="llm", max_models=2)
        async with provider:  # This will call initialize() and cleanup()
            # Register the provider and model
            ResourceRegistry.register_resource("provider", "llm", provider)
            ResourceRegistry.register_resource("model", "flow_generator", FlowGeneratorModel())
            logger.debug("Registered flow_generator model")
            
            try:
                logger.info("Starting flow generation...")
                
                # Stage 1: Generate Flow Description
                async with FlowDescriptionGenerator() as generator:
                    logger.info("Generating flow description...")
                    flow_description = await generator.generate_description(input_data)
                    logger.info("Generated flow description with %d components", len(flow_description.components))
                
                # Stage 2: Design Pipeline Models
                async with PipelineModelDesigner() as designer:
                    logger.info("Designing pipeline models...")
                    pipeline_models = await designer.design_models(flow_description)
                    logger.info("Generated pipeline models")
                
                # Stage 3: Plan Stages
                async with StagePlanner() as planner:
                    logger.info("Planning flow stages...")
                    stage_sequence = await planner.plan_stages(flow_description, pipeline_models)
                    logger.info("Planned %d stages", len(stage_sequence.stages))
                
                # Stage 4: Generate Stage Models
                async with StageModelGenerator() as model_generator:
                    logger.info("Generating stage models...")
                    stage_models = await model_generator.generate_stage_models(
                        flow_description,
                        pipeline_models,
                        stage_sequence
                    )
                    logger.info("Generated models for %d stages", len(stage_models.interfaces))
                
                # Stage 5: Validate Flow
                async with FlowValidator() as validator:
                    logger.info("Validating flow...")
                    validation_result = await validator.validate_flow(
                        flow_description,
                        pipeline_models,
                        stage_sequence,
                        stage_models
                    )
                    logger.info("Flow validation status: %s", validation_result.overall_status)
                    
                    if validation_result.overall_status == "invalid":
                        logger.error("Critical issues found:")
                        for issue in validation_result.critical_issues:
                            logger.error("- %s", issue)
                        return FlowGeneratorOutput(
                            status="failed",
                            errors=validation_result.critical_issues
                        )
                
                # Stage 6: Generate Stage Implementations
                async with StageImplementationGenerator() as implementation_generator:
                    logger.info("Generating stage implementations...")
                    implementations = await implementation_generator.generate_implementations(
                        flow_description,
                        stage_sequence,
                        stage_models
                    )
                    logger.info("Generated implementations for %d stages", len(implementations.stages))
                
                # Stage 7: Generate Pipeline
                async with PipelineGenerator() as pipeline_generator:
                    logger.info("Generating pipeline...")
                    pipeline = await pipeline_generator.generate_pipeline(
                        flow_description,
                        stage_sequence,
                        stage_models,
                        implementations
                    )
                    logger.info("Generated pipeline")
                
                # Stage 8: Generate Support Files
                async with SupportFileGenerator() as support_generator:
                    logger.info("Generating support files...")
                    support_files = await support_generator.generate_support_files(
                        flow_description,
                        pipeline
                    )
                    logger.info("Generated support files")
                
                # Stage 9: Write Output Files
                async with FileOutputGenerator() as file_generator:
                    logger.info("Writing output files...")
                    generated_files = await file_generator.write_output_files(
                        support_files,
                        implementations
                    )
                    logger.info("Written %d files", len(generated_files))
                
                return FlowGeneratorOutput(
                    status="success",
                    flow_description=flow_description,
                    pipeline_models=pipeline_models,
                    stage_sequence=stage_sequence,
                    stage_models=stage_models,
                    validation_result=validation_result,
                    implementations=implementations,
                    pipeline=pipeline,
                    support_files=support_files,
                    generated_files=generated_files
                )
                
            except Exception as e:
                logger.error(f"Flow generation failed: {str(e)}")
                return FlowGeneratorOutput(
                    status="failed",
                    error=str(e)
                ) 