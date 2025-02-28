"""Stage Implementation Generator stage implementation."""

from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, StageSequence, StageModels,
    StageImplementation, GeneratedStage, StageImplementations,
    PromptTemplateList, StageTestCaseList
)

logger = logging.getLogger(__name__)

@flow("stage_implementation_generator")
class StageImplementationGenerator:
    """Generates implementation code for each stage in the flow."""
    
    def __init__(self):
        """Initialize the generator."""
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
    
    async def __aenter__(self) -> 'StageImplementationGenerator':
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
    
    @stage(output_model=StageImplementation)
    async def generate_stage_code(
        self,
        stage_name: str,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels
    ) -> StageImplementation:
        """Generate implementation code for a single stage."""
        stage_def = next(s for s in stage_sequence.stages if s.name == stage_name)
        stage_interface = stage_models.interfaces[stage_name]
        
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_stage_code.txt").format(
                stage_name=stage_name,
                stage_purpose=stage_def.purpose,
                stage_description=stage_def.description,
                requirements="\n".join(f"- {r}" for r in stage_def.requirements),
                uses_llm=stage_def.uses_llm,
                input_model=self._format_schema(stage_interface.input_model.model_dump()),
                output_model=self._format_schema(stage_interface.output_model.model_dump()),
                internal_models="\n".join(
                    f"- {name}: {model.description}"
                    for name, model in stage_interface.internal_models.items()
                ),
                error_models="\n".join(
                    f"- {name}: {model.description}"
                    for name, model in stage_interface.error_models.items()
                ),
                schema=StageImplementation.model_json_schema()
            ),
            model_name=self.model_name,
            response_model=StageImplementation,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @stage(output_model=PromptTemplateList)
    async def generate_prompt_templates(
        self,
        stage_name: str,
        stage_implementation: StageImplementation,
        stage_models: StageModels
    ) -> PromptTemplateList:
        """Generate prompt templates for an LLM-using stage."""
        stage_interface = stage_models.interfaces[stage_name]
        llm_interfaces = {
            k: v for k, v in stage_models.llm_interfaces.items()
            if k.startswith(f"{stage_name}_")
        }
        
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_prompt_templates.txt").format(
                stage_name=stage_name,
                stage_methods="\n".join(
                    f"- {m.name}:\n  Purpose: {m.description}\n  Returns: {m.return_type}"
                    for m in stage_implementation.methods
                ),
                llm_interfaces="\n".join(
                    f"- {name}:\n  {model.description}"
                    for name, model in llm_interfaces.items()
                ),
                input_model=self._format_schema(stage_interface.input_model.model_dump()),
                output_model=self._format_schema(stage_interface.output_model.model_dump()),
                schema=self._format_schema(PromptTemplateList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=PromptTemplateList,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=StageTestCaseList)
    async def generate_stage_tests(
        self,
        stage_name: str,
        stage_implementation: StageImplementation,
        stage_models: StageModels
    ) -> StageTestCaseList:
        """Generate test cases for a stage."""
        stage_interface = stage_models.interfaces[stage_name]
        
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_test_cases.txt").format(
                stage_name=stage_name,
                stage_methods="\n".join(
                    f"- {m.name}:\n  Purpose: {m.description}\n  Returns: {m.return_type}"
                    for m in stage_implementation.methods
                ),
                input_model=self._format_schema(stage_interface.input_model.model_dump()),
                output_model=self._format_schema(stage_interface.output_model.model_dump()),
                error_handlers="\n".join(
                    f"- {h.error_types}: {h.handling_code}"
                    for h in stage_implementation.error_handlers
                ),
                schema=self._format_schema(StageTestCaseList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=StageTestCaseList,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @pipeline(
        input_model=tuple([FlowDescription, StageSequence, StageModels]),
        output_model=StageImplementations
    )
    async def generate_implementations(
        self,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels
    ) -> StageImplementations:
        """Generate implementations for all stages.
        
        Args:
            flow_description: Description of the flow
            stage_sequence: Sequence of stages
            stage_models: Stage interface models
            
        Returns:
            Complete set of stage implementations
        """
        stages = {}
        common_code = []
        common_requirements = []
        integration_tests = []
        
        # Generate implementations for each stage
        for stage in stage_sequence.stages:
            # Generate stage implementation
            implementation = await self.generate_stage_code(
                stage.name,
                flow_description,
                stage_sequence,
                stage_models
            )
            logger.info(f"Generated implementation for stage {stage.name}")
            
            # Generate prompt templates if stage uses LLM
            prompts = []
            if stage.uses_llm:
                prompt_list = await self.generate_prompt_templates(
                    stage.name,
                    implementation,
                    stage_models
                )
                logger.info(f"Generated {len(prompt_list.items)} prompt templates for stage {stage.name}")
                implementation.prompts = {p.name: p for p in prompt_list.items}
            
            # Generate test cases
            tests = await self.generate_stage_tests(
                stage.name,
                implementation,
                stage_models
            )
            logger.info(f"Generated {len(tests.items)} test cases for stage {stage.name}")
            
            # Create documentation
            documentation = f"""# {stage.name}

## Purpose
{stage.purpose}

## Description
{stage.description}

## Requirements
{chr(10).join(f'- {r}' for r in stage.requirements)}

## Methods
{chr(10).join(f'### {m.name}{chr(10)}{m.description}{chr(10)}Parameters: {m.parameters}{chr(10)}Returns: {m.return_type}' for m in implementation.methods)}

## Error Handling
{chr(10).join(f'- {h.error_types}:\n  Handling: {h.handling_code}\n  Recovery: {h.recovery_strategy}' for h in implementation.error_handlers)}
"""
            
            # Add performance notes
            performance_notes = [
                f"Stage uses LLM: {stage.uses_llm}",
                f"Number of methods: {len(implementation.methods)}",
                f"Number of error handlers: {len(implementation.error_handlers)}"
            ]
            
            # Create examples
            examples = []
            for test in tests.items:
                if test.is_example:
                    try:
                        examples.append({
                            "name": test.name,
                            "description": test.description,
                            "input": test.input_data,
                            "expected_output": test.expected_output
                        })
                    except AttributeError as e:
                        logger.warning(f"Skipping malformed test case {test.name}: {str(e)}")
            
            # Add to stages dictionary
            stages[stage.name] = GeneratedStage(
                implementation=implementation,
                tests=tests.items,
                requirements=[],  # Will be filled by dependency analysis
                documentation=documentation,
                examples=examples,
                performance_notes=performance_notes
            )
        
        # Analyze common code and requirements
        # This could be enhanced with more sophisticated analysis
        common_code = []  # Placeholder for common code analysis
        common_requirements = []  # Placeholder for common requirements analysis
        
        # Generate integration tests
        # This could be enhanced with more sophisticated test generation
        integration_tests = []  # Placeholder for integration test generation
        
        return StageImplementations(
            stages=stages,
            common_code=common_code,
            common_requirements=common_requirements,
            integration_tests=integration_tests
        ) 