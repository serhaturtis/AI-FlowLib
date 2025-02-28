"""Flow Validator stage implementation."""

from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, PipelineModels, StageSequence, StageModels,
    FlowValidation, DataFlowValidation,
    SecurityValidation, PerformanceValidation, ValidationCheckList, ValidationDiagramList
)

logger = logging.getLogger(__name__)

@flow("flow_validator")
class FlowValidator:
    """Validates the complete flow design and data connections."""
    
    def __init__(self):
        """Initialize the validator."""
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
    
    async def __aenter__(self) -> 'FlowValidator':
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
    
    @stage(output_model=ValidationCheckList)
    async def validate_connections(
        self,
        flow_description: FlowDescription,
        pipeline_models: PipelineModels,
        stage_sequence: StageSequence,
        stage_models: StageModels
    ) -> ValidationCheckList:
        """Validate all connections between stages."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("validate_connections.txt").format(
                overview=flow_description.overview,
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}\n  Requirements: {s.requirements}"
                    for s in stage_sequence.stages
                ),
                data_flow="\n".join(
                    f"- {dep.source_stage} → {dep.target_stage}: {dep.data_fields}"
                    for dep in stage_models.data_flow.dependencies
                ),
                interfaces="\n".join(
                    f"- {name}:\n  Input: {interface.input_model.name}\n  Output: {interface.output_model.name}"
                    for name, interface in stage_models.interfaces.items()
                ),
                schema=self._format_schema(ValidationCheckList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ValidationCheckList,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=SecurityValidation)
    async def validate_security(
        self,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels
    ) -> SecurityValidation:
        """Validate security aspects of the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("validate_security.txt").format(
                overview=flow_description.overview,
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}\n  Uses LLM: {s.uses_llm}"
                    for s in stage_sequence.stages
                ),
                data_flow="\n".join(
                    f"- {dep.source_stage} → {dep.target_stage}: {dep.data_fields}"
                    for dep in stage_models.data_flow.dependencies
                ),
                schema=SecurityValidation.model_json_schema()
            ),
            model_name=self.model_name,
            response_model=SecurityValidation,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=PerformanceValidation)
    async def validate_performance(
        self,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels
    ) -> PerformanceValidation:
        """Validate performance aspects of the flow."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("validate_performance.txt").format(
                overview=flow_description.overview,
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}\n  Uses LLM: {s.uses_llm}"
                    for s in stage_sequence.stages
                ),
                parallel_groups="\n".join(
                    f"- Group {i}: {', '.join(group)}"
                    for i, group in enumerate(stage_sequence.parallel_groups)
                ),
                data_flow="\n".join(
                    f"- {dep.source_stage} → {dep.target_stage}: {dep.data_fields}"
                    for dep in stage_models.data_flow.dependencies
                ),
                schema=PerformanceValidation.model_json_schema()
            ),
            model_name=self.model_name,
            response_model=PerformanceValidation,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ValidationDiagramList)
    async def generate_diagrams(
        self,
        flow_description: FlowDescription,
        stage_sequence: StageSequence,
        stage_models: StageModels,
        validation_checks: ValidationCheckList
    ) -> ValidationDiagramList:
        """Generate validation diagrams."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_validation_diagrams.txt").format(
                overview=flow_description.overview,
                stages="\n".join(
                    f"- {s.name}:\n  Purpose: {s.purpose}"
                    for s in stage_sequence.stages
                ),
                data_flow="\n".join(
                    f"- {dep.source_stage} → {dep.target_stage}: {dep.data_fields}"
                    for dep in stage_models.data_flow.dependencies
                ),
                validation_checks="\n".join(
                    f"- {check.component}: {check.status} - {check.details}"
                    for check in validation_checks.items
                ),
                schema=self._format_schema(ValidationDiagramList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ValidationDiagramList,
            max_tokens=2048,
            temperature=0.7
        )
        return result
    
    @pipeline(
        input_model=tuple([FlowDescription, PipelineModels, StageSequence, StageModels]),
        output_model=FlowValidation
    )
    async def validate_flow(
        self,
        flow_description: FlowDescription,
        pipeline_models: PipelineModels,
        stage_sequence: StageSequence,
        stage_models: StageModels
    ) -> FlowValidation:
        """Validate the complete flow design.
        
        Args:
            flow_description: Description of the flow
            pipeline_models: Pipeline input/output models
            stage_sequence: Sequence of stages
            stage_models: Stage interface models
            
        Returns:
            Complete flow validation results
        """
        # Validate stage connections
        validation_checks = await self.validate_connections(
            flow_description,
            pipeline_models,
            stage_sequence,
            stage_models
        )
        logger.info(f"Completed {len(validation_checks.items)} validation checks")
        
        # Analyze data flow validation
        data_flow_validation = DataFlowValidation(
            missing_data=[
                check.details for check in validation_checks.items
                if check.check_type == "data_flow" and check.status == "fail"
            ],
            type_mismatches=[
                check.details for check in validation_checks.items
                if check.check_type == "model" and check.status == "fail"
            ],
            unused_outputs=[
                check.details for check in validation_checks.items
                if check.check_type == "requirement" and check.status == "warning"
            ],
            circular_dependencies=[],  # Already checked in stage planner
            validation_gaps=[
                check.details for check in validation_checks.items
                if check.check_type == "validation" and check.status == "warning"
            ]
        )
        
        # Validate security aspects
        security_validation = await self.validate_security(
            flow_description,
            stage_sequence,
            stage_models
        )
        logger.info("Completed security validation")
        
        # Validate performance aspects
        performance_validation = await self.validate_performance(
            flow_description,
            stage_sequence,
            stage_models
        )
        logger.info("Completed performance validation")
        
        # Generate validation diagrams
        diagrams = await self.generate_diagrams(
            flow_description,
            stage_sequence,
            stage_models,
            validation_checks
        )
        logger.info(f"Generated {len(diagrams.items)} validation diagrams")
        
        # Determine overall status
        critical_issues = [
            check.details for check in validation_checks.items
            if check.status == "fail"
        ]
        
        if critical_issues:
            overall_status = "invalid"
        elif any(check.status == "warning" for check in validation_checks.items):
            overall_status = "valid_with_warnings"
        else:
            overall_status = "valid"
        
        # Collect recommendations
        recommendations = []
        for check in validation_checks.items:
            recommendations.extend(check.recommendations)
        recommendations.extend(performance_validation.optimization_suggestions)
        
        return FlowValidation(
            checks=validation_checks.items,
            data_flow=data_flow_validation,
            security=security_validation,
            performance=performance_validation,
            diagrams=diagrams.items,
            overall_status=overall_status,
            critical_issues=critical_issues,
            recommendations=list(set(recommendations))  # Remove duplicates
        ) 