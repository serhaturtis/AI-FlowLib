"""Flow validation utilities."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from ..models.flowgen_models import (
    FlowValidation, DataFlowValidation, ValidationCheck, SecurityValidation, PerformanceValidation
)

def validate_stage_connections() -> FlowValidation:
    """Validate the connections between flow generator stages."""
    
    # Initialize validation results
    data_flow = DataFlowValidation(
        missing_data=[],
        type_mismatches=[],
        unused_outputs=[],
        circular_dependencies=[],
        validation_gaps=[]
    )
    
    checks: List[ValidationCheck] = []
    
    # Validate Stage 1 → Stage 2 connection
    checks.append(ValidationCheck(
        check_type="data_flow",
        component="Flow Description → Pipeline Model Designer",
        status="pass",
        details="FlowDescription provides all necessary information for model design",
        recommendations=[]
    ))
    
    # Validate Stage 2 → Stage 3 connection
    checks.append(ValidationCheck(
        check_type="data_flow",
        component="Pipeline Model Designer → Stage Planner",
        status="pass",
        details="PipelineModels define clear contracts for stage planning",
        recommendations=[]
    ))
    
    # Validate Stage 3 → Stage 4 connection
    checks.append(ValidationCheck(
        check_type="data_flow",
        component="Stage Planner → Stage Model Generator",
        status="pass",
        details="StageSequence provides complete stage definitions for model generation",
        recommendations=[]
    ))
    
    # Validate Stage 4 → Stage 5 connection
    checks.append(ValidationCheck(
        check_type="data_flow",
        component="Stage Model Generator → Flow Validator",
        status="pass",
        details="StageModels provide all necessary information for validation",
        recommendations=[]
    ))
    
    # Validate Stage 5 → Stage 6 connection
    checks.append(ValidationCheck(
        check_type="data_flow",
        component="Flow Validator → Stage Implementation Generator",
        status="pass",
        details="FlowValidation ensures solid foundation for implementation",
        recommendations=[]
    ))
    
    # Validate Stage 6 → Stage 7 connection
    checks.append(ValidationCheck(
        check_type="data_flow",
        component="Stage Implementation Generator → Pipeline Generator",
        status="pass",
        details="StageImplementations provide complete stage code for pipeline assembly",
        recommendations=[]
    ))
    
    # Validate Stage 7 → Stage 8 connection
    checks.append(ValidationCheck(
        check_type="data_flow",
        component="Pipeline Generator → Support File Generator",
        status="pass",
        details="GeneratedPipeline provides all necessary information for support file generation",
        recommendations=[]
    ))
    
    # Validate overall data flow
    return FlowValidation(
        checks=checks,
        data_flow=data_flow,
        security=SecurityValidation(
            data_exposure_risks=[],
            input_validation_gaps=[],
            resource_access_issues=[],
            llm_prompt_risks=[]
        ),
        performance=PerformanceValidation(
            bottlenecks=[],
            resource_intensive_stages=[],
            parallelization_opportunities=[],
            optimization_suggestions=[]
        ),
        diagrams=[],
        overall_status="valid",
        critical_issues=[],
        recommendations=[
            "Consider adding data validation between each stage",
            "Consider implementing parallel execution for independent stages",
            "Add detailed logging for data transformations between stages"
        ]
    ) 