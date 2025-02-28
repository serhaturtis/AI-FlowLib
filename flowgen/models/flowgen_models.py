"""Stage-specific models for flow generation."""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


# Flow Description Generator Models
class DataTransformation(BaseModel):
    """Description of a data transformation in the flow."""
    input_data: str = Field(..., description="Description of input data")
    output_data: str = Field(..., description="Description of output data")
    transformation: str = Field(..., description="Description of the transformation process")
    requirements: List[str] = Field(default_factory=list, description="Requirements for this transformation")

class ComponentDescription(BaseModel):
    """Description of a flow component."""
    name: str = Field(..., description="Name of the component")
    purpose: str = Field(..., description="Purpose of this component")
    responsibilities: List[str] = Field(..., description="List of component responsibilities")
    dependencies: List[str] = Field(default_factory=list, description="Other components this depends on")

class ArchitecturalDecision(BaseModel):
    """Description of an architectural decision."""
    decision: str = Field(..., description="The architectural decision made")
    rationale: str = Field(..., description="Reasoning behind the decision")
    implications: List[str] = Field(..., description="Implications of this decision")
    alternatives_considered: Optional[List[str]] = Field(default=None, description="Other options that were considered")

class FlowDescription(BaseModel):
    """Complete flow description output."""
    overview: str = Field(..., description="High-level overview of the flow")
    components: List[ComponentDescription] = Field(..., description="Flow components")
    data_transformations: List[DataTransformation] = Field(..., description="Data transformations")
    architectural_decisions: List[ArchitecturalDecision] = Field(..., description="Key architectural decisions")
    constraints: List[str] = Field(..., description="Flow constraints and limitations")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made in the design")

# Pipeline Model Designer Models
class FieldValidation(BaseModel):
    """Validation rule for a model field."""
    type: str = Field(..., description="Type of validation (e.g., range, regex, custom)")
    rule: str = Field(..., description="The validation rule")
    error_message: str = Field(..., description="Error message when validation fails")

class FieldDefinition(BaseModel):
    """Definition of a model field."""
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (e.g., str, int, List[str])")
    description: str = Field(..., description="Field description")
    required: bool = Field(..., description="Whether the field is required")
    default: Optional[str] = Field(default=None, description="Default value if any")
    validations: List[FieldValidation] = Field(default_factory=list, description="Field validation rules")
    example: Optional[str] = Field(default=None, description="Example value")

class ModelDefinition(BaseModel):
    """Definition of a Pydantic model."""
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    fields: List[FieldDefinition] = Field(..., description="Model fields")
    base_classes: List[str] = Field(default_factory=lambda: ["BaseModel"], description="Base classes")
    referenced_models: List[str] = Field(default_factory=list, description="Names of other models referenced by this model")

class PipelineModels(BaseModel):
    """Complete pipeline model definitions."""
    input_model: ModelDefinition = Field(..., description="Flow input model definition")
    output_model: ModelDefinition = Field(..., description="Flow output model definition")
    shared_models: Dict[str, ModelDefinition] = Field(
        default_factory=dict,
        description="Shared models used by both input and output"
    )
    validation_notes: List[str] = Field(
        default_factory=list,
        description="Notes about model validation and usage"
    )

# Stage Planner Models
class StageRequirement(BaseModel):
    """Requirement for a stage."""
    type: Literal["data", "resource", "processing"] = Field(..., description="Type of requirement")
    description: str = Field(..., description="Description of the requirement")
    source: Optional[str] = Field(default=None, description="Source of the requirement (e.g., previous stage)")
    priority: Literal["must_have", "should_have", "nice_to_have"] = Field(
        default="must_have",
        description="Priority of this requirement"
    )

class StageDependency(BaseModel):
    """Dependency between stages."""
    dependent_stage: str = Field(..., description="Name of the dependent stage")
    required_stage: str = Field(..., description="Name of the required stage")
    data_dependencies: List[str] = Field(..., description="Data items required from the stage")
    dependency_type: Literal["strict", "optional"] = Field(
        default="strict",
        description="Whether this dependency is required"
    )

class StageDefinition(BaseModel):
    """Definition of a flow stage."""
    name: str = Field(..., description="Stage name")
    purpose: str = Field(..., description="Stage purpose")
    description: str = Field(..., description="Detailed description of what this stage does")
    requirements: List[StageRequirement] = Field(..., description="Stage requirements")
    outputs: List[str] = Field(..., description="Data outputs produced by this stage")
    uses_llm: bool = Field(..., description="Whether this stage requires LLM")
    dependencies: List[StageDependency] = Field(
        default_factory=list,
        description="Dependencies on other stages"
    )
    error_handling: List[str] = Field(
        default_factory=list,
        description="Error scenarios to handle"
    )

class StageSequence(BaseModel):
    """Complete sequence of flow stages."""
    stages: List[StageDefinition] = Field(..., description="List of all stages")
    execution_order: List[str] = Field(..., description="Order of stage execution")
    parallel_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups of stages that can run in parallel"
    )
    validation_points: List[str] = Field(
        default_factory=list,
        description="Points in sequence where validation should occur"
    )
    error_recovery: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Recovery actions for each stage on error"
    )

# Stage Model Generator Models
class DataDependency(BaseModel):
    """Represents a data dependency between stages."""
    source_stage: str = Field(..., description="Stage that produces the data")
    target_stage: str = Field(..., description="Stage that consumes the data")
    data_fields: List[str] = Field(..., description="Fields being passed between stages")
    is_required: bool = Field(default=True, description="Whether this data is required for the target stage")
    validation_rules: List[str] = Field(default_factory=list, description="Rules for validating the data transfer")

class StageInterface(BaseModel):
    """Interface definition for a stage."""
    stage_name: str = Field(..., description="Name of the stage")
    input_model: ModelDefinition = Field(..., description="Input data model for the stage")
    output_model: ModelDefinition = Field(..., description="Output data model for the stage")
    internal_models: Dict[str, ModelDefinition] = Field(
        default_factory=dict,
        description="Models used internally by the stage"
    )
    error_models: Dict[str, ModelDefinition] = Field(
        default_factory=dict,
        description="Models for stage-specific errors"
    )

class DataFlow(BaseModel):
    """Represents the flow of data through stages."""
    dependencies: List[DataDependency] = Field(..., description="Data dependencies between stages")
    shared_models: Dict[str, ModelDefinition] = Field(
        default_factory=dict,
        description="Models shared across multiple stages"
    )
    validation_chain: List[str] = Field(
        default_factory=list,
        description="Sequence of validation steps for data flow"
    )

class StageModels(BaseModel):
    """Complete set of models for all stages."""
    interfaces: Dict[str, StageInterface] = Field(..., description="Interface definitions for each stage")
    data_flow: DataFlow = Field(..., description="Data flow between stages")
    common_types: Dict[str, ModelDefinition] = Field(
        default_factory=dict,
        description="Common data types used across stages"
    )
    llm_interfaces: Dict[str, ModelDefinition] = Field(
        default_factory=dict,
        description="Models for LLM input/output in relevant stages"
    )
    validation_models: Dict[str, ModelDefinition] = Field(
        default_factory=dict,
        description="Models used for data validation between stages"
    )

# Flow Validator Models
class ValidationCheck(BaseModel):
    """Individual validation check result."""
    check_type: Literal["data_flow", "model", "stage", "requirement", "security"] = Field(
        ...,
        description="Type of validation check"
    )
    component: str = Field(..., description="Component being validated")
    status: Literal["pass", "fail", "warning"] = Field(..., description="Result of the check")
    details: str = Field(..., description="Detailed explanation of the check result")
    recommendations: List[str] = Field(default_factory=list, description="Suggested improvements if any")

class DataFlowValidation(BaseModel):
    """Validation of data flow between stages."""
    missing_data: List[str] = Field(default_factory=list, description="Data required but not provided")
    type_mismatches: List[str] = Field(default_factory=list, description="Incompatible data types between stages")
    unused_outputs: List[str] = Field(default_factory=list, description="Stage outputs not consumed by any stage")
    circular_dependencies: List[List[str]] = Field(default_factory=list, description="Circular dependencies between stages")
    validation_gaps: List[str] = Field(default_factory=list, description="Points where data validation is missing")

class SecurityValidation(BaseModel):
    """Security-related validation results."""
    data_exposure_risks: List[str] = Field(default_factory=list, description="Potential data exposure points")
    input_validation_gaps: List[str] = Field(default_factory=list, description="Missing input validation")
    resource_access_issues: List[str] = Field(default_factory=list, description="Unsafe resource access patterns")
    llm_prompt_risks: List[str] = Field(default_factory=list, description="Potential risks in LLM prompt handling")

class PerformanceValidation(BaseModel):
    """Performance-related validation results."""
    bottlenecks: List[str] = Field(default_factory=list, description="Potential performance bottlenecks")
    resource_intensive_stages: List[str] = Field(default_factory=list, description="Stages with high resource usage")
    parallel_opportunities: List[str] = Field(default_factory=list, description="Opportunities for parallelization")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Suggested optimizations")

class ValidationDiagram(BaseModel):
    """Diagram for flow validation."""
    diagram_type: Literal["data_flow", "sequence", "component", "state"] = Field(..., description="Type of diagram")
    content: str = Field(..., description="Diagram content in specified format (e.g., mermaid)")
    format: str = Field(..., description="Format of the diagram (e.g., mermaid, dot)")
    description: str = Field(..., description="Description of what the diagram shows")
    highlights: List[str] = Field(default_factory=list, description="Important points highlighted in the diagram")

class FlowValidation(BaseModel):
    """Complete flow validation results."""
    checks: List[ValidationCheck] = Field(..., description="Individual validation checks")
    data_flow: DataFlowValidation = Field(..., description="Data flow validation results")
    security: SecurityValidation = Field(..., description="Security validation results")
    performance: PerformanceValidation = Field(..., description="Performance validation results")
    diagrams: List[ValidationDiagram] = Field(..., description="Validation diagrams")
    overall_status: Literal["valid", "valid_with_warnings", "invalid"] = Field(
        ...,
        description="Overall validation status"
    )
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues that must be addressed")
    recommendations: List[str] = Field(default_factory=list, description="Suggested improvements")

# Stage Implementation Generator Models
class ImportStatement(BaseModel):
    """Import statement in generated code."""
    module: str = Field(..., description="Module to import from")
    names: List[str] = Field(..., description="Names to import")
    is_from_import: bool = Field(default=True, description="Whether to use 'from module import names' syntax")
    alias: Optional[str] = Field(default=None, description="Optional alias for the import")

class MethodImplementation(BaseModel):
    """Implementation of a method in generated code."""
    name: str = Field(..., description="Method name")
    description: str = Field(..., description="Method docstring")
    parameters: List[str] = Field(..., description="Method parameters")
    return_type: str = Field(..., description="Return type annotation")
    code: str = Field(..., description="Method implementation code")
    decorators: List[str] = Field(default_factory=list, description="Method decorators")
    raises: List[str] = Field(default_factory=list, description="Exceptions that may be raised")
    async_method: bool = Field(default=True, description="Whether this is an async method")

class PromptTemplate(BaseModel):
    """LLM prompt template for a stage."""
    name: str = Field(..., description="Template name")
    template: str = Field(..., description="The prompt template")
    required_context: List[str] = Field(..., description="Required context variables")
    format_instructions: str = Field(..., description="Instructions for formatting the prompt")
    example_context: Dict[str, str] = Field(
        default_factory=dict,
        description="Example context values"
    )
    fallback_strategy: str = Field(
        default="retry",
        description="What to do if LLM call fails"
    )

class ErrorHandler(BaseModel):
    """Error handling code for a stage."""
    error_types: List[str] = Field(..., description="Types of errors to handle")
    handling_code: str = Field(..., description="Error handling implementation")
    recovery_strategy: str = Field(..., description="How to recover from the error")
    logging_level: Literal["debug", "info", "warning", "error"] = Field(
        ...,
        description="Logging level for this error"
    )

class StageImplementation(BaseModel):
    """Complete implementation of a stage."""
    stage_name: str = Field(..., description="Name of the stage")
    imports: List[ImportStatement] = Field(..., description="Required imports")
    methods: List[MethodImplementation] = Field(..., description="Stage methods")
    prompts: Dict[str, PromptTemplate] = Field(
        default_factory=dict,
        description="LLM prompt templates if stage uses LLM"
    )
    error_handlers: List[ErrorHandler] = Field(..., description="Error handlers for the stage")
    helper_functions: List[MethodImplementation] = Field(
        default_factory=list,
        description="Helper functions used by the stage"
    )
    class_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Class-level variables and constants"
    )

class StageTestCase(BaseModel):
    """Test case for a stage."""
    name: str = Field(..., description="Test case name")
    description: str = Field(..., description="Test case description")
    input_data: Dict[str, Any] = Field(..., description="Test input data")
    expected_output: Dict[str, Any] = Field(..., description="Expected output")
    mocks: Dict[str, str] = Field(default_factory=dict, description="Required mocks")
    setup_code: str = Field(default="", description="Test setup code")
    cleanup_code: str = Field(default="", description="Test cleanup code")
    is_example: bool = Field(default=False, description="Whether this test case should be used as an example")

class GeneratedStage(BaseModel):
    """Complete generated stage package."""
    implementation: StageImplementation = Field(..., description="Stage implementation")
    tests: List[StageTestCase] = Field(..., description="Test cases for the stage")
    requirements: List[str] = Field(
        default_factory=list,
        description="Additional package requirements"
    )
    documentation: str = Field(..., description="Stage documentation")
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Usage examples"
    )
    performance_notes: List[str] = Field(
        default_factory=list,
        description="Notes about performance characteristics"
    )

class StageImplementations(BaseModel):
    """Collection of all generated stages."""
    stages: Dict[str, GeneratedStage] = Field(..., description="Generated stages by name")
    common_code: List[str] = Field(
        default_factory=list,
        description="Code shared between stages"
    )
    common_requirements: List[str] = Field(
        default_factory=list,
        description="Requirements shared by multiple stages"
    )
    integration_tests: List[StageTestCase] = Field(
        default_factory=list,
        description="Tests for stage integration"
    )

# Pipeline Generator Models
class ResourceDefinition(BaseModel):
    """Definition of a resource used by the flow."""
    name: str = Field(..., description="Resource name")
    type: str = Field(..., description="Resource type")
    configuration: Dict[str, Any] = Field(..., description="Resource configuration")
    initialization_code: str = Field(..., description="Code to initialize the resource")
    cleanup_code: str = Field(..., description="Code to clean up the resource")
    error_handling: Dict[str, str] = Field(
        default_factory=dict,
        description="Error handling for resource operations"
    )

class FlowState(BaseModel):
    """State management for the flow."""
    variables: Dict[str, str] = Field(..., description="State variables and their types")
    initialization: str = Field(..., description="State initialization code")
    update_points: Dict[str, str] = Field(
        default_factory=dict,
        description="Points where state is updated"
    )
    cleanup: str = Field(
        default="",
        description="State cleanup code"
    )

class StageOrchestration(BaseModel):
    """Orchestration logic for stages."""
    stage_initialization: Dict[str, str] = Field(..., description="Initialization code for each stage")
    execution_order: List[str] = Field(..., description="Order of stage execution")
    data_passing: Dict[str, str] = Field(..., description="Code for passing data between stages")
    parallel_execution: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Stages that can be executed in parallel"
    )
    error_handling: Dict[str, str] = Field(
        default_factory=dict,
        description="Error handling for stage execution"
    )

class MonitoringSetup(BaseModel):
    """Monitoring configuration for the flow."""
    metrics: List[str] = Field(..., description="Metrics to track")
    logging_points: Dict[str, str] = Field(..., description="Points where logging occurs")
    performance_tracking: Dict[str, str] = Field(
        default_factory=dict,
        description="Performance monitoring code"
    )
    alert_conditions: Dict[str, str] = Field(
        default_factory=dict,
        description="Conditions that trigger alerts"
    )

class FlowImplementation(BaseModel):
    """Complete flow implementation."""
    flow_name: str = Field(..., description="Name of the flow")
    imports: List[ImportStatement] = Field(..., description="Required imports")
    resources: Dict[str, ResourceDefinition] = Field(..., description="Resource definitions")
    state_management: FlowState = Field(..., description="Flow state management")
    orchestration: StageOrchestration = Field(..., description="Stage orchestration logic")
    monitoring: MonitoringSetup = Field(..., description="Monitoring configuration")
    pipeline_method: MethodImplementation = Field(..., description="Main pipeline method")
    helper_methods: List[MethodImplementation] = Field(
        default_factory=list,
        description="Helper methods for the flow"
    )
    error_handlers: List[ErrorHandler] = Field(
        default_factory=list,
        description="Flow-level error handlers"
    )

class FlowTestSuite(BaseModel):
    """Test suite for the flow."""
    unit_tests: List[StageTestCase] = Field(..., description="Unit tests for flow methods")
    integration_tests: List[StageTestCase] = Field(..., description="Integration tests for the flow")
    performance_tests: List[StageTestCase] = Field(
        default_factory=list,
        description="Performance tests"
    )
    error_scenario_tests: List[StageTestCase] = Field(
        default_factory=list,
        description="Tests for error scenarios"
    )

class GeneratedPipeline(BaseModel):
    """Complete generated pipeline package."""
    implementation: FlowImplementation = Field(..., description="Flow implementation")
    tests: FlowTestSuite = Field(..., description="Test suite")
    configuration: Dict[str, Any] = Field(..., description="Flow configuration")
    documentation: str = Field(..., description="Flow documentation")
    deployment: Dict[str, str] = Field(
        default_factory=dict,
        description="Deployment configuration and scripts"
    )
    monitoring_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Monitoring and alerting configuration"
    )
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Usage examples"
    )

class DocumentationSection(BaseModel):
    """Section of documentation."""
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    format: str = Field(..., description="Content format (markdown, rst, etc.)")
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Named sections of content"
    )
    code_examples: List[str] = Field(
        default_factory=list,
        description="Code examples"
    )

class APIDocumentation(BaseModel):
    """API documentation for a flow component."""
    component_name: str = Field(..., description="Name of the component being documented")
    description: str = Field(..., description="Component description")
    methods: List[Dict[str, str]] = Field(..., description="Method documentation")
    parameters: List[Dict[str, str]] = Field(..., description="Parameter documentation")
    returns: Dict[str, str] = Field(..., description="Return value documentation")
    exceptions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Documentation of possible exceptions"
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Usage examples"
    )

class ConfigurationTemplate(BaseModel):
    """Template for a configuration file."""
    filename: str = Field(..., description="Name of the configuration file")
    format: str = Field(..., description="File format (yaml, json, etc.)")
    template: str = Field(..., description="Template content")
    config_schema: Dict[str, Any] = Field(..., description="Configuration schema")
    default_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default configuration values"
    )
    documentation: str = Field(..., description="Configuration documentation")

class ExampleScript(BaseModel):
    """Example script demonstrating flow usage."""
    filename: str = Field(..., description="Script filename")
    description: str = Field(..., description="Script purpose and usage")
    code: str = Field(..., description="Example code")
    requirements: List[str] = Field(
        default_factory=list,
        description="Additional requirements for the example"
    )
    expected_output: str = Field(..., description="Expected script output")
    setup_instructions: str = Field(
        default="",
        description="Instructions for setting up the example"
    )

class DeploymentFile(BaseModel):
    """Deployment or environment setup file."""
    filename: str = Field(..., description="File name")
    type: str = Field(..., description="File type (Dockerfile, docker-compose, etc.)")
    content: str = Field(..., description="File content")
    purpose: str = Field(..., description="Purpose of this file")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Required dependencies or services"
    )
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Required environment variables"
    )

class TestDocumentation(BaseModel):
    """Documentation for tests."""
    test_type: str = Field(..., description="Type of test (unit, integration, etc.)")
    description: str = Field(..., description="Test description")
    setup_requirements: List[str] = Field(..., description="Setup requirements")
    test_cases: List[Dict[str, str]] = Field(..., description="Documented test cases")
    mocking_guide: Optional[str] = Field(
        default=None,
        description="Guide for mocking dependencies"
    )
    troubleshooting: List[str] = Field(
        default_factory=list,
        description="Troubleshooting tips"
    )

class SupportFiles(BaseModel):
    """Complete set of support files."""
    readme: DocumentationSection = Field(..., description="Main README documentation")
    api_docs: List[APIDocumentation] = Field(..., description="API documentation")
    configuration_templates: List[ConfigurationTemplate] = Field(
        ...,
        description="Configuration file templates"
    )
    example_scripts: List[ExampleScript] = Field(..., description="Example scripts")
    deployment_files: List[DeploymentFile] = Field(..., description="Deployment files")
    test_documentation: List[TestDocumentation] = Field(..., description="Test documentation")
    additional_docs: Dict[str, DocumentationSection] = Field(
        default_factory=dict,
        description="Additional documentation files"
    )

# Wrapper models for list responses
class ComponentDescriptionList(BaseModel):
    """List of component descriptions."""
    items: List[ComponentDescription] = Field(..., description="List of component descriptions")

class DataTransformationList(BaseModel):
    """List of data transformations."""
    items: List[DataTransformation] = Field(..., description="List of data transformations")

class ArchitecturalDecisionList(BaseModel):
    """List of architectural decisions."""
    items: List[ArchitecturalDecision] = Field(..., description="List of architectural decisions")

class ValidationCheckList(BaseModel):
    """List of validation checks."""
    items: List[ValidationCheck] = Field(..., description="List of validation checks")

class ValidationDiagramList(BaseModel):
    """List of validation diagrams."""
    items: List[ValidationDiagram] = Field(..., description="List of validation diagrams")

class APIDocumentationList(BaseModel):
    """List of API documentation items."""
    items: List[APIDocumentation] = Field(..., description="List of API documentation items")

class ConfigurationTemplateList(BaseModel):
    """List of configuration templates."""
    items: List[ConfigurationTemplate] = Field(..., description="List of configuration templates")

class ExampleScriptList(BaseModel):
    """List of example scripts."""
    items: List[ExampleScript] = Field(..., description="List of example scripts")

class DeploymentFileList(BaseModel):
    """List of deployment files."""
    items: List[DeploymentFile] = Field(..., description="List of deployment files")

class TestDocumentationList(BaseModel):
    """List of test documentation items."""
    items: List[TestDocumentation] = Field(..., description="List of test documentation items")

class StageDefinitionList(BaseModel):
    """List of stage definitions."""
    items: List[StageDefinition] = Field(..., description="List of stage definitions")

class StageOrderList(BaseModel):
    """List of stage names in execution order."""
    items: List[str] = Field(..., description="List of stage names in execution order")

class ParallelGroupList(BaseModel):
    """List of parallel stage groups."""
    items: List[List[str]] = Field(..., description="List of parallel stage groups")

# Wrapper models for dictionary responses
class StageInterfaceDict(BaseModel):
    """Dictionary of stage interfaces."""
    items: Dict[str, StageInterface] = Field(..., description="Dictionary of stage interfaces by stage name")

class ModelDefinitionDict(BaseModel):
    """Dictionary of model definitions."""
    items: Dict[str, ModelDefinition] = Field(..., description="Dictionary of model definitions by name")

class PromptTemplateList(BaseModel):
    """List of prompt templates."""
    items: List[PromptTemplate] = Field(..., description="List of prompt templates")

class StageTestCaseList(BaseModel):
    """List of stage test cases."""
    items: List[StageTestCase] = Field(..., description="List of stage test cases")

class FlowRequirement(BaseModel):
    """Requirement specification for flow generation."""
    description: str = Field(..., description="Detailed description of the requirement")
    constraints: Optional[List[str]] = Field(default=None, description="Any specific constraints or conditions")
    examples: Optional[List[str]] = Field(default=None, description="Example values or scenarios")

class ResourceRequirement(BaseModel):
    """Resource requirement specification."""
    type: str = Field(..., description="Type of resource (e.g., 'llm', 'database', 'api')")
    description: str = Field(..., description="Description of how the resource will be used")
    constraints: Optional[Dict[str, str]] = Field(default=None, description="Any specific constraints for this resource")

class FlowGeneratorInput(BaseModel):
    """Input for flow generation."""
    task_description: str = Field(
        ...,
        description="Detailed explanation of what the flow should accomplish"
    )
    input_requirements: List[FlowRequirement] = Field(
        ...,
        description="Description of data the flow will receive"
    )
    output_requirements: List[FlowRequirement] = Field(
        ...,
        description="Description of data the flow should produce"
    )
    performance_requirements: Optional[List[str]] = Field(
        default=None,
        description="Any specific performance requirements or constraints"
    )
    resource_requirements: Optional[List[ResourceRequirement]] = Field(
        default=None,
        description="Any specific resource requirements (LLM, DB, etc.)"
    )
    integration_requirements: Optional[List[str]] = Field(
        default=None,
        description="Any specific integration requirements"
    )

class GeneratedFile(BaseModel):
    """A generated file."""
    path: str = Field(..., description="Relative path where the file should be created")
    content: str = Field(..., description="Content of the file")
    description: str = Field(..., description="Description of the file's purpose")
    is_executable: bool = Field(default=False, description="Whether the file should be executable")

class PromptTemplate(BaseModel):
    """A prompt template for LLM stages."""
    name: str = Field(..., description="Name of the prompt template")
    template: str = Field(..., description="The actual prompt template")
    description: str = Field(..., description="Description of what this prompt does")
    required_context: List[str] = Field(..., description="List of context variables required by this template")
    example_context: Optional[Dict[str, str]] = Field(default=None, description="Example context values")

class ValidationArtifact(BaseModel):
    """Validation artifact from flow generation."""
    type: str = Field(..., description="Type of validation artifact (diagram, documentation, etc.)")
    content: str = Field(..., description="Content of the validation artifact")
    format: str = Field(..., description="Format of the content (markdown, mermaid, etc.)")
    description: str = Field(..., description="Description of what this artifact validates")

class FlowGeneratorOutput(BaseModel):
    """Output from flow generation."""
    files: Dict[str, GeneratedFile] = Field(
        ...,
        description="Generated implementation files"
    )
    prompt_templates: Dict[str, PromptTemplate] = Field(
        ...,
        description="Generated prompt templates for LLM stages"
    )
    validation_artifacts: List[ValidationArtifact] = Field(
        ...,
        description="Generated validation artifacts"
    )
    documentation: Dict[str, str] = Field(
        ...,
        description="Generated documentation (README, usage examples, etc.)"
    ) 

# We'll add models for subsequent stages as we design them iteratively
# Each stage's models will be added here, using the context from the previous stage
# This ensures proper data flow between stages 