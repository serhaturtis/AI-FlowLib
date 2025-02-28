"""Support File Generator stage implementation."""

from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline
from flowlib.core.resources import ResourceRegistry

from ..models.flowgen_models import (
    FlowDescription, GeneratedPipeline, SupportFiles, DocumentationSection, 
    DeploymentFile, APIDocumentationList, ConfigurationTemplateList, ExampleScriptList,
    DeploymentFileList, TestDocumentationList
)

logger = logging.getLogger(__name__)

@flow("support_file_generator")
class SupportFileGenerator:
    """Generates all supporting files and documentation for a flow."""
    
    def __init__(self):
        """Initialize the generator."""
        self.provider = ResourceRegistry.get_resource("provider", "llm")
        self.model_name = "flow_generator"  # Use the model name defined in config/models.py
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
    
    async def __aenter__(self) -> 'SupportFileGenerator':
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
    
    @stage(output_model=DocumentationSection)
    async def generate_readme(
        self,
        flow_description: FlowDescription,
        pipeline: GeneratedPipeline
    ) -> DocumentationSection:
        """Generate a comprehensive README documentation for the flow."""
        section_prompts = {
            "Overview": "Provide a clear introduction to the flow, its purpose, and key features. Include badges for build status, version, and license.",
            "Installation": "Detail the installation process, including prerequisites, dependencies, and step-by-step instructions.",
            "Configuration": "Explain how to configure the flow, including environment variables, config files, and validation rules. Reference the config_schema.",
            "Usage": "Demonstrate how to use the flow with practical examples, code snippets, and common use cases.",
            "Architecture": "Describe the flow's architecture, components, data flow, and integration points.",
            "Development": "Provide guidelines for development, testing, and contributing to the flow.",
            "Deployment": "Explain deployment options, requirements, and best practices.",
            "Troubleshooting": "List common issues, their solutions, and debugging tips."
        }

        # Prepare context for section generation
        context = {
            "overview": flow_description.overview,
            "components": "\n".join(f"- {c.name}: {c.purpose}" for c in flow_description.components),
            "configuration": str(pipeline.configuration),
            "examples": "\n".join(str(e) for e in pipeline.examples),
            "schema": self._format_schema(DocumentationSection.model_json_schema())
        }

        sections = {}
        content_parts = []
        
        # Extract project name from overview for title
        project_name = flow_description.overview.split(".")[0] or "Flow Documentation"
        content_parts.append(f"# {project_name}\n")
        
        # Generate and add each section
        for section_name, section_prompt in section_prompts.items():
            # Generate each section with a focused prompt
            section_content = await self.provider.generate_structured(
                prompt=self._load_prompt("generate_readme_section.txt").format(
                    section_name=section_name,
                    section_prompt=section_prompt,
                    **context
                ),
                model_name=self.model_name,
                response_model=DocumentationSection,
                max_tokens=2048,
                temperature=0.7
            )
            
            # Store the section content
            sections[section_name] = section_content.content.strip()
            
            # Add to main content
            content_parts.append(f"\n## {section_name}\n")
            content_parts.append(section_content.content.strip())

        # Add badges at the end
        content_parts.append("\n## Badges\n")
        content_parts.append("![Build Status](https://img.shields.io/badge/build-passing-brightgreen)")
        content_parts.append("![Version](https://img.shields.io/badge/version-1.0.0-blue)")
        content_parts.append("![License](https://img.shields.io/badge/license-MIT-green)")

        return DocumentationSection(
            title=project_name,
            content="\n".join(content_parts),
            format="markdown",
            sections=sections
        )
    
    @stage(output_model=APIDocumentationList)
    async def generate_api_docs(
        self,
        flow_description: FlowDescription,
        pipeline: GeneratedPipeline
    ) -> APIDocumentationList:
        """Generate API documentation for all components."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_api_docs.txt").format(
                overview=flow_description.overview,
                components="\n".join(
                    f"- {c.name}:\n  {c.purpose}\n  {c.responsibilities}"
                    for c in flow_description.components
                ),
                implementation=str(pipeline.implementation),
                schema=self._format_schema(APIDocumentationList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=APIDocumentationList,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ConfigurationTemplateList)
    async def generate_config_templates(
        self,
        flow_description: FlowDescription,
        pipeline: GeneratedPipeline
    ) -> ConfigurationTemplateList:
        """Generate configuration file templates."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_config_templates.txt").format(
                overview=flow_description.overview,
                configuration=str(pipeline.configuration),
                resources="\n".join(
                    f"- {name}: {resource.type}\n  Configuration: {resource.configuration}"
                    for name, resource in pipeline.implementation.resources.items()
                ),
                schema=self._format_schema(ConfigurationTemplateList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ConfigurationTemplateList,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @stage(output_model=ExampleScriptList)
    async def generate_example_scripts(
        self,
        flow_description: FlowDescription,
        pipeline: GeneratedPipeline
    ) -> ExampleScriptList:
        """Generate example usage scripts."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_example_scripts.txt").format(
                overview=flow_description.overview,
                implementation=str(pipeline.implementation),
                configuration=str(pipeline.configuration),
                examples="\n".join(str(e) for e in pipeline.examples),
                schema=self._format_schema(ExampleScriptList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=ExampleScriptList,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @stage(output_model=DeploymentFileList)
    async def generate_deployment_files(
        self,
        flow_description: FlowDescription,
        pipeline: GeneratedPipeline
    ) -> DeploymentFileList:
        """Generate deployment and environment setup files."""
        # Generate files in separate chunks to avoid token limits
        deployment_files = []

        # 1. Generate Dockerfile
        dockerfile = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_dockerfile.txt").format(
                overview=flow_description.overview,
                configuration=str(pipeline.configuration),
                resources="\n".join(
                    f"- {name}: {resource.type}\n  Configuration: {resource.configuration}"
                    for name, resource in pipeline.implementation.resources.items()
                )
            ),
            model_name=self.model_name,
            response_model=DeploymentFile,
            max_tokens=2048,
            temperature=0.7
        )
        deployment_files.append(dockerfile)

        # 2. Generate docker-compose.yml
        compose = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_docker_compose.txt").format(
                overview=flow_description.overview,
                configuration=str(pipeline.configuration)
            ),
            model_name=self.model_name,
            response_model=DeploymentFile,
            max_tokens=2048,
            temperature=0.7
        )
        deployment_files.append(compose)

        # 3. Generate Kubernetes manifests
        k8s = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_kubernetes.txt").format(
                overview=flow_description.overview,
                configuration=str(pipeline.configuration),
                resources="\n".join(
                    f"- {name}: {resource.type}\n  Configuration: {resource.configuration}"
                    for name, resource in pipeline.implementation.resources.items()
                )
            ),
            model_name=self.model_name,
            response_model=DeploymentFile,
            max_tokens=2048,
            temperature=0.7
        )
        deployment_files.append(k8s)

        # 4. Generate environment setup files
        env_setup = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_env_setup.txt").format(
                overview=flow_description.overview,
                configuration=str(pipeline.configuration)
            ),
            model_name=self.model_name,
            response_model=DeploymentFile,
            max_tokens=2048,
            temperature=0.7
        )
        deployment_files.append(env_setup)

        return DeploymentFileList(items=deployment_files)
    
    @stage(output_model=TestDocumentationList)
    async def generate_test_docs(
        self,
        flow_description: FlowDescription,
        pipeline: GeneratedPipeline
    ) -> TestDocumentationList:
        """Generate test documentation."""
        result = await self.provider.generate_structured(
            prompt=self._load_prompt("generate_test_docs.txt").format(
                overview=flow_description.overview,
                tests=str(pipeline.tests),
                implementation=str(pipeline.implementation),
                schema=self._format_schema(TestDocumentationList.model_json_schema())
            ),
            model_name=self.model_name,
            response_model=TestDocumentationList,
            max_tokens=4096,
            temperature=0.7
        )
        return result
    
    @pipeline(
        input_model=tuple([FlowDescription, GeneratedPipeline]),
        output_model=SupportFiles
    )
    async def generate_support_files(
        self,
        flow_description: FlowDescription,
        pipeline: GeneratedPipeline
    ) -> SupportFiles:
        """Generate all supporting files and documentation.
        
        Args:
            flow_description: Description of the flow
            pipeline: Generated pipeline implementation
            
        Returns:
            Complete set of support files
        """
        # Generate main README
        readme = await self.generate_readme(flow_description, pipeline)
        logger.info("Generated README documentation")
        
        # Generate API documentation
        api_docs = await self.generate_api_docs(flow_description, pipeline)
        logger.info(f"Generated {len(api_docs.items)} API documentation files")
        
        # Generate configuration templates
        config_templates = await self.generate_config_templates(flow_description, pipeline)
        logger.info(f"Generated {len(config_templates.items)} configuration templates")
        
        # Generate example scripts
        example_scripts = await self.generate_example_scripts(flow_description, pipeline)
        logger.info(f"Generated {len(example_scripts.items)} example scripts")
        
        # Generate deployment files
        deployment_files = await self.generate_deployment_files(flow_description, pipeline)
        logger.info(f"Generated {len(deployment_files.items)} deployment files")
        
        # Generate test documentation
        test_docs = await self.generate_test_docs(flow_description, pipeline)
        logger.info(f"Generated {len(test_docs.items)} test documentation files")
        
        # Create additional documentation sections
        additional_docs = {
            "architecture": DocumentationSection(
                title="Architecture Overview",
                content="\n".join(
                    f"## {d.decision}\n{d.rationale}\n\nImplications:\n" +
                    "\n".join(f"- {i}" for i in d.implications)
                    for d in flow_description.architectural_decisions
                ),
                format="markdown",
                sections={}  # Initialize with empty sections dict
            ),
            "data_flow": DocumentationSection(
                title="Data Flow",
                content="\n".join(
                    f"## {t.input_data} → {t.output_data}\n{t.transformation}"
                    for t in flow_description.data_transformations
                ),
                format="markdown",
                sections={}  # Initialize with empty sections dict
            )
        }
        
        return SupportFiles(
            readme=readme,
            api_docs=api_docs.items,
            configuration_templates=config_templates.items,
            example_scripts=example_scripts.items,
            deployment_files=deployment_files.items,
            test_documentation=test_docs.items,
            additional_docs=additional_docs
        ) 