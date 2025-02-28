"""File Output Generator stage implementation."""

from typing import Dict, List, Optional
from pathlib import Path
import logging
import os

from flowlib import flow, stage, pipeline

from ..models.flowgen_models import (
    StageImplementations, SupportFiles
)

from ..models.flowgen_models import (
    GeneratedFile
)

logger = logging.getLogger(__name__)

@flow("file_output_generator")
class FileOutputGenerator:
    """Generates and writes all output files to disk."""
    
    def __init__(self):
        """Initialize the generator."""
        # Get workspace root path
        self.workspace_root = Path(os.getenv("ROOT_FOLDER", "."))
        self.prompts_dir = Path(os.getenv("PROMPTS_FOLDER", "."))
        self.output_dir = self.workspace_root / "output"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def __aenter__(self) -> 'FileOutputGenerator':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass

    def _write_file(self, file: GeneratedFile) -> None:
        """Write a generated file to disk.
        
        Args:
            file: The file to write
        """
        file_path = self.output_dir / file.path
        os.makedirs(file_path.parent, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(file.content)
            
        if file.is_executable:
            os.chmod(file_path, 0o755)
            
        logger.info(f"Written {file_path}")

    @pipeline(
        input_model=tuple([SupportFiles, StageImplementations]),
        output_model=Dict[str, GeneratedFile]
    )
    async def write_output_files(
        self,
        support_files: SupportFiles,
        implementations: StageImplementations
    ) -> Dict[str, GeneratedFile]:
        """Write all generated files to disk.
        
        Args:
            support_files: Generated support files
            implementations: Generated stage implementations
            
        Returns:
            Dictionary mapping file paths to GeneratedFile objects
        """
        generated_files = {}
        
        # Write README
        readme_file = GeneratedFile(
            path="README.md",
            content=support_files.readme.content,
            description="Main README documentation",
            is_executable=False
        )
        self._write_file(readme_file)
        generated_files["README.md"] = readme_file
        
        # Write API documentation
        for doc in support_files.api_docs:
            file = GeneratedFile(
                path=f"docs/api/{doc.component_name}.md",
                content=doc.description,
                description=f"API documentation for {doc.component_name}",
                is_executable=False
            )
            self._write_file(file)
            generated_files[file.path] = file
            
        # Write configuration templates
        for config in support_files.configuration_templates:
            file = GeneratedFile(
                path=f"config/{config.filename}",
                content=config.template,
                description=config.description,
                is_executable=False
            )
            self._write_file(file)
            generated_files[file.path] = file
            
        # Write example scripts
        for script in support_files.example_scripts:
            file = GeneratedFile(
                path=f"examples/{script.filename}",
                content=script.code,
                description=script.description,
                is_executable=True
            )
            self._write_file(file)
            generated_files[file.path] = file
            
        # Write deployment files
        for deploy_file in support_files.deployment_files:
            file = GeneratedFile(
                path=f"deploy/{deploy_file.filename}",
                content=deploy_file.content,
                description=deploy_file.purpose,
                is_executable=False
            )
            self._write_file(file)
            generated_files[file.path] = file
            
        # Write test documentation
        for test_doc in support_files.test_documentation:
            file = GeneratedFile(
                path=f"tests/docs/{test_doc.test_type}.md",
                content=test_doc.description,
                description=f"Documentation for {test_doc.test_type} tests",
                is_executable=False
            )
            self._write_file(file)
            generated_files[file.path] = file
            
        # Write additional documentation
        for name, doc in support_files.additional_docs.items():
            file = GeneratedFile(
                path=f"docs/{name}.md",
                content=doc.content,
                description=doc.title,
                is_executable=False
            )
            self._write_file(file)
            generated_files[file.path] = file
            
        # Write stage implementations
        for stage_name, stage in implementations.stages.items():
            file = GeneratedFile(
                path=f"app/flows/stages/{stage_name}.py",
                content=stage.implementation.code,
                description=f"Implementation of {stage_name} stage",
                is_executable=False
            )
            self._write_file(file)
            generated_files[file.path] = file
            
        return generated_files 