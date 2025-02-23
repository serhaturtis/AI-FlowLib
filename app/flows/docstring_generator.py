"""Docstring generator implementation."""

import ast
import json
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from flowlib import flow, stage, pipeline, managed
from flowlib.providers.llm import ModelConfig

from ..models.docstring import FunctionInfo, DocstringResult, FileResult
from ..config.app_config import AppConfig

logger = logging.getLogger(__name__)

@flow("docstring_generator")
@managed
class DocstringGenerator:
    """Docstring generator using flow framework."""
    
    def __init__(self):
        """Initialize generator."""
        # Load configuration
        self.config = AppConfig.load()
        
        # Create model config
        model_configs = {
            "analysis_model": ModelConfig(
                path=self.config.Provider.Models.ANALYSIS_MODEL,
                n_ctx=self.config.Provider.Models.N_CTX,
                n_threads=self.config.Provider.Models.N_THREADS,
                n_batch=self.config.Provider.Models.N_BATCH,
                use_gpu=self.config.Provider.Models.USE_GPU
            )
        }
        
        # Setup provider
        self.provider = managed.factory.llm(
            name=self.config.Provider.NAME,
            model_configs=model_configs,
            max_models=self.config.Provider.MAX_MODELS
        )
    
    async def __aenter__(self) -> 'DocstringGenerator':
        """Async context manager entry."""
        await self.provider.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.provider.cleanup()
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        prompt_path = Path("./prompts") / filename
        with open(prompt_path) as f:
            return f.read()
    
    def _format_schema(self, schema: Dict) -> str:
        """Format the JSON schema for better LLM understanding.
        
        Args:
            schema: The raw JSON schema dictionary
            
        Returns:
            A formatted string representation of the schema
        """
        # Extract only the relevant parts of the schema
        properties = schema.get("properties", {})
        formatted_props = {}
        
        for prop, details in properties.items():
            formatted_props[prop] = {
                "type": details.get("type"),
                "description": details.get("description")
            }
            if "minimum" in details:
                formatted_props[prop]["minimum"] = details["minimum"]
            if "maximum" in details:
                formatted_props[prop]["maximum"] = details["maximum"]
        
        simplified_schema = {
            "type": "object",
            "properties": formatted_props,
            "required": schema.get("required", [])
        }
        
        return json.dumps(simplified_schema, indent=2)
    
    def _get_function_context(self, node: ast.FunctionDef, source_lines: List[str]) -> str:
        """Extract function code with context.
        
        Args:
            node: The AST node for the function
            source_lines: The source file lines
            
        Returns:
            Function code with context
        """
        # Get decorator lines if any
        decorator_lines = []
        first_decorator_line = node.lineno
        if node.decorator_list:
            # Find the first decorator's line number
            first_decorator_line = min(d.lineno for d in node.decorator_list)
            # Get all lines from first decorator to function def
            decorator_lines = source_lines[first_decorator_line - 1:node.lineno - 1]
        
        # Get the function definition and body
        func_lines = source_lines[node.lineno - 1:node.end_lineno]
        
        # Try to get comments (but not docstrings) before the function/decorators
        start_line = max(0, first_decorator_line - 3)
        context_before = []
        for line in source_lines[start_line:first_decorator_line - 1]:
            line = line.strip()
            # Only collect actual comments, not docstrings
            if line and line.startswith('#'):
                context_before.append(line)
        
        # Combine everything in the correct order
        full_code = []
        if context_before:
            full_code.extend(context_before)
            if context_before[-1].strip():  # Add empty line after comments if needed
                full_code.append('')
        if decorator_lines:
            full_code.extend(decorator_lines)
        full_code.extend(func_lines)
        
        # Clean up the code - remove any imports that got included
        cleaned_lines = []
        for line in full_code:
            # Skip import statements that might have been included
            if line.strip().startswith(('import ', 'from ')):
                continue
            cleaned_lines.append(line)
            
        return '\n'.join(line.rstrip() for line in cleaned_lines)

    def _has_misplaced_docstring(self, node: ast.FunctionDef, source_lines: List[str]) -> bool:
        """Check if function has a misplaced docstring (before the function definition).
        
        Args:
            node: The AST node for the function
            source_lines: The source file lines
            
        Returns:
            True if a misplaced docstring is found
        """
        # Get the actual start line, considering decorators
        start_line = node.lineno
        if node.decorator_list:
            start_line = min(d.lineno for d in node.decorator_list)
        
        # Look at lines before the function/decorator
        start_check = max(0, start_line - 5)
        for line_num in range(start_check, start_line):
            line = source_lines[line_num].strip()
            # Check for docstring markers, including malformed ones with code blocks
            if ('"""' in line or "'''" in line or 
                (line.startswith('"""') and "```" in line)):
                return True
        
        # Also check between decorators and function definition
        if node.decorator_list:
            for line_num in range(start_line, node.lineno):
                line = source_lines[line_num].strip()
                if '"""' in line or "'''" in line:
                    return True
        
        return False

    def _get_function_definition_line(self, node: ast.FunctionDef, source_lines: List[str]) -> int:
        """Get the actual function definition line number.
        
        Args:
            node: The AST node for the function
            source_lines: The source file lines
            
        Returns:
            The line number where the function definition starts
        """
        # Start from the AST's reported line number
        def_line = node.lineno - 1
        
        # Look for the actual 'def' keyword
        while def_line < len(source_lines):
            if source_lines[def_line].lstrip().startswith('def '):
                break
            def_line += 1
        
        return def_line

    def _extract_functions(self, file_path: str) -> Tuple[List[FunctionInfo], str]:
        """Extract function information from Python file."""
        with open(file_path, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        source_lines = content.splitlines()
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    # Skip if function already has a docstring
                    if ast.get_docstring(node):
                        continue
                    
                    # Get function info
                    args = [arg.arg for arg in node.args.args]
                    returns = ast.unparse(node.returns) if node.returns else None
                    
                    # Find the complete function signature
                    def_line = node.lineno - 1
                    end_line = def_line
                    
                    # Find where the function body starts (after all parameters)
                    while end_line < len(source_lines):
                        line = source_lines[end_line].strip()
                        if line.endswith(':'):  # Found the end of signature
                            break
                        end_line += 1
                    
                    # Get the complete signature
                    signature_lines = source_lines[def_line:end_line + 1]
                    complete_signature = '\n'.join(signature_lines)
                    
                    functions.append(FunctionInfo(
                        name=node.name,
                        args=args,
                        returns=returns,
                        code=complete_signature,  # Use complete signature
                        line_number=def_line + 1  # Convert back to 1-based
                    ))
                    
                except Exception as e:
                    logger.error(f"Error processing function {node.name}: {str(e)}")
                    raise
        
        return functions, content
    
    def _validate_docstring(self, docstring: str, function: FunctionInfo) -> float:
        """Validate the generated docstring and return a confidence score.
        
        Args:
            docstring: The generated docstring
            function: Information about the function
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not docstring or docstring.isspace():
            return 0.0
            
        score = 1.0
        required_sections = []
        
        # Check for one-line summary
        lines = docstring.split('\n')
        if len(lines) < 1 or not lines[0].strip():
            score *= 0.5
            
        # Check for Args section if function has arguments
        if function.args and 'Args:' not in docstring:
            score *= 0.7
            required_sections.append('Args')
            
        # Check for Returns section if function has a return type
        if function.returns and 'Returns:' not in docstring:
            score *= 0.7
            required_sections.append('Returns')
            
        # Check for parameter documentation
        if function.args:
            documented_args = [
                arg for arg in function.args 
                if arg in docstring
            ]
            if len(documented_args) < len(function.args):
                score *= 0.8
                
        # Check for generic/placeholder text
        generic_phrases = [
            "Generated docstring",
            "This is a docstring",
            "This function",
            "A function that"
        ]
        if any(phrase in docstring for phrase in generic_phrases):
            score *= 0.5
            
        return max(0.1, min(score, 1.0))  # Ensure score is between 0.1 and 1.0
    
    @stage(output_model=DocstringResult)
    async def generate_docstring(self, function: FunctionInfo) -> DocstringResult:
        """Generate docstring for a single function."""
        # Get and format the schema
        schema = DocstringResult.model_json_schema()
        formatted_schema = self._format_schema(schema)
        
        # Format the prompt
        formatted_prompt = self._load_prompt("docstring_generation.txt").format(
            function_name=function.name,
            args=function.args,
            returns=function.returns,
            code=function.code,
            schema=formatted_schema
        )
        
        # Log the formatted prompt
        logger.info(f"\nGenerating docstring for function: {function.name}")
        logger.info("=" * 80)
        logger.info("Formatted Prompt:")
        logger.info("-" * 40)
        logger.info(formatted_prompt)
        logger.info("=" * 80)
        
        # Generate docstring
        result = await self.provider.generate_structured(
            prompt=formatted_prompt,
            model_name=self.config.Flow.MODEL_NAME,
            response_model=DocstringResult,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=self.config.Flow.Generation.TEMPERATURE,
            top_p=self.config.Flow.Generation.TOP_P,
            top_k=self.config.Flow.Generation.TOP_K,
            repeat_penalty=self.config.Flow.Generation.REPEAT_PENALTY
        )
        
        # Log the result
        logger.info("LLM Response:")
        logger.info("-" * 40)
        logger.info(json.dumps(result.model_dump(), indent=2))
        logger.info("=" * 80)
        
        # Adjust confidence based on validation
        result.confidence = self._validate_docstring(
            result.docstring,
            function
        )
        
        return result
    
    def _update_file_content(self, content: str, functions: List[FunctionInfo], docstrings: List[DocstringResult]) -> str:
        """Update file content with generated docstrings."""
        if not functions:
            return content
            
        lines = content.splitlines()
        updates = list(zip(functions, docstrings))
        updates.sort(key=lambda x: x[0].line_number, reverse=True)
        
        for func, doc_result in updates:
            if not doc_result.docstring or doc_result.docstring.isspace():
                continue
                
            # Get function line and indentation
            def_line = func.line_number - 1
            indent = len(lines[def_line]) - len(lines[def_line].lstrip())
            
            # Find the end of function definition (where the colon is)
            end_def_line = def_line
            while end_def_line < len(lines):
                if lines[end_def_line].rstrip().endswith(':'):
                    break
                end_def_line += 1
            
            # Format docstring with proper indentation and line breaks
            docstring_lines = []
            base_indent = ' ' * (indent + 4)  # Base indentation for docstring content
            
            # Clean up the docstring: replace escaped newlines and normalize sections
            clean_docstring = (
                doc_result.docstring
                .replace('\\n', '\n')  # Replace escaped newlines with actual newlines
                .replace('\n\n', '\n')  # Normalize multiple newlines
            )
            
            # Split into sections and clean up
            sections = []
            current_section = []
            
            for line in clean_docstring.split('\n'):
                line = line.strip()
                if not line:
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    continue
                
                # Check if this is a section header
                if line.startswith(('Args:', 'Returns:', 'Raises:')):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    sections.append(line)
                    continue
                
                # Check if this is a parameter description
                if ': ' in line and any(line.startswith(param + ' ') for param in func.args):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    sections.append('    ' + line)  # Extra indent for params
                    continue
                
                current_section.append(line)
            
            if current_section:
                sections.append('\n'.join(current_section))
            
            # Format the final docstring
            docstring_lines = [base_indent + '"""']
            
            # Add each section with proper spacing
            for i, section in enumerate(sections):
                if i > 0:  # Add blank line between sections
                    docstring_lines.append('')
                
                if section.startswith(('Args:', 'Returns:', 'Raises:')):
                    docstring_lines.append(base_indent + section)
                elif section.startswith('    '):  # Parameter description
                    docstring_lines.append(base_indent + section)
                else:
                    # Wrap long lines
                    words = section.split()
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 > 80:  # Standard Python line length
                            docstring_lines.append(base_indent + ' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            current_line.append(word)
                            current_length += len(word) + 1
                    
                    if current_line:
                        docstring_lines.append(base_indent + ' '.join(current_line))
            
            docstring_lines.append(base_indent + '"""')
            
            # Insert docstring after the complete function definition
            lines.insert(end_def_line + 1, '')  # Empty line after def
            for line in reversed(docstring_lines):
                lines.insert(end_def_line + 1, line)
        
        return '\n'.join(lines)
    
    @pipeline(output_model=FileResult)
    async def process_file(self, file_path: str, update_file: bool = False) -> FileResult:
        """Process a single Python file.
        
        Args:
            file_path: Path to the Python file
            update_file: Whether to update the source file with generated docstrings
            
        Returns:
            Processing result for the file
        """
        # Extract functions and get original content
        functions, content = self._extract_functions(file_path)
        
        # Skip if no functions need docstrings
        if not functions:
            return FileResult(
                filepath=file_path,
                functions=[],
                requires_review=False
            )
        
        # Generate docstrings for each function
        results = []
        for func in functions:
            result = await self.generate_docstring(func)
            results.append(result)
        
        # Update file if requested and we have results
        if update_file and results:
            try:
                updated_content = self._update_file_content(content, functions, results)
                with open(file_path, 'w') as f:
                    f.write(updated_content)
            except Exception as e:
                raise RuntimeError(f"Failed to update {file_path}: {str(e)}")
        
        # Check if review is needed
        requires_review = any(r.confidence < 0.7 for r in results)
        review_comments = None
        if requires_review:
            low_confidence = [
                f"{functions[i].name} (confidence: {results[i].confidence:.2f})"
                for i in range(len(results))
                if results[i].confidence < 0.7
            ]
            review_comments = f"Low confidence for functions: {', '.join(low_confidence)}"
        
        return FileResult(
            filepath=file_path,
            functions=results,
            requires_review=requires_review,
            review_comments=review_comments
        ) 