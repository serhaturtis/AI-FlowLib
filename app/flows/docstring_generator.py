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
from ..config.analyzer_config import AnalyzerConfig

logger = logging.getLogger(__name__)

@flow("docstring_generator")
@managed
class DocstringGenerator:
    """Docstring generator using flow framework."""
    
    def __init__(self):
        """Initialize generator."""
        # Load configuration
        self.config = AnalyzerConfig.load()
        
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
        
        # Try to get comments before the function/decorators
        start_line = max(0, first_decorator_line - 3)
        context_before = []
        for line in source_lines[start_line:first_decorator_line - 1]:
            line = line.strip()
            if line and (line.startswith('#') or line.startswith('"""') or line.startswith("'''")):
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
    
    def _extract_functions(self, file_path: str) -> Tuple[List[FunctionInfo], str]:
        """Extract function information from Python file."""
        logger.debug(f"Extracting functions from {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Get source lines for context extraction
        source_lines = content.splitlines()
        logger.debug(f"File has {len(source_lines)} lines")
        
        functions = []
        tree = ast.parse(content)
        
        # First, collect imports and top-level definitions
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
        logger.debug(f"Found {len(imports)} imports")
        
        # Helper to find parent class
        def find_parent_class(node):
            parent = getattr(node, 'parent', None)
            while parent and not isinstance(parent, ast.ClassDef):
                parent = getattr(parent, 'parent', None)
            return parent
        
        # Then process functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    logger.debug(f"Processing function: {node.name} at line {node.lineno}")
                    
                    # Skip if function already has a docstring
                    if ast.get_docstring(node):
                        logger.debug(f"Skipping {node.name} - already has docstring")
                        continue
                        
                    args = [arg.arg for arg in node.args.args]
                    returns = None
                    if node.returns:
                        returns = ast.unparse(node.returns)
                    
                    # Get function code with context
                    code = self._get_function_context(node, source_lines)
                    logger.debug(f"Got context for {node.name}:\n{code}")
                    
                    # For class methods, add class context
                    parent_class = find_parent_class(node)
                    if parent_class:
                        logger.debug(f"Found parent class for {node.name}: {parent_class.name}")
                        # Get class definition line
                        class_def = source_lines[parent_class.lineno - 1].rstrip()
                        # Add class context with proper indentation
                        class_lines = [class_def + ':']
                        # Add an empty line after class definition for better readability
                        class_lines.append('')
                        # Indent the function code
                        indented_code = '\n'.join(f"    {line}" for line in code.splitlines())
                        code = '\n'.join(class_lines + [indented_code])
                    
                    # Add relevant imports at the top if they're used in the function
                    func_code = ast.unparse(node)
                    relevant_imports = []
                    for imp in imports:
                        try:
                            # Extract imported names more accurately
                            if ' import ' not in imp:
                                continue  # Skip malformed imports
                            
                            if imp.startswith('from '):
                                # Handle 'from module import name1, name2'
                                module_part, names_part = imp.split(' import ', 1)
                                names = [n.strip().split(' as ')[0] for n in names_part.split(',')]
                            else:
                                # Handle 'import module' or 'import module as alias'
                                module_names = imp.split(' import ', 1)[1]
                                names = [n.strip().split(' as ')[0] for n in module_names.split(',')]
                            
                            # Check if any of the imported names are used in the function
                            if any(name in func_code for name in names):
                                relevant_imports.append(imp)
                        except Exception as e:
                            logger.debug(f"Skipping malformed import '{imp}': {str(e)}")
                            continue
                    
                    if relevant_imports:
                        logger.debug(f"Adding {len(relevant_imports)} relevant imports for {node.name}")
                        code = '\n'.join(relevant_imports) + '\n\n' + code
                    
                    functions.append(FunctionInfo(
                        name=node.name,
                        args=args,
                        returns=returns,
                        code=code,
                        line_number=node.lineno
                    ))
                    logger.debug(f"Successfully processed function: {node.name}")
                except Exception as e:
                    logger.error(f"Error processing function {node.name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
        
        logger.debug(f"Found {len(functions)} functions to process")
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
        logger.info(json.dumps(result, indent=2))
        logger.info("=" * 80)
        
        # Validate and potentially adjust confidence
        validated_result = DocstringResult.model_validate(result)
        validated_result.confidence = self._validate_docstring(
            validated_result.docstring,
            function
        )
        
        return validated_result
    
    def _update_file_content(self, content: str, functions: List[FunctionInfo], docstrings: List[DocstringResult]) -> str:
        """Update file content with generated docstrings.
        
        Args:
            content: Original file content
            functions: List of extracted functions
            docstrings: List of generated docstrings
            
        Returns:
            Updated file content with new docstrings
        """
        if not functions:  # No functions to update
            return content
            
        # Convert content to lines for easier manipulation
        lines = content.splitlines()
        
        # Sort functions by line number in reverse order to avoid
        # line number changes affecting subsequent insertions
        updates = list(zip(functions, docstrings))
        updates.sort(key=lambda x: x[0].line_number, reverse=True)
        
        for func, doc_result in updates:
            # Skip if docstring is empty or just whitespace
            if not doc_result.docstring or doc_result.docstring.isspace():
                continue
                
            # Find the line after function definition
            def_line = func.line_number - 1  # Convert to 0-based index
            
            # Skip if there's already a docstring
            next_lines = lines[def_line:def_line + 3]
            if any('"""' in line or "'''" in line for line in next_lines):
                continue
                
            # Get proper indentation
            indent = len(lines[def_line]) - len(lines[def_line].lstrip())
            
            # Format docstring with proper indentation
            docstring_lines = [' ' * indent + '"""' + doc_result.docstring.split('\n')[0] + '"""']
            if '\n' in doc_result.docstring:
                docstring_lines = [
                    ' ' * indent + '"""' + doc_result.docstring.split('\n')[0],
                    *[' ' * indent + line for line in doc_result.docstring.split('\n')[1:-1]],
                    ' ' * indent + '"""'
                ]
            
            # Insert docstring after function definition
            lines[def_line:def_line] = [''] + docstring_lines + ['']
        
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