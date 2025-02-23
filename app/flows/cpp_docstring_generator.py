"""C++ docstring generator implementation."""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from flowlib import flow, stage, pipeline, managed
from flowlib.providers.llm import ModelConfig

from ..models.cpp_docstring import CppFunctionInfo, CppDocstringResult, CppFileResult
from ..config.app_config import AppConfig

logger = logging.getLogger(__name__)

@flow("cpp_docstring_generator")
@managed
class CppDocstringGenerator:
    """C++ docstring generator using flow framework."""
    
    def __init__(self):
        """Initialize generator."""
        # Load configuration
        self.config = AppConfig.load()
        
        # Create model config
        model_configs = {
            "analysis_model": ModelConfig(
                path=self.config.Provider.Models.ANALYSIS_MODEL,
                model_type=self.config.Provider.Models.MODEL_TYPE,
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
    
    async def __aenter__(self) -> 'CppDocstringGenerator':
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
        """Format the JSON schema for better LLM understanding."""
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
    
    def _extract_functions(self, file_path: str) -> Tuple[List[CppFunctionInfo], str]:
        """Extract function information from C++ file."""
        with open(file_path, 'r') as f:
            content = f.read()
            
        functions = []
        lines = content.splitlines()
        
        # Regular expressions for C++ parsing
        class_pattern = r'class\s+(\w+)'
        function_pattern = r'(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?(?:explicit\s+)?(?:const\s+)?(\w+(?:::\w+)?)\s+(\w+)\s*\((.*?)\)(?:\s*const)?(?:\s*=\s*0)?(?:\s*override)?(?:\s*final)?(?:\s*noexcept)?(?:\s*\{|\s*;)'
        
        current_class = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Track class context
            class_match = re.search(class_pattern, line)
            if class_match:
                current_class = class_match.group(1)
            elif line.startswith('};'):
                current_class = None
            
            # Look for function definitions
            # First, collect the full function signature which might span multiple lines
            if not line.startswith('//') and not line.startswith('/*'):
                full_signature = line
                j = i + 1
                while j < len(lines) and not (')' in full_signature and ('{' in full_signature or ';' in full_signature)):
                    full_signature += ' ' + lines[j].strip()
                    j += 1
                
                # Check if this is a function definition
                func_match = re.search(function_pattern, full_signature)
                if func_match:
                    # Skip if function already has a docstring
                    prev_lines = '\n'.join(lines[max(0, i-3):i]).strip()
                    if '/**' in prev_lines or '*/' in prev_lines:
                        i = j
                        continue
                    
                    return_type = func_match.group(1)
                    name = func_match.group(2)
                    params = func_match.group(3)
                    
                    # Parse parameters
                    args = []
                    if params.strip():
                        # Handle multi-line parameters
                        param_list = params.split(',')
                        for param in param_list:
                            param = param.strip()
                            if param:
                                args.append(param)
                    
                    # Get the complete function signature
                    signature_lines = lines[i:j]
                    complete_signature = '\n'.join(signature_lines)
                    
                    functions.append(CppFunctionInfo(
                        name=name,
                        args=args,
                        returns=return_type,
                        code=complete_signature,
                        line_number=i + 1,
                        is_method=bool(current_class),
                        class_name=current_class
                    ))
                    
                    i = j - 1
            
            i += 1
        
        return functions, content
    
    def _validate_docstring(self, docstring: str, function: CppFunctionInfo) -> float:
        """Validate the generated docstring and return a confidence score."""
        if not docstring or docstring.isspace():
            return 0.0
            
        score = 1.0
        required_sections = []
        
        # Check for brief description
        if '/**' not in docstring or '*/' not in docstring:
            score *= 0.5
            
        # Check for @param tags if function has arguments
        if function.args:
            param_count = len(function.args)
            found_params = len(re.findall(r'@param', docstring))
            if found_params < param_count:
                score *= (found_params / param_count)
                required_sections.append('@param')
            
        # Check for @return tag if function has a return type
        if function.returns and function.returns != 'void' and '@return' not in docstring:
            score *= 0.7
            required_sections.append('@return')
            
        # Check for generic/placeholder text
        generic_phrases = [
            "Generated docstring",
            "This is a docstring",
            "This function",
            "A function that"
        ]
        if any(phrase in docstring for phrase in generic_phrases):
            score *= 0.5
            
        return max(0.1, min(score, 1.0))
    
    @stage(output_model=CppDocstringResult)
    async def generate_docstring(self, function: CppFunctionInfo) -> CppDocstringResult:
        """Generate docstring for a single function."""
        # Get and format the schema
        schema = CppDocstringResult.model_json_schema()
        formatted_schema = self._format_schema(schema)
        
        # Format the prompt
        formatted_prompt = self._load_prompt("cpp_docstring_generation.txt").format(
            function_name=function.name,
            class_name=function.class_name or "",
            args=function.args,
            returns=function.returns,
            code=function.code,
            schema=formatted_schema
        )
        
        # Generate docstring
        result = await self.provider.generate_structured(
            prompt=formatted_prompt,
            model_name=self.config.Flow.MODEL_NAME,
            response_model=CppDocstringResult,
            max_tokens=self.config.Flow.Generation.MAX_TOKENS,
            temperature=self.config.Flow.Generation.TEMPERATURE,
            top_p=self.config.Flow.Generation.TOP_P,
            top_k=self.config.Flow.Generation.TOP_K,
            repeat_penalty=self.config.Flow.Generation.REPEAT_PENALTY
        )
        
        # Adjust confidence based on validation
        result.confidence = self._validate_docstring(
            result.docstring,
            function
        )
        
        return result
    
    def _update_file_content(self, content: str, functions: List[CppFunctionInfo], docstrings: List[CppDocstringResult]) -> str:
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
            
            # Clean up the docstring: replace escaped newlines and normalize sections
            clean_docstring = (
                doc_result.docstring
                .replace('\\n', '\n')  # Replace escaped newlines with actual newlines
                .replace('\\t', '\t')  # Replace escaped tabs
                .replace('\\r', '\r')  # Replace escaped carriage returns
                .replace('\\"', '"')   # Replace escaped quotes
                .replace('\\\\', '\\') # Replace escaped backslashes
                .replace('\n\n', '\n') # Normalize multiple newlines
            )
            
            # Format docstring with proper indentation
            docstring_lines = clean_docstring.split('\n')
            formatted_lines = []
            
            for i, line in enumerate(docstring_lines):
                line = line.rstrip()  # Remove trailing whitespace
                if i == 0:  # First line with /**
                    formatted_lines.append(' ' * indent + line.strip())
                else:  # Subsequent lines
                    if line.strip():  # Only add asterisk for non-empty lines
                        formatted_lines.append(' ' * indent + ' * ' + line.strip().lstrip('* '))
                    else:
                        formatted_lines.append(' ' * indent + ' *')  # Empty line in comment
            
            # Ensure proper comment closure
            if formatted_lines and not formatted_lines[-1].rstrip().endswith('*/'):
                formatted_lines.append(' ' * indent + ' */')
            
            # Insert docstring before function
            for line in reversed(formatted_lines):
                lines.insert(def_line, line)
            
            # Add empty line after docstring for readability
            lines.insert(def_line + len(formatted_lines), '')
        
        return '\n'.join(lines)
    
    @pipeline(output_model=CppFileResult)
    async def process_file(self, file_path: str, update_file: bool = False) -> CppFileResult:
        """Process a single C++ file.
        
        Args:
            file_path: Path to the C++ file
            update_file: Whether to update the source file with generated docstrings
            
        Returns:
            Processing result for the file
        """
        # Extract functions and get original content
        functions, content = self._extract_functions(file_path)
        
        # Skip if no functions need docstrings
        if not functions:
            return CppFileResult(
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
        
        return CppFileResult(
            filepath=file_path,
            functions=results,
            requires_review=requires_review,
            review_comments=review_comments
        ) 