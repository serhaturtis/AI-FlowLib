"""LlamaCpp provider implementation for local language models.

This module implements a provider for local language models using the llama-cpp-python
library, which provides efficient inference on consumer hardware.
"""

import logging
import inspect
import os
import json
from typing import Any, Dict, Optional, Type
from llama_cpp import LlamaGrammar

from ...core.errors import ProviderError, ErrorContext
from ..decorators import provider
from ..constants import ProviderType
from .base import LLMProvider, ModelType
from .base import LLMProviderSettings
from ...resources.decorators import PromptTemplate

logger = logging.getLogger(__name__)


class LlamaCppSettings(LLMProviderSettings):
    """Settings for the LlamaCpp provider.
    
    Attributes:
        n_ctx: Context length in tokens
        n_threads: Number of CPU threads to use
        n_batch: Batch size for processing
        use_gpu: Whether to use GPU acceleration
        n_gpu_layers: Number of layers to offload to GPU
        chat_format: Chat format template to use (default: None)
        verbose: Whether to enable verbose logging
    """
    
    n_ctx: int = 2048
    n_threads: int = 4
    n_batch: int = 512
    use_gpu: bool = False
    n_gpu_layers: int = 0
    chat_format: Optional[str] = None
    verbose: bool = False


@provider(provider_type=ProviderType.LLM, name="llamacpp")
class LlamaCppProvider(LLMProvider[LlamaCppSettings]):
    """Provider for local inference using llama-cpp-python.
    
    This provider supports:
    1. Text generation with various LLM architectures (llama, phi, mistral, etc.)
    2. Structured output generation with format guidance
    3. Optional GPU acceleration with Metal or CUDA
    """
    
    def __init__(self, name: str = "llamacpp", settings: Optional[LlamaCppSettings] = None, **kwargs):
        """Initialize LlamaCpp provider.
        
        Args:
            name: Unique provider name
            settings: Provider settings
            **kwargs: Additional settings as keyword arguments
        """
        # Create settings first to avoid issues with _default_settings() method
        settings = settings or LlamaCppSettings()
        
        # Initialize parent with provider_type="llm"
        super().__init__(name=name, settings=settings)
        
        # Store settings for local use
        self._models = {}
        self._settings = settings
        
        # Update settings with any provided kwargs
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self._settings, key):
                    setattr(self._settings, key, value)
            
    async def _initialize_model(self, model_name: str):
        """Initialize a specific model.
        
        Args:
            model_name: Name of the model to initialize
            
        Raises:
            ProviderError: If initialization fails
        """
        if model_name in self._models:
            return
            
        try:
            # Import here to avoid requiring llama-cpp-python for all users
            from llama_cpp import Llama
            
            # Get model configuration from registry
            model_config = await self.get_model_config(model_name)
            
            # Check if model_config is a dictionary or object
            if isinstance(model_config, dict):
                model_path = model_config.get('path')
                model_type = model_config.get('model_type', 'default')
                n_ctx = model_config.get('n_ctx', self._settings.n_ctx)
                n_threads = model_config.get('n_threads', self._settings.n_threads)
                n_batch = model_config.get('n_batch', self._settings.n_batch)
                use_gpu = model_config.get('use_gpu', self._settings.use_gpu)
                n_gpu_layers = model_config.get('n_gpu_layers', self._settings.n_gpu_layers)
                verbose = model_config.get('verbose', self._settings.verbose)
            else:
                # Handle object with attributes
                model_path = getattr(model_config, 'path', None)
                model_type = getattr(model_config, 'model_type', 'default')
                n_ctx = getattr(model_config, 'n_ctx', self._settings.n_ctx)
                n_threads = getattr(model_config, 'n_threads', self._settings.n_threads)
                n_batch = getattr(model_config, 'n_batch', self._settings.n_batch)
                use_gpu = getattr(model_config, 'use_gpu', self._settings.use_gpu)
                n_gpu_layers = getattr(model_config, 'n_gpu_layers', self._settings.n_gpu_layers)
                verbose = getattr(model_config, 'verbose', self._settings.verbose)
            
            # Check if model path exists
            if not model_path or not os.path.exists(model_path):
                raise ProviderError(
                    message=f"Model path does not exist: {model_path}",
                    provider_name=self.name,
                    context=ErrorContext.create(model_name=model_name)
                )
                
            # Load model with specified settings
            logger.info(f"Loading LlamaCpp model '{model_name}' from: {model_path}")
            model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers if use_gpu else 0,
                verbose=verbose
            )
            
            # Store model and its configuration
            self._models[model_name] = {
                "model": model,
                "config": model_config,
                "type": model_type
            }
            
            logger.info(f"Loaded LlamaCpp model: {model_name} ({os.path.basename(model_path)})")
            
        except ImportError as e:
            raise ProviderError(
                message="llama-cpp-python package not installed",
                provider_name=self.name,
                context=ErrorContext.create(
                    help="Install with: pip install llama-cpp-python"
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to initialize LlamaCpp model '{model_name}': {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    model_name=model_name
                ),
                cause=e
            )
            
    async def initialize(self):
        """Initialize the provider."""
        self._initialized = True
        
    async def _initialize(self):
        """Initialize the provider.
        
        This implements the required abstract method from the Provider base class.
        The actual model initialization is done lazily in _initialize_model when needed.
        """
        # The LlamaCpp provider uses lazy initialization of models
        # when they are first requested, so there's no work to do here
        pass
        
    async def shutdown(self):
        """Release model resources."""
        for model_name, model_data in self._models.items():
            logger.info(f"Released LlamaCpp model: {model_name}")
        
        self._models = {}
        self._initialized = False
        
    async def generate(self, prompt: PromptTemplate, model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> str:
        """Generate text completion.
        
        Args:
            prompt: Prompt template object with template and config attributes
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Generated text
            
        Raises:
            ProviderError: If generation fails
            TypeError: If prompt is not a valid template object
        """
        # Make sure model is initialized
        if model_name not in self._models:
            await self._initialize_model(model_name)
            
        model_data = self._models[model_name]
        model = model_data["model"]
        model_config = model_data["config"]
            
        try:
            # Validate prompt is a template object with template attribute
            if not hasattr(prompt, 'template'):
                raise TypeError(f"prompt must be a template object with 'template' attribute, got {type(prompt).__name__}")
            
            # Get template string from prompt
            template_str = prompt.template
            
            # Get config from prompt object if available
            if hasattr(prompt, 'config') and prompt.config:
                # Override model_config with ALL values from prompt.config
                for param_name, param_value in prompt.config.items():
                    setattr(model_config, param_name, param_value)
            
            # Format prompt with variables if provided
            formatted_prompt = template_str
            if prompt_variables:
                # Format the template with variables
                formatted_prompt = self.format_template(template_str, {"variables": prompt_variables})
            
            # Get generation parameters from model config
            max_tokens = getattr(model_config, "max_tokens", 512)
            temperature = getattr(model_config, "temperature", 0.7)
            top_p = getattr(model_config, "top_p", 0.9)
            top_k = getattr(model_config, "top_k", 40)
            repeat_penalty = getattr(model_config, "repeat_penalty", 1.1)
            
            # Run generation
            result = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=[]
            )
            
            # Extract generated text
            if isinstance(result, dict) and "choices" in result:
                # Extract from completion format
                return result["choices"][0]["text"]
            elif isinstance(result, list) and len(result) > 0:
                # Extract from list format
                return result[0]["text"] if "text" in result[0] else ""
            else:
                # Handle unexpected response format
                return str(result)
                
        except Exception as e:
            if isinstance(e, TypeError) and ("must be a template object" in str(e) or "output_type must be a class" in str(e)):
                # Re-raise TypeError for invalid prompt or output_type
                raise
            raise ProviderError(
                message=f"Generation failed: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    model_name=model_name,
                    prompt_length=len(str(prompt)) if prompt else 0
                ),
                cause=e
            )
            
    async def generate_structured(self, prompt: PromptTemplate, output_type: Type[ModelType], model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> ModelType:
        """Generate structured output using a JSON grammar.
        
        Args:
            prompt: Prompt template object with template and config attributes
            output_type: Pydantic model for output validation
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Validated response model instance
            
        Raises:
            ProviderError: If generation fails
            TypeError: If prompt is not a valid template object or if output_type is not a class
        """
        # Ensure output_type is a class, not an instance
        if not inspect.isclass(output_type):
            raise TypeError(f"output_type must be a class, not an instance of {type(output_type)}")
        
        # Make sure model is initialized
        if model_name not in self._models:
            await self._initialize_model(model_name)
        
        model_data = self._models[model_name]
        model = model_data["model"]
        model_config = model_data["config"]
        model_type = model_data["type"]
        
        # Validate prompt is a template object with template attribute
        if not hasattr(prompt, 'template'):
            raise TypeError(f"prompt must be a template object with 'template' attribute, got {type(prompt).__name__}")
        
        # Get template string from prompt
        template_str = prompt.template
        
        # Get config from prompt object if available
        if hasattr(prompt, 'config') and prompt.config:
            # Override model_config with ALL values from prompt.config
            for param_name, param_value in prompt.config.items():
                setattr(model_config, param_name, param_value)
        
        # Format prompt with variables if provided
        formatted_prompt_text = template_str
        if prompt_variables:
            # Format the template with variables
            formatted_prompt_text = self.format_template(template_str, {"variables": prompt_variables})
        
        # Format the prompt according to model type
        formatted_prompt = self._format_prompt(formatted_prompt_text, model_type, output_type)
        
        # Get generation parameters from model config
        max_tokens = getattr(model_config, "max_tokens", 1024)
        temperature = getattr(model_config, "temperature", 0.2)
        top_p = getattr(model_config, "top_p", 0.95)
        top_k = getattr(model_config, "top_k", 40)
        repeat_penalty = getattr(model_config, "repeat_penalty", 1.1)
        
        # Log generation parameters
        logger.info("Starting LLM structured generation with parameters:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Model Type: {model_type}")
        logger.info(f"  Response Model: {output_type.__name__}")
        logger.info("  Generation Parameters:")
        for name, value in {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }.items():
            logger.info(f"    {name}: {value}")

        # Get schema from model
        try:
            schema = output_type.model_json_schema()
        except AttributeError as e:
            raise ProviderError(
                message=f"Cannot generate structured output: {str(e)}. Model type does not support schema generation.",
                provider_name=self.name,
                context=ErrorContext.create(
                    input_text=formatted_prompt[:100],
                    model_type=output_type.__name__ if hasattr(output_type, "__name__") else str(output_type)
                )
            ) from e
        
        # Create grammar from schema
        schema_str = json.dumps(schema)
        try:
            grammar = LlamaGrammar.from_json_schema(schema_str)
        except Exception as e:
            raise ProviderError(
                message=f"Failed to create grammar from schema: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    schema=schema_str
                )
            ) from e
        
        # Log the formatted prompt
        logger.info("=============== FORMATTED PROMPT ===============")
        logger.info(formatted_prompt)
        logger.info("================================================")
        
        # Generate with grammar
        try:
            result = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                grammar=grammar
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to generate with grammar: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    input_text=formatted_prompt[:100]
                )
            ) from e
        
        # Extract generated text
        if isinstance(result, dict) and "choices" in result:
            generated_text = result["choices"][0]["text"]
        elif isinstance(result, list) and len(result) > 0:
            generated_text = result[0]["text"] if "text" in result[0] else str(result[0])
        else:
            generated_text = str(result)
        
        logger.info(f"Generated text: {generated_text[:200]}...")
        
        # Parse JSON directly - no fallback or extraction attempts
        try:
            parsed_data = json.loads(generated_text)
        except json.JSONDecodeError as e:
            raise ProviderError(
                message=f"Failed to parse JSON from response: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    response_text=generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                )
            ) from e
        
        # Sanitize strings in parsed data
        sanitized_data = self._sanitize_strings(parsed_data)
        
        # Validate against target model
        try:
            # Create an instance of the output_type class using the parsed data
            validated_response = output_type.model_validate(sanitized_data)
            return validated_response
        except Exception as e:
            raise ProviderError(
                message=f"Failed to validate response against model {output_type.__name__}: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    response_text=json.dumps(sanitized_data)[:200] + "..." if len(json.dumps(sanitized_data)) > 200 else json.dumps(sanitized_data)
                )
            ) from e
            
    def _format_prompt(self, prompt: str, model_type: str = "default", output_type: Optional[Type[ModelType]] = None) -> str:
        """Format a prompt according to model-specific requirements.
        
        Applies model-specific formatting templates.
        
        Args:
            prompt: The main prompt text
            model_type: The type/name of the model
            output_type: Optional Pydantic model type for structured output
            
        Returns:
            Formatted prompt string
        """
        # First, let the base class potentially add JSON structure information
        prompt_with_json = super()._format_prompt(prompt, model_type, output_type)
        
        # Get template for model type or use default
        templates = self._get_model_templates()
        template = templates.get(model_type.lower(), templates["default"])
        
        # Apply template formatting
        formatted = template["pre_prompt"] + prompt_with_json + template["post_prompt"]
        
        return formatted
        
    def _get_model_templates(self) -> Dict[str, Dict[str, str]]:
        """Get model-specific prompt templates for different LLM architectures.
        
        Returns:
            Dictionary mapping model_type to pre/post prompt templates
        """
        return {
            "default": {
                "pre_prompt": "",
                "post_prompt": ""
            },
            "llama2": {
                "pre_prompt": "<s>[INST] ",
                "post_prompt": " [/INST]"
            },
            "phi2": {
                "pre_prompt": "Instruct: ",
                "post_prompt": "\nOutput: "
            },
            "phi4": {
                "pre_prompt": "<|im_start|>user<|im_sep|>",
                "post_prompt": "<|im_end|><|im_start|>assistant<|im_sep|>"
            },
            "mistral": {
                "pre_prompt": "<|user|>\n",
                "post_prompt": "\n</s><|assistant|>\n"
            },
            "chatml": {
                "pre_prompt": "<|im_start|>user\n",
                "post_prompt": "\n<|im_end|><|im_start|>assistant\n"
            }
        }
        
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string or empty string if none found
        """
        # Look for JSON object in the response
        import re
        
        # Try to find a JSON object using regex
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            potential_json = json_match.group(0)
            try:
                # Validate it's actually JSON
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON object found, try the whole text
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
            
        return ""

    def _sanitize_strings(self, obj):
        """Sanitize strings in parsed data to ensure consistent formatting.
        
        Args:
            obj: Object to sanitize (dict, list, string, or primitive)
            
        Returns:
            Sanitized object with special characters in strings properly escaped
        """
        if isinstance(obj, str):
            # Remove any special formatting characters that might cause issues
            return obj.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        elif isinstance(obj, dict):
            return {k: self._sanitize_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_strings(item) for item in obj]
        else:
            # Return other types unchanged
            return obj 