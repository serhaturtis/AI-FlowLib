"""LlamaCpp provider implementation for local language models.

This module implements a provider for local language models using the llama-cpp-python
library, which provides efficient inference on consumer hardware.
"""

import logging
import os
import json
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
from pydantic import BaseModel, Field, ValidationError, validate_arguments

from ...core.errors import ProviderError, ErrorContext
from ...core.registry.decorators import provider
from ...core.models.settings import LLMProviderSettings
from ...core.registry.constants import ProviderType
from .base import LLMProvider, ModelType

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
        
    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        """Generate text completion.
        
        Args:
            prompt: Input prompt
            model_name: Name of the model to use
            **kwargs: Generation parameters
            
        Returns:
            Generated text
            
        Raises:
            ProviderError: If generation fails
        """
        # Make sure model is initialized
        if model_name not in self._models:
            await self._initialize_model(model_name)
            
        model_data = self._models[model_name]
        model = model_data["model"]
        model_config = model_data["config"]
            
        try:
            # Get generation parameters with defaults
            max_tokens = kwargs.get("max_tokens", getattr(model_config, "max_tokens", 512))
            temperature = kwargs.get("temperature", getattr(model_config, "temperature", 0.7))
            top_p = kwargs.get("top_p", getattr(model_config, "top_p", 0.9))
            top_k = kwargs.get("top_k", getattr(model_config, "top_k", 40))
            repeat_penalty = kwargs.get("repeat_penalty", getattr(model_config, "repeat_penalty", 1.1))
            
            # Run generation
            result = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=kwargs.get("stop", [])
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
            raise ProviderError(
                message=f"Generation failed: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    model_name=model_name,
                    prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt
                ),
                cause=e
            ) 
            
    async def generate_structured(self, prompt: str, output_type: Type[ModelType], model_name: str, **kwargs) -> ModelType:
        """Generate structured output using a JSON grammar.
        
        Args:
            prompt: Input prompt
            output_type: Pydantic model for output validation
            model_name: Name of the model to use
            **kwargs: Generation parameters
            
        Returns:
            Validated response model instance
            
        Raises:
            ProviderError: If generation fails
        """
        # Make sure model is initialized
        if model_name not in self._models:
            await self._initialize_model(model_name)
            
        model_data = self._models[model_name]
        model = model_data["model"]
        model_config = model_data["config"]
        model_type = model_data["type"]
            
        try:
            # Format the prompt according to model type
            formatted_prompt = self._format_prompt(prompt, model_type)
            
            # DEBUG: Print the formatted prompt after model-specific formatting
            print("\n===== MODEL-FORMATTED PROMPT (TO BE SENT TO LLM) =====")
            # Print a shortened version to avoid overwhelming the console
            if len(formatted_prompt) > 2000:
                print(formatted_prompt[:1000] + "\n[...]\n" + formatted_prompt[-1000:])
            else:
                print(formatted_prompt)
            print("===== END MODEL-FORMATTED PROMPT =====\n")
            
            # Get generation parameters with defaults
            max_tokens = kwargs.get("max_tokens", getattr(model_config, "max_tokens", 1024))
            temperature = kwargs.get("temperature", getattr(model_config, "temperature", 0.2))
            top_p = kwargs.get("top_p", getattr(model_config, "top_p", 0.95))
            top_k = kwargs.get("top_k", getattr(model_config, "top_k", 40))
            repeat_penalty = kwargs.get("repeat_penalty", getattr(model_config, "repeat_penalty", 1.1))
            
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
            
            # Create grammar from response model schema
            try:
                from llama_cpp import LlamaGrammar
                
                # Get schema from model
                if hasattr(output_type, "model_json_schema"):
                    # Pydantic v2
                    schema = output_type.model_json_schema()
                else:
                    # Pydantic v1
                    schema = output_type.schema()
                
                # Create grammar from schema
                schema_str = json.dumps(schema)
                grammar = LlamaGrammar.from_json_schema(schema_str)
                
                # Generate with grammar
                result = model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    grammar=grammar
                )
            except (ImportError, AttributeError) as e:
                # Fall back to regular generation if grammar not supported
                logger.warning(f"Grammar-based generation not available: {str(e)}. Falling back to standard generation.")
                result = model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty
                )
            
            # Extract generated text
            if isinstance(result, dict) and "choices" in result:
                generated_text = result["choices"][0]["text"]
            elif isinstance(result, list) and len(result) > 0:
                generated_text = result[0]["text"] if "text" in result[0] else ""
            else:
                generated_text = str(result)
                
            logger.info(f"Generated text: {generated_text[:200]}...")
            
            # Parse and validate response
            try:
                # Debug prints for JSON parsing
                print("\n===== PARSING LLM RESPONSE =====")
                print(f"Raw generated text (first 300 chars):\n{generated_text[:300]}")
                
                # Extract JSON from the text if necessary
                if not generated_text.strip().startswith('{') and not generated_text.strip().startswith('['):
                    # Try to extract JSON from the text
                    import re
                    print("Text doesn't start with { or [ - attempting to extract JSON with regex")
                    json_match = re.search(r'(\{.*\}|\[.*\])', generated_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        print(f"JSON extracted with regex (first 300 chars):\n{json_str[:300]}")
                    else:
                        print("Failed to extract JSON with regex")
                        raise ProviderError(
                            message="Failed to extract JSON from response",
                            provider_name=self.name,
                            context=ErrorContext.create(
                                response_text=generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                            )
                        )
                else:
                    json_str = generated_text
                    print("Text appears to be JSON format already")
                
                # Parse JSON with lenient decoder
                print("Attempting to parse JSON with lenient decoder")
                try:
                    decoder = json.JSONDecoder(strict=False)
                    parsed_response = decoder.decode(json_str)
                    print("JSON parsing successful")
                    print(f"Parsed response type: {type(parsed_response)}")
                    if isinstance(parsed_response, dict):
                        print(f"Keys in parsed response: {list(parsed_response.keys())}")
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error: {str(json_err)}")
                    print(f"Error position: {json_err.pos}")
                    print(f"Error line/col: Line {json_err.lineno}, Column {json_err.colno}")
                    print(f"Document context: {json_str[max(0, json_err.pos-50):json_err.pos+50]}")
                    raise ProviderError(
                        message=f"Failed to parse JSON response: {str(json_err)}",
                        provider_name=self.name,
                        context=ErrorContext.create(
                            response_text=generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                            error=str(json_err)
                        ),
                        cause=json_err
                    )
                
                # Sanitize strings in the parsed data
                print("Sanitizing strings in parsed data")
                def sanitize_strings(obj):
                    if isinstance(obj, str):
                        return obj.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    elif isinstance(obj, dict):
                        return {k: sanitize_strings(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [sanitize_strings(item) for item in obj]
                    return obj
                
                sanitized_data = sanitize_strings(parsed_response)
                
                # Validate with response model
                print(f"Validating against model: {output_type.__name__}")
                try:
                    if hasattr(output_type, "model_validate"):
                        # Pydantic v2
                        print("Using Pydantic v2 model_validate")
                        result = output_type.model_validate(sanitized_data)
                    else:
                        # Pydantic v1
                        print("Using Pydantic v1 parse_obj")
                        result = output_type.parse_obj(sanitized_data)
                    
                    print("Model validation successful")
                    print("===== PARSING COMPLETE =====\n")
                    return result
                    
                except Exception as validation_err:
                    print(f"Validation error: {str(validation_err)}")
                    if hasattr(validation_err, '__cause__') and validation_err.__cause__:
                        print(f"Cause: {validation_err.__cause__}")
                    raise ProviderError(
                        message=f"Failed to validate response: {str(validation_err)}",
                        provider_name=self.name,
                        context=ErrorContext.create(
                            error=str(validation_err),
                            error_type=type(validation_err).__name__,
                            response=generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                        ),
                        cause=validation_err
                    )
                    
            except Exception as e:
                if not isinstance(e, ProviderError):
                    raise ProviderError(
                        message=f"Structured generation failed: {str(e)}",
                        provider_name=self.name,
                        context=ErrorContext.create(
                            prompt_length=len(prompt),
                            model_name=model_name,
                            model_type=model_type,
                            error=str(e),
                            error_type=type(e).__name__
                        ),
                        cause=e
                    )
                raise
                
        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError(
                    message=f"Structured generation failed: {str(e)}",
                    provider_name=self.name,
                    context=ErrorContext.create(
                        prompt_length=len(prompt),
                        model_name=model_name,
                        model_type=model_type,
                        error=str(e),
                        error_type=type(e).__name__
                    ),
                    cause=e
                )
            raise
            
    def _format_prompt(self, prompt: str, model_type: str) -> str:
        """Format a prompt according to model-specific requirements.
        
        Args:
            prompt: The main prompt text
            model_type: The type/name of the model
            
        Returns:
            Formatted prompt string
        """
        # Model-specific formatting templates
        templates = {
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
        
        # Get template for model type or use default
        template = templates.get(model_type.lower(), templates["default"])
        
        # Apply template formatting
        formatted = template["pre_prompt"] + prompt + template["post_prompt"]
        
        return formatted
        
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
        
    def format_template(self, template: str, kwargs: Dict[str, Any]) -> str:
        """Format a template with variables.
        
        Replaces {{variable}} in the template with the corresponding value.
        
        Args:
            template: Template string with {{variable}} placeholders
            kwargs: Dict containing variables and their values
            
        Returns:
            Formatted template string
        """
        if "variables" in kwargs and isinstance(kwargs["variables"], dict):
            variables = kwargs["variables"]
            result = template
            
            # Replace {{variable}} with corresponding value
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                result = result.replace(placeholder, str(value))
                
            return result
        
        # No variables provided, return template as-is
        return template 