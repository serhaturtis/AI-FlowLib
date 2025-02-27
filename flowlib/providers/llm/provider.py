from typing import Dict, Any, Type, Optional
import gc
import sys
import json
import logging
import threading
from llama_cpp import Llama, LlamaGrammar

from pydantic import BaseModel
from ...core.errors.base import ResourceError, ErrorContext
from ..base import Provider
from .utils import GPUConfigManager
from .prompt_templates import format_prompt
from ...core.resources import ResourceRegistry

from .models import ModelConfig

logger = logging.getLogger(__name__)

class LLMProvider(Provider):
    """LLM provider implementation."""
    
    # Supported generation parameters for llama-cpp
    SUPPORTED_PARAMS = {
        "max_tokens": int,
        "temperature": float,
        "top_p": float,
        "top_k": int,
        "repeat_penalty": float
    }
    
    def __init__(
        self,
        name: str,
        max_models: int = 2
    ):
        """Initialize provider.
        
        Args:
            name: Provider name
            max_models: Maximum number of models to keep loaded
        """
        self.name = name
        self.max_models = max_models
        self._models = {}
        self._model_locks = {}
        self._gpu_manager = GPUConfigManager()
    
    async def initialize(self) -> None:
        """Initialize provider."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        for model_name in list(self._models.keys()):
            self._cleanup_model(model_name)
    
    def validate_model_name(self, model_name: str) -> None:
        """Validate that a model exists.
        
        Args:
            model_name: Name of the model to validate
            
        Raises:
            ResourceError: If model configuration is not found
        """
        model_config = ResourceRegistry.get_resource('model', model_name)
        if not model_config:
            raise ResourceError(
                f"Model '{model_name}' not found. Define it using @model decorator.",
                ErrorContext.create(provider_name=self.name)
            )
    
    def _load_model(self, model_name: str) -> Llama:
        """Load a model into memory.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model instance
            
        Raises:
            ResourceError: If model loading fails
        """
        try:
            # Get model configuration
            model_config = ResourceRegistry.get_resource('model', model_name)
            if not model_config:
                raise ResourceError(
                    f"Model '{model_name}' not found",
                    ErrorContext.create(provider_name=self.name)
                )
            
            # Create model instance
            model = Llama(
                model_path=model_config.path,
                n_ctx=model_config.n_ctx,
                n_threads=model_config.n_threads,
                n_batch=model_config.n_batch,
                n_gpu_layers=model_config.n_gpu_layers if model_config.use_gpu else 0
            )
            
            # Create lock for this model
            self._model_locks[model_name] = threading.Lock()
            
            return model
            
        except Exception as e:
            raise ResourceError(
                f"Failed to load model '{model_name}': {str(e)}",
                ErrorContext.create(provider_name=self.name)
            ) from e
    
    def _cleanup_model(self, model_name: str) -> None:
        """Clean up a loaded model.
        
        Args:
            model_name: Name of the model to clean up
        """
        if model_name in self._models:
            del self._models[model_name]
            if model_name in self._model_locks:
                del self._model_locks[model_name]
            gc.collect()
    
    def get_model(self, model_name: str) -> Llama:
        """Get a loaded model instance.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            Model instance
        """
        # Load model if not already loaded
        if model_name not in self._models:
            # Clean up old models if we're at capacity
            if len(self._models) >= self.max_models:
                oldest_model = next(iter(self._models))
                self._cleanup_model(oldest_model)
            
            # Load new model
            self._models[model_name] = self._load_model(model_name)
        
        return self._models[model_name]
    
    def _validate_param_type(self, name: str, value: Any) -> None:
        """Validate a generation parameter type.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Raises:
            ResourceError: If parameter type is invalid
        """
        expected_type = self.SUPPORTED_PARAMS.get(name)
        if expected_type and not isinstance(value, expected_type):
            raise ResourceError(
                f"Invalid type for parameter '{name}'. Expected {expected_type.__name__}, got {type(value).__name__}",
                ErrorContext.create(provider_name=self.name)
            )
    
    def _filter_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and validate generation parameters.
        
        Args:
            params: Generation parameters
            
        Returns:
            Filtered parameters
            
        Raises:
            ResourceError: If parameter validation fails
        """
        filtered = {}
        for name, value in params.items():
            if name in self.SUPPORTED_PARAMS:
                self._validate_param_type(name, value)
                filtered[name] = value
        return filtered
    
    async def generate_structured(
        self,
        prompt: str,
        model_name: str,
        response_model: Type[BaseModel],
        **generation_params: Any
    ) -> BaseModel:
        """Generate structured output using an LLM.
        
        Args:
            prompt: Input prompt
            model_name: Name of the model to use
            response_model: Pydantic model for output validation
            **generation_params: Generation parameters
            
        Returns:
            Validated response
            
        Raises:
            ResourceError: If generation fails
        """
        try:
            # Get model configuration
            model_config = ResourceRegistry.get_resource('model', model_name)
            if not model_config:
                raise ResourceError(
                    f"Model '{model_name}' not found",
                    ErrorContext.create(provider_name=self.name)
                )
            
            # Get model instance
            model = self.get_model(model_name)
            
            # Filter and validate generation parameters
            params = self._filter_generation_params(generation_params)
            
            # Log generation parameters
            logger.info("Starting LLM generation with parameters:")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Response Model: {response_model.__name__}")
            logger.info("  Generation Parameters:")
            for name, value in params.items():
                logger.info(f"    {name}: {value}")
            
            # Format prompt based on model type
            formatted_prompt = format_prompt(prompt, model_config.model_type)
            
            # Create grammar from response model schema
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema)
            grammar = LlamaGrammar.from_json_schema(schema_str)
            
            # Add grammar to parameters
            params["grammar"] = grammar
            
            # Generate and process response within the same lock
            with self._model_locks[model_name]:
                # Generate response
                response = model.create_completion(
                    formatted_prompt,
                    **params
                )
                
                logger.info(f"Raw response: {response}")
                
                if not response or not isinstance(response, dict) or 'choices' not in response or not response['choices']:
                    raise ResourceError(
                        "Empty or invalid response from model",
                        ErrorContext.create(
                            provider_name=self.name,
                            prompt_length=len(prompt),
                            model_name=model_name,
                            response=str(response)
                        )
                    )
                
                # Extract and process text while still holding the lock
                generated_text = response['choices'][0]['text'].strip()
                logger.info(f"Generated text: {generated_text}")
                
                try:
                    # Parse JSON with more lenient decoder
                    decoder = json.JSONDecoder(strict=False)
                    parsed_response = decoder.decode(generated_text)
                    
                    # Sanitize strings in the parsed data
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
                    return response_model.model_validate(sanitized_data)
                    
                except json.JSONDecodeError as e:
                    raise ResourceError(
                        f"Failed to parse JSON response: {str(e)}",
                        ErrorContext.create(
                            provider_name=self.name,
                            response_text=generated_text,
                            error=str(e)
                        )
                    ) from e
                    
                except Exception as e:
                    raise ResourceError(
                        f"Failed to validate response: {str(e)}",
                        ErrorContext.create(
                            provider_name=self.name,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                    ) from e
                
        except Exception as e:
            if not isinstance(e, ResourceError):
                raise ResourceError(
                    f"Generation failed: {str(e)}",
                    ErrorContext.create(
                        provider_name=self.name,
                        model_name=model_name,
                        prompt_length=len(prompt),
                        error=str(e),
                        error_type=type(e).__name__
                    )
                ) from e
            raise 