from typing import Dict, Any, Type
from pathlib import Path
import gc
import sys
import json
import logging
import threading
from llama_cpp import Llama, LlamaGrammar

from pydantic import BaseModel
from ...core.errors.base import ResourceError, ErrorContext
from ..base import Provider
from .models import ModelConfig, GenerationParams
from .utils import GPUConfigManager

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
        model_configs: Dict[str, ModelConfig],
        max_models: int = 2
    ):
        """Initialize LLM provider.
        
        Args:
            name: Provider name
            model_configs: Model configurations
            max_models: Maximum number of models to keep loaded
        """
        super().__init__(name)
        self.model_configs = model_configs
        self.max_models = max_models
        self._models: Dict[str, Llama] = {}
        self._load_order: list[str] = []  # Track loading order for LRU
        self._lock = threading.Lock()  # Thread safety for model management
        self._gpu_manager = GPUConfigManager()

    async def initialize(self) -> None:
        """Initialize provider resources."""
        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up provider resources."""
        with self._lock:
            for model_name in list(self._models.keys()):
                self._cleanup_model(model_name)
        await super().cleanup()

    def validate_model_name(self, model_name: str) -> None:
        """Validate model name and configuration.
        
        Args:
            model_name: Name of model to validate
            
        Raises:
            ResourceError: If model configuration is invalid
        """
        if model_name not in self.model_configs:
            raise ResourceError(
                f"Unknown model: {model_name}",
                {"available_models": list(self.model_configs.keys())}
            )
        
        config = self.model_configs[model_name]
        model_path = Path(config.path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        
        if not model_path.exists():
            raise ResourceError(
                f"Model file not found: {model_path}",
                {"model_name": model_name}
            )

    def _load_model(self, model_name: str) -> Llama:
        """Load a model."""
        try:
            config = self.model_configs[model_name]
            model_path = Path(config.path)
            
            if not model_path.exists():
                raise ResourceError(
                    f"Model file not found: {model_path}",
                    {"model_name": model_name}
                )
            
            # Get optimal GPU configuration if GPU is enabled
            model_params = {}
            if config.use_gpu:
                try:
                    gpu_config = self._gpu_manager.get_optimal_config(model_path)
                    model_params.update(gpu_config)
                except Exception as e:
                    print(f"Warning: Failed to get GPU config: {e}. Falling back to CPU.", file=sys.stderr)
            
            # Override with user-specified parameters
            model_params.update({
                "n_ctx": config.n_ctx,
                "n_threads": config.n_threads,
                "n_batch": config.n_batch
            })
                
            model = Llama(
                model_path=str(model_path),
                **model_params
            )
            return model
        except Exception as e:
            if isinstance(e, ResourceError):
                raise
            raise ResourceError(
                f"Failed to load model {model_name}: {str(e)}",
                {
                    "model_name": model_name,
                    "model_config": self.model_configs[model_name].model_dump()
                }
            )

    def _cleanup_model(self, model_name: str) -> None:
        """Clean up a specific model."""
        if model_name in self._models:
            try:
                model = self._models[model_name]
                # Ensure model resources are freed
                model.__del__()
            except Exception as e:
                # Log error but continue with cleanup
                print(f"Warning: Error during model cleanup: {e}", file=sys.stderr)
            finally:
                del self._models[model_name]
                self._load_order.remove(model_name)
                gc.collect()  # Force garbage collection

    def get_model(self, model_name: str) -> Llama:
        """Get a model instance.
        
        Args:
            model_name: Name of model to get
            
        Returns:
            Model instance
            
        Raises:
            ResourceError: If model cannot be loaded
        """
        self.check_initialized()
        self.validate_model_name(model_name)

        with self._lock:
            # Return existing model if loaded
            if model_name in self._models:
                # Update LRU order
                self._load_order.remove(model_name)
                self._load_order.append(model_name)
                return self._models[model_name]

            # Clean up least recently used model if at capacity
            if len(self._models) >= self.max_models and self._load_order:
                self._cleanup_model(self._load_order[0])

            # Load new model
            model = self._load_model(model_name)
            self._models[model_name] = model
            self._load_order.append(model_name)
            return model

    def _validate_param_type(self, name: str, value: Any) -> None:
        """Validate parameter type."""
        expected_type = self.SUPPORTED_PARAMS.get(name)
        if expected_type and not isinstance(value, expected_type):
            raise ResourceError(
                f"Invalid type for parameter {name}",
                {
                    "parameter": name,
                    "expected_type": expected_type.__name__,
                    "actual_type": type(value).__name__,
                    "value": str(value)
                }
            )

    def _filter_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and validate generation parameters.
        
        Args:
            params: Generation parameters to filter
            
        Returns:
            Filtered parameters
            
        Raises:
            ResourceError: If parameters are invalid
        """
        filtered = {}
        for name, value in params.items():
            if name in self.SUPPORTED_PARAMS:
                try:
                    self._validate_param_type(name, value)
                    filtered[name] = value
                except ResourceError as e:
                    raise ResourceError(
                        "Invalid generation parameter",
                        {**e.context.details}
                    )
        
        if not filtered and params:
            raise ResourceError(
                "No valid generation parameters after filtering",
                {
                    "provided_params": list(params.keys()),
                    "supported_params": list(self.SUPPORTED_PARAMS.keys())
                }
            )
        
        return filtered

    async def generate_structured(
        self,
        prompt: str,
        model_name: str,
        response_model: Type[BaseModel],
        **generation_params: Any
    ) -> BaseModel:
        """Generate structured output using LlamaGrammar.
        
        Args:
            prompt: Input prompt
            model_name: Name of model to use
            response_model: Pydantic model for output validation
            **generation_params: Generation parameters
            
        Returns:
            Validated instance of the response_model
            
        Raises:
            ResourceError: If generation fails
            ValidationError: If output validation fails
        """
        try:
            model = self.get_model(model_name)
            
            # Create grammar from pydantic model
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema)
            grammar = LlamaGrammar.from_json_schema(schema_str)
            
            # Filter and validate generation parameters
            params = self._filter_generation_params(generation_params)
            
            # Log generation parameters
            logger.info("Starting LLM generation with parameters:")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Response Model: {response_model.__name__}")
            logger.info("  Generation Parameters:")
            for name, value in params.items():
                logger.info(f"    {name}: {value}")
            
            # Generate with grammar
            output = model.create_completion(
                prompt=prompt,
                grammar=grammar,
                **params
            )
            
            if not output or not output.get('choices'):
                raise ResourceError(
                    "Empty response from model",
                    ErrorContext.create(
                        prompt_length=len(prompt),
                        model_name=model_name
                    )
                )
            
            text = output['choices'][0].get('text', '').strip()
            
            try:
                # Parse JSON with more lenient decoder settings
                decoder = json.JSONDecoder(strict=False)
                data = json.loads(text, strict=False)
                
                # Recursively sanitize strings in the parsed data
                def sanitize_strings(obj):
                    if isinstance(obj, str):
                        # Replace common control characters with their escaped versions
                        return obj.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    elif isinstance(obj, dict):
                        return {k: sanitize_strings(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [sanitize_strings(item) for item in obj]
                    return obj
                
                sanitized_data = sanitize_strings(data)
                
                # Validate with pydantic model
                validated_model = response_model.model_validate(sanitized_data)
                return validated_model
                
            except Exception as e:
                raise ResourceError(
                    f"Failed to parse or validate model output: {str(e)}",
                    ErrorContext.create(
                        response_text=text,
                        model_name=model_name,
                        error=str(e)
                    )
                )
            
        except Exception as e:
            if isinstance(e, ResourceError):
                raise
            raise ResourceError(
                f"Generation failed: {str(e)}",
                ErrorContext.create(
                    model_name=model_name,
                    prompt_length=len(prompt),
                    error=str(e),
                    error_type=type(e).__name__
                )
            ) 