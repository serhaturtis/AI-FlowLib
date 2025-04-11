"""LLM provider base class and related functionality.

This module provides the base class for implementing local LLM providers 
such as LlamaCpp, which share common functionality for generating responses 
and extracting structured data.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field, field_validator

from ...core.errors import ProviderError, ErrorContext
from ..base import ProviderSettings
from ..base import Provider

from ...resources.registry import resource_registry
from ...resources.decorators import PromptTemplate

from ...utils.pydantic.schema import model_to_simple_json_schema

logger = logging.getLogger(__name__)

class LLMProviderSettings(ProviderSettings):
    """Settings for LLM providers.
    
    This class provides:
    1. Model configuration
    2. Generation parameters
    3. Token management
    """
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Token management
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    
    # Streaming settings
    stream: bool = False
    
    # Advanced settings
    stop_sequences: List[str] = Field(default_factory=list)
    
    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature."""
        if v < 0 or v > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @field_validator("top_p")
    def validate_top_p(cls, v: float) -> float:
        """Validate top_p."""
        if v < 0 or v > 1:
            raise ValueError("Top_p must be between 0 and 1")
        return v


T = TypeVar('T', bound=LLMProviderSettings)
ModelType = TypeVar('ModelType', bound=BaseModel)


class LLMProvider(Provider[T]):
    """Base class for local LLM backends.
    
    This class provides the interface for:
    1. Structured generation with Pydantic models
    2. Grammar-based parsing and validation
    3. Type-safe response handling
    """
    
    def __init__(self, name: str = "llm", settings: Optional[T] = None):
        """Initialize LLM provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Pass provider_type="llm" to the parent class
        super().__init__(name=name, settings=settings, provider_type="llm")
        self._initialized = False
        self._models = {}
        
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
        
    async def initialize(self):
        """Initialize the provider.
        
        This method should be implemented by subclasses.
        """
        self._initialized = True
        
    async def shutdown(self):
        """Clean up resources.
        
        This method should be implemented by subclasses.
        """
        self._initialized = False
        
    async def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model from the resource registry.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ProviderError: If model is not found or invalid
        """
        try:
            model_config = resource_registry.get_sync(model_name, resource_type="model")
            
            # Log the model config for debugging
            logger.info(f"Retrieved model config for '{model_name}': {model_config}")
            
            # If model_config is a class (not instance), create an instance
            if isinstance(model_config, type):
                logger.info(f"Model '{model_name}' is a class, creating instance")
                model_config = model_config()
                
            return model_config
        except Exception as e:
            logger.error(f"Error retrieving model '{model_name}': {str(e)}")
            
            # Check if the model exists in registry
            if hasattr(resource_registry, 'contains') and resource_registry.contains(model_name, resource_type="model"):
                logger.info(f"Model '{model_name}' exists in registry but couldn't be retrieved")
            
            # For debugging - log what's in the registry
            if hasattr(resource_registry, 'list_resources'):
                available = resource_registry.list_resources("model")
                logger.info(f"Available models in registry: {available}")
            
            raise ProviderError(
                message=f"Error retrieving model configuration for '{model_name}': {str(e)}",
                context=ErrorContext.create(
                    model_name=model_name
                ),
                cause=e
            )
        
    async def generate(self, prompt: PromptTemplate, model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt template to generate from
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Generated text response
            
        Raises:
            ProviderError: If generation fails
        """
        raise NotImplementedError("Subclasses must implement generate()")
        
    async def generate_structured(self, prompt: PromptTemplate, output_type: Type[ModelType], model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> ModelType:
        """Generate a structured response from the LLM.
        
        Args:
            prompt: The prompt template to generate from
            output_type: Pydantic model to parse the response into
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Pydantic model instance parsed from response
            
        Raises:
            ProviderError: If generation or parsing fails
        """
        raise NotImplementedError("Subclasses must implement generate_structured()")
        
    def format_template(self, template: str, kwargs: Dict[str, Any]) -> str:
        """Format a template with variables.
        
        Replaces {{variable}} placeholders in the template with corresponding values.
        Uses double curly braces to avoid conflicts with JSON formatting.
        
        Args:
            template: Template string with {{variable}} placeholders
            kwargs: Dict containing variables and their values
            
        Returns:
            Formatted template string
        """
        # Debug: Log what variables we're receiving
        print("\n===== DEBUG: FORMAT_TEMPLATE VARIABLES =====")
        print(f"Template length: {len(template)}")
        print(f"Variables available: {list(kwargs.get('variables', {}).keys())}")
        
        if "variables" in kwargs and isinstance(kwargs["variables"], dict):
            variables = kwargs["variables"]
            result = template
            
            # Replace {{variable}} with corresponding values
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                if placeholder in result:
                    print(f"Replacing placeholder: {placeholder}")
                    result = result.replace(placeholder, str(value))
                else:
                    print(f"Warning: Placeholder {placeholder} not found in template")
            
            # Debug: Log if any {{...}} remain in the template
            import re
            remaining = re.findall(r'\{\{([^}]+)\}\}', result)
            if remaining:
                print("Warning: Unreplaced placeholders remain:")
                for placeholder in remaining:
                    print(f"  - {placeholder}")
            
            print("===== END DEBUG: FORMAT_TEMPLATE =====\n")
            return result
        
        # No variables provided, return template as-is
        print("Warning: No 'variables' key in kwargs or not a dict")
        print("===== END DEBUG: FORMAT_TEMPLATE =====\n")
        return template
        
    def _format_prompt(self, prompt: str, model_type: str = "default", output_type: Optional[Type[ModelType]] = None) -> str:
        """Format a prompt according to model-specific requirements.
        
        This is a base implementation that provides common formatting patterns.
        If output_type is provided, automatically appends JSON structure instructions.
        
        Args:
            prompt: The main prompt text
            model_type: The type/name of the model
            output_type: Optional Pydantic model type for structured output
            
        Returns:
            Formatted prompt string
        """
        # If output_type is provided, append JSON structure information
        if output_type is not None and hasattr(output_type, 'model_json_schema'):
            try:                
                example_json = model_to_simple_json_schema(output_type)
                
                # Append the structure information to the prompt
                json_instructions = f"\n\nPlease format your response as a JSON object with the following structure:\n{example_json}\n"
                return prompt + json_instructions
            except Exception as e:
                # If there's an error generating the schema example, log it but continue
                logger.warning(f"Failed to generate JSON schema example: {str(e)}")
                
        # Default formatting does nothing
        return prompt
        
        
    def _get_model_templates(self) -> Dict[str, Dict[str, str]]:
        """Get model-specific prompt templates.
        
        Override this in subclasses to provide templates for different model types.
        
        Returns:
            Dictionary mapping model_type to pre/post prompt templates
        """
        return {
            "default": {
                "pre_prompt": "",
                "post_prompt": ""
            }
        } 