"""LLM provider base class and related functionality.

This module provides the base class for implementing local LLM providers 
such as LlamaCpp, which share common functionality for generating responses 
and extracting structured data.
"""

import logging
import re
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
import asyncio
from pydantic import BaseModel
from enum import Enum

from ...core.errors import ProviderError, ErrorContext
from ...core.models.settings import LLMProviderSettings
from ...core.registry import resource_registry as registry
from ..base import Provider

logger = logging.getLogger(__name__)

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
        """Get model configuration from the resource registry.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ProviderError: If model is not found or invalid
        """
        try:
            # Use registry.get which is async in AsyncResourceRegistry
            model_config = await registry.get(model_name, resource_type="model")
            
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
            if hasattr(registry, 'contains') and registry.contains(model_name, resource_type="model"):
                logger.info(f"Model '{model_name}' exists in registry but couldn't be retrieved")
            
            # For debugging - log what's in the registry
            if hasattr(registry, 'list_resources'):
                available = registry.list_resources("model")
                logger.info(f"Available models in registry: {available}")
            
            raise ProviderError(
                message=f"Error retrieving model configuration for '{model_name}': {str(e)}",
                context=ErrorContext.create(
                    model_name=model_name
                ),
                cause=e
            )
        
    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to generate from
            model_name: Name of the model to use
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated text response
            
        Raises:
            ProviderError: If generation fails
        """
        raise NotImplementedError("Subclasses must implement generate()")
        
    async def generate_structured(self, prompt: str, output_type: Type[ModelType], model_name: str, **kwargs) -> ModelType:
        """Generate a structured response from the LLM.
        
        Args:
            prompt: The prompt to generate from
            output_type: Pydantic model to parse the response into
            model_name: Name of the model to use
            **kwargs: Additional parameters for generation
            
        Returns:
            Pydantic model instance parsed from response
            
        Raises:
            ProviderError: If generation or parsing fails
        """
        raise NotImplementedError("Subclasses must implement generate_structured()") 