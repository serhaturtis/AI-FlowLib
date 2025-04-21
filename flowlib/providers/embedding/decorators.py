"""
Decorators for registering embedding models and providers.
"""

import inspect
import logging
from typing import Any, Dict, Type, Callable

from ..registry import provider_registry
from ..constants import ProviderType
from .base import EmbeddingProvider
from .llama_cpp_provider import LlamaCppEmbeddingProvider # Import concrete provider

logger = logging.getLogger(__name__)

# Mapping from a implementation name (if specified) to provider class
# This might need expansion if more embedding providers are added
EMBEDDING_PROVIDER_MAP = {
    "llamacpp": LlamaCppEmbeddingProvider,
    # Add other implementations here, e.g., "sentence_transformers", etc.
}

def embedding_model(name: str, implementation: str = "llamacpp") -> Callable[[Type], Type]:
    """Class decorator to register an embedding model config via a factory.

    This decorator registers a factory function for a specific embedding provider 
    implementation (e.g., LlamaCppEmbeddingProvider) using the configuration 
    defined in the decorated class.
    
    Args:
        name: Name to register the embedding model provider instance under.
        implementation: The type of embedding provider to instantiate (default: llamacpp).

    Returns:
        Decorator function.
    """
    def decorator(cls: Type) -> Type:
        if not inspect.isclass(cls):
            raise TypeError("@embedding_model can only decorate classes.")
            
        logger.debug(f"Registering embedding model factory '{name}' (impl: {implementation}) with config from {cls.__name__}")
        
        # Extract configuration details from the class definition
        config = {}
        for key, value in inspect.getmembers(cls):
            if not key.startswith('__') and not callable(value):
                config[key] = value
                
        if not config:
            logger.warning(f"Embedding model configuration class {cls.__name__} for '{name}' has no config attributes.")

        # Get the provider class based on the implementation key
        provider_cls = EMBEDDING_PROVIDER_MAP.get(implementation.lower())
        if not provider_cls:
            raise ValueError(f"Unsupported embedding provider implementation: '{implementation}'. Supported: {list(EMBEDDING_PROVIDER_MAP.keys())}")

        # Define the factory function
        def factory() -> EmbeddingProvider:
            # Instantiate the concrete provider using the extracted config
            try:
                return provider_cls(name=name, config=config)
            except Exception as e:
                logger.error(f"Failed to create embedding provider '{name}' via factory: {e}", exc_info=True)
                # Re-raise as a more specific error? For now, re-raise original.
                raise

        # Register the factory function with the config stored in metadata
        provider_registry.register_factory(
            provider_type=ProviderType.EMBEDDING,
            name=name,
            factory=factory,
            implementation=implementation, # Store implementation type
            config=config # Store config in metadata for inspection if needed
        )
        return cls
        
    return decorator

# TODO: Add @embedding_provider decorator if needed for direct class registration 