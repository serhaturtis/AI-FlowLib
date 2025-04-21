"""
Embedding provider using llama-cpp-python.
"""

import logging
from typing import List, Union, Any, Dict, Optional, Generic, TypeVar
import asyncio
from pydantic import Field
from abc import ABC, abstractmethod

from .base import EmbeddingProvider
from ..base import Provider, ProviderSettings
from ...core.errors import ProviderError, ConfigurationError

# Lazy import llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)

# --- Settings Model ---
class LlamaCppEmbeddingProviderSettings(ProviderSettings):
    """Settings specific to LlamaCppEmbeddingProvider."""
    path: str = Field(description="Path to the GGUF model file.")
    n_ctx: int = Field(default=512, description="Context size for the model.")
    n_threads: Optional[int] = Field(default=None, description="Number of threads to use.")
    n_batch: int = Field(default=512, description="Batch size for processing.")
    use_mlock: bool = Field(default=False, description="Use mlock to keep model in memory.")
    n_gpu_layers: int = Field(default=0, description="Number of layers to offload to GPU (-1 for all).")
    verbose: bool = Field(default=False, description="Enable llama.cpp verbose logging.")
    # Add other relevant llama_cpp parameters if needed

# Type variable for settings
SettingsType = TypeVar('SettingsType', bound=LlamaCppEmbeddingProviderSettings)

# --- Provider Implementation ---
class LlamaCppEmbeddingProvider(EmbeddingProvider[LlamaCppEmbeddingProviderSettings]):
    """Provides embeddings using a GGUF model via llama-cpp-python.
    
    Requires llama-cpp-python to be installed with embedding support.
    """
    
    def __init__(
        self,
        name: str,
        # Config dict is passed to Provider base which handles settings creation
        config: Optional[Dict[str, Any]] = None, 
        **kwargs # Allow direct settings override via kwargs
    ):
        """Initialize the LlamaCppEmbeddingProvider.
        
        Args:
            name: Provider instance name.
            config: Configuration dictionary (used by Provider base to create settings).
            **kwargs: Direct settings overrides.
        """
        # Combine config dict and kwargs for settings creation
        init_settings = {**(config or {}), **kwargs}

        # Initialize Provider base first - it handles settings creation/assignment
        # It will use LlamaCppEmbeddingProviderSettings based on Generic type hint
        super().__init__(name=name, settings=init_settings)
        
        # --- Post-initialization validation and setup ---
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Please install it with embedding support: "
                "pip install llama-cpp-python[server] or similar."
            )
        
        # Validate required config (path) - Base init should raise if path missing now
        if not self.settings.path:
             raise ConfigurationError(
                 message="'path' is required in LlamaCppEmbeddingProvider config.",
                 provider_name=name,
                 config_key="path"
             )
            
        self._model_path = self.settings.path
        self._model: Optional[Llama] = None
        self._lock = asyncio.Lock()
        
        logger.info(f"LlamaCppEmbeddingProvider '{name}' configured with model: {self._model_path}")

    async def _initialize(self) -> None:
        """Load the Llama model for embeddings."""
        async with self._lock:
            if self._model:
                return # Already initialized
            
            logger.info(f"Loading embedding model: {self._model_path}...")
            try:
                # Prepare arguments for Llama constructor using self.settings
                llama_args = {
                    "model_path": self.settings.path,
                    "embedding": True, # Crucial for embedding models
                    "n_ctx": self.settings.n_ctx,
                    "n_threads": self.settings.n_threads,
                    "n_batch": self.settings.n_batch,
                    "use_mlock": self.settings.use_mlock,
                    "n_gpu_layers": self.settings.n_gpu_layers,
                    "verbose": self.settings.verbose
                }
                # Remove None values
                llama_args = {k: v for k, v in llama_args.items() if v is not None}

                self._model = Llama(**llama_args)
                logger.info(f"Embedding model loaded successfully: {self._model_path}")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self._model_path}': {e}", exc_info=True)
                self._model = None # Ensure model is None on failure
                raise ProviderError(
                    message=f"Failed to load embedding model '{self._model_path}': {e}",
                    provider_name=self.name,
                    cause=e
                )

    async def _shutdown(self) -> None:
        """Release the Llama model."""
        async with self._lock:
            if self._model:
                # llama-cpp-python doesn't have an explicit close/shutdown
                # relies on garbage collection (del self._model)
                del self._model
                self._model = None
                logger.info(f"Embedding model resources released for: {self._model_path}")
            
    async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for the given text(s)."""
        if not self.initialized or not self._model:
            raise ProviderError(
                message="Embedding provider is not initialized.", 
                provider_name=self.name
            )
            
        async with self._lock:
            try:
                # llama-cpp expects a list, even for a single string
                input_texts = [text] if isinstance(text, str) else text
                if not input_texts:
                    return []
                
                # Get embeddings
                # Note: Llama.embed() might be synchronous depending on version/setup
                # If it blocks significantly, consider running in a thread pool executor
                embeddings = self._model.embed(input_texts)
                
                # Ensure the output is List[List[float]]
                if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
                     raise ProviderError("Llama model did not return expected embedding format.")
                 
                return embeddings
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
                raise ProviderError(
                    message=f"Failed to generate embeddings: {e}",
                    provider_name=self.name,
                    cause=e
                ) 