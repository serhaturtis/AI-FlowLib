"""
Base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Any, Generic, TypeVar

from ..base import Provider
from ...core.errors import ProviderError

SettingsType = TypeVar('SettingsType')

class EmbeddingProvider(Provider[SettingsType], ABC, Generic[SettingsType]):
    """Abstract base class for embedding providers.
    
    Subclasses must implement the 'embed' method.
    """
    
    @abstractmethod
    async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for the given text(s).
        
        Args:
            text: A single string or a list of strings to embed.
            
        Returns:
            A list of embeddings (each embedding is a list of floats).
            If input was a single string, returns a list containing one embedding.
            
        Raises:
            ProviderError: If embedding generation fails.
        """
        pass

    # Optional: Add methods for specific embedding tasks if needed
    # async def embed_query(self, query: str) -> List[float]: ...
    # async def embed_documents(self, documents: List[str]) -> List[List[float]]: ...

    # Default initialize/shutdown can be inherited from Provider
    # if no specific logic is needed for the base class. 