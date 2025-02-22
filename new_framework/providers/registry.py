from typing import Dict, Type, Optional
from .base import Provider

class ProviderRegistry:
    """Registry for managing provider instances."""
    
    def __init__(self):
        """Initialize registry."""
        self._providers: Dict[str, Provider] = {}
        self._provider_types: Dict[str, Type[Provider]] = {}
    
    def register_provider_type(self, name: str, provider_type: Type[Provider]) -> None:
        """Register a provider type.
        
        Args:
            name: Provider type name
            provider_type: Provider class
        """
        self._provider_types[name] = provider_type
    
    def get_provider_type(self, name: str) -> Optional[Type[Provider]]:
        """Get a provider type by name.
        
        Args:
            name: Provider type name
            
        Returns:
            Provider class if found, None otherwise
        """
        return self._provider_types.get(name)
    
    def register_provider(self, provider: Provider) -> None:
        """Register a provider instance.
        
        Args:
            provider: Provider instance
        """
        self._providers[provider.name] = provider
    
    def get_provider(self, name: str) -> Optional[Provider]:
        """Get a provider instance by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance if found, None otherwise
        """
        return self._providers.get(name)
    
    def unregister_provider(self, name: str) -> None:
        """Unregister a provider instance.
        
        Args:
            name: Provider name
        """
        if name in self._providers:
            del self._providers[name]
    
    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        self._provider_types.clear()

# Global registry instance
registry = ProviderRegistry() 