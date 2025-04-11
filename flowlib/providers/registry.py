"""Provider registry implementation for provider instances.

This module provides a concrete implementation of the BaseRegistry for
managing provider instances with specialized provider-related functionality.
This includes asynchronous initialization and lifecycle management.
"""

import asyncio
import logging
import importlib
import pkgutil
from inspect import isclass
from typing import Any, Dict, List, Optional, Type, TypeVar, cast, Callable, Tuple

from ..core.registry import BaseRegistry
from ..core.errors import ExecutionError
from .base import Provider

logger = logging.getLogger(__name__)


T = TypeVar('T')



class ProviderRegistry(BaseRegistry[Any]):
    """Registry for provider instances.
    
    This class implements the BaseRegistry interface with additional
    provider-specific functionality including:
    1. Type-based retrieval and factory management
    2. Lazy asynchronous initialization of providers
    3. Thread-safe provider initialization with locks
    4. Provider lifecycle management (initialization and shutdown)
    """
    
    def __init__(self):
        """Initialize provider registry."""
        # Main storage for providers: (provider_type, name) -> provider
        self._providers: Dict[Tuple[str, str], Provider] = {}
        # Storage for provider metadata
        self._metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Storage for provider factories
        self._factories: Dict[Tuple[str, str], Callable[[], Provider]] = {}
        # Factory metadata
        self._factory_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Async functionality
        self._initialized_providers: Dict[Tuple[str, str], Provider] = {}
        self._initialization_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        
    def register(self, provider: Provider, **metadata: Any) -> None:
        """Register a provider.
        
        Args:
            provider: Provider to register
            **metadata: Additional metadata about the provider
            
        Raises:
            ValueError: If provider name is not set or has no provider_type
        """
        # Check if provider has necessary attributes
        if not hasattr(provider, 'name') or not provider.name:
            raise ValueError("Provider must have a name")
            
        if not hasattr(provider, 'provider_type') or not provider.provider_type:
            raise ValueError("Provider must have a provider_type")
            
        provider_name = provider.name
        provider_type = provider.provider_type
        key = (provider_type, provider_name)
        
        # Store provider and metadata
        self._providers[key] = provider
        self._metadata[key] = metadata
        
        logger.info(f"Registered provider: {provider_name} (type: {provider_type})")
        
    def register_factory(self, provider_type: str, name: str, factory: Callable[[], Provider], **metadata: Any) -> None:
        """Register a factory for creating providers.
        
        Args:
            provider_type: Type of provider (e.g., llm, database)
            name: Unique name for this provider
            factory: Factory function that creates the provider
            **metadata: Additional metadata about the provider
        """
        key = (provider_type, name)
        
        # Store factory and metadata
        self._factories[key] = factory
        self._factory_metadata[key] = {
            "provider_type": provider_type,
            **metadata
        }
        
        logger.info(f"Registered provider factory: {name} (type: {provider_type})")
    
    async def get(self, provider_type: str, name: str, expected_type: Optional[Type] = None) -> Provider:
        """Get a provider by type and name with automatic initialization.
        
        This method enhances the base get method by automatically initializing
        providers when they're retrieved. It provides a unified interface for
        accessing providers regardless of their initialization state.
        
        Args:
            provider_type: Type of provider
            name: Name of the provider
            expected_type: Optional type for validation
            
        Returns:
            Initialized provider instance
            
        Raises:
            KeyError: If provider doesn't exist
            TypeError: If provider doesn't match expected type
        """
        key = (provider_type, name)
        
        # Check if this is a provider that needs initialization
        if key in self._providers or key in self._factories:
            return await self.get_provider_async(provider_type, name)
        
        # If we get here, the key was not found in providers or factories
        raise KeyError(f"Provider '{name}' of type '{provider_type}' not found")
    
    def get_sync(self, provider_type: str, name: str, expected_type: Optional[Type] = None) -> Provider:
        """Get a provider by type and name without initialization (synchronous).
        
        Note: This method returns the provider instance as registered, which 
        may not be initialized. For initialized providers, use get or get_provider_async.
        
        Args:
            provider_type: Type of provider
            name: Name of the provider
            expected_type: Optional type for validation
            
        Returns:
            Provider instance
            
        Raises:
            KeyError: If provider doesn't exist
            TypeError: If provider doesn't match expected type
        """
        key = (provider_type, name)
        
        if key not in self._providers:
            raise KeyError(f"Provider '{name}' of type '{provider_type}' not found")
        
        provider = self._providers[key]
        
        # Type checking if expected_type is provided
        if expected_type and not isinstance(provider, expected_type):
            raise TypeError(f"Provider '{name}' is not of expected type {expected_type.__name__}")
        
        return provider
    
    def get_factory(self, provider_type: str, name: str) -> Callable[[], Provider]:
        """Get a provider factory function.
        
        Args:
            provider_type: Type of provider
            name: Name of the factory
            
        Returns:
            Factory function
            
        Raises:
            KeyError: If factory doesn't exist
        """
        key = (provider_type, name)
        
        if key not in self._factories:
            raise KeyError(f"Provider factory '{name}' of type '{provider_type}' not found")
        
        return self._factories[key]
    
    def get_factory_metadata(self, provider_type: str, name: str) -> Dict[str, Any]:
        """Get metadata for a provider factory.
        
        Args:
            provider_type: Type of provider
            name: Name of the factory
            
        Returns:
            Metadata dictionary
            
        Raises:
            KeyError: If factory doesn't exist
        """
        key = (provider_type, name)
        
        if key not in self._factories:
            raise KeyError(f"Provider factory '{name}' of type '{provider_type}' not found")
        
        return self._factory_metadata.get(key, {})
    
    def contains(self, provider_type: str, name: str) -> bool:
        """Check if a provider exists.
        
        Args:
            provider_type: Type of provider
            name: Name to check
            
        Returns:
            True if the provider exists, False otherwise
        """
        key = (provider_type, name)
        return key in self._providers
    
    def contains_factory(self, provider_type: str, name: str) -> bool:
        """Check if a provider factory exists.
        
        Args:
            provider_type: Type of provider
            name: Name to check
            
        Returns:
            True if the factory exists, False otherwise
        """
        key = (provider_type, name)
        return key in self._factories
    
    def list(self, provider_type: Optional[str] = None) -> List[str]:
        """List registered providers matching criteria.
        
        Args:
            provider_type: Optional provider type to filter by
                
        Returns:
            List of provider names matching the criteria
        """
        result = []
        for key in self._providers.keys():
            if provider_type is None or key[0] == provider_type:
                result.append(key[1])
                
        return result
    
    def list_factories(self, provider_type: Optional[str] = None) -> List[str]:
        """List registered factories matching criteria.
        
        Args:
            provider_type: Optional provider type to filter by
                
        Returns:
            List of factory names matching the criteria
        """
        result = []
        for key in self._factories.keys():
            if provider_type is None or key[0] == provider_type:
                result.append(key[1])
                
        return result
    
    def list_provider_types(self) -> List[str]:
        """List all provider types in the registry.
        
        Returns:
            List of provider types
        """
        provider_types = set()
        for key in self._providers.keys():
            provider_types.add(key[0])
        for key in self._factories.keys():
            provider_types.add(key[0])
        return list(provider_types)
    
    def get_by_type(self, provider_type: str) -> Dict[str, Provider]:
        """Get all providers of a specific type.
        
        Args:
            provider_type: Type of providers to retrieve
            
        Returns:
            Dictionary of provider names to providers
        """
        result = {}
        for key, provider in self._providers.items():
            if key[0] == provider_type:
                result[key[1]] = provider
        return result
    
    async def get_typed(self, provider_type: str, name: str, expected_type: Type[T]) -> T:
        """Get a provider with type validation, casting, and automatic initialization.
        
        Args:
            provider_type: Type of provider
            name: Name of the provider
            expected_type: Expected provider type
            
        Returns:
            The initialized provider cast to the expected type
            
        Raises:
            KeyError: If provider doesn't exist
            TypeError: If provider doesn't match expected type
        """
        provider = await self.get(provider_type, name)
        
        if not isinstance(provider, expected_type):
            raise TypeError(f"Provider '{name}' is not of expected type {expected_type.__name__}")
        
        return cast(T, provider)
    
    def get_typed_sync(self, provider_type: str, name: str, expected_type: Type[T]) -> T:
        """Get a provider with type validation and casting (synchronous).
        
        Note: This method returns the provider instance as registered, which 
        may not be initialized. For initialized providers, use get_typed or get_provider_typed_async.
        
        Args:
            provider_type: Type of provider
            name: Name of the provider
            expected_type: Expected provider type
            
        Returns:
            The provider cast to the expected type
            
        Raises:
            KeyError: If provider doesn't exist
            TypeError: If provider doesn't match expected type
        """
        provider = self.get_sync(provider_type, name, expected_type)
        return cast(T, provider)
    
    async def get_provider_async(self, provider_type: str, name: str) -> Provider:
        """Get a provider with automatic initialization.
        
        This method:
        1. Retrieves a provider instance
        2. Initializes it if needed (thread-safe)
        3. Caches the initialized provider
        
        Args:
            provider_type: Type of provider
            name: Provider name
            
        Returns:
            Initialized provider instance
            
        Raises:
            KeyError: If provider not found
            ExecutionError: If initialization fails
        """
        key = (provider_type, name)
        
        # Check if already initialized
        if key in self._initialized_providers:
            return self._initialized_providers[key]
        
        # Get or create lock for this provider
        if key not in self._initialization_locks:
            self._initialization_locks[key] = asyncio.Lock()
        
        # Acquire lock to prevent concurrent initialization
        async with self._initialization_locks[key]:
            # Check again after acquiring lock (double-checked locking)
            if key in self._initialized_providers:
                return self._initialized_providers[key]
            
            # Get the provider
            try:
                # First try direct provider
                if self.contains(provider_type, name):
                    provider = self.get_sync(provider_type, name)
                    
                    # Initialize if not already
                    if hasattr(provider, 'initialized') and not provider.initialized:
                        await provider.initialize()
                    
                    self._initialized_providers[key] = provider
                    return provider
                
                # Try factory if direct provider not found
                elif self.contains_factory(provider_type, name):
                    factory = self.get_factory(provider_type, name)
                    metadata = self.get_factory_metadata(provider_type, name)
                    
                    # Create provider using factory
                    provider = factory()
                    
                    # Initialize provider
                    if hasattr(provider, 'initialize'):
                        await provider.initialize()
                    
                    # Store in provider registry and initialized cache
                    self.register(provider)
                    self._initialized_providers[key] = provider
                    
                    return provider
                else:
                    raise KeyError(f"Provider '{name}' of type '{provider_type}' not found and could not be initialized")
                
            except Exception as e:
                # Catch and wrap initialization errors
                raise ExecutionError(
                    message=f"Failed to initialize provider '{name}' of type '{provider_type}': {str(e)}",
                    context={
                        "provider_name": name,
                        "provider_type": provider_type,
                        "available_providers": self.list(provider_type),
                        "available_factories": self.list_factories(provider_type)
                    },
                    cause=e
                )
    
    async def get_provider_typed_async(self, provider_type: str, name: str, expected_type: Type[T]) -> T:
        """Get a provider with automatic initialization and type checking.
        
        Args:
            provider_type: Type of provider
            name: Provider name
            expected_type: Expected provider type
            
        Returns:
            Initialized provider cast to the expected type
            
        Raises:
            KeyError: If provider not found
            TypeError: If provider doesn't match expected type
            ExecutionError: If initialization fails
        """
        provider = await self.get_provider_async(provider_type, name)
        
        if not isinstance(provider, expected_type):
            raise TypeError(f"Provider '{name}' is not of expected type {expected_type.__name__}")
            
        return cast(T, provider)
    
    async def initialize_all(self, provider_type: Optional[str] = None) -> None:
        """Initialize all registered providers of a given type or all types.
        
        Args:
            provider_type: Optional provider type to initialize
            
        Raises:
            ExecutionError: If initialization of any provider fails
        """
        # Get providers to initialize
        if provider_type:
            providers = self.get_by_type(provider_type)
        else:
            # Get all providers across all types
            providers = {}
            for key, provider in self._providers.items():
                providers[key[1]] = provider
        
        # Initialize them all
        for name, provider in providers.items():
            if hasattr(provider, 'initialized') and not provider.initialized:
                try:
                    provider_type = provider.provider_type
                    await self.get_provider_async(provider_type, name)
                except Exception as e:
                    logger.warning(f"Failed to initialize provider '{name}': {str(e)}")
                    
    async def shutdown_all(self, provider_type: Optional[str] = None) -> None:
        """Shutdown all initialized providers.
        
        Args:
            provider_type: Optional provider type to shutdown
        """
        # Filter initialized providers by type if specified
        providers_to_shutdown = []
        for key, provider in self._initialized_providers.items():
            if provider_type is None or key[0] == provider_type:
                providers_to_shutdown.append((key, provider))
                
        # Shutdown each provider
        for (provider_type, name), provider in providers_to_shutdown:
            if hasattr(provider, 'shutdown'):
                try:
                    await provider.shutdown()
                    logger.info(f"Shut down provider '{name}' of type '{provider_type}'")
                except Exception as e:
                    logger.warning(f"Error shutting down provider '{name}' of type '{provider_type}': {str(e)}")
                    
            # Remove from initialized providers
            key = (provider_type, name)
            self._initialized_providers.pop(key, None)
            self._initialization_locks.pop(key, None)
    
    def discover_providers(self, package_name: str) -> List[Type]:
        """Discover provider classes in a package.
        
        Args:
            package_name: Python package to search
            
        Returns:
            List of discovered provider classes
        """
        discovered = []
        
        try:
            package = importlib.import_module(package_name)
            
            # Get the package path
            if hasattr(package, '__path__'):
                # Recursive discovery in package
                for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
                    if is_pkg:
                        # Recursively discover in subpackage
                        discovered.extend(self.discover_providers(name))
                    else:
                        # Import module and find provider classes
                        try:
                            module = importlib.import_module(name)
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                # Check if attribute is a class and is a subclass of Provider
                                # We use string comparison for now to avoid circular imports
                                if (isclass(attr) and 
                                    hasattr(attr, "__mro__") and
                                    any(base.__name__ == "Provider" for base in attr.__mro__) and
                                    attr.__name__ != "Provider"):
                                    discovered.append(attr)
                        except Exception as e:
                            logger.warning(f"Error discovering providers in {name}: {str(e)}")
            
            logger.info(f"Discovered {len(discovered)} provider classes in {package_name}")
            
        except Exception as e:
            logger.error(f"Error discovering providers in {package_name}: {str(e)}")
        
        return discovered 
    
provider_registry = ProviderRegistry()