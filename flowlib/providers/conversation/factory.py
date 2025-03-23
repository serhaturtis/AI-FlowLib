"""Factory for creating conversation providers.

This module provides functions for creating and configuring conversation providers.
"""

import logging
from typing import Optional, Dict, Any, Union, Type

from ...core.registry import provider_registry
from ...core.registry.constants import ProviderType
from .base import ConversationProvider, ConversationProviderSettings
from .cli import CLIConversationProviderSettings
from .web import WebConversationProviderSettings
from .api import APIConversationProviderSettings

logger = logging.getLogger(__name__)

def create_conversation_provider(
    provider_name: str,
    settings: Optional[Dict[str, Any]] = None
) -> Optional[ConversationProvider]:
    """Create a conversation provider from the registry.
    
    Args:
        provider_name: Name of the provider to create
        settings: Optional settings for the provider
        
    Returns:
        A conversation provider instance, or None if the provider could not be created
    """
    try:
        # Convert dictionary settings to the appropriate settings class
        typed_settings = None
        if settings:
            if provider_name == "cli":
                typed_settings = CLIConversationProviderSettings(**settings)
            elif provider_name == "web":
                typed_settings = WebConversationProviderSettings(**settings)
            elif provider_name == "api":
                typed_settings = APIConversationProviderSettings(**settings)
            else:
                # For unknown providers, create a base settings class
                typed_settings = ConversationProviderSettings(**settings)
        
        # Get the provider from the registry
        factory = provider_registry.get_factory(ProviderType.CONVERSATION, provider_name)
        if not factory:
            logger.error(f"Conversation provider '{provider_name}' not found in registry")
            return None
        
        # Create provider instance with typed settings
        provider = factory()
        if typed_settings:
            provider.settings = typed_settings
            
        return provider
    except Exception as e:
        logger.error(f"Error creating conversation provider '{provider_name}': {e}")
        return None

def get_available_providers() -> Dict[str, str]:
    """Get a dictionary of available conversation providers.
    
    Returns:
        Dictionary mapping provider names to their descriptions
    """
    providers = {}
    
    # Get all conversation providers from the registry
    provider_names = provider_registry.list_factories(ProviderType.CONVERSATION)
    
    for name in provider_names:
        try:
            # Get the factory and create a temporary instance
            factory = provider_registry.get_factory(ProviderType.CONVERSATION, name)
            provider = factory()
            providers[name] = provider.__class__.__doc__ or "No description available"
        except Exception as e:
            logger.warning(f"Error getting description for provider '{name}': {e}")
            providers[name] = "Description unavailable"
            
    return providers 