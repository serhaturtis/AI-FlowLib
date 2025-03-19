"""Factory for creating conversation providers.

This module provides functions for creating and configuring conversation providers.
"""

import logging
from typing import Optional, Dict, Any, Union, Type

from ...core.registry import provider_registry
from .base import ConversationProvider

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
        # Get the provider from the registry
        provider = provider_registry.get("conversation", provider_name)
        if not provider:
            logger.error(f"Conversation provider '{provider_name}' not found in registry")
            return None
            
        # Initialize with settings
        if settings:
            provider._settings = settings
            
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
    for name, factory in provider_registry.get_providers_by_type("conversation").items():
        try:
            # Create a temporary instance to get the description
            provider = factory()
            providers[name] = provider.__class__.__doc__ or "No description available"
        except Exception as e:
            logger.warning(f"Error getting description for provider '{name}': {e}")
            providers[name] = "Description unavailable"
            
    return providers 