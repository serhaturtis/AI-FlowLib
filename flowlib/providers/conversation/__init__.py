"""Conversation providers for flowlib.

This package provides various conversation providers for agent interactions,
including CLI, web, and API interfaces.
"""

from .base import ConversationProvider, ConversationProviderSettings
from .cli import CLIConversationProvider, CLIConversationProviderSettings
from .web import WebConversationProvider, WebConversationProviderSettings
from .api import APIConversationProvider, APIConversationProviderSettings
from .factory import create_conversation_provider, get_available_providers

__all__ = [
    # Base classes
    "ConversationProvider",
    "ConversationProviderSettings",
    
    # Provider implementations
    "CLIConversationProvider",
    "WebConversationProvider",
    "APIConversationProvider",
    
    # Settings classes
    "CLIConversationProviderSettings",
    "WebConversationProviderSettings",
    "APIConversationProviderSettings",
    
    # Factory functions
    "create_conversation_provider",
    "get_available_providers",
] 