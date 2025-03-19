"""Base conversation provider for flowlib.

This module defines the base class for conversation providers, which
handle interactions between users and agents through different interfaces.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ...core.models import Context
from ...core.errors import ExecutionError

logger = logging.getLogger(__name__)

class ConversationProvider(ABC):
    """Base class for conversation providers.
    
    Conversation providers handle the interaction between users and agents
    through different interfaces (CLI, web, API, etc.).
    """
    
    def __init__(self, name: str, settings: Optional[Dict[str, Any]] = None):
        """Initialize the conversation provider.
        
        Args:
            name: Provider name
            settings: Optional provider settings
        """
        self.name = name
        self.settings = settings or {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the conversation provider."""
        self.initialized = True
    
    @abstractmethod
    async def get_next_input(self) -> Optional[str]:
        """Get the next input from the conversation source.
        
        Returns:
            User input text or None if conversation should end
        """
        pass
    
    @abstractmethod
    async def send_response(self, response: str):
        """Send a response to the conversation destination.
        
        Args:
            response: Response text to send
        """
        pass
    
    async def show_details(self, details: Dict[str, Any]):
        """Show additional details about execution.
        
        Args:
            details: Dictionary of execution details
        """
        pass
    
    async def handle_error(self, error: Exception) -> str:
        """Handle errors during conversation.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Error message to display to the user
        """
        if isinstance(error, ExecutionError):
            logger.error(f"Execution error in conversation: {error.message}")
            return f"I encountered an error: {error.message}"
        else:
            logger.error(f"Error in conversation: {str(error)}")
            return f"I encountered an unexpected error: {str(error)}"
            
    async def shutdown(self):
        """Clean up any resources used by the provider.
        
        This method should be overridden by provider implementations
        that need to perform cleanup operations before shutting down.
        """
        logger.info(f"Shutting down {self.name} conversation provider")
        self.initialized = False 