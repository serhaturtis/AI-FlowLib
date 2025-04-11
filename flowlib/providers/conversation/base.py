"""Base conversation provider for flowlib.

This module defines the base class for conversation providers, which
handle interactions between users and agents through different interfaces.
"""

import logging
from abc import abstractmethod
from typing import Optional, Dict, Any, TypeVar, Generic

from ...flows.base import FlowSettings
from ...core.errors import ExecutionError
from ..base import Provider

logger = logging.getLogger(__name__)

# Define a type variable for conversation provider settings
T = TypeVar('T', bound=FlowSettings)

class ConversationProviderSettings(FlowSettings):
    """Base settings class for conversation providers."""
    pass

class ConversationProvider(Provider[T], Generic[T]):
    """Base class for conversation providers.
    
    Conversation providers handle the interaction between users and agents
    through different interfaces (CLI, web, API, etc.).
    """
    
    async def _initialize(self) -> None:
        """Initialize the conversation provider.
        
        This method is called by the initialize() method in the Provider base class.
        Derived classes should override this to perform their initialization.
        """
        logger.info(f"Initializing conversation provider: {self.name}")
    
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
            
    async def start_conversation(self):
        """Start a new conversation session."""
        logger.info(f"Starting conversation with provider: {self.name}")
    
    async def end_conversation(self):
        """End the current conversation session."""
        logger.info(f"Ending conversation with provider: {self.name}")
    
    async def shutdown(self):
        """Clean up resources used by the provider.
        
        This method overrides the Provider base class method.
        """
        logger.info(f"Shutting down {self.name} conversation provider")
        await super().shutdown() 