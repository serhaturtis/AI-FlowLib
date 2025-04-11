"""CLI-based conversation provider.

This module provides a CLI-based implementation of the conversation provider interface.
It allows for conversation interactions via the command line interface.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator

from ..decorators import conversation_provider
from .base import ConversationProvider, ConversationProviderSettings
from ...utils.formatting import (
    process_escape_sequences,
    format_agent_execution_details
)

logger = logging.getLogger(__name__)


class CLIConversationProviderSettings(ConversationProviderSettings):
    """Settings for CLI conversation provider.
    
    Attributes:
        show_execution_details: Whether to show execution details in the console
    """
    show_execution_details: bool = True


@conversation_provider("cli")
class CLIConversationProvider(ConversationProvider[CLIConversationProviderSettings]):
    """CLI-based conversation provider implementation.
    
    This provider enables conversation interaction via command line interface.
    It handles sending and receiving messages, as well as displaying execution details.
    """
    
    def __init__(
        self,
        name: str = "cli",
        settings: Optional[CLIConversationProviderSettings] = None,
        provider_type: str = "conversation"
    ):
        """Initialize CLI conversation provider.
        
        Args:
            name: Provider name
            settings: Provider settings
            provider_type: Provider type
        """
        # Use default settings if none provided
        if settings is None:
            settings = CLIConversationProviderSettings()
        
        super().__init__(name=name, settings=settings, provider_type=provider_type)
        
    async def _initialize(self) -> None:
        """Initialize the CLI conversation provider."""
        await super()._initialize()
        # No additional initialization needed for CLI provider
        
    async def get_next_input(self) -> str:
        """Get user input from command line.
        
        Returns:
            User input string
        """
        return input("User: ")
        
    async def send_response(self, response: str):
        """Send a response to the user via command line.
        
        Args:
            response: Response text to display
        """
        processed_response = process_escape_sequences(response)
        print(f"\nAgent: {processed_response}")
    
    async def handle_error(self, error: Any) -> str:
        """Handle an error by displaying it properly in the CLI.
        
        Args:
            error: Error object or message
            
        Returns:
            Formatted error message
        """
        # Convert to string if it's not already
        error_message = str(error) if not isinstance(error, str) else error
        
        if not error_message:
            return ""
            
        processed_message = process_escape_sequences(error_message)
        print(f"\nError: {processed_message}")
        return processed_message
        
    async def show_details(self, details: Dict[str, Any]):
        """Display execution details in console.
        
        Args:
            details: Dictionary of execution details
        """
        if not self.settings.show_execution_details:
            return
            
        # Format and display the execution details
        formatted_details = format_agent_execution_details(details)
        print(f"\n{formatted_details}")
        
    async def start_conversation(self):
        """Start a new conversation session."""
        await super().start_conversation()
        print("\nNew conversation started. Type 'exit' to end the conversation.")
        
    async def end_conversation(self):
        """End the current conversation session."""
        await super().end_conversation()
        print("\nConversation ended.")
        
    async def send_thinking_indicator(self) -> AsyncIterator[bool]:
        """Send a thinking indicator and yield control while the agent is processing.
        
        Yields:
            True while thinking is in progress
        """
        print("\nThinking...", end="", flush=True)
        try:
            while True:
                await asyncio.sleep(0.5)
                print(".", end="", flush=True)
                yield True
        finally:
            print("", flush=True)  # New line after thinking

    # Remove _process_escape_sequences as it's now imported from the formatting module 