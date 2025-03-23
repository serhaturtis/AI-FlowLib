"""CLI conversation provider implementation.

This module provides a command-line interface for agent conversations.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List

from ...core.registry import conversation_provider
from .base import ConversationProvider

logger = logging.getLogger(__name__)

@conversation_provider("cli")
class CLIConversationProvider(ConversationProvider):
    """CLI-based conversation provider.
    
    This provider facilitates conversations through a command-line interface,
    with support for customizing prompts and displaying execution details.
    """
    
    def __init__(self, name: str = "cli", settings: Optional[Dict[str, Any]] = None):
        """Initialize the CLI conversation provider.
        
        Args:
            name: Provider name
            settings: Optional provider settings including:
                - prompt: Input prompt text (default: "You: ")
                - exit_commands: List of commands to exit (default: ["exit", "quit", "bye"])
                - show_execution_details: Whether to show execution details (default: True)
        """
        super().__init__(name, settings)
        
        # Get settings with defaults
        self.prompt = self.settings.get("prompt", "You: ")
        self.exit_commands = self.settings.get("exit_commands", ["exit", "quit", "bye"])
        self.show_execution_details = self.settings.get("show_execution_details", True)
    
    async def get_next_input(self) -> Optional[str]:
        """Get input from command line."""
        # Run input in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            user_input = await loop.run_in_executor(None, lambda: input(f"\n{self.prompt}"))
            
            # Check if user wants to exit
            if user_input.lower() in self.exit_commands:
                print("Goodbye!")
                return None
                
            return user_input
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return None
    
    async def send_response(self, response: str):
        """Print response to console."""
        # Process the response to replace literal escape sequences
        processed_response = self._process_escape_sequences(response)
        print(f"\nAgent: {processed_response}")
    
    def _process_escape_sequences(self, text: str) -> str:
        """Process any literal escape sequences in the text.
        
        This method replaces literal escape sequences like '\\n' with
        their actual character representation.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text with proper escape sequences
        """
        import re
        
        # Replace common literal escape sequences with their character representation
        # Start with double backslashes (\\n) as they may appear in JSON strings
        text = text.replace('\\\\n', '\n')
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\r', '\r')
        
        # Handle unicode escape sequences like \u00A0
        text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
        
        return text
    
    async def show_details(self, details: Dict[str, Any]):
        """Display execution details in console."""
        if not self.show_execution_details:
            return
            
        # Implementation similar to display_agent_execution_details function
        print("\n--- Agent Execution Details ---")
        
        state = details.get("state")
        if not state:
            print("No detailed agent execution information available.")
            print("-----------------------------")
            return
            
        # Show execution progress
        progress = getattr(state, "progress", 0)
        is_complete = getattr(state, "is_complete", False)
        print(f"Progress: {progress}%, Complete: {is_complete}")
        
        # Get execution history
        execution_history = details.get("execution_history", [])
        if not execution_history:
            print("No execution history available.")
            print("-----------------------------")
            return
        
        # Show planning information
        planning_entries = [e for e in execution_history if e.get("action") == "plan"]
        if planning_entries:
            latest_plan = planning_entries[-1]
            print("\n📋 Planning:")
            reasoning = self._process_escape_sequences(latest_plan.get('reasoning', 'No reasoning')[:100])
            print(f"  Reasoning: {reasoning}...")
            flow = self._process_escape_sequences(latest_plan.get('flow', 'No flow selected'))
            print(f"  Selected flow: {flow}")
        
        # Show execution information
        print("\n⚙️ Recent Executions:")
        for i, execution in enumerate(execution_history[-3:]):
            action = self._process_escape_sequences(execution.get("action", "unknown"))
            flow = self._process_escape_sequences(execution.get("flow", "unknown"))
            print(f"  {i+1}. Action: {action}, Flow: {flow}")
        
        # Show reflection information
        reflection_entries = [e for e in execution_history if e.get("action") == "reflect"]
        if reflection_entries:
            latest_reflection = reflection_entries[-1]
            reflection = self._process_escape_sequences(latest_reflection.get("reflection", "No reflection available"))
            
            print("\n🔍 Latest Reflection:")
            reflection_lines = reflection.split('\n')
            for line in reflection_lines[:3]:
                if line.strip():
                    print(f"  - {line.strip()}")
            if len(reflection_lines) > 3:
                print("  - ...")
            
            # New information handling has been removed as it's now handled by memory extraction flow
        
        print("-----------------------------")
    
    async def handle_error(self, error: Exception) -> str:
        """Handle errors during conversation.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Error message to display to the user
        """
        # Get the error message from the base implementation
        error_message = await super().handle_error(error)
        
        # Process the error message to replace escape sequences
        processed_message = self._process_escape_sequences(error_message)
        
        # Print the processed error message
        print(f"\nError: {processed_message}")
        
        return error_message 