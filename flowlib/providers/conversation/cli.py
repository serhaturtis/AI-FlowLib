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
        print(f"\nAgent: {response}")
    
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
            print(f"  Reasoning: {latest_plan.get('reasoning', 'No reasoning')[:100]}...")
            print(f"  Selected flow: {latest_plan.get('flow', 'No flow selected')}")
        
        # Show execution information
        print("\n⚙️ Recent Executions:")
        for i, execution in enumerate(execution_history[-3:]):
            action = execution.get("action", "unknown")
            flow = execution.get("flow", "unknown")
            print(f"  {i+1}. Action: {action}, Flow: {flow}")
        
        # Show reflection information
        reflection_entries = [e for e in execution_history if e.get("action") == "reflect"]
        if reflection_entries:
            latest_reflection = reflection_entries[-1]
            reflection = latest_reflection.get("reflection", "No reflection available")
            
            print("\n🔍 Latest Reflection:")
            reflection_lines = reflection.split('\n')
            for line in reflection_lines[:3]:
                if line.strip():
                    print(f"  - {line.strip()}")
            if len(reflection_lines) > 3:
                print("  - ...")
            
            # Show new information
            new_info = latest_reflection.get("new_information", [])
            if new_info:
                print("\n💡 New Information:")
                for info in new_info[:3]:
                    print(f"  - {info}")
                if len(new_info) > 3:
                    print(f"  - ... ({len(new_info) - 3} more items)")
        
        print("-----------------------------") 