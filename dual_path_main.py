"""
Dual-Path Conversational Agent using FlowLib

This script creates and runs a dual-path agent that can handle both simple conversations
and complex tasks with separate pathways for optimal processing.
"""

import asyncio
import logging
import datetime
import os
import uuid
from typing import Dict, Any, Optional

from flowlib.agent.models.config import (
    StatePersistenceConfig, 
    AgentConfig,
    PlannerConfig, 
    ReflectionConfig, 
    EngineConfig, 
)
from flowlib.agent.core.dual_path_agent import DualPathAgent
from flowlib.agent.persistence.factory import create_state_persister

# Import the new agent decorator and registry
from flowlib.agent.decorators.base import dual_path_agent
from flowlib.agent.registry import agent_registry

# Import flow components
from flowlib.agent.conversation.flow import ConversationFlow
# Import the new runner
from flowlib.agent.runners import run_interactive_session # Import from package
# Import the persistence utility
from flowlib.agent.persistence.utils import list_saved_states_metadata
# Import the config loader
from flowlib.agent.config_utils import load_agent_config
# Import AgentConfig for type hint
from flowlib.agent.models.config import AgentConfig

# Import the correct decorator
from flowlib.providers.embedding.decorators import embedding_model
from flowlib.resources.decorators import model # Keep for LLM

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
STATE_DIR = "./states"

# ===================== Define Embedding Model Configuration =====================
@embedding_model("default_embedding")
class EmbeddingModelConfig:
    """Configuration for the default embedding model."""
    # IMPORTANT: Replace this with the actual path to your embedding model file
    path = "/home/swr/tools/models/embedding/bge-m3-q8_0.gguf"
    # Adjust other parameters as needed for the embedding model
    model_type = "embedding" # Or appropriate type if different
    n_ctx = 512  # Often smaller for embedding models
    n_threads = 4
    n_batch = 512
    use_gpu = True  # Or False
    n_gpu_layers = -1 # Or specific number
    # Embedding models usually don't use temperature/max_tokens in the same way

# ===================== Define LLM Model Configuration =====================
@model("default")
class LLMModelConfig:
    """Configuration for the conversation LLM."""
    path = "path"
    #path = "path"
    model_type = "llama3"
    n_ctx = 4096
    n_threads = 4
    n_batch = 512
    use_gpu = True
    n_gpu_layers = -1
    temperature = 0.7
    max_tokens = 1000

# ===================== Define Agent Class =====================
@dual_path_agent(name="DualPathAssistant", description="An assistant that handles conversation and tasks.")
class MyDualPathAssistant(DualPathAgent):
    # This class now represents our agent implementation.
    # DualPathAgent base class provides the core logic.
    # We can add custom methods or override base methods here if needed.
    pass

# ===================== Main Agent Setup =====================
async def setup_agent(task_id: Optional[str] = None) -> DualPathAgent:
    """Set up and configure the dual-path agent.
    
    Args:
        task_id: Optional task ID to resume a conversation
        
    Returns:
        Configured agent ready for conversation
    """
    # Load the base configuration from the YAML file
    try:
        config: AgentConfig = load_agent_config("agent_config.yaml")
        logger.info(f"Agent configuration loaded successfully for '{config.name}'")
        # Debug print: Show loaded config
        logger.debug(f"Loaded config object: {config}")
    except FileNotFoundError:
        logger.error("CRITICAL: agent_config.yaml not found. Please create it.")
        raise
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load or validate agent_config.yaml: {e}", exc_info=True)
        raise

    # --- Application-specific Overrides --- 
    # Override task_id and auto_load if resuming a conversation
    if task_id:
        logger.info(f"Resuming conversation, setting task_id to {task_id}")
        config.task_id = task_id
        if config.state_config:
             config.state_config.auto_load = True
        else:
             logger.warning("Task ID provided for resume, but no state_config found in agent_config.yaml")
    # Ensure state directory exists based on loaded config
    
    # Debug print: Show config after potential overrides
    logger.debug(f"Final config before agent instantiation: {config}")
    
    if config.state_config and config.state_config.persistence_type == 'file' and config.state_config.base_path:
         os.makedirs(config.state_config.base_path, exist_ok=True)
    # ----------------------------------------
    
    # Create the agent with configuration
    # --- Get agent class from registry and instantiate --- 
    AgentClass = agent_registry.get_agent_class("DualPathAssistant")
    if not AgentClass:
        raise RuntimeError("Agent 'DualPathAssistant' not found in registry. Ensure it's decorated.")
    
    # Instantiate the registered agent class
    agent = AgentClass(config=config) 
    # -------------------------------------------------------
    
    # Create conversation flow
    conversation_flow = ConversationFlow()
    
    # Register flows
    agent.register_flow(conversation_flow)
    
    # Initialize agent
    await agent.initialize()
    
    # Load state if task_id provided
    if task_id:
        await agent.load_state(task_id)
        
    return agent

# ===================== Main Function =====================
async def main():
    """Main entry point for the conversational agent."""
    # Get task ID
    task_id = await prompt_for_conversation()
    
    # Set up the agent
    agent = await setup_agent(task_id)
    
    # Run the interactive session using the library function
    await run_interactive_session(agent)

async def prompt_for_conversation() -> Optional[str]:
    """Prompt user to select an existing conversation or start a new one.
    
    Returns:
        Selected task_id or None for a new conversation
    """
    # List available conversations using the persistence utility
    # Get list of metadata dictionaries
    states_metadata = await list_saved_states_metadata(persister_type="file", base_path=STATE_DIR)
    
    # Convert list to dictionary indexed by task_id for easier lookup/sorting
    conversations = {s["task_id"]: s for s in states_metadata}
    
    if not conversations:
        print("\nNo saved conversations found. Starting a new conversation.")
        return None
        
    # Display available conversations
    print("\n=== Available Conversations ===")
    print("0. Start a new conversation")
    
    sorted_conversations = sorted(
        conversations.items(), 
        key=lambda x: x[1].get("timestamp", ""), 
        reverse=True
    )
    
    for i, (task_id, metadata) in enumerate(sorted_conversations, 1):
        timestamp = metadata.get("timestamp", "Unknown date")
        description = metadata.get("task_description", "No description")
        print(f"{i}. {description} (Last updated: {timestamp})")
    
    # Get user selection
    while True:
        try:
            choice = input("\nSelect a conversation (0-{}): ".format(len(sorted_conversations)))
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(sorted_conversations):
                return sorted_conversations[choice_num-1][0]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

if __name__ == "__main__":
    asyncio.run(main()) 
