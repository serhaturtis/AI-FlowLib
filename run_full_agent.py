"""
Full Agent Conversation Example with Planning, Execution, and Reflection.

This script demonstrates a complete agent with conversation capabilities,
short-term and long-term memory, and full planning, execution, and reflection.
"""

import asyncio
import logging
import os
import argparse
from typing import List, Optional, Dict, Any

import flowlib as fl
from flowlib.core.models import Context
# Import all flow classes explicitly to ensure their decorators register them
from flowlib.agents.flows import (
    ConversationInput, ConversationOutput, 
    ConversationFlow, AgentPlanningFlow, AgentInputGenerationFlow, AgentReflectionFlow
)
from flowlib.agents.models import AgentState
from flowlib.agents import agent, MemoryContext, FullConversationalAgent
from flowlib.agents.config import AgentConfig, ModelConfig
from flowlib.core.registry.constants import ProviderType
from flowlib.providers.conversation import (
    CLIConversationProvider, WebConversationProvider, APIConversationProvider,
    create_conversation_provider, get_available_providers
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Model Configuration - will be used if no config file is provided
@fl.model("default")
class LLMModelConfig:
    """Configuration for the LLM model."""
    path = os.environ.get("MODEL_PATH", "/home/swr/tools/models/my_biz/my_models/phi-4-Q5_K_M.gguf")
    model_type = "chatml"
    n_ctx = 8192
    n_threads = 4
    n_batch = 512
    use_gpu = True
    n_gpu_layers = -1
    temperature = 0.7

# Function to debug flows and registry information
def debug_flow_registry(agent):
    """Log detailed information about flows and registry for debugging purposes."""
    # Log what flows were found
    flow_names = list(agent.flows.keys())
    logger.info(f"Agent loaded these flows: {', '.join(flow_names)}")
    
    # Check registered flows in stage_registry
    from flowlib.flows.registry import stage_registry
    registered_flow_names = stage_registry.get_flows()
    logger.info(f"Flows in stage_registry: {', '.join(registered_flow_names)}")
    registered_instances = stage_registry.get_flow_instances()
    logger.info(f"Flow instances in registry: {list(registered_instances.keys())}")

    # Check if flow.name attribute was set correctly
    for flow_name, flow_instance in agent.flows.items():
        logger.info(f"Agent flow '{flow_name}' has name attribute: {getattr(flow_instance, 'name', 'NOT_SET')}")
        if hasattr(flow_instance, "__flow_name__"):
            logger.info(f"  __flow_name__: {flow_instance.__flow_name__}")
        logger.info(f"  class name: {flow_instance.__class__.__name__}")

async def run_agent_with_provider(agent: FullConversationalAgent, provider_type: str, provider_settings: Optional[Dict[str, Any]] = None):
    """Run the agent with the specified conversation provider.
    
    Args:
        agent: The agent to run
        provider_type: Type of conversation provider to use (cli, web, api)
        provider_settings: Optional settings for the provider
    """
    provider_settings = provider_settings or {}
    
    # Create the appropriate conversation provider
    if provider_type == "cli":
        provider = CLIConversationProvider(
            settings=provider_settings or {
                "prompt": "You: ",
                "exit_commands": ["exit", "quit", "bye"],
                "show_execution_details": True
            }
        )
    elif provider_type == "web":
        provider = WebConversationProvider(
            settings=provider_settings or {
                "host": "localhost",
                "port": 8080,
                "static_path": "./static",
                "show_execution_details": True
            }
        )
    elif provider_type == "api":
        provider = APIConversationProvider(
            settings=provider_settings or {
                "host": "localhost",
                "port": 8081,
                "conversation_timeout": 1800
            }
        )
    else:
        logger.error(f"Unknown provider type: {provider_type}")
        return
    
    # Initialize the provider
    await provider.initialize()
    
    try:
        # Welcome message for CLI provider
        if provider_type == "cli":
            print("\nWelcome! I'm your conversational agent. Type 'exit', 'quit', or 'bye' to end the conversation.")
        
        # Main conversation loop
        while True:
            # Get user input
            user_input = await provider.get_next_input()
            if user_input is None:
                break
                
            try:
                # Handle the message and get response
                response = await agent.handle_message(user_input)
                
                # Send response to user
                await provider.send_response(response)
                
                # Show execution details if available
                if hasattr(agent, "state") and agent.state:
                    await provider.show_details({
                        "state": agent.state,
                        "execution_history": agent.state.execution_history
                    })
                    
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await provider.handle_error(e)
        
        # For web and API providers, we need to keep the application running
        if provider_type in ["web", "api"]:
            logger.info(f"Provider {provider_type} is running. Press Ctrl+C to stop.")
            try:
                # Keep the application running until interrupted
                while True:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.info(f"Stopping {provider_type} provider...")
    
    finally:
        # Clean up resources
        if hasattr(provider, 'shutdown'):
            logger.info("Shutting down provider...")
            await provider.shutdown()
            logger.info("Provider shutdown complete.")

def generate_config_template(output_path: str):
    """Generate a config template file.
    
    Args:
        output_path: Path to save the template to
    """
    # Create a default config
    config = AgentConfig()
    
    # Set some values to make the template more useful
    config.name = "my_agent"
    config.description = "Custom conversational agent"
    
    # Add model configs
    config.models = {
        "default": ModelConfig(
            name="default",
            path="/path/to/your/model.gguf",
            model_type="chatml",
            temperature=0.7
        ),
        "creative": ModelConfig(
            name="creative",
            path="/path/to/your/model.gguf",
            model_type="chatml",
            temperature=0.9,
            top_p=0.95
        )
    }
    
    # Save to file
    config.save_to_file(output_path)
    print(f"Config template saved to {output_path}")

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run agent with different conversation providers")
    parser.add_argument("--provider", "-p", default="cli", choices=["cli", "web", "api"],
                        help="Type of conversation provider to use")
    parser.add_argument("--host", default="localhost", help="Host for web/API provider")
    parser.add_argument("--port", type=int, help="Port for web/API provider")
    parser.add_argument("--list-providers", action="store_true", help="List available conversation providers and exit")
    parser.add_argument("--model-path", help="Path to the LLM model file")
    parser.add_argument("--model-temperature", type=float, help="Temperature setting for the LLM", default=0.7)
    parser.add_argument("--config", "-c", help="Path to agent configuration file")
    parser.add_argument("--generate-config", help="Generate a config template file at the specified path and exit")
    args = parser.parse_args()
    
    # Generate config template if requested
    if args.generate_config:
        generate_config_template(args.generate_config)
        return
    
    # List available providers if requested
    if args.list_providers:
        available_providers = get_available_providers()
        print("\nAvailable Conversation Providers:")
        print("================================")
        for name, description in available_providers.items():
            print(f"- {name}: {description.split('.')[0]}")
        return
    
    # Override model path if provided
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    
    # Set up provider settings based on arguments
    provider_settings = {}
    if args.provider in ["web", "api"]:
        provider_settings["host"] = args.host
        if args.port:
            provider_settings["port"] = args.port
    
    # Show which provider we're using
    print(f"\nUsing {args.provider.upper()} conversation provider")
    if args.provider == "web":
        print(f"Web interface will be available at: http://{args.host}:{provider_settings.get('port', 8080)}")
    elif args.provider == "api":
        print(f"API interface will be available at: http://{args.host}:{provider_settings.get('port', 8081)}/api/conversations")
    
    try:
        # Load config if provided
        config = None
        if args.config:
            print(f"Loading configuration from {args.config}")
            config = AgentConfig.from_file(args.config)
            print(f"Using agent: {config.name} - {config.description}")
            
            # Override with command line parameters if provided
            if args.model_path:
                if "default" in config.models:
                    config.models["default"].path = args.model_path
                    print(f"Overriding model path to: {args.model_path}")
                    
            if args.model_temperature is not None:
                if "default" in config.models:
                    config.models["default"].temperature = args.model_temperature
                    print(f"Overriding model temperature to: {args.model_temperature}")
        
        # Initialize the agent
        agent = FullConversationalAgent()
        
        # Update model temperature if no config file
        if not config and hasattr(LLMModelConfig, "temperature") and args.model_temperature is not None:
            LLMModelConfig.temperature = args.model_temperature
            print(f"Setting model temperature to: {args.model_temperature}")
        
        # Initialize the agent - this is required before handling messages
        logger.info("Initializing agent...")
        await agent.initialize()
        logger.info("Agent initialized successfully")
        
        # Debug flow registry information (optional)
        debug_flow_registry(agent)
        
        # Run the agent with the specified provider
        await run_agent_with_provider(agent, args.provider, provider_settings)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 