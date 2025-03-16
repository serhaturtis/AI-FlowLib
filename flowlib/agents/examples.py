"""Example usage of the agent system.

This module demonstrates how to create and use agents 
with the consolidated flowlib library.
"""

import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ..core.registry import ResourceRegistry
from ..flows import Stage, CompositeFlow
from ..providers.llm import LlamaProvider

from .models import AgentConfig
from .llm_agent import LLMAgent

# Example: Define some models for flows
class SearchQuery(BaseModel):
    """Search query input."""
    query: str = Field(..., description="The search query to execute")
    max_results: int = Field(5, description="Maximum number of results to return")

class SearchResult(BaseModel):
    """Search result output."""
    results: List[Dict[str, Any]] = Field(..., description="List of search results")
    total_found: int = Field(..., description="Total number of results found")

class SummarizeInput(BaseModel):
    """Input for summarization."""
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(100, description="Maximum length of summary")

class SummaryOutput(BaseModel):
    """Output from summarization."""
    summary: str = Field(..., description="Generated summary")
    keywords: List[str] = Field(..., description="Key topics or themes")

# Example: Create some simple flows
async def search_function(context):
    """Simulated search flow."""
    query = context.data.query
    max_results = context.data.max_results
    
    # Simulate search results
    results = [
        {"title": f"Result {i} for {query}", "url": f"https://example.com/{i}", "snippet": f"This is result {i} for query '{query}'"}
        for i in range(1, min(max_results + 1, 4))
    ]
    
    return SearchResult(
        results=results,
        total_found=len(results)
    )

async def summarize_function(context):
    """Simulated summarization flow."""
    text = context.data.text
    
    # Simple summarization (in a real case this would use an LLM)
    summary = f"Summary of: {text[:30]}..." if len(text) > 30 else text
    keywords = ["example", "summary", "automated"]
    
    return SummaryOutput(
        summary=summary,
        keywords=keywords
    )

def create_example_agent():
    """Create an example agent with sample flows.
    
    Returns:
        Configured agent instance
    """
    # Define flows
    search_flow = Stage(
        name="search",
        process_fn=search_function,
        input_schema=SearchQuery,
        output_schema=SearchResult,
        metadata={"description": "Search for information on a topic"}
    )
    
    summarize_flow = Stage(
        name="summarize",
        process_fn=summarize_function,
        input_schema=SummarizeInput,
        output_schema=SummaryOutput,
        metadata={"description": "Generate a summary of text"}
    )
    
    # Register LLM provider (assuming it's already configured)
    # In a real application, you would register your LLM provider before creating the agent
    if not ResourceRegistry.has_resource("provider", "llm"):
        # This is just an example placeholder
        # In a real application, you would configure and register a proper LLM provider
        llm_provider = LlamaProvider(
            name="llm",
            model_path="/path/to/model.gguf"  # Replace with actual path
        )
        ResourceRegistry.register_resource("provider", "llm", llm_provider)
    
    # Configure the agent
    agent_config = AgentConfig(
        provider_name="llm",
        planner_model="llama3",  # Replace with actual model name
        input_generator_model="llama3",
        reflection_model="llama3",
        default_system_prompt="You are a helpful AI assistant that helps users find and summarize information."
    )
    
    # Create the agent
    agent = LLMAgent(
        flows=[search_flow, summarize_flow],
        config=agent_config,
        task_description="Search for information about climate change and summarize the key points."
    )
    
    return agent

async def run_example():
    """Run the example agent.
    
    This demonstrates the full agent execution cycle.
    """
    try:
        agent = create_example_agent()
        
        print("Starting agent execution...")
        result = await agent.execute()
        
        print("\nAgent execution complete!")
        print(f"Final result: {result}")
        
        # Show agent state summary
        print("\nAgent state summary:")
        state_summary = agent.get_state_summary()
        for key, value in state_summary.items():
            print(f"- {key}: {value}")
            
    except Exception as e:
        print(f"Error running agent: {str(e)}")

# To run the example
if __name__ == "__main__":
    asyncio.run(run_example()) 