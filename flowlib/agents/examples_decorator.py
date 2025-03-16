"""Example usage of the agent decorator.

This module demonstrates how to create and use agents with the decorator-based
approach in the consolidated flowlib library.
"""

import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ..core.registry import ResourceRegistry
from ..flows import Stage, CompositeFlow, stage
from ..providers.llm import LlamaProvider

from .models import AgentState, PlanningResponse, ReflectionResponse
from .decorators import agent

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

# Create stage-based flows using decorators
@stage(input_schema=SearchQuery, output_schema=SearchResult)
async def search_flow(context):
    """Search for information on a topic."""
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

@stage(input_schema=SummarizeInput, output_schema=SummaryOutput)
async def summarize_flow(context):
    """Generate a summary of text."""
    text = context.data.text
    
    # Simple summarization (in a real case this would use an LLM)
    summary = f"Summary of: {text[:30]}..." if len(text) > 30 else text
    keywords = ["example", "summary", "automated"]
    
    return SummaryOutput(
        summary=summary,
        keywords=keywords
    )

# Define a custom agent using the decorator
@agent(
    provider_name="llm",
    planner_model="llama3",  # Replace with actual model
    default_system_prompt="You are a helpful AI assistant that helps users find and summarize information."
)
class ResearchAgent:
    """Agent that can search for information and summarize it."""
    
    # You can define a default model name for the agent
    model_name = "llama3"
    
    def __init__(self, custom_config=None):
        """Initialize custom agent properties."""
        self.custom_config = custom_config or {}
    
    # Optional method to customize planning
    async def plan_next_action(self, state, provider, config, system_prompt, user_prompt):
        """Custom planning logic to decide the next step."""
        # Here you could add custom logic before/after calling the provider
        # For this example, we'll use the standard LLM approach
        
        response = await provider.generate_structured_output(
            model=config.planner_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=PlanningResponse,
            max_retries=config.max_retries
        )
        
        # You could modify the response here if needed
        print(f"Planning decision: {response.decision} (confidence: {response.confidence})")
        
        return response
    
    # Optional method to customize input generation
    async def generate_flow_inputs(self, flow, state, provider, config, system_prompt, user_prompt, last_result):
        """Custom input generation logic."""
        # Here you could add custom logic based on the flow type
        
        if flow.name == "search_flow" and not state.history:
            # For the first search, we could hardcode or customize inputs
            print("Generating custom inputs for first search")
            return SearchQuery(
                query="climate change impacts",
                max_results=3
            )
        
        # Otherwise use the standard approach
        if flow.input_schema:
            return await provider.generate_structured_output(
                model=config.input_generator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=flow.input_schema,
                max_retries=config.max_retries
            )
        else:
            response = await provider.generate(
                model=config.input_generator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=config.max_retries
            )
            
            # Try to parse as JSON
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return response
    
    # Optional method to customize reflection
    async def reflect_on_result(self, result, state, provider, config, system_prompt, user_prompt):
        """Custom reflection logic."""
        # Standard approach using the LLM provider
        reflection = await provider.generate_structured_output(
            model=config.reflection_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=ReflectionResponse,
            max_retries=config.max_retries
        )
        
        # Add custom logging
        print(f"Reflection insights: {', '.join(reflection.key_insights)}")
        
        return reflection

async def run_example():
    """Run the example with the decorated agent."""
    try:
        # Set up LLM provider (placeholder)
        if not ResourceRegistry.has_resource("provider", "llm"):
            # This is just an example placeholder
            llm_provider = LlamaProvider(
                name="llm",
                model_path="/path/to/model.gguf"  # Replace with actual path
            )
            ResourceRegistry.register_resource("provider", "llm", llm_provider)
        
        # Create agent using class method provided by decorator
        agent = ResearchAgent.create(
            flows=[search_flow, summarize_flow],
            task_description="Search for information about climate change and summarize the key points.",
            custom_config={"max_steps": 5}  # Custom parameter passed to agent
        )
        
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