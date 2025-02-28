"""Example script for generating flows using the FlowGenerator flow."""

import asyncio
import logging
import os
from pathlib import Path

from .models.flowgen_models import (
    FlowGeneratorInput, FlowRequirement, ResourceRequirement
)
from .flows.flow_generator import FlowGenerator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set workspace root environment variable
os.environ["ROOT_FOLDER"] = str(Path(__file__))

async def main_async():
    """Run the flow generator example."""
    
    # Example input for generating a simple document analyzer flow
    input_data = FlowGeneratorInput(
        task_description="""
        Create a flow that analyzes documents to extract key information. The flow should:
        1. Extract main topics and themes
        2. Analyze sentiment and tone
        3. Generate a concise summary
        4. Identify key entities and relationships
        """,
        input_requirements=[
            FlowRequirement(
                description="Text content to analyze",
                examples=["Article text", "Blog post", "Research paper"]
            )
        ],
        output_requirements=[
            FlowRequirement(
                description="Topic analysis with main themes and their relevance scores",
                examples=["[{topic: 'AI', relevance: 0.85}, {topic: 'Ethics', relevance: 0.72}]"]
            ),
            FlowRequirement(
                description="Sentiment analysis with overall tone and confidence",
                examples=["{'sentiment': 'positive', 'confidence': 0.92, 'tone': 'professional'}"]
            ),
            FlowRequirement(
                description="Concise summary of the main points",
                examples=["3-5 sentence summary capturing key insights"]
            )
        ],
        resource_requirements=[
            ResourceRequirement(
                type="llm",
                description="LLM for text analysis and summarization",
                constraints={"context_length": "8192 tokens", "response_format": "structured"}
            )
        ]
    )
    
    try:
        logger.info("Starting flow generation...")
        
        # Create and run the flow generator
        async with FlowGenerator() as generator:
            result = await generator.generate_flow(input_data)
            
            if result["status"] == "success":
                logger.info("Flow generation completed successfully!")
                logger.info("Generated %d files", len(result["generated_files"]))
            else:
                logger.error("Flow generation failed:")
                if "errors" in result:
                    for error in result["errors"]:
                        logger.error("- %s", error)
                else:
                    logger.error(result["error"])
                
    except Exception as e:
        logger.error(f"Flow generation failed: {str(e)}")
        raise

def main():
    """Main entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 