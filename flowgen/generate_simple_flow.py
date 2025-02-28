"""Example script for generating a simple calculator flow."""

import asyncio
import logging
import os
from pathlib import Path

from .models.flowgen_models import (
    FlowGeneratorInput, FlowRequirement
)
from .flows.flow_generator import FlowGenerator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main_async():
    """Run the flow generator example."""
    
    # Example input for generating a simple calculator flow
    input_data = FlowGeneratorInput(
        task_description="""
        Create a simple calculator flow that performs basic arithmetic operations. The flow should:
        1. Validate input numbers and operation
        2. Perform the calculation
        3. Format and return the result
        """,
        input_requirements=[
            FlowRequirement(
                description="First number for calculation",
                examples=["5", "3.14"]
            ),
            FlowRequirement(
                description="Second number for calculation",
                examples=["2", "1.618"]
            ),
            FlowRequirement(
                description="Operation to perform",
                examples=["add", "subtract", "multiply", "divide"]
            )
        ],
        output_requirements=[
            FlowRequirement(
                description="Calculation result with operation details",
                examples=["{'operation': 'add', 'result': 7.0, 'success': true}"]
            )
        ],
        # No LLM required for this simple flow
        resource_requirements=[]
    )
    
    try:
        logger.info("Starting flow generation...")
        
        # Create and run the flow generator
        async with FlowGenerator() as generator:
            result = await generator.generate_flow(input_data)
            
            if result.status == "success":
                logger.info("Flow generation completed successfully!")
                logger.info("Generated %d files", len(result.generated_files))
                # Log the components for visibility
                logger.info("Flow components:")
                for comp in result.flow_description.components:
                    logger.info("- %s: %s", comp.name, comp.purpose)
            else:
                logger.error("Flow generation failed:")
                if result.errors:
                    for error in result.errors:
                        logger.error("- %s", error)
                else:
                    logger.error(result.error)
                
    except Exception as e:
        print(e)
        logger.error(f"Flow generation failed: {str(e)}")
        raise

def main():
    """Main entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 