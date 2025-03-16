"""Examples demonstrating the Flow system with enforced rules.

This module shows how to create and use flows following the new rules:
1. Each flow must have exactly one pipeline method
2. Only execute() can be called from outside the flow
"""

import asyncio
from pydantic import BaseModel
from typing import Dict, Any, List

from ..core.models.context import Context
from . import flow, stage, pipeline


# Example 1: Basic Flow with Stages
@flow
class SimpleFlow:
    """A simple flow demonstrating the standard pattern."""
    
    @stage
    async def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the input data."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
        if "text" not in data:
            raise ValueError("Input must contain a 'text' field")
        return {"validated_text": data["text"]}
    
    @stage
    async def process_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the validated text."""
        processed = data["validated_text"].upper()
        return {"processed_text": processed}
    
    @pipeline
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the flow pipeline."""
        # Note: We call the stage methods with leading underscore
        # as they've been renamed by the @flow decorator
        validated = await self._validate_input(data)
        processed = await self._process_text(validated)
        return {
            "result": processed["processed_text"],
            "original": data["text"]
        }


# Example 2: Flow with Type Validation
class TextInput(BaseModel):
    """Input model for text processing."""
    text: str
    options: Dict[str, Any] = {}

class TextOutput(BaseModel):
    """Output model for text processing."""
    result: str
    metadata: Dict[str, Any] = {}

@flow
class TypedFlow:
    """A flow with input and output type validation."""
    
    @stage(input_model=TextInput, output_model=Dict)
    async def process(self, data: TextInput) -> Dict[str, Any]:
        """Process the input data."""
        return {
            "processed": data.text.upper(),
            "options_used": data.options
        }
    
    @stage(input_model=Dict, output_model=TextOutput)
    async def format_output(self, data: Dict[str, Any]) -> TextOutput:
        """Format the output data."""
        return TextOutput(
            result=data["processed"],
            metadata={"options": data["options_used"]}
        )
    
    @pipeline(input_model=TextInput, output_model=TextOutput)
    async def execute_pipeline(self, data: TextInput) -> TextOutput:
        """Execute the pipeline with type checking."""
        processed = await self._process(data)
        return await self._format_output(processed)


# Example 3: Composite Flow
@flow
class CompositeFlow:
    """A flow that combines multiple flows."""
    
    def __init__(self):
        self.simple_flow = SimpleFlow()
        self.typed_flow = TypedFlow()
    
    @pipeline
    async def run_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all flows in sequence."""
        # We can only call execute() on the flow instances
        simple_result = await self.simple_flow.execute(Context(data=data))
        
        # Prepare input for the typed flow
        typed_input = TextInput(
            text=simple_result.data["result"],
            options={"from_simple_flow": True}
        )
        
        # Execute the typed flow
        typed_result = await self.typed_flow.execute(Context(data=typed_input))
        
        # Combine results
        return {
            "simple_result": simple_result.data,
            "typed_result": typed_result.data
        }


# Example usage
async def run_examples():
    """Run the example flows."""
    # Example 1
    simple_flow = SimpleFlow()
    simple_result = await simple_flow.execute(Context(data={"text": "hello world"}))
    print(f"Simple Flow Result: {simple_result.data}")
    
    # Example 2
    typed_flow = TypedFlow()
    typed_input = TextInput(text="typed input", options={"process": True})
    typed_result = await typed_flow.execute(Context(data=typed_input))
    print(f"Typed Flow Result: {typed_result.data}")
    
    # Example 3
    composite_flow = CompositeFlow()
    composite_result = await composite_flow.execute(Context(data={"text": "composite example"}))
    print(f"Composite Flow Result: {composite_result.data}")
    
    # Try to access a private method directly (should fail)
    try:
        # This should raise an AttributeError
        await simple_flow.process_text({"validated_text": "test"})
        print("ERROR: Private method was accessible!")
    except AttributeError:
        print("SUCCESS: Private method access prevented as expected")


if __name__ == "__main__":
    asyncio.run(run_examples()) 