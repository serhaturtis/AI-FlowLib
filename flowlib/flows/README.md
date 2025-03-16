# FlowLib Flow System

The FlowLib Flow System provides a structured way to build and execute flows with LLMs. This document explains the architecture and usage of the flow system, particularly the new rules for creating flows.

## Key Rules

1. **One Pipeline Per Flow**: Each flow must implement exactly one pipeline method using the `@pipeline` decorator. This is the entry point for the flow.

2. **Restricted Method Access**: Only the `execute` method is available to external callers. All other methods in a flow class are private and can only be called from within the flow.

## Flow Architecture

The flow system is organized around these key components:

1. **Flow**: Base class for all flow components
2. **Stage**: Individual execution units within a flow
3. **Pipeline**: The primary execution path in a flow, composed of stages

### Decorators

- `@flow`: Marks a class as a flow and enforces flow rules
- `@pipeline`: Marks a method as the flow's execution entry point
- `@stage`: Marks a method as a stage in the flow

## Usage Examples

### Creating a Flow

```python
import flowlib as fl

@fl.flow
class MyFlow:
    @fl.stage
    async def process_data(self, data):
        # This method is private and can only be called from within the flow
        return {"processed": data}
    
    @fl.stage
    async def generate_response(self, data):
        # This method is private and can only be called from within the flow
        return {"response": f"Generated from {data['processed']}"}
    
    @fl.pipeline
    async def run(self, data):
        # This is the entry point for the flow
        processed = await self._process_data(data)
        response = await self._generate_response(processed)
        return response
```

### Executing a Flow

```python
from flowlib_consolidated.core.models.context import Context

# Create flow instance
flow = MyFlow()

# Execute flow (this is the only method that should be called from outside)
result = await flow.execute(Context(data={"input": "example"}))

# Access results
print(result.data)  # {"response": "Generated from example"}
```

## Best Practices

1. **Use the Pipeline for Orchestration**: Your pipeline method should orchestrate the flow by calling stages in the right order.

2. **Keep Stages Focused**: Each stage should perform a single, well-defined task.

3. **Use Type Validation**: Specify input and output models for stages and pipelines for better type safety:

```python
from pydantic import BaseModel

class InputData(BaseModel):
    text: str

class OutputData(BaseModel):
    result: str

@fl.flow
class TypedFlow:
    @fl.stage(input_model=InputData, output_model=OutputData)
    async def process(self, data):
        return OutputData(result=f"Processed: {data.text}")
    
    @fl.pipeline(input_model=InputData, output_model=OutputData)
    async def run(self, data):
        return await self._process(data)
```

4. **Compositions**: For complex flows, compose multiple flows together:

```python
@fl.flow
class ComposedFlow:
    def __init__(self):
        self.flow1 = Flow1()
        self.flow2 = Flow2()
    
    @fl.pipeline
    async def run(self, data):
        # Execute other flows via their execute method
        result1 = await self.flow1.execute(Context(data=data))
        result2 = await self.flow2.execute(Context(data=result1.data))
        return result2.data
``` 