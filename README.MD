# FlowLib: A Framework for Type-Safe Data Processing Pipelines

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/pydantic-v2-green)](https://pydantic-docs.helpmanual.io/)


FlowLib is a Python framework for constructing type-safe data processing pipelines with an emphasis on artificial intelligence and machine learning workflows. The framework implements a declarative approach to pipeline construction, enforcing strict typing through Pydantic models while providing built-in mechanisms for error handling, validation, and execution monitoring.

## 1. Introduction

Modern data processing pipelines, particularly those involving AI components, often require complex orchestration of multiple processing stages. Each stage typically has specific input and output requirements, and errors at any stage can propagate through the system in unpredictable ways. FlowLib addresses these challenges by providing a structured framework that enforces type safety, enables clear error propagation, and simplifies the composition of complex multi-stage processes.

## 2. Design Philosophy

The design of FlowLib is guided by several key principles:

- **Type Safety**: All data flowing through the pipeline is validated against defined schemas
- **Composability**: Processing stages can be combined in flexible ways to create complex workflows
- **Error Traceability**: Errors are captured with contextual information about their origin
- **Declarative Definition**: Pipelines are defined using Python decorators for clarity and conciseness
- **Separation of Concerns**: Clear distinction between flow definition, execution, and error handling

## 3. Core Components

### 3.1 Flows

Flows represent the top-level containers for data processing pipelines. A flow:
- Is defined with a unique identifier
- Contains multiple processing stages
- Includes a pipeline method that coordinates stage execution
- May specify input and output schemas for validation

### 3.2 Stages

Stages represent individual processing steps within a flow. A stage:
- Performs a specific transformation on input data
- Defines explicit input and output schemas
- Can be executed independently or as part of a flow
- Supports both synchronous and asynchronous execution

### 3.3 Context

Context objects facilitate data transfer between stages, containing:
- The data payload
- Metadata about the execution environment
- Error information when applicable

### 3.4 Type Validation

FlowLib utilizes Pydantic models for type validation, providing:
- Input and output schema definitions
- Runtime data validation
- Automatic type conversion when possible
- Detailed validation error reporting

## 4. Implementation

### 4.1 Basic Implementation Pattern

Below is a minimal example demonstrating the implementation of a simple flow:

```python
import flowlib as fl
from pydantic import BaseModel, Field
from typing import List

# Define input/output schemas
class InputText(BaseModel):
    text: str = Field(..., description="Raw text to process")

class ProcessedText(BaseModel):
    tokens: List[str] = Field(..., description="Tokenized text")
    word_count: int = Field(..., description="Word count")

# Define a flow with stages
@fl.flow(name="text-processing-flow")
class TextProcessingFlow:
    @fl.stage(input_model=InputText, output_model=ProcessedText)
    async def process_text(self, context: fl.Context) -> ProcessedText:
        """Process input text."""
        input_data = context.data
        tokens = input_data.text.split()
        
        return ProcessedText(
            tokens=tokens,
            word_count=len(tokens)
        )
    
    @fl.pipeline(input_model=InputText, output_model=ProcessedText)
    async def run_pipeline(self, input_data: InputText) -> ProcessedText:
        """Execute the text processing pipeline."""
        input_context = fl.Context(data=input_data)
        process_stage = self.get_stage("process_text")
        result = await process_stage.execute(input_context)
        return result.data

# Execution example
async def main():
    input_data = InputText(text="Hello world, this is FlowLib!")
    flow = TextProcessingFlow()
    result = await flow.execute(fl.Context(data=input_data))
    print(f"Tokens: {result.data.tokens}")
    print(f"Word count: {result.data.word_count}")

import asyncio
asyncio.run(main())
```

### 4.2 Advanced Implementation: Multi-Stage Processing

This example demonstrates a more complex implementation involving language model integration:

```python
import flowlib as fl
from flowlib.providers.llm import LlamaCppProvider
from pydantic import BaseModel, Field
from typing import List, Dict

# Define models for pipeline stages
class InputText(BaseModel):
    text: str = Field(..., description="Raw text to process")

class ExtractedInfo(BaseModel):
    entities: List[Dict[str, str]] = Field(..., description="Extracted entities")
    key_facts: List[str] = Field(..., description="Key facts")

class SentimentResult(BaseModel):
    sentiment: str = Field(..., description="Overall sentiment")
    score: float = Field(..., description="Sentiment score")

class FinalReport(BaseModel):
    entities: List[Dict[str, str]] = Field(..., description="Extracted entities")
    sentiment: str = Field(..., description="Overall sentiment")
    recommendations: List[str] = Field(..., description="Recommendations")

# Register language model prompts
@fl.prompt("extraction")
class ExtractionPrompt:
    template = """
    Extract entities and facts from the following text:
    {text}
    """

@fl.prompt("sentiment")
class SentimentPrompt:
    template = """
    Analyze the sentiment of the following entities:
    {entities}
    """

# Define the multi-stage flow
@fl.flow(name="nlp-pipeline")
class NLPFlow:
    @fl.stage(input_model=InputText, output_model=ExtractedInfo)
    async def extract_information(self, context: fl.Context) -> ExtractedInfo:
        input_data = context.data
        extraction_prompt = await fl.resource_registry.get("extraction")
        llm = await fl.provider_registry.get(fl.ProviderType.LLM, "llamacpp")
        
        result = await llm.generate_structured(
            extraction_prompt.format(text=input_data.text),
            ExtractedInfo,
            temperature=0.2
        )
        
        return result
        
    @fl.stage(input_model=ExtractedInfo, output_model=SentimentResult)
    async def analyze_sentiment(self, context: fl.Context) -> SentimentResult:
        # Implementation details omitted for brevity
        pass

    @fl.pipeline(input_model=InputText, output_model=FinalReport)
    async def run_pipeline(self, input_data: InputText) -> FinalReport:
        # Implementation details omitted for brevity
        pass
```

## 5. Advanced Configuration

### 5.1 Resource Registration

The framework provides registries for managing computational resources:

```python
# Model registration
@fl.model("text-model")
class ModelConfig:
    path = "/path/to/model.gguf"
    model_type = "llama"
    
# Provider registration
@fl.provider(provider_type=fl.ProviderType.LLM, name="custom-provider")
class CustomProvider(fl.LLMProvider):
    # Provider implementation
```

### 5.2 Error Management

The error handling system can be extended with custom handlers:

```python
# Custom error handler implementation
class CustomErrorHandler(fl.ErrorHandler):
    async def handle(self, error: fl.BaseError) -> None:
        # Error handling logic
        pass

# Handler registration
flow.add_error_handler(fl.ValidationError, CustomErrorHandler())
```

## 6. Monitoring and Observability

The framework provides mechanisms for monitoring execution:

```python
# Execution metrics retrieval
result = await flow.execute(context)
execution_time = result.metadata.get("execution_time")

# Flow execution status
if result.is_success():
    print("Flow executed successfully")
else:
    print(f"Flow failed: {result.error}")
```

## 7. Limitations and Future Work

Current limitations and areas for future development include:
- Performance optimization for high-throughput scenarios
- Distributed execution capabilities
- Integration with additional model serving frameworks
- Enhanced visualization of flow execution

## 8. Conclusion

FlowLib provides a structured approach to building complex data processing pipelines with strong type safety guarantees. By combining declarative flow definitions with robust error handling and validation, the framework addresses common challenges in building reliable AI/ML pipelines.

## References

- Pydantic: [Documentation](https://pydantic-docs.helpmanual.io/)
- Asynchronous Programming in Python: [PEP 3156](https://peps.python.org/pep-3156/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
