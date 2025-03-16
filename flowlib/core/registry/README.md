# FlowLib Registry System

The FlowLib Registry System provides a centralized way to manage resources and providers in your applications. This document explains the architecture and how to use the registry system effectively.

## Architecture

The registry system consists of two main components:

1. **ResourceRegistry**: Manages non-provider resources like models, prompts, and configurations
2. **ProviderRegistry**: Manages provider instances and provider factories with built-in asynchronous initialization

### Key Features

- Clear separation of concerns between resource types and provider types
- Type-safe retrieval with validation
- Standardized constant types using enums
- Decorator-based registration
- Lazy asynchronous initialization of providers
- Thread-safe provider management

## Usage Examples

### Registering and Accessing Resources

```python
import flowlib_consolidated as fl

# Register a model configuration
@fl.model("my-model")
class MyModelConfig:
    model_path = "/path/to/model.gguf"
    temperature = 0.7

# Access the model
model_config = fl.resource_registry.get("my-model", resource_type=fl.ResourceType.MODEL)
```

### Registering and Using Providers

```python
import flowlib_consolidated as fl

# Register a provider with a decorator
@fl.llm_provider("my-llm-provider", model_name="my-model")
class MyLLMProviderFactory:
    """Factory for creating LLM providers."""
    pass

# Get the provider (initializes automatically)
llm = await fl.provider_registry.get_provider_async("my-llm-provider")

# Use the provider
response = await llm.generate("Hello, world!")
```

### Creating Providers Directly

```python
import flowlib_consolidated as fl

# Create and register a provider programmatically
provider = fl.create_provider(
    provider_type=fl.ProviderType.VECTOR_DB,
    name="my-vector-db",
    implementation="chromadb",
    persist_directory="./data/vectors"
)

# Access and initialize the provider
initialized_provider = await fl.provider_registry.get_provider_async("my-vector-db")

# Or create and initialize in one step
provider = await fl.create_and_initialize_provider(
    provider_type=fl.ProviderType.VECTOR_DB,
    name="my-vector-db-2",
    implementation="chromadb",
    persist_directory="./data/vectors2"
)
```

### Using Type-Safe Provider Retrieval

```python
from flowlib_consolidated.providers.vector import VectorDBProvider
import flowlib_consolidated as fl

# Get provider with type checking and initialization
vector_db = await fl.provider_registry.get_provider_typed_async("my-vector-db", VectorDBProvider)

# Now you have proper type hints
await vector_db.add_documents(documents)
```

### Provider Lifecycle Management

```python
import flowlib_consolidated as fl

# Initialize all providers of a specific type
await fl.provider_registry.initialize_all(provider_type=fl.ProviderType.LLM)

# Shutdown all initialized providers
await fl.provider_registry.shutdown_all()
```

## Resource Types

The `ResourceType` enum provides standardized resource types:

- `ResourceType.MODEL`: Model configurations
- `ResourceType.PROMPT`: Prompt templates
- `ResourceType.CONFIG`: General configurations 
- `ResourceType.EMBEDDING`: Embedding models

## Provider Types

The `ProviderType` enum provides standardized provider types:

- `ProviderType.LLM`: Language model providers
- `ProviderType.VECTOR_DB`: Vector database providers
- `ProviderType.DATABASE`: Database providers
- `ProviderType.CACHE`: Cache providers
- `ProviderType.STORAGE`: Storage providers
- `ProviderType.MESSAGE_QUEUE`: Message queue providers
- `ProviderType.GPU`: GPU management providers
- `ProviderType.API`: External API providers

## Best Practices

1. Use the decorator-based registration when possible for cleaner code
2. Use specialized decorators like `@fl.llm_provider` rather than the general `@fl.provider`
3. Access providers through the `get_provider_async` methods to ensure proper initialization
4. Use constants from `ResourceType` and `ProviderType` for consistency
5. Provide meaningful metadata when registering resources and providers 