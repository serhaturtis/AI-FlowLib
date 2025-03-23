# Formatting Utilities for FlowLib

This module provides a collection of utilities for formatting various types of data in the FlowLib ecosystem. It centralizes string formatting operations that were previously duplicated across different components of the library.

## Modules

The formatting package includes the following modules:

### `text.py`

Text processing utilities for handling escape sequences and other text transformations:

- `process_escape_sequences(text)`: Converts literal escape sequences like `\\n` to their actual character representation
- `truncate_text(text, max_length, add_ellipsis)`: Truncates text to a specified length
- `format_key_value_pairs(pairs, delimiter)`: Formats key-value pairs into a string

### `entities.py`

Entity formatting utilities for display and prompt context:

- `format_entity_for_display(entity, detailed)`: Formats an entity for human-readable display
- `format_entities_as_context(entities, include_relationships)`: Formats multiple entities for prompt injection
- `format_entity_list(entities, compact)`: Formats a list of entities in a compact or detailed format

### `conversation.py`

Conversation formatting utilities for history, state, and flows:

- `format_conversation(conversation)`: Formats conversation history into a string for prompts
- `format_state(state)`: Formats agent state for prompt
- `format_history(history)`: Formats execution history for prompt
- `format_flows(flows)`: Formats available flows for prompt
- `format_agent_execution_details(details)`: Formats agent execution details for CLI display

### `json.py`

JSON extraction and formatting utilities:

- `extract_json(text)`: Extracts JSON data from text as Python objects
- `extract_json_str(text)`: Extracts JSON string from text
- `format_json(data, indent)`: Formats Python data as pretty-printed JSON

### `schema.py`

Schema and model formatting utilities:

- `format_schema_model(model, error_context)`: Formats a schema model into a readable string representation
- `format_schema_description(model)`: Extracts schema description from model

### `serialization.py`

Object serialization utilities for web display or API responses:

- `make_serializable(obj)`: Converts complex objects to JSON serializable values
- `format_execution_details(details)`: Formats execution details for web display

## Examples

See the `examples.py` module for usage examples of these formatting utilities.

## Usage

Import the utilities directly from the package:

```python
from flowlib.utils.formatting import (
    process_escape_sequences,
    format_entity_for_display,
    format_conversation,
    extract_json,
    format_schema_model,
    make_serializable
)
```

Or import from specific modules for clarity:

```python
from flowlib.utils.formatting.text import process_escape_sequences
from flowlib.utils.formatting.entities import format_entity_for_display
from flowlib.utils.formatting.json import extract_json
``` 