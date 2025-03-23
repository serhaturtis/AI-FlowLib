# Migration Guide: String Formatting Utilities

This guide provides instructions for migrating existing string formatting code to use the new centralized `flowlib.utils.formatting` utilities. These utilities consolidate duplicated formatting operations across the FlowLib ecosystem.

## Why Migrate?

- **Consistency**: Ensure consistent formatting across the library
- **Maintainability**: Simplify maintenance by centralizing formatting logic
- **Features**: Take advantage of additional formatting features and enhancements
- **DRY Principle**: Avoid duplicating common formatting code

## Quick Reference

Below is a quick reference for common migration patterns:

| Old Code | New Code |
|----------|----------|
| `_process_escape_sequences(text)` | `from flowlib.utils.formatting import process_escape_sequences` |
| `format_entity_for_display(entity)` | `from flowlib.utils.formatting import format_entity_for_display` |
| `_format_conversation(conversation)` | `from flowlib.utils.formatting import format_conversation` |
| `_extract_json(text)` | `from flowlib.utils.formatting import extract_json` |
| `_format_schema_model(model)` | `from flowlib.utils.formatting import format_schema_model` |
| Custom JSON serialization | `from flowlib.utils.formatting import make_serializable` |

## Step-by-Step Migration

### 1. Add Imports

Add imports for the formatting utilities you need:

```python
# Import all formatting utilities
from flowlib.utils.formatting import (
    # Text formatting
    process_escape_sequences,
    
    # Entity formatting
    format_entity_for_display,
    format_entities_as_context,
    
    # Conversation formatting
    format_conversation,
    format_state,
    format_history,
    format_flows,
    
    # JSON formatting
    extract_json,
    
    # Schema formatting
    format_schema_model,
    
    # Serialization
    make_serializable
)

# Or import specific modules
from flowlib.utils.formatting.text import process_escape_sequences
from flowlib.utils.formatting.entities import format_entity_for_display
```

### 2. Replace Text Formatting Code

Replace custom escape sequence processing:

```python
# Old code
def _process_escape_sequences(text):
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\t')
    # ...
    return text

# New code
processed_text = process_escape_sequences(text)
```

### 3. Replace Entity Formatting

Replace entity formatting methods:

```python
# Old code
def format_entity_for_display(entity, detailed=False):
    lines = [f"Entity: {entity.type.title()} - {entity.id}"]
    # ...
    return "\n".join(lines)

# New code
formatted_entity = format_entity_for_display(entity, detailed=False)
```

### 4. Replace Conversation Formatting

Replace conversation formatting:

```python
# Old code
def _format_conversation(conversation):
    formatted = []
    for message in conversation:
        speaker = message.get("speaker", "Unknown")
        content = message.get("content", "")
        formatted.append(f"{speaker}: {content}")
    return "\n".join(formatted)

# New code
formatted_conversation = format_conversation(conversation)
```

### 5. Replace JSON Extraction

Replace JSON extraction code:

```python
# Old code
def _extract_json(text):
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    # ...

# New code
json_data = extract_json(text)
```

### 6. Replace Schema Formatting

Replace schema model formatting:

```python
# Old code
def _format_schema_model(model):
    if hasattr(model, "model_fields"):
        # ...
    elif hasattr(model, "__fields__"):
        # ...
    return model_name

# New code
formatted_schema = format_schema_model(model)
```

### 7. Replace Serialization Logic

Replace custom serialization code:

```python
# Old code
def _make_serializable(obj):
    if hasattr(obj, '__dict__'):
        # ...
    elif isinstance(obj, dict):
        # ...
    return str(obj)

# New code
serialized_data = make_serializable(obj)
```

## Common Migration Patterns

### CLI Provider

```python
# Old code
from flowlib.providers.conversation.base import ConversationProvider

class CLIConversationProvider(ConversationProvider):
    def send_response(self, response: str):
        processed_response = self._process_escape_sequences(response)
        print(f"\nAgent: {processed_response}")
        
    def _process_escape_sequences(self, text: str) -> str:
        # Implementation...

# New code
from flowlib.providers.conversation.base import ConversationProvider
from flowlib.utils.formatting import process_escape_sequences

class CLIConversationProvider(ConversationProvider):
    def send_response(self, response: str):
        processed_response = process_escape_sequences(response)
        print(f"\nAgent: {processed_response}")
```

### Memory Utilities

```python
# Old code
def format_entity_for_display(entity: Entity, detailed: bool = False) -> str:
    lines = [f"Entity: {entity.type.title()} - {entity.id}"]
    # Implementation...
    
# New code
from flowlib.utils.formatting import format_entity_for_display

# Just use the imported function directly
```

## Testing Your Migration

After migrating, verify that your code works as expected:

1. Run your test suite to ensure functionality is maintained
2. Check that formatting behaves consistently with previous implementations
3. Verify that error handling is preserved

## Need Help?

If you have questions or need assistance with migration, please:

1. Check the utilities' docstrings for usage examples
2. Review the example usage in `flowlib/utils/formatting/examples.py`
3. Refer to the README for a complete list of available utilities 