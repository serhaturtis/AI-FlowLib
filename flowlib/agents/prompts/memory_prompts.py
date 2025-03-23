"""Prompt templates for memory operations.

This module provides standardized prompt templates for entity extraction,
memory retrieval, memory updates, and reflective processes.
"""

from flowlib.core.registry.decorators import prompt
from flowlib.core.registry.constants import ResourceType

@prompt("entity-extraction")
class EntityExtractionPrompt:
    """Prompt template for extracting entities from conversation.
    
    This prompt guides the model to extract structured entity information
    from conversation text, identifying objects, people, concepts, etc.
    """
    template = """
You are an entity extraction assistant. Your task is to analyze a conversation and identify entities mentioned in it.
Entities can be people, places, organizations, concepts, events, products, etc.

For each entity, identify:
1. The entity type (person, place, organization, concept, event, product, etc.)
2. Key attributes of the entity - VERY IMPORTANT: EVERY entity MUST have at least one attribute
3. Relationships with other entities

Return the extracted entities in the following JSON format:
```json
{{
  "entities": [
    {{
      "entity_id": "optional-id",
      "entity_type": "the-entity-type",
      "name": "entity-name",
      "attributes": [
        {{
          "name": "attribute-name",
          "value": "attribute-value",
          "confidence": 0.9
        }},
        {{
          "name": "another-attribute",
          "value": "another-value",
          "confidence": 0.8
        }}
      ],
      "relationships": [
        {{
          "relation_type": "relation-type",
          "target_entity": "target-entity-name",
          "confidence": 0.8
        }}
      ],
      "confidence": 0.85
    }}
  ]
}}
```

CRITICALLY IMPORTANT GUIDELINES:
1. Each entity MUST have at least one attribute with "name" and "value" fields.
2. If you only know the entity's name, add a "name" attribute with that name as the value.
3. For people entities, always include attributes like "full_name", "role", or "description".
4. For organization entities, include attributes like "type", "purpose", or "description".
5. Ensure that relationships refer to other entities you're extracting.
6. Make sure the JSON structure is exactly as shown above.
7. BE GENEROUS in what you consider an entity - even in simple conversations, identify people and assistants.

Examples of good attributes:
- For a person: {{"name": "full_name", "value": "John Smith"}}, {{"name": "role", "value": "CEO"}}
- For a company: {{"name": "industry", "value": "technology"}}, {{"name": "founded", "value": "2010"}}
- For a product: {{"name": "category", "value": "smartphone"}}, {{"name": "manufacturer", "value": "Apple"}}

Here is the conversation to analyze:

{conversation}
"""
    
    # Configuration for LLM generation with a higher temperature
    config = {
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 1.0,
        "top_k": 40
    }

@prompt("memory-retrieval")
class MemoryRetrievalPrompt:
    """Prompt template for generating memory retrieval queries.
    
    This prompt guides the model to analyze conversation context and
    determine what information would be helpful to retrieve from memory.
    """
    template = """
You are an AI memory expert helping to retrieve relevant information from a knowledge base.
Your task is to analyze the conversation and generate a search query that will retrieve the most relevant information.

IMPORTANT INSTRUCTIONS:
1. If the conversation is just starting or contains only greetings (like "Hello", "Hi there", etc.), 
   return an EMPTY query.
2. ONLY generate a substantive query if there is a specific topic, question, or request for information.
3. Focus on extracting key concepts, terms, and questions that need factual answers.
4. Your query should be concise and focused on the most important information needs in the conversation.
5. Avoid making up topics that aren't actually mentioned in the conversation.

The query will be used to search a memory database of entities, facts, and relationships.

Conversation:
{conversation}

Generate an appropriate search query (1-2 sentences maximum) that captures what information would be most useful to retrieve,
or return an empty string if no substantive query is needed:
"""

@prompt("memory-update")
class MemoryUpdatePrompt:
    """Prompt template for updating existing memories.
    
    This prompt guides the model to analyze new information and suggest
    updates to existing entity records in memory.
    """
    template = """
You are a memory management assistant. Your task is to analyze existing information about an entity
and new information from a conversation, then determine how to update the entity record.

Existing entity information:
{existing_entity}

New information from conversation:
{new_information}

Analyze the new information and suggest updates to the entity record.
Your output should be in the following JSON format:

```json
{{
  "entity_id": "the-entity-id",
  "updates": [
    {{
      "attribute": "attribute-name",
      "old_value": "old-value",
      "new_value": "new-value",
      "action": "add|update|remove",
      "confidence": 0.9,
      "reasoning": "Brief explanation of why this update should be made"
    }}
  ],
  "new_relationships": [
    {{
      "relation_type": "relation-type",
      "target_entity": "target-entity-name",
      "target_entity_type": "target-entity-type",
      "confidence": 0.8,
      "reasoning": "Brief explanation of why this relationship should be added"
    }}
  ],
  "remove_relationships": [
    {{
      "relation_type": "relation-type",
      "target_entity": "target-entity-name",
      "reasoning": "Brief explanation of why this relationship should be removed"
    }}
  ],
  "summary": "Brief summary of the changes and why they're needed"
}}
```

Only suggest changes if there's clear evidence for the update in the new information.
If no updates are needed, return an empty updates array.
"""

@prompt("memory-reflection")
class MemoryReflectionPrompt:
    """Prompt template for reflecting on memory usage.
    
    This prompt guides the model to analyze how memory was used in a
    conversation and suggest improvements.
    """
    template = """
You are a memory analysis expert. Your task is to analyze how memory was used in a conversation
and suggest improvements for future interactions.

Conversation:
{conversation}

Memory retrieval queries used:
{retrieval_queries}

Entities retrieved:
{retrieved_entities}

Entities extracted:
{extracted_entities}

Based on the above information, analyze:
1. How well did the retrieved memories help answer the user's questions?
2. Were there any important entities that should have been retrieved but weren't?
3. Were any irrelevant entities retrieved?
4. How could memory retrieval be improved in future interactions?

Provide your analysis in the following JSON format:

```json
{{
  "retrieval_quality": {{
    "score": 0-10,
    "strengths": ["list of things that worked well"],
    "weaknesses": ["list of things that could be improved"]
  }},
  "extraction_quality": {{
    "score": 0-10,
    "strengths": ["list of things that worked well"],
    "weaknesses": ["list of things that could be improved"]
  }},
  "missed_entities": [
    {{
      "entity_type": "type",
      "description": "description of entity that should have been retrieved"
    }}
  ],
  "irrelevant_entities": [
    "id-of-irrelevant-entity"
  ],
  "improvement_suggestions": [
    "concrete suggestions for improving memory usage"
  ],
  "summary": "overall assessment of memory usage in this conversation"
}}
```

Be specific and constructive in your analysis.
""" 