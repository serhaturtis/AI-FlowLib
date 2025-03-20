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
2. Key attributes of the entity
3. Relationships with other entities

Return the extracted entities in the following JSON format:
```json
{
  "entities": [
    {
      "entity_id": "optional-id",
      "entity_type": "the-entity-type",
      "name": "entity-name",
      "attributes": [
        {
          "name": "attribute-name",
          "value": "attribute-value",
          "confidence": 0.9
        }
      ],
      "relationships": [
        {
          "relation_type": "relation-type",
          "target_entity": "target-entity-name",
          "target_entity_type": "target-entity-type",
          "confidence": 0.8
        }
      ],
      "confidence": 0.85
    }
  ]
}
```

Only extract entities that are clearly mentioned and have useful information.
Ensure that relationships refer to other entities mentioned in the text.

Here is the conversation to analyze:

{conversation}
"""

@prompt("memory-retrieval")
class MemoryRetrievalPrompt:
    """Prompt template for generating memory retrieval queries.
    
    This prompt guides the model to analyze conversation context and
    determine what information would be helpful to retrieve from memory.
    """
    template = """
You are an AI memory expert helping to retrieve relevant information from a knowledge base.
Your task is to analyze the conversation and generate a search query that will retrieve the most relevant information.

Focus on extracting key concepts, terms, and questions that need answers.
Your query should be concise and focused on the most important information needs in the conversation.

The query will be used to search a memory database of entities, facts, and relationships.

Conversation:
{conversation}

Generate a search query (2-3 sentences maximum) that captures what information would be most useful to retrieve:
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
{
  "entity_id": "the-entity-id",
  "updates": [
    {
      "attribute": "attribute-name",
      "old_value": "old-value",
      "new_value": "new-value",
      "action": "add|update|remove",
      "confidence": 0.9,
      "reasoning": "Brief explanation of why this update should be made"
    }
  ],
  "new_relationships": [
    {
      "relation_type": "relation-type",
      "target_entity": "target-entity-name",
      "target_entity_type": "target-entity-type",
      "confidence": 0.8,
      "reasoning": "Brief explanation of why this relationship should be added"
    }
  ],
  "remove_relationships": [
    {
      "relation_type": "relation-type",
      "target_entity": "target-entity-name",
      "reasoning": "Brief explanation of why this relationship should be removed"
    }
  ],
  "summary": "Brief summary of the changes and why they're needed"
}
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
{
  "retrieval_quality": {
    "score": 0-10,
    "strengths": ["list of things that worked well"],
    "weaknesses": ["list of things that could be improved"]
  },
  "extraction_quality": {
    "score": 0-10,
    "strengths": ["list of things that worked well"],
    "weaknesses": ["list of things that could be improved"]
  },
  "missed_entities": [
    {
      "entity_type": "type",
      "description": "description of entity that should have been retrieved"
    }
  ],
  "irrelevant_entities": [
    "id-of-irrelevant-entity"
  ],
  "improvement_suggestions": [
    "concrete suggestions for improving memory usage"
  ],
  "summary": "overall assessment of memory usage in this conversation"
}
```

Be specific and constructive in your analysis.
""" 