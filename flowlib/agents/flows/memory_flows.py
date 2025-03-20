"""Memory flow implementations for entity extraction and retrieval.

This module provides flows for entity extraction from conversations and
memory retrieval based on query or context.
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any

from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

import flowlib as fl
from flowlib.core.errors import ExecutionError, ErrorContext
from flowlib.core.registry.constants import ProviderType, ResourceType
from flowlib.flows.decorators import flow, pipeline
from flowlib.core.models.result import FlowResult
from flowlib.flows.base import Flow
from flowlib.core.models.context import Context
from flowlib.agents.memory.models import Entity, EntityAttribute, EntityRelationship
from flowlib.agents.memory.utils import (
    normalize_entity_id, generate_entity_id, validate_entity
)

logger = logging.getLogger(__name__)

# --- Input/Output Models ---

class ConversationInput(BaseModel):
    """Input model for conversation-based memory operations.
    
    Attributes:
        conversation_history: The conversation history as a list of message dicts
        latest_message: The latest message in the conversation (for retrieval)
        source: Optional source identifier for extracted entities
    """
    conversation_history: List[Dict[str, str]] = Field(
        description="List of conversation messages, each with 'speaker' and 'content'"
    )
    latest_message: Optional[Dict[str, str]] = Field(
        default=None,
        description="The most recent message in the conversation (for targeted retrieval)"
    )
    source: Optional[str] = Field(
        default="conversation",
        description="Source identifier for extracted entities"
    )

class EntityRetrievalQuery(BaseModel):
    """Explicit query model for entity retrieval.
    
    Attributes:
        query: The text query to search for related entities
        limit: Maximum number of entities to retrieve
    """
    query: str = Field(
        description="Text query to search for related entities"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of entities to retrieve"
    )

class MemorySearchInput(BaseModel):
    """Input model for combined memory search operations.
    
    This model supports either conversation-based search or explicit query.
    
    Attributes:
        conversation: Optional conversation context for search
        query: Optional explicit query for search
    """
    conversation: Optional[ConversationInput] = Field(
        default=None,
        description="Conversation context for search"
    )
    query: Optional[EntityRetrievalQuery] = Field(
        default=None,
        description="Explicit query for search"
    )

class ExtractedEntities(BaseModel):
    """Output model for entity extraction flow.
    
    Attributes:
        entities: List of extracted entities
        summary: Text summary of what was extracted
    """
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of extracted entities"
    )
    summary: str = Field(
        default="",
        description="Text summary of what was extracted"
    )

class RetrievedMemories(BaseModel):
    """Output model for memory retrieval flow.
    
    Attributes:
        entities: List of retrieved entities
        context: Formatted context for prompt injection
        relevance_scores: Relevance scores for retrieved entities
    """
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of retrieved entities"
    )
    context: str = Field(
        default="",
        description="Formatted context for prompt injection"
    )
    relevance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores for retrieved entities"
    )

# --- Flow Implementations ---

@flow(name="memory-extraction")
class MemoryExtractionFlow(Flow):
    """Flow for extracting entities from conversations and storing in memory.
    
    This flow:
    1. Formats conversation history into a prompt
    2. Generates a structured entity extraction prompt using an LLM
    3. Extracts and validates entities from the LLM response
    4. Stores the entities in the memory system
    """
    
    def __init__(self, llm_provider_name: str = "llamacpp", name_or_instance: str = "memory-extraction"):
        """Initialize the memory extraction flow.
        
        Args:
            llm_provider_name: Name of the LLM provider to use
            name_or_instance: Name of the flow to register or an existing instance
        """
        super().__init__(name_or_instance=name_or_instance)
        self.llm_provider_name = llm_provider_name
    
    @pipeline(input_model=ConversationInput, output_model=ExtractedEntities)
    async def run(self, context_or_data: Union[Context, ConversationInput]) -> FlowResult[ExtractedEntities]:
        """Extract entities from conversation history.
        
        Args:
            context_or_data: Context with conversation data or direct input
            
        Returns:
            FlowResult with extracted entities
            
        Raises:
            ExecutionError: If entity extraction fails
        """
        # Handle context
        if isinstance(context_or_data, Context):
            context = context_or_data
            data = context.data
        else:
            context = Context(data=context_or_data)
            data = context_or_data
            
        if not isinstance(data, ConversationInput):
            raise ExecutionError(
                message="Invalid input type for MemoryExtractionFlow",
                context=ErrorContext.create(
                    expected_type="ConversationInput",
                    actual_type=str(type(data))
                )
            )
            
        try:
            # Format conversation for the prompt
            formatted_conversation = self._format_conversation(data.conversation_history)
            
            # Get entity extraction prompt from resource registry
            extraction_prompt = await fl.resource_registry.get(
                "entity-extraction", 
                ResourceType.PROMPT
            )
            
            # Format the prompt with conversation
            formatted_prompt = extraction_prompt.template.format(
                conversation=formatted_conversation
            )
            
            # Get LLM provider 
            llm = await fl.provider_registry.get(
                ProviderType.LLM, 
                self.llm_provider_name
            )
            
            # Generate entities using the LLM
            logger.info("Extracting entities from conversation with LLM")
            extracted_data = await llm.generate_text(
                formatted_prompt,
                **extraction_prompt.config
            )
            
            # Extract JSON data from the response
            entity_data = self._extract_json_data(extracted_data)
            
            # Convert to entity objects
            entities = []
            for item in entity_data.get("entities", []):
                try:
                    # Skip if missing required fields
                    if not item.get("entity_type") or not item.get("attributes", []):
                        logger.warning(f"Skipping invalid entity: {item}")
                        continue
                        
                    # Create entity
                    entity = self._create_entity_from_data(item, data.source)
                    if entity:
                        entities.append(entity)
                        
                except ValidationError as e:
                    logger.warning(f"Invalid entity data: {e}")
                    
            # Create result with summary
            result = ExtractedEntities(
                entities=entities,
                summary=f"Extracted {len(entities)} entities from conversation"
            )
            
            return FlowResult.success(result)
            
        except Exception as e:
            raise ExecutionError(
                message="Failed to extract entities from conversation",
                context=ErrorContext.create(
                    conversation_length=len(data.conversation_history),
                    llm_provider=self.llm_provider_name
                ),
                cause=e
            )
            
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation history into a string for the prompt.
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Formatted conversation string
        """
        formatted = []
        for message in conversation:
            speaker = message.get("speaker", "Unknown")
            content = message.get("content", "")
            formatted.append(f"{speaker}: {content}")
            
        return "\n".join(formatted)
        
    def _extract_json_data(self, text: str) -> Dict[str, Any]:
        """Extract JSON data from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted JSON data
        """
        try:
            # Try to extract JSON object if embedded in text
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
                return json.loads(json_str)
            elif "[" in text and "]" in text:
                # Maybe it's a list of entities without the wrapper
                start = text.find("[")
                end = text.rfind("]") + 1
                json_str = text[start:end]
                entity_list = json.loads(json_str)
                return {"entities": entity_list}
            else:
                # No JSON found
                logger.warning("No JSON data found in LLM response")
                return {"entities": []}
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return {"entities": []}
    
    def _create_entity_from_data(self, item: Dict[str, Any], source: str) -> Optional[Entity]:
        """Convert raw entity data to Entity object.
        
        Args:
            item: Dictionary with entity data
            source: Source identifier
            
        Returns:
            Entity object or None if invalid
        """
        try:
            # Create attributes
            attributes = []
            for attr in item.get("attributes", []):
                if "name" in attr and "value" in attr:
                    attributes.append(
                        EntityAttribute(
                            name=attr["name"],
                            value=attr["value"],
                            confidence=attr.get("confidence", 0.8)
                        )
                    )
            
            # Create relationships
            relationships = []
            for rel in item.get("relationships", []):
                if "relation_type" in rel and "target_entity" in rel:
                    relationships.append(
                        EntityRelationship(
                            relation_type=rel["relation_type"],
                            target_entity=rel["target_entity"],
                            target_entity_type=rel.get("target_entity_type"),
                            confidence=rel.get("confidence", 0.7)
                        )
                    )
            
            # Generate or normalize entity ID
            entity_id = item.get("entity_id")
            if not entity_id:
                entity_id = generate_entity_id(
                    item.get("entity_type", "unknown"),
                    item.get("name", "")
                )
            else:
                entity_id = normalize_entity_id(entity_id)
                
            # Create entity
            entity = Entity(
                id=entity_id,
                type=item["entity_type"],
                attributes={},
                relationships=[],
                source=source,
                importance=item.get("confidence", 0.8),
                last_updated=datetime.now().isoformat()
            )
            
            # Add attributes to the entity
            for attr in attributes:
                entity.attributes[attr.name] = attr
                
            # Add relationships to the entity
            for rel in relationships:
                # Convert to the correct relationship format
                entity_rel = EntityRelationship(
                    relation_type=rel.relation_type,
                    target_id=rel.target_entity,
                    confidence=rel.confidence,
                    source=source
                )
                entity.relationships.append(entity_rel)
            
            # Validate entity
            is_valid, error_messages = validate_entity(entity)
            if not is_valid:
                logger.warning(f"Invalid entity: {error_messages}")
                return None
                
            return entity
            
        except ValidationError as e:
            logger.warning(f"Failed to create entity: {e}")
            return None

@flow(name="memory-retrieval")
class MemoryRetrievalFlow(Flow):
    """Flow for retrieving relevant memories based on conversation context.
    
    This flow:
    1. Generates search queries from conversation or uses explicit queries
    2. Searches relevant memories using vector and graph database
    3. Returns formatted memories for prompt injection
    """
    
    def __init__(
        self, 
        llm_provider_name: str = "llamacpp", 
        hybrid_memory_manager = None,
        max_entities: int = 5,
        name_or_instance: str = "memory-retrieval"
    ):
        """Initialize the memory retrieval flow.
        
        Args:
            llm_provider_name: Name of the LLM provider to use
            hybrid_memory_manager: Optional memory manager instance
            max_entities: Maximum number of entities to retrieve
            name_or_instance: Name of the flow to register or an existing instance
        """
        super().__init__(name_or_instance=name_or_instance)
        self.llm_provider_name = llm_provider_name
        self.hybrid_memory_manager = hybrid_memory_manager
        self.max_entities = max_entities
    
    @pipeline(input_model=MemorySearchInput, output_model=RetrievedMemories)
    async def run(self, context_or_data: Union[Context, MemorySearchInput]) -> FlowResult[RetrievedMemories]:
        """Retrieve relevant memories based on context.
        
        Args:
            context_or_data: Context or input data
            
        Returns:
            FlowResult with retrieved memories
            
        Raises:
            ExecutionError: If memory retrieval fails
        """
        # Handle context
        if isinstance(context_or_data, Context):
            context = context_or_data
            data = context.data
        else:
            context = Context(data=context_or_data)
            data = context_or_data
            
        if not isinstance(data, MemorySearchInput):
            raise ExecutionError(
                message="Invalid input type for MemoryRetrievalFlow",
                context=ErrorContext.create(
                    expected_type="MemorySearchInput",
                    actual_type=str(type(data))
                )
            )
            
        try:
            # Get memory manager
            memory_manager = await self._get_memory_manager()
            
            # Handle explicit query if provided
            if data.query:
                logger.info(f"Searching memories with explicit query: {data.query.query}")
                search_result = await memory_manager.search_memory(
                    query=data.query.query,
                    entity_types=None,  # We don't filter by entity type from explicit query
                    limit=data.query.limit,
                    min_relevance=0.7,
                    include_related=True
                )
                
            # Handle conversation-based query
            elif data.conversation:
                # Generate search query from conversation
                search_query = await self._generate_search_query(data.conversation)
                logger.info(f"Searching memories with generated query: {search_query}")
                
                # Search memories
                search_result = await memory_manager.search_memory(
                    query=search_query,
                    entity_types=None,  # No entity type filtering 
                    limit=self.max_entities,
                    min_relevance=0.7,
                    include_related=True
                )
            else:
                raise ExecutionError(
                    message="No valid input provided for memory retrieval",
                    context=ErrorContext.create(
                        has_query=data.query is not None,
                        has_conversation=data.conversation is not None
                    )
                )
                
            # Format retrieved memories
            result = RetrievedMemories(
                entities=search_result.entities,
                context=search_result.context,
                relevance_scores=search_result.relevance_scores
            )
            
            return FlowResult.success(result)
            
        except Exception as e:
            raise ExecutionError(
                message="Failed to retrieve memories",
                context=ErrorContext.create(
                    has_query=data.query is not None,
                    has_conversation=data.conversation is not None
                ),
                cause=e
            )
    
    async def _generate_search_query(self, conversation: ConversationInput) -> str:
        """Generate a search query from conversation context.
        
        Args:
            conversation: Conversation input
            
        Returns:
            Generated search query
        """
        try:
            # Get retrieval prompt from resource registry
            retrieval_prompt = await fl.resource_registry.get(
                "memory-retrieval", 
                ResourceType.PROMPT
            )
            
            # Format the conversation for the prompt
            formatted_conversation = []
            for message in conversation.conversation_history[-5:]:  # Last 5 messages
                speaker = message.get("speaker", "Unknown")
                content = message.get("content", "")
                formatted_conversation.append(f"{speaker}: {content}")
                
            conversation_text = "\n".join(formatted_conversation)
            
            # Format the prompt
            formatted_prompt = retrieval_prompt.template.format(
                conversation=conversation_text
            )
            
            # Get LLM
            llm = await fl.provider_registry.get(
                ProviderType.LLM, 
                self.llm_provider_name
            )
            
            # Generate query
            query = await llm.generate_text(
                formatted_prompt,
                **retrieval_prompt.config
            )
            
            return query.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate search query: {e}")
            
            # Fallback: use the last message as the query
            if conversation.latest_message:
                return conversation.latest_message.get("content", "")
            elif conversation.conversation_history:
                return conversation.conversation_history[-1].get("content", "")
            else:
                return ""
    
    async def _get_memory_manager(self):
        """Get the memory manager instance.
        
        Returns:
            Initialized memory manager
            
        Raises:
            ExecutionError: If memory manager is not available
        """
        if self.hybrid_memory_manager:
            # Use provided manager
            if not getattr(self.hybrid_memory_manager, "_initialized", False):
                await self.hybrid_memory_manager.initialize()
            return self.hybrid_memory_manager
            
        raise ExecutionError(
            message="Memory manager not available",
            context=ErrorContext.create(
                hybrid_memory_manager=self.hybrid_memory_manager
            )
        ) 