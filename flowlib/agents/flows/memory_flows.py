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
from flowlib.core.errors import ExecutionError, ErrorContext, ResourceError
from flowlib.core.registry.constants import ProviderType, ResourceType
from flowlib.flows.decorators import flow, pipeline
from flowlib.core.models.result import FlowResult
from flowlib.flows.base import Flow
from flowlib.core.models.context import Context
from flowlib.agents.memory.models import Entity, EntityAttribute, EntityRelationship
from flowlib.agents.memory.utils import (
    normalize_entity_id, generate_entity_id, validate_entity
)
from flowlib.utils.formatting import format_conversation, extract_json

from ..memory_manager import MemoryManager
from .models import ExtractedEntities, RetrievedMemories

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

class SearchQueryResult(BaseModel):
    """Model for search query generation results.
    
    Attributes:
        query: The generated search query
    """
    query: str = Field(
        description="The generated search query"
    )

class EntityExtractionResult(BaseModel):
    """Model for entity extraction results.
    
    Attributes:
        entities: List of extracted entities
    """
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted entities"
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

# --- Flow Implementations ---

@flow(name="memory-extraction", is_infrastructure=True)
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
    async def run(self, context_or_data: Union[Context, ConversationInput]) -> ExtractedEntities:
        """Extract entities from conversation history.
        
        Args:
            context_or_data: Context with conversation data or direct input
            
        Returns:
            ExtractedEntities with entities and summary
            
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
            
            # Generate entities using the LLM with structured output
            logger.info("Extracting entities from conversation with LLM")
            try:
                print("\n===== DEBUG: MEMORY EXTRACTION LLM CALL =====")
                print(f"Formatted prompt (first 100 chars): {formatted_prompt[:100]}...")
                print(f"LLM provider: {self.llm_provider_name}")
                print(f"Output type: {EntityExtractionResult}")
                print("Calling generate_structured...")
                
                raw_result = await llm.generate_structured(
                    formatted_prompt,
                    output_type=EntityExtractionResult,
                    model_name="default",
                    **extraction_prompt.config
                )
                
                print("\n===== DEBUG: RAW LLM RESULT =====")
                print(f"Result type: {type(raw_result)}")
                print(f"Result attributes: {dir(raw_result)}")
                print(f"Result as string: {str(raw_result)}")
                if hasattr(raw_result, "entities"):
                    print(f"Entities type: {type(raw_result.entities)}")
                    print(f"Number of entities: {len(raw_result.entities)}")
                    if raw_result.entities:
                        print(f"First entity (sample): {raw_result.entities[0]}")
                else:
                    print("No 'entities' attribute found in result")
                
                result = raw_result
            except Exception as e:
                print("\n===== DEBUG: LLM CALL EXCEPTION =====")
                print(f"Exception type: {type(e)}")
                print(f"Exception message: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Process the extracted entities
            print("\n===== DEBUG: PROCESSING ENTITIES =====")
            entity_data = {"entities": result.entities}
            print(f"Entity data: {entity_data}")
            
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
                        print(f"Created valid entity: {entity.id} of type {entity.type}")
                except ValidationError as e:
                    logger.warning(f"Invalid entity data: {e}")
                    
            # Create and return the result object directly
            print(f"\n===== DEBUG: FINAL RESULT =====")
            print(f"Total valid entities created: {len(entities)}")
            result_obj = ExtractedEntities(
                entities=entities,
                summary=f"Extracted {len(entities)} entities from conversation"
            )
            print(f"Result object: {result_obj}")
            print("===== END DEBUG =====\n")
            return result_obj
            
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
        # Use the shared formatting utility
        return format_conversation(conversation)
        
    def _extract_json_data(self, text: str) -> Dict[str, Any]:
        """Extract JSON data from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted JSON data
        """
        # Use the shared JSON extraction utility 
        json_data = extract_json(text)
        
        if not json_data:
            logger.warning("No JSON data found in LLM response")
            return {"entities": []}
            
        # If it's already a dict with 'entities', return as is
        if isinstance(json_data, dict) and "entities" in json_data:
            return json_data
            
        # If it's a list, assume it's a list of entities
        if isinstance(json_data, list):
            return {"entities": json_data}
            
        # Otherwise, wrap in entities dict
        return {"entities": [json_data]}
    
    def _create_entity_from_data(self, item: Dict[str, Any], source: str) -> Optional[Entity]:
        """Convert raw entity data to Entity object.
        
        Args:
            item: Dictionary with entity data
            source: Source identifier
            
        Returns:
            Entity object or None if invalid
        """
        try:
            # Print the incoming entity data for debugging
            print(f"Creating entity from data: {item}")
            
            # Create attributes
            attributes = []
            for attr in item.get("attributes", []):
                if "name" in attr and "value" in attr:
                    attributes.append(
                        EntityAttribute(
                            name=attr["name"],
                            value=attr["value"],
                            confidence=attr.get("confidence", 0.8),
                            source=source
                        )
                    )
            
            # Always add the name as an attribute if provided
            if item.get("name"):
                # Check if a name attribute already exists
                has_name_attr = any(a.name == "name" for a in attributes)
                if not has_name_attr:
                    attributes.append(
                        EntityAttribute(
                            name="name",
                            value=item["name"],
                            confidence=item.get("confidence", 0.8),
                            source=source
                        )
                    )
                    
                # Also add a "description" attribute as fallback to ensure we have at least one attribute
                has_desc_attr = any(a.name == "description" for a in attributes)
                if not has_desc_attr:
                    attributes.append(
                        EntityAttribute(
                            name="description",
                            value=f"A {item['entity_type']} named {item['name']}",
                            confidence=item.get("confidence", 0.7),
                            source=source
                        )
                    )
            
            # Skip entity ONLY if it has no name AND no attributes
            if not attributes and not item.get("name"):
                logger.warning(f"Skipping entity with no attributes or name: {item}")
                return None
                
            # Create relationships
            relationships = []
            for rel in item.get("relationships", []):
                if "relation_type" in rel and "target_entity" in rel:
                    relationships.append(
                        EntityRelationship(
                            relation_type=rel["relation_type"],
                            target_entity=rel["target_entity"],
                            confidence=rel.get("confidence", 0.7),
                            source=source
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
            entity.relationships = relationships
            
            # Debug pre-validation
            print(f"Entity before validation: id={entity.id}, type={entity.type}, attributes={len(entity.attributes)}, relationships={len(entity.relationships)}")
            
            # Validate entity
            is_valid, error_messages = validate_entity(entity)
            if not is_valid:
                print(f"VALIDATION ERRORS for entity {entity.id}: {error_messages}")
                logger.warning(f"Invalid entity: {error_messages}")
                return None
                
            # Debug successful validation
            print(f"Entity validation SUCCESS for {entity.id}")
            return entity
            
        except ValidationError as e:
            logger.warning(f"Failed to create entity: {e}")
            return None

@flow(name="memory-retrieval", is_infrastructure=True)
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
    async def run(self, context_or_data: Union[Context, MemorySearchInput]) -> RetrievedMemories:
        """Retrieve relevant memories based on context.
        
        Args:
            context_or_data: Context or input data
            
        Returns:
            RetrievedMemories with entities and formatted context
            
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
                
            # Format retrieved memories and return the model directly
            return RetrievedMemories(
                entities=search_result.entities,
                context=search_result.context,
                relevance_scores=search_result.relevance_scores
            )
            
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
        print("\n===== DEBUG: MEMORY RETRIEVAL QUERY GENERATION =====")
        print(f"Conversation history length: {len(conversation.conversation_history)}")
        if conversation.latest_message:
            print(f"Latest message: {conversation.latest_message.get('content', '')}")
        
        try:
            # Get retrieval prompt from resource registry
            try:
                retrieval_prompt = await fl.resource_registry.get(
                    "memory-retrieval", 
                    ResourceType.PROMPT
                )
                print(f"Retrieved prompt template 'memory-retrieval'")
                
                # Format the conversation for the prompt
                formatted_conversation = []
                for message in conversation.conversation_history[-5:]:  # Last 5 messages
                    speaker = message.get("speaker", "Unknown")
                    content = message.get("content", "")
                    formatted_conversation.append(f"{speaker}: {content}")
                    
                conversation_text = "\n".join(formatted_conversation)
                print(f"Formatted conversation (first 100 chars): {conversation_text[:100]}...")
                
                # Format the prompt
                formatted_prompt = retrieval_prompt.template.format(
                    conversation=conversation_text
                )
                print(f"Formatted prompt (first 100 chars): {formatted_prompt[:100]}...")
                
                # Get LLM
                llm = await fl.provider_registry.get(
                    ProviderType.LLM, 
                    self.llm_provider_name
                )
                print(f"Retrieved LLM provider: {self.llm_provider_name}")
                
                # Generate query using generate_structured with required parameters
                print("Calling generate_structured for query generation...")
                try:
                    result = await llm.generate_structured(
                        formatted_prompt,
                        output_type=SearchQueryResult,
                        model_name="default",
                        **retrieval_prompt.config
                    )
                    
                    print(f"Result type: {type(result)}")
                    print(f"Result dir: {dir(result)}")
                    if hasattr(result, 'query'):
                        print(f"Generated query: {result.query}")
                        query = result.query.strip()
                    else:
                        print("No 'query' attribute in result")
                        query = ""
                        
                except Exception as gen_err:
                    print(f"Query generation error: {type(gen_err).__name__}: {str(gen_err)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                print(f"Final query: {query}")
                print("===== END DEBUG: QUERY GENERATION =====\n")
                return query
                
            except ResourceError as e:
                # Handle missing prompt template
                print(f"Resource error: {str(e)}")
                logger.warning(f"Failed to get memory-retrieval prompt from resource registry: {e}")
                logger.info("Using fallback method for query generation")
                
                # Fallback: use the last message content or summarize recent messages
                fallback_query = ""
                if conversation.latest_message:
                    fallback_query = conversation.latest_message.get("content", "")
                elif conversation.conversation_history:
                    # Take the most recent messages
                    recent_content = [msg.get("content", "") for msg in conversation.conversation_history[-3:]]
                    fallback_query = " ".join(recent_content)
                
                print(f"Using fallback query: {fallback_query}")
                print("===== END DEBUG: QUERY GENERATION (FALLBACK) =====\n")
                return fallback_query
                
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {str(e)}")
            logger.warning(f"Failed to generate search query: {str(e)}")
            
            # Fallback: use the last message as the query
            fallback_query = ""
            if conversation.latest_message:
                fallback_query = conversation.latest_message.get("content", "")
            elif conversation.conversation_history:
                fallback_query = conversation.conversation_history[-1].get("content", "")
            
            print(f"Using emergency fallback query: {fallback_query}")
            print("===== END DEBUG: QUERY GENERATION (EMERGENCY FALLBACK) =====\n")
            return fallback_query
    
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