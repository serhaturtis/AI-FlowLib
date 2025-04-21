"""
Knowledge base memory implementation.

This module provides a graph-based memory system for storing structured knowledge
using graph database providers.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Generic, TypeVar

from ...core.context import Context
from ...providers.constants import ProviderType
from ...providers.registry import provider_registry
from ...providers.graph.models import Entity, EntityAttribute, EntityRelationship
from ..core.errors import MemoryError, ProviderError
from .base import BaseMemory
from .models import (
    MemoryItem, 
    MemoryStoreRequest, 
    MemoryRetrieveRequest, 
    MemorySearchRequest,
    MemorySearchResult
)
from ...providers.graph.base import GraphDBProvider, GraphDBProviderSettings

logger = logging.getLogger(__name__)

# Define Settings TypeVar based on GraphDBProviderSettings
SettingsType = TypeVar('SettingsType', bound=GraphDBProviderSettings)

class KnowledgeBaseMemory(BaseMemory, Generic[SettingsType]):
    """Graph-based knowledge memory system.
    
    This component provides a structured knowledge base using graph database providers
    to store entities, attributes, and relationships.
    """
    
    def __init__(
        self,
        graph_provider: GraphDBProvider,
        name: str = "knowledge_memory"
    ):
        """Initialize knowledge base memory.
        
        Args:
            graph_provider: Graph database provider instance
            name: Component name
        """
        super().__init__(name)
        
        if not graph_provider:
            raise ValueError("A GraphDBProvider instance must be provided.")
            
        self._graph_provider = graph_provider
        # Assume provider is already initialized by AgentCore
        
    async def _initialize_impl(self) -> None:
        """Initialize the knowledge base memory (verify provider)."""
        # Provider should be initialized by the caller (AgentCore)
        if not self._graph_provider or not self._graph_provider.initialized:
            logger.error(f"Graph provider ('{self._graph_provider.name if self._graph_provider else 'None'}') not initialized before passing to {self.name}.")
            # Attempt to initialize? Or just raise? Let's raise for stricter control.
            raise MemoryError(f"Graph provider must be initialized before use in {self.name}")
            
        logger.debug(f"{self.name} initialized using provider: {self._graph_provider.name}")
                
    async def _shutdown_impl(self) -> None:
        """Shutdown knowledge base memory (delegated to provider)."""
        # Shutdown is handled by AgentCore managing the provider instance
        logger.debug(f"{self.name} shutdown (provider managed externally).")
        pass # Provider lifecycle managed elsewhere
    
    async def _store_impl(
        self, 
        key: str, 
        value: Any, 
        context: str, 
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> None:
        """Store knowledge in the graph database.
        
        This method handles different types of values:
        - If value is an Entity, it's stored directly
        - If value is a dict with entity information, it's converted to Entity
        - Other values are stored as attributes of an entity with the key as ID
        
        Args:
            key: Entity ID or knowledge key
            value: Entity, entity dict, or attribute value
            context: Context path (used as entity type if not specified)
            metadata: Additional metadata
            **kwargs: Additional parameters, including entity_type, attributes, relationships
        """
        if not self._graph_provider:
            raise MemoryError("Graph provider not initialized")
        
        try:
            # Handle Entity objects directly
            if isinstance(value, Entity):
                entity = value
                # Only update the ID if not already set
                if not entity.id:
                    entity.id = key
                
                # Add to graph
                await self._graph_provider.add_entity(entity)
                logger.debug(f"Stored entity with ID '{entity.id}' of type '{entity.type}'")
                return
                
            # Handle dictionary with entity information
            if isinstance(value, dict) and 'type' in value:
                # Convert to Entity
                entity_type = value.get('type')
                attributes = value.get('attributes', {})
                relationships = value.get('relationships', [])
                tags = value.get('tags', [])
                
                # Create entity attributes
                entity_attrs = {}
                for attr_name, attr_value in attributes.items():
                    if isinstance(attr_value, dict) and 'value' in attr_value:
                        # Already in EntityAttribute format
                        entity_attrs[attr_name] = EntityAttribute(**attr_value)
                    else:
                        # Convert to EntityAttribute
                        entity_attrs[attr_name] = EntityAttribute(
                            name=attr_name,
                            value=str(attr_value),
                            source=metadata.get('source', 'system') if metadata else 'system'
                        )
                
                # Create entity relationships
                entity_rels = []
                for rel in relationships:
                    if isinstance(rel, dict) and 'relation_type' in rel and 'target_entity' in rel:
                        # Already in EntityRelationship format
                        entity_rels.append(EntityRelationship(**rel))
                    elif isinstance(rel, dict) and 'type' in rel and 'target' in rel:
                        # Convert from simplified format
                        entity_rels.append(EntityRelationship(
                            relation_type=rel['type'],
                            target_entity=rel['target'],
                            source=metadata.get('source', 'system') if metadata else 'system'
                        ))
                
                # Create entity
                entity = Entity(
                    id=key,
                    type=entity_type,
                    attributes=entity_attrs,
                    relationships=entity_rels,
                    tags=tags,
                    importance=kwargs.get('importance', 0.7)
                )
                
                # Add to graph
                await self._graph_provider.add_entity(entity)
                logger.debug(f"Stored entity with ID '{key}' of type '{entity_type}'")
                return
                
            # Handle attribute value for existing entity or create new entity
            entity_type = kwargs.get('entity_type', context.split('/')[-1])
            
            # Try to get existing entity
            entity = await self._graph_provider.get_entity(key)
            
            if not entity:
                # Create new entity
                entity = Entity(
                    id=key,
                    type=entity_type,
                    importance=kwargs.get('importance', 0.7)
                )
            
            # Add attribute
            attribute_name = kwargs.get('attribute_name', 'value')
            entity.attributes[attribute_name] = EntityAttribute(
                name=attribute_name,
                value=str(value),
                source=metadata.get('source', 'system') if metadata else 'system'
            )
            
            # Update entity
            await self._graph_provider.add_entity(entity)
            logger.debug(f"Updated entity '{key}' with attribute '{attribute_name}'")
            
        except Exception as e:
            raise MemoryError(
                f"Failed to store knowledge: {str(e)}",
                operation="store",
                key=key,
                context=context
            ) from e
    
    async def _retrieve_impl(
        self, 
        key: str, 
        context: str,
        **kwargs
    ) -> Any:
        """Retrieve knowledge from the graph database.
        
        Args:
            key: Entity ID to retrieve
            context: Context path (used as fallback type filter)
            **kwargs: Additional retrieval parameters
            
        Returns:
            Entity object if found, None otherwise
        """
        if not self._graph_provider:
            raise MemoryError("Graph provider not initialized")
            
        try:
            # Get entity by ID
            entity = await self._graph_provider.get_entity(key)
            
            if not entity:
                logger.debug(f"Entity with ID '{key}' not found")
                return None
                
            # Check if we're requesting a specific attribute
            attribute_name = kwargs.get('attribute_name')
            if attribute_name and attribute_name in entity.attributes:
                return entity.attributes[attribute_name].value
                
            # Return full entity
            return entity
            
        except Exception as e:
            raise MemoryError(
                f"Failed to retrieve knowledge: {str(e)}",
                operation="retrieve",
                key=key,
                context=context
            ) from e
    
    async def _search_impl(
        self, 
        query: str, 
        context: str,
        limit: int = 10,
        **kwargs
    ) -> List[MemoryItem]:
        """Search for knowledge in the graph database using the provider's search method.
        
        Args:
            query: Search query text.
            context: Context path (used as fallback entity type filter).
            limit: Maximum number of results.
            **kwargs: Additional search parameters including:
                      - entity_type: Explicit entity type to filter by.
                      - tags: List of tags to filter by.
                      - related_to: Entity ID for relationship traversal (optional).
                      - relation: Relation type for traversal (optional).
                      - max_depth: Max depth for traversal (optional).
            
        Returns:
            List of memory items with matching entities.
            
        Raises:
            MemoryError: If the search fails or the provider is not initialized.
        """
        if not self._graph_provider:
            raise MemoryError("Graph provider not initialized")
            
        items = []
        try:
            # --- Handle Traversal Search (if requested) --- 
            if 'related_to' in kwargs:
                related_to_id = kwargs['related_to']
                relation_type = kwargs.get('relation')
                max_depth = kwargs.get('max_depth', 2)
                
                logger.debug(f"Performing traversal search from '{related_to_id}'")
                # Assuming graph_provider has a 'traverse' method
                if not hasattr(self._graph_provider, 'traverse'):
                     raise NotImplementedError(f"Graph provider '{self._graph_provider.name}' does not support traverse.")
                     
                related_entities = await self._graph_provider.traverse(
                    start_id=related_to_id,
                    relation_types=[relation_type] if relation_type else None,
                    max_depth=max_depth
                )
                
                # Convert results to MemoryItem
                for entity in related_entities[:limit]:
                    items.append(MemoryItem(
                        key=entity.id,
                        value=entity, # Store the full Entity object
                        context=context or 'knowledge',
                        metadata={
                            'source': 'graph_traversal',
                            'entity_type': entity.type,
                            'tags': entity.tags,
                            'importance': entity.importance
                        }
                    ))
                logger.debug(f"Traversal search found {len(items)} related entities.")
                return items

            # --- Handle Standard Entity Search --- 
            entity_type = kwargs.get('entity_type', context.split('/')[-1] if context else None)
            tags = kwargs.get('tags', [])
            
            logger.debug(f"Performing entity search with query: '{query}', type: '{entity_type}', tags: {tags}")
            # Use the provider's dedicated search method
            if not hasattr(self._graph_provider, 'search_entities'):
                 raise NotImplementedError(f"Graph provider '{self._graph_provider.name}' does not support search_entities.")
                 
            search_results: List[Entity] = await self._graph_provider.search_entities(
                query=query,
                entity_type=entity_type,
                tags=tags,
                limit=limit
            )
            
            # Convert results to MemoryItem
            for entity in search_results:
                items.append(MemoryItem(
                    key=entity.id,
                    value=entity, # Store the full Entity object
                    context=context or 'knowledge',
                    metadata={
                        'source': 'graph_search',
                        'entity_type': entity.type,
                        'tags': entity.tags,
                        'importance': entity.importance
                    }
                ))
            
            logger.debug(f"Entity search found {len(items)} entities.")
            return items
            
        except Exception as e:
            # Avoid leaking raw provider errors unless it's already a MemoryError
            error_message = f"Failed to search knowledge: {str(e)}"
            if isinstance(e, MemoryError):
                 raise e
            elif isinstance(e, NotImplementedError):
                 error_message = f"Search failed: {str(e)}"
                 
            raise MemoryError(
                message=error_message,
                operation="search",
                context=context,
                cause=e # Preserve original exception cause
            ) from e
    
    async def add_relationship(
        self, 
        source_id: str, 
        relation_type: str, 
        target_id: str,
        **kwargs
    ) -> None:
        """Add a relationship between entities.
        
        Args:
            source_id: ID of the source entity
            relation_type: Type of relationship
            target_id: ID of the target entity
            **kwargs: Additional parameters including:
                      - confidence: Confidence score
                      - source: Information source
                      
        Raises:
            MemoryError: If relationship creation fails
        """
        if not self._graph_provider:
            raise MemoryError("Graph provider not initialized")
            
        try:
            # Get source entity
            source_entity = await self._graph_provider.get_entity(source_id)
            if not source_entity:
                raise MemoryError(f"Source entity '{source_id}' not found")
                
            # Create relationship
            relationship = EntityRelationship(
                relation_type=relation_type,
                target_entity=target_id,
                confidence=kwargs.get('confidence', 0.9),
                source=kwargs.get('source', 'system')
            )
            
            # Add to source entity
            source_entity.relationships.append(relationship)
            
            # Update entity
            await self._graph_provider.add_entity(source_entity)
            logger.debug(f"Added relationship '{relation_type}' from '{source_id}' to '{target_id}'")
            
        except Exception as e:
            raise MemoryError(
                f"Failed to add relationship: {str(e)}",
                operation="add_relationship",
                source=source_id,
                target=target_id
            ) from e
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            MemoryError: If entity retrieval fails
        """
        if not self._graph_provider:
            raise MemoryError("Graph provider not initialized")
            
        try:
            return await self._graph_provider.get_entity(entity_id)
        except Exception as e:
            raise MemoryError(
                f"Failed to get entity: {str(e)}",
                operation="get_entity",
                key=entity_id
            ) from e

    # TODO: Add wipe_context to graph provider interface if needed
    async def _wipe_context_impl(self, context: str) -> None:
        raise NotImplementedError("Wiping specific context not yet implemented for KnowledgeBaseMemory")

    async def _wipe_all_impl(self) -> None:
        """Wipe all data from the knowledge base via the provider."""
        if not self._graph_provider:
            raise MemoryError("Graph provider not initialized")
        
        if not hasattr(self._graph_provider, 'wipe'):
            raise NotImplementedError(f"Graph provider '{self._graph_provider.name}' does not support wipe operation.")
            
        try:
            logger.warning(f"Wiping ALL data from knowledge base provider: {self._graph_provider.name}")
            await self._graph_provider.wipe()
            logger.info(f"Successfully wiped knowledge base: {self._graph_provider.name}")
        except Exception as e:
            raise MemoryError(
                f"Failed to wipe knowledge base: {str(e)}",
                operation="wipe_all"
            ) from e 