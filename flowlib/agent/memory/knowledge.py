"""
Knowledge base memory implementation.

This module provides a graph-based memory system for storing structured knowledge
using graph database providers.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Set, Tuple

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

logger = logging.getLogger(__name__)


class KnowledgeBaseMemory(BaseMemory):
    """Graph-based knowledge memory system.
    
    This component provides a structured knowledge base using graph database providers
    to store entities, attributes, and relationships.
    """
    
    def __init__(
        self,
        provider_name: str = "neo4j",
        name: str = "knowledge_memory"
    ):
        """Initialize knowledge base memory.
        
        Args:
            provider_name: Name of the graph database provider to use
            name: Component name
        """
        super().__init__(name)
        
        self._provider_name = provider_name
        self._graph_provider = None
        
    async def _initialize_impl(self) -> None:
        """Initialize the knowledge base memory."""
        # Import here to avoid circular imports
        from ...providers.registry import provider_registry
        from ...providers.constants import ProviderType
        
        try:
            # Initialize graph database provider
            if not self._provider_name:
                raise MemoryError("No graph database provider specified")
                
            self._graph_provider = await provider_registry.get(
                ProviderType.GRAPH_DB, 
                self._provider_name
            )
            
            if not self._graph_provider:
                raise ProviderError(f"Graph database provider not found: {self._provider_name}")
                
            # Initialize the provider if it has its own initialize method
            if hasattr(self._graph_provider, 'initialize'):
                await self._graph_provider.initialize()
                
            logger.debug(f"Initialized {self.name} with provider={self._provider_name}")
                
        except Exception as e:
            raise MemoryError(f"Failed to initialize knowledge base memory: {str(e)}") from e
    
    async def _shutdown_impl(self) -> None:
        """Shutdown knowledge base memory."""
        # Shutdown graph provider if it has its own shutdown method
        if self._graph_provider and hasattr(self._graph_provider, 'shutdown'):
            await self._graph_provider.shutdown()
            
        self._graph_provider = None
        
        logger.debug(f"Shut down {self.name}")
    
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
        """Search for knowledge in the graph database.
        
        Args:
            query: Search query
            context: Context path (used as entity type filter)
            limit: Maximum number of results
            **kwargs: Additional search parameters including:
                      - relation: Relation type to filter by
                      - entity_type: Entity type to filter by
                      - tags: Tags to filter by
            
        Returns:
            List of memory items with matching entities
        """
        if not self._graph_provider:
            raise MemoryError("Graph provider not initialized")
            
        try:
            items = []
            
            # Check if we're searching for related entities
            if 'related_to' in kwargs:
                related_to = kwargs['related_to']
                relation_type = kwargs.get('relation')
                
                # Get related entities
                related_entities = await self._graph_provider.traverse(
                    start_id=related_to,
                    relation_types=[relation_type] if relation_type else None,
                    max_depth=kwargs.get('max_depth', 2)
                )
                
                # Convert to memory items
                for entity in related_entities[:limit]:
                    item = MemoryItem(
                        key=entity.id,
                        value=entity,
                        context=context or 'knowledge',
                        metadata={
                            'entity_type': entity.type,
                            'tags': entity.tags,
                            'importance': entity.importance
                        }
                    )
                    items.append(item)
                    
                return items
                
            # Use native query capabilities if possible
            entity_type = kwargs.get('entity_type', context.split('/')[-1] if context else None)
            tags = kwargs.get('tags', [])
            
            # Construct query based on provider capabilities
            if hasattr(self._graph_provider, 'query'):
                # Simple query language parsing
                query_parts = []
                params = {}
                
                if query and query.strip():
                    query_parts.append("MATCH (e:Entity)")
                    query_parts.append("WHERE e.id CONTAINS $query OR ANY(attr IN KEYS(e.attributes) WHERE e.attributes[attr].value CONTAINS $query)")
                    params['query'] = query
                    
                if entity_type:
                    if query_parts:
                        query_parts.append("AND e.type = $type")
                    else:
                        query_parts.append("MATCH (e:Entity)")
                        query_parts.append("WHERE e.type = $type")
                    params['type'] = entity_type
                    
                if tags:
                    if query_parts:
                        query_parts.append("AND ANY(tag IN e.tags WHERE tag IN $tags)")
                    else:
                        query_parts.append("MATCH (e:Entity)")
                        query_parts.append("WHERE ANY(tag IN e.tags WHERE tag IN $tags)")
                    params['tags'] = tags
                    
                if not query_parts:
                    query_parts.append("MATCH (e:Entity)")
                    
                query_parts.append(f"RETURN e LIMIT {limit}")
                
                # Execute query
                query_results = await self._graph_provider.query(" ".join(query_parts), params)
                
                # Convert to entities and then to memory items
                for result in query_results:
                    if 'e' in result and isinstance(result['e'], Entity):
                        entity = result['e']
                    elif 'e' in result and isinstance(result['e'], dict):
                        # Convert dict to Entity
                        entity = Entity(**result['e'])
                    else:
                        continue
                        
                    item = MemoryItem(
                        key=entity.id,
                        value=entity,
                        context=context or 'knowledge',
                        metadata={
                            'entity_type': entity.type,
                            'tags': entity.tags,
                            'importance': entity.importance
                        }
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            raise MemoryError(
                f"Failed to search knowledge: {str(e)}",
                operation="search",
                context=context
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