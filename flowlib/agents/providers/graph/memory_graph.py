"""In-memory graph database provider implementation.

This module provides a simple in-memory implementation of the graph database provider
interface for testing and small-scale use cases.
"""

import asyncio
import logging
import copy
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Union

from pydantic import ValidationError

from flowlib.core.errors import ProviderError, ErrorContext
from flowlib.core.registry.decorators import provider
from flowlib.core.registry.constants import ProviderType
from flowlib.agents.memory.models import Entity, EntityAttribute, EntityRelationship
from .base import GraphDBProvider, GraphDBProviderSettings

logger = logging.getLogger(__name__)

@provider(provider_type=ProviderType.GRAPH_DB, name="memory-graph")
class MemoryGraphProvider(GraphDBProvider):
    """In-memory graph database provider for testing and small-scale use.
    
    This provider stores all entities and relationships in memory,
    providing a lightweight implementation for development and testing.
    """
    
    def __init__(self, name: str = "memory-graph", settings: Optional[GraphDBProviderSettings] = None):
        """Initialize in-memory graph provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Create settings explicitly if not provided to avoid TypeVar issues
        settings = settings or GraphDBProviderSettings()
        
        super().__init__(name=name, settings=settings)
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        
    async def _initialize(self) -> None:
        """Initialize provider.
        
        Nothing to initialize for in-memory implementation.
        """
        pass
    
    async def _shutdown(self) -> None:
        """Clean up resources.
        
        Clear entities and relationships.
        """
        self.entities = {}
        self.relationships = {}
        
    async def add_entity(self, entity: Entity) -> str:
        """Add or update an entity node.
        
        Args:
            entity: Entity to add or update
            
        Returns:
            ID of the created/updated entity
            
        Raises:
            ProviderError: If entity is invalid
        """
        try:
            async with self._lock:
                # Store a copy of the entity
                self.entities[entity.id] = copy.deepcopy(entity)
                
                # Ensure the entity has an entry in relationships
                if entity.id not in self.relationships:
                    self.relationships[entity.id] = []
                    
                # Add relationships
                for rel in entity.relationships:
                    await self.add_relationship(
                        entity.id, 
                        rel.target_id, 
                        rel.relation_type,
                        {
                            "confidence": rel.confidence,
                            "source": rel.source,
                            "timestamp": rel.timestamp
                        }
                    )
                
                return entity.id
                
        except ValidationError as e:
            raise ProviderError(
                message=f"Invalid entity data: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity.id,
                    entity_type=entity.type
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity.id
                ),
                cause=e
            )
        
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            Entity object if found, None otherwise
        """
        async with self._lock:
            entity = self.entities.get(entity_id)
            if entity:
                # Return a copy to prevent external modification
                return copy.deepcopy(entity)
            return None
        
    async def add_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relation_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a relationship between entities.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relation_type: Type of relationship
            properties: Optional properties for the relationship
            
        Raises:
            ProviderError: If source or target entity doesn't exist
        """
        try:
            async with self._lock:
                # Ensure both source and target entities exist
                if source_id not in self.entities:
                    raise ProviderError(
                        message=f"Source entity {source_id} does not exist",
                        provider_name=self.name
                    )
                
                if target_id not in self.entities:
                    raise ProviderError(
                        message=f"Target entity {target_id} does not exist",
                        provider_name=self.name
                    )
                    
                # Ensure both source and target have entries in relationships
                if source_id not in self.relationships:
                    self.relationships[source_id] = []
                if target_id not in self.relationships:
                    self.relationships[target_id] = []
                
                # Default properties
                props = properties or {}
                if "timestamp" not in props:
                    props["timestamp"] = datetime.now().isoformat()
                
                # Create relationship objects for both directions
                outgoing = {
                    "source": source_id,
                    "target": target_id,
                    "type": relation_type,
                    "direction": "outgoing",
                    "properties": props
                }
                
                incoming = {
                    "source": target_id,
                    "target": source_id,
                    "type": relation_type,
                    "direction": "incoming",
                    "properties": props
                }
                
                # Add relationships (avoiding duplicates)
                if not any(r["target"] == target_id and r["type"] == relation_type 
                        for r in self.relationships[source_id]):
                    self.relationships[source_id].append(outgoing)
                    
                if not any(r["target"] == source_id and r["type"] == relation_type 
                        for r in self.relationships[target_id]):
                    self.relationships[target_id].append(incoming)
                    
                # Add to entity's relationships list if not already there
                source_entity = self.entities[source_id]
                target_entity = self.entities[target_id]
                
                # Source -> Target relationship
                if not any(r.target_id == target_id and r.relation_type == relation_type 
                         for r in source_entity.relationships):
                    rel = EntityRelationship(
                        relation_type=relation_type,
                        target_id=target_id,
                        confidence=props.get("confidence", 0.9),
                        source=props.get("source", "memory-graph"),
                        timestamp=props.get("timestamp")
                    )
                    source_entity.relationships.append(rel)
                
                # Don't automatically add the inverse relationship to target entity
                # as the relationship might not be bidirectional
                
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type
                ),
                cause=e
            )
        
    async def query_relationships(
        self, 
        entity_id: str, 
        relation_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """Query relationships for an entity.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional type to filter by
            direction: 'outgoing' or 'incoming'
            
        Returns:
            List of relationship information dicts
            
        Raises:
            ProviderError: If entity doesn't exist
        """
        try:
            async with self._lock:
                if entity_id not in self.relationships:
                    return []
                    
                results = []
                for rel in self.relationships[entity_id]:
                    if rel["direction"] == direction:
                        if relation_type is None or rel["type"] == relation_type:
                            results.append(copy.deepcopy(rel))
                            
                return results
        except Exception as e:
            raise ProviderError(
                message=f"Failed to query relationships: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity_id,
                    relation_type=relation_type,
                    direction=direction
                ),
                cause=e
            )
        
    async def traverse(
        self, 
        start_id: str, 
        relation_types: Optional[List[str]] = None, 
        max_depth: int = 2
    ) -> List[Entity]:
        """Traverse the graph starting from an entity.
        
        Args:
            start_id: ID of the starting entity
            relation_types: Optional list of relation types to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            List of entities found in traversal
            
        Raises:
            ProviderError: If traversal fails
        """
        try:
            async with self._lock:
                if start_id not in self.entities:
                    return []
                    
                visited = set()
                results = []
                
                async def dfs(entity_id: str, depth: int) -> None:
                    """Depth-first search traversal."""
                    if depth > max_depth or entity_id in visited:
                        return
                        
                    visited.add(entity_id)
                    
                    if entity_id in self.entities:
                        # Add a copy of the entity to results
                        results.append(copy.deepcopy(self.entities[entity_id]))
                        
                    if entity_id in self.relationships and depth < max_depth:
                        for rel in self.relationships[entity_id]:
                            if rel["direction"] == "outgoing":
                                if relation_types is None or rel["type"] in relation_types:
                                    await dfs(rel["target"], depth + 1)
                
                await dfs(start_id, 1)
                return results
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    start_id=start_id,
                    max_depth=max_depth
                ),
                cause=e
            )
        
    async def query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query in the graph.
        
        For the in-memory provider, this is a basic implementation
        that supports simple entity filtering queries.
        
        Args:
            query: Query string (simple filtering expressions)
            params: Optional query parameters
            
        Returns:
            List of matching entity dictionaries
            
        Raises:
            ProviderError: If query syntax is invalid
        """
        try:
            # For in-memory implementation, we only support basic filtering
            async with self._lock:
                results = []
                query_params = params or {}
                
                # Simple parsing of equality conditions
                conditions = []
                if query:
                    parts = [p.strip() for p in query.split("AND")]
                    for part in parts:
                        if "=" in part:
                            field, value = [p.strip() for p in part.split("=", 1)]
                            # Handle parameter substitution
                            if value.startswith(":"):
                                param_name = value[1:]
                                if param_name in query_params:
                                    value = query_params[param_name]
                                else:
                                    continue  # Skip if parameter not provided
                            elif value.startswith("'") and value.endswith("'"):
                                # String literal
                                value = value[1:-1]
                            conditions.append((field, value))
                
                # Filter entities based on conditions
                for entity_id, entity in self.entities.items():
                    entity_dict = entity.dict()
                    match = True
                    
                    for field, value in conditions:
                        # Handle dotted notation for nested fields
                        if "." in field:
                            parts = field.split(".")
                            curr = entity_dict
                            for part in parts[:-1]:
                                if part in curr:
                                    curr = curr[part]
                                else:
                                    match = False
                                    break
                            if match and parts[-1] in curr:
                                if str(curr[parts[-1]]) != str(value):
                                    match = False
                            else:
                                match = False
                        else:
                            # Top-level field
                            if field in entity_dict:
                                if str(entity_dict[field]) != str(value):
                                    match = False
                            else:
                                match = False
                    
                    if match:
                        results.append(entity_dict)
                
                return results
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query
                ),
                cause=e
            )
        
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
        """
        try:
            async with self._lock:
                if entity_id not in self.entities:
                    return False
                
                # Find all relationships with this entity
                if entity_id in self.relationships:
                    # For each outgoing relationship, remove the corresponding incoming relationship
                    for rel in self.relationships[entity_id]:
                        if rel["direction"] == "outgoing":
                            target_id = rel["target"]
                            if target_id in self.relationships:
                                # Remove the incoming relationship
                                self.relationships[target_id] = [
                                    r for r in self.relationships[target_id]
                                    if not (r["direction"] == "incoming" and 
                                            r["source"] == entity_id and
                                            r["type"] == rel["type"])
                                ]
                    
                    # Delete entity's relationships
                    del self.relationships[entity_id]
                
                # Delete entity
                del self.entities[entity_id]
                return True
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
        
    async def delete_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relation_type: Optional[str] = None
    ) -> bool:
        """Delete relationship(s) between entities.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relation_type: Optional type to filter by
            
        Returns:
            True if relationships were deleted, False if none found
        """
        try:
            async with self._lock:
                if source_id not in self.relationships or target_id not in self.relationships:
                    return False
                
                deleted = False
                
                # Filter relationships to keep
                new_source_rels = []
                for rel in self.relationships[source_id]:
                    if (rel["direction"] == "outgoing" and
                        rel["target"] == target_id and
                        (relation_type is None or rel["type"] == relation_type)):
                        # This is a relationship to delete
                        deleted = True
                    else:
                        new_source_rels.append(rel)
                
                # Only update if we found relationships to delete
                if deleted:
                    self.relationships[source_id] = new_source_rels
                    
                    # Also remove the corresponding relationships from target
                    new_target_rels = []
                    for rel in self.relationships[target_id]:
                        if (rel["direction"] == "incoming" and
                            rel["source"] == source_id and
                            (relation_type is None or rel["type"] == relation_type)):
                            # This is a relationship to delete
                            pass
                        else:
                            new_target_rels.append(rel)
                            
                    self.relationships[target_id] = new_target_rels
                    
                    # Update entity relationship lists
                    if source_id in self.entities:
                        source_entity = self.entities[source_id]
                        source_entity.relationships = [
                            r for r in source_entity.relationships
                            if not (r.target_id == target_id and
                                   (relation_type is None or r.relation_type == relation_type))
                        ]
                        
                return deleted
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type
                ),
                cause=e
            ) 