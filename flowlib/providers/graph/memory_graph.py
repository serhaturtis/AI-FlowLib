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
                        rel.target_entity, 
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
        target_entity: str,
        relation_type: str,
        properties: Dict[str, Any] = {}
    ) -> None:
        """Add a relationship between two entities.
        
        Args:
            source_id: ID of the source entity
            target_entity: Name or identifier of the target entity
            relation_type: Type of relationship
            properties: Properties for the relationship
            
        Raises:
            ProviderError: If source entity doesn't exist
        """
        # Check if source entity exists
        if source_id not in self.entities:
            raise ProviderError(
                message=f"Source entity {source_id} does not exist",
                provider_name=self.name
            )
            
        # Check if target entity exists - skip if it doesn't
        if target_entity not in self.entities:
            logger.warning(f"Skipping relationship creation: Target entity '{target_entity}' does not exist")
            return
            
        # Initialize relationships for source entity if needed
        if source_id not in self.relationships:
            self.relationships[source_id] = []
            
        if target_entity not in self.relationships:
            self.relationships[target_entity] = []
            
        # Create outgoing relationship
        outgoing = {
            "type": relation_type,
            "target": target_entity,
            "direction": "outgoing",
            "properties": properties
        }
        
        # Create incoming relationship
        incoming = {
            "type": relation_type,
            "target": source_id,
            "direction": "incoming",
            "properties": properties
        }
        
        # Add relationships if they don't already exist
        if not any(r["target"] == target_entity and r["type"] == relation_type 
                  for r in self.relationships[source_id]):
            self.relationships[source_id].append(outgoing)
            
        if not any(r["target"] == source_id and r["type"] == relation_type
                  for r in self.relationships[target_entity]):
            self.relationships[target_entity].append(incoming)
            
        # Update entity objects
        source_entity = self.entities[source_id]
        target_entity_obj = self.entities[target_entity]
        
        # Add relationship to source entity if not already present
        if not any(r.target_entity == target_entity and r.relation_type == relation_type
                  for r in source_entity.relationships):
            source_entity.relationships.append(
                EntityRelationship(
                    relation_type=relation_type,
                    target_entity=target_entity,
                    confidence=0.9,
                    source="system"
                )
            )
            
        # Add relationship to target entity if not already present
        if not any(r.target_entity == source_id and r.relation_type == relation_type
                  for r in target_entity_obj.relationships):
            target_entity_obj.relationships.append(
                EntityRelationship(
                    relation_type=relation_type,
                    target_entity=source_id,
                    confidence=0.9,
                    source="system"
                )
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
                            results.append({
                                "source": entity_id,
                                "target": rel["target"],
                                "type": rel["type"],
                                "properties": rel["properties"]
                            })
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
                    
                visited_ids: Set[str] = set()
                results: List[Entity] = []
                
                # Define recursive DFS function
                async def dfs(entity_id: str, depth: int) -> None:
                    if depth > max_depth or entity_id in visited_ids:
                        return
                        
                    visited_ids.add(entity_id)
                    
                    if entity_id in self.entities:
                        results.append(copy.deepcopy(self.entities[entity_id]))
                        
                    if depth < max_depth:
                        # Get outgoing relationships
                        rels = await self.query_relationships(
                            entity_id,
                            relation_type=None if relation_types is None else None,  # Query all types
                            direction="outgoing"
                        )
                        
                        for rel in rels:
                            # Check if relation type matches filter
                            if relation_types is None or rel["type"] in relation_types:
                                await dfs(rel["target"], depth + 1)
                
                # Start traversal
                await dfs(start_id, 0)
                
                return results
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    start_id=start_id,
                    relation_types=relation_types,
                    max_depth=max_depth
                ),
                cause=e
            )
        
    async def query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a simple query language for the in-memory graph.
        
        Supports a limited set of commands:
            - "find_entities type={entity_type}" - Find entities by type
            - "find_entities name={name}" - Find entities by name
            - "neighbors id={id} [relation={relation_type}]" - Find neighboring entities
            - "path from={id} to={id} [max_depth={n}]" - Find path between entities
        
        Args:
            query: Query string
            params: Optional query parameters
            
        Returns:
            Query results
            
        Raises:
            ProviderError: If query parsing fails
        """
        try:
            async with self._lock:
                # Parse the query
                query = query.strip().lower()
                
                # Override query parameters with explicit params if provided
                if params:
                    param_dict = params
                else:
                    # Extract parameters from query string
                    param_parts = [part.strip() for part in query.split(" ") if "=" in part]
                    param_dict = {}
                    for part in param_parts:
                        key, value = part.split("=", 1)
                        param_dict[key] = value
                
                # Find entities by type
                if query.startswith("find_entities") and "type" in param_dict:
                    entity_type = param_dict["type"]
                    return self._find_entities_by_type(entity_type)
                    
                # Find entities by name
                elif query.startswith("find_entities") and "name" in param_dict:
                    name = param_dict["name"]
                    return self._find_entities_by_name(name)
                    
                # Find neighbors
                elif query.startswith("neighbors") and "id" in param_dict:
                    entity_id = param_dict["id"]
                    relation_type = param_dict.get("relation")
                    return await self._find_neighbors(entity_id, relation_type)
                    
                # Find path between entities
                elif query.startswith("path") and "from" in param_dict and "to" in param_dict:
                    from_id = param_dict["from"]
                    to_id = param_dict["to"]
                    max_depth = int(param_dict.get("max_depth", "3"))
                    return await self._find_path(from_id, to_id, max_depth)
                    
                else:
                    raise ProviderError(
                        message=f"Unsupported query: {query}",
                        provider_name=self.name
                    )
                    
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query,
                    params=params
                ),
                cause=e
            )
            
    def _find_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Find entities by type."""
        results = []
        for entity_id, entity in self.entities.items():
            if entity.type.lower() == entity_type.lower():
                results.append({"id": entity_id, "entity": entity.dict()})
        return results
        
    def _find_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find entities by name."""
        results = []
        for entity_id, entity in self.entities.items():
            if name.lower() in entity.name.lower():
                results.append({"id": entity_id, "entity": entity.dict()})
        return results
        
    async def _find_neighbors(self, entity_id: str, relation_type: Optional[str]) -> List[Dict[str, Any]]:
        """Find neighboring entities."""
        if entity_id not in self.entities:
            return []
            
        # Get relationships
        rels = await self.query_relationships(
            entity_id,
            relation_type=relation_type,
            direction="outgoing"
        )
        
        results = []
        for rel in rels:
            target_id = rel["target"]
            if target_id in self.entities:
                results.append({
                    "id": target_id,
                    "relation": rel["type"],
                    "entity": self.entities[target_id].dict()
                })
                
        return results
        
    async def _find_path(self, from_id: str, to_id: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find path between entities."""
        if from_id not in self.entities or to_id not in self.entities:
            return []
            
        # BFS to find path
        queue = [(from_id, [])]
        visited = {from_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            # Check if reached target
            if current_id == to_id:
                # Construct result
                result_path = []
                for i, node_id in enumerate(path + [current_id]):
                    result_path.append({
                        "position": i,
                        "id": node_id,
                        "entity": self.entities[node_id].dict()
                    })
                return result_path
                
            # Stop if max depth reached
            if len(path) >= max_depth:
                continue
                
            # Get outgoing relationships
            rels = await self.query_relationships(
                current_id,
                relation_type=None,
                direction="outgoing"
            )
            
            for rel in rels:
                next_id = rel["target"]
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [current_id]))
                    
        return []  # No path found
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
        """
        async with self._lock:
            if entity_id not in self.entities:
                return False
                
            # Delete the entity
            del self.entities[entity_id]
            
            # Collect all relationships involving this entity
            to_remove = []
            for source_id, relations in self.relationships.items():
                for rel in relations:
                    if rel["target"] == entity_id:
                        to_remove.append((source_id, rel))
                        
            # Remove relationships
            for source_id, rel in to_remove:
                if source_id in self.relationships:
                    self.relationships[source_id] = [
                        r for r in self.relationships[source_id]
                        if not (r["target"] == entity_id and r["type"] == rel["type"])
                    ]
                    
            # Remove entity from relationships dict
            if entity_id in self.relationships:
                del self.relationships[entity_id]
                
            return True
            
    async def delete_relationship(
        self, 
        source_id: str, 
        target_entity: str, 
        relation_type: Optional[str] = None
    ) -> bool:
        """Delete relationship(s) between entities.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Optional type to filter by
            
        Returns:
            True if relationships were deleted, False if none found
        """
        async with self._lock:
            # Check if entities exist
            if source_id not in self.relationships or target_entity not in self.relationships:
                return False
                
            deleted = False
            
            # Filter relationships to remove from source
            if source_id in self.relationships:
                original_count = len(self.relationships[source_id])
                filtered_rels = []
                
                for rel in self.relationships[source_id]:
                    should_keep = True
                    
                    if rel["target"] == target_entity:
                        if relation_type is None or rel["type"] == relation_type:
                            should_keep = False
                            
                    if should_keep:
                        filtered_rels.append(rel)
                        
                self.relationships[source_id] = filtered_rels
                deleted = deleted or (original_count > len(filtered_rels))
                
            # Filter relationships to remove from target
            if target_entity in self.relationships:
                original_count = len(self.relationships[target_entity])
                filtered_rels = []
                
                for rel in self.relationships[target_entity]:
                    should_keep = True
                    
                    if rel["target"] == source_id:
                        if relation_type is None or rel["type"] == relation_type:
                            should_keep = False
                            
                    if should_keep:
                        filtered_rels.append(rel)
                        
                self.relationships[target_entity] = filtered_rels
                deleted = deleted or (original_count > len(filtered_rels))
                
            # Update entity objects if they exist
            if source_id in self.entities:
                source_entity = self.entities[source_id]
                original_count = len(source_entity.relationships)
                
                source_entity.relationships = [
                    r for r in source_entity.relationships
                    if not (r.target_entity == target_entity and 
                           (relation_type is None or r.relation_type == relation_type))
                ]
                
                deleted = deleted or (original_count > len(source_entity.relationships))
                
            if target_entity in self.entities:
                target_entity_obj = self.entities[target_entity]
                original_count = len(target_entity_obj.relationships)
                
                target_entity_obj.relationships = [
                    r for r in target_entity_obj.relationships
                    if not (r.target_entity == source_id and 
                           (relation_type is None or r.relation_type == relation_type))
                ]
                
                deleted = deleted or (original_count > len(target_entity_obj.relationships))
                
            return deleted
            
    async def remove_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str
    ) -> None:
        """Remove a relationship between two entities.
        
        Args:
            source_id: ID of the source entity
            target_entity: Name or identifier of the target entity
            relation_type: Type of relationship
            
        Raises:
            ProviderError: If relationship removal fails
        """
        try:
            result = await self.delete_relationship(source_id, target_entity, relation_type)
            if not result:
                logger.warning(
                    f"No relationship found to remove: {source_id} -> {relation_type} -> {target_entity}"
                )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to remove relationship: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    source_id=source_id,
                    target_entity=target_entity,
                    relation_type=relation_type
                ),
                cause=e
            )

    async def _update_entity_relationships(self, entity_id: str, entity: Entity) -> None:
        """Update entity relationships from entity object.
        
        Used internally to sync relationships structure with entity object.
        
        Args:
            entity_id: ID of the entity
            entity: Entity object with relationships
        """
        # Clear existing relationships for this entity
        if entity_id in self.relationships:
            outgoing_rels = [r for r in self.relationships[entity_id] if r["direction"] == "outgoing"]
            for rel in outgoing_rels:
                await self.delete_relationship(entity_id, rel["target"], rel["type"])
                
        # Add new relationships
        for rel in entity.relationships:
            try:
                await self.add_relationship(
                    entity_id,
                    rel.target_entity,
                    rel.relation_type,
                    {
                        "confidence": rel.confidence,
                        "source": rel.source,
                        "timestamp": rel.timestamp
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to add relationship: {str(e)}") 