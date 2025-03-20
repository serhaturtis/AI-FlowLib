"""Hybrid memory management for entity-centric agent memory.

This module provides the HybridMemoryManager, which integrates vector databases 
and graph databases for comprehensive entity-centric memory capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple, TypeVar, Generic, Callable
import uuid

from pydantic import BaseModel, Field

import flowlib as fl
from flowlib.core.errors import ProviderError, ErrorContext
from flowlib.core.registry.constants import ProviderType
from flowlib.providers.vector.base import VectorDBProvider
from flowlib.agents.providers.graph.base import GraphDBProvider
from flowlib.agents.memory.models import Entity, EntityAttribute, EntityRelationship

logger = logging.getLogger(__name__)

class MemorySearchResult(BaseModel):
    """Result from a memory search operation.
    
    This model represents the combined results from vector search
    and graph traversal operations.
    
    Attributes:
        entities: List of relevant entities
        context: Formatted context that can be injected into prompts
        sources: Source information for retrieved entities
        relevance_scores: Relevance scores for entities (if available)
    """
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of relevant entities retrieved from memory"
    )
    context: str = Field(
        default="",
        description="Formatted context that can be injected into prompts"
    )
    sources: Dict[str, str] = Field(
        default_factory=dict,
        description="Source information for retrieved entities"
    )
    relevance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores for entities (if available)"
    )

class WorkingMemoryItem(BaseModel):
    """Item in working memory cache.
    
    Attributes:
        entity: The entity object
        last_accessed: Timestamp of last access
        expiry: Optional expiry timestamp
    """
    entity: Entity
    last_accessed: float = Field(default_factory=time.time)
    expiry: Optional[float] = None

class HybridMemoryManager:
    """Hybrid memory manager integrating vector and graph capabilities.
    
    This class provides comprehensive memory management for agents,
    combining semantic search via vector databases and relationship
    traversal via graph databases with a working memory cache.
    """
    
    def __init__(
        self,
        vector_provider_name: Optional[str] = None,
        graph_provider_name: Optional[str] = None,
        working_memory_ttl: Optional[int] = 3600,  # 1 hour default TTL
        max_working_memory_items: int = 100,
    ):
        """Initialize the hybrid memory manager.
        
        Args:
            vector_provider_name: Name of the vector DB provider to use
            graph_provider_name: Name of the graph DB provider to use
            working_memory_ttl: Time-to-live in seconds for working memory items (None for no expiry)
            max_working_memory_items: Maximum number of items in working memory
        """
        self._vector_provider_name = vector_provider_name
        self._graph_provider_name = graph_provider_name
        self._working_memory_ttl = working_memory_ttl
        self._max_working_memory_items = max_working_memory_items
        
        # Will be initialized in initialize()
        self._vector_provider: Optional[VectorDBProvider] = None
        self._graph_provider: Optional[GraphDBProvider] = None
        self._working_memory: Dict[str, WorkingMemoryItem] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the memory manager and its providers.
        
        This method initializes the vector and graph providers
        if provider names were provided during initialization.
        
        Raises:
            ProviderError: If provider initialization fails
        """
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            # Initialize providers if specified
            try:
                if self._vector_provider_name:
                    self._vector_provider = await fl.provider_registry.get(
                        ProviderType.VECTOR_DB, 
                        self._vector_provider_name
                    )
                    
                if self._graph_provider_name:
                    self._graph_provider = await fl.provider_registry.get(
                        ProviderType.GRAPH_DB,
                        self._graph_provider_name
                    )
                    
                self._initialized = True
                logger.info(
                    f"Initialized HybridMemoryManager with "
                    f"vector={self._vector_provider_name}, "
                    f"graph={self._graph_provider_name}"
                )
                
            except Exception as e:
                raise ProviderError(
                    message=f"Failed to initialize memory manager: {str(e)}",
                    provider_name="hybrid_memory",
                    context=ErrorContext.create(
                        vector_provider=self._vector_provider_name,
                        graph_provider=self._graph_provider_name
                    ),
                    cause=e
                )
    
    async def shutdown(self) -> None:
        """Shutdown the memory manager and its providers.
        
        This method doesn't actually shut down the providers
        since they may be shared, but clears the working memory.
        """
        async with self._lock:
            self._working_memory.clear()
            self._initialized = False
            logger.info("Shut down HybridMemoryManager")
    
    async def _ensure_initialized(self) -> None:
        """Ensure the memory manager is initialized.
        
        Raises:
            ProviderError: If not initialized
        """
        if not self._initialized:
            await self.initialize()
            
        if not self._initialized:
            raise ProviderError(
                message="Memory manager not initialized",
                provider_name="hybrid_memory"
            )
    
    async def store_entity(self, entity: Entity) -> Entity:
        """Store an entity in memory.
        
        This method stores the entity in both the vector database
        and the graph database if available, as well as in working memory.
        
        Args:
            entity: The entity to store
            
        Returns:
            The stored entity (with any updates from storage)
            
        Raises:
            ProviderError: If entity storage fails
        """
        await self._ensure_initialized()
        
        try:
            # Store in graph database if available
            if self._graph_provider:
                await self._graph_provider.add_entity(entity)
                
            # Store in vector database if available
            # For vector DB we store each attribute as a separate vector
            if self._vector_provider:
                # Implementation will depend on vector provider interface
                # This is a placeholder for actual vector storage logic
                pass
                
            # Store in working memory
            await self._update_working_memory(entity)
            
            return entity
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to store entity: {str(e)}",
                provider_name="hybrid_memory",
                context=ErrorContext.create(
                    entity_id=entity.id,
                    entity_type=entity.type
                ),
                cause=e
            )
    
    async def store_entities(self, entities: List[Entity]) -> List[Entity]:
        """Store multiple entities in memory.
        
        Args:
            entities: List of entities to store
            
        Returns:
            List of stored entities
            
        Raises:
            ProviderError: If entity storage fails
        """
        await self._ensure_initialized()
        
        try:
            stored_entities = []
            
            # Store in graph database if available (bulk operation)
            if self._graph_provider:
                await self._graph_provider.bulk_add_entities(entities)
                
            # Store in vector database if available
            if self._vector_provider:
                # Implementation will depend on vector provider interface
                # This is a placeholder for actual vector storage logic
                pass
                
            # Store in working memory
            for entity in entities:
                await self._update_working_memory(entity)
                stored_entities.append(entity)
                
            return stored_entities
            
        except Exception as e:
            entity_ids = [entity.id for entity in entities]
            raise ProviderError(
                message=f"Failed to store entities: {str(e)}",
                provider_name="hybrid_memory",
                context=ErrorContext.create(
                    entity_ids=entity_ids
                ),
                cause=e
            )
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.
        
        This method checks working memory first, then graph database,
        and updates working memory if the entity is found.
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            ProviderError: If entity retrieval fails
        """
        await self._ensure_initialized()
        
        try:
            # Check working memory first
            if entity_id in self._working_memory:
                item = self._working_memory[entity_id]
                item.last_accessed = time.time()
                return item.entity
                
            # Check graph database if available
            entity = None
            if self._graph_provider:
                entity = await self._graph_provider.get_entity(entity_id)
                
            # If found, update working memory
            if entity:
                await self._update_working_memory(entity)
                
            return entity
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                provider_name="hybrid_memory",
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from memory.
        
        This method deletes the entity from both the vector database
        and the graph database if available, as well as from working memory.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
            
        Raises:
            ProviderError: If entity deletion fails
        """
        await self._ensure_initialized()
        
        try:
            # Delete from working memory
            if entity_id in self._working_memory:
                del self._working_memory[entity_id]
                
            # Delete from graph database if available
            graph_deleted = False
            if self._graph_provider:
                graph_deleted = await self._graph_provider.delete_entity(entity_id)
                
            # Delete from vector database if available
            vector_deleted = False
            if self._vector_provider:
                # Implementation will depend on vector provider interface
                # This is a placeholder for actual vector deletion logic
                pass
                
            return graph_deleted or vector_deleted
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                provider_name="hybrid_memory",
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
    
    async def add_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relation_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a relationship between entities.
        
        This method adds a relationship in the graph database if available,
        and updates the entities in working memory.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relation_type: Type of relationship
            properties: Optional properties for the relationship
            
        Raises:
            ProviderError: If relationship creation fails
        """
        await self._ensure_initialized()
        
        try:
            # Add relationship in graph database if available
            if self._graph_provider:
                await self._graph_provider.add_relationship(
                    source_id, target_id, relation_type, properties
                )
                
                # Update entities in working memory
                source_entity = await self.get_entity(source_id)
                target_entity = await self.get_entity(target_id)
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                provider_name="hybrid_memory",
                context=ErrorContext.create(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type
                ),
                cause=e
            )
    
    async def search_memory(
        self, 
        query: str, 
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
        min_relevance: float = 0.7,
        include_related: bool = True,
        max_related_depth: int = 1
    ) -> MemorySearchResult:
        """Search memory for relevant entities.
        
        This method combines vector search and graph traversal to find
        entities relevant to the query.
        
        Args:
            query: The search query
            entity_types: Optional list of entity types to filter by
            limit: Maximum number of primary results
            min_relevance: Minimum relevance score (0.0-1.0)
            include_related: Whether to include related entities
            max_related_depth: Maximum depth for related entities
            
        Returns:
            Memory search results
            
        Raises:
            ProviderError: If memory search fails
        """
        await self._ensure_initialized()
        
        try:
            result = MemorySearchResult()
            
            # Search vector database if available
            if self._vector_provider:
                # This is a placeholder for actual vector search logic
                # Vector search implementation will depend on provider interface
                pass
                
            # If no vector results or graph provider only, use graph query
            if not result.entities and self._graph_provider:
                # Basic query implementation - this would be enhanced based on provider capabilities
                graph_results = await self._graph_provider.query(
                    f"type IN :types",
                    {"types": entity_types} if entity_types else {}
                )
                
                # Convert results to Entity objects
                for item in graph_results[:limit]:
                    entity_id = item.get("id")
                    if entity_id:
                        entity = await self.get_entity(entity_id)
                        if entity:
                            result.entities.append(entity)
                            result.relevance_scores[entity_id] = 0.8  # Placeholder score
            
            # Get related entities if requested
            if include_related and result.entities and self._graph_provider:
                related_entities = []
                
                for entity in result.entities:
                    # Traverse graph to find related entities
                    traversal_results = await self._graph_provider.traverse(
                        entity.id,
                        max_depth=max_related_depth
                    )
                    
                    # Filter out duplicates and original entities
                    primary_ids = {e.id for e in result.entities}
                    for related in traversal_results:
                        if related.id not in primary_ids and related not in related_entities:
                            related_entities.append(related)
                
                # Add related entities to results
                result.entities.extend(related_entities)
                
                # Add relevance scores for related entities
                for entity in related_entities:
                    result.relevance_scores[entity.id] = 0.6  # Lower score for related entities
            
            # Format context for prompt injection
            if result.entities:
                context_parts = ["Relevant memory information:"]
                
                for entity in result.entities:
                    # Create formatted string representation of entity
                    entity_str = f"Entity: {entity.type.title()} - {entity.id}\n"
                    
                    # Add attributes
                    if entity.attributes:
                        entity_str += "Attributes:\n"
                        for name, attr in entity.attributes.items():
                            entity_str += f"  {name}: {attr.value}"
                            if attr.source:
                                entity_str += f" (from {attr.source})"
                            entity_str += "\n"
                    
                    # Add relationships
                    if entity.relationships:
                        entity_str += "Relationships:\n"
                        for rel in entity.relationships:
                            entity_str += f"  {rel.relation_type} -> {rel.target_id}\n"
                    
                    context_parts.append(entity_str)
                
                result.context = "\n".join(context_parts)
                
                # Add sources
                for entity in result.entities:
                    if entity.source:
                        result.sources[entity.id] = entity.source
            
            return result
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search memory: {str(e)}",
                provider_name="hybrid_memory",
                context=ErrorContext.create(
                    query=query,
                    entity_types=entity_types
                ),
                cause=e
            )
    
    async def _update_working_memory(self, entity: Entity) -> None:
        """Update working memory with an entity.
        
        This method adds or updates an entity in working memory,
        managing TTL and evicting old items if needed.
        
        Args:
            entity: Entity to add or update
        """
        async with self._lock:
            # Determine expiry time if TTL is set
            expiry = None
            if self._working_memory_ttl is not None:
                expiry = time.time() + self._working_memory_ttl
                
            # Add or update entity
            self._working_memory[entity.id] = WorkingMemoryItem(
                entity=entity,
                last_accessed=time.time(),
                expiry=expiry
            )
            
            # Cleanup expired or excess items
            await self._cleanup_working_memory()
    
    async def _cleanup_working_memory(self) -> None:
        """Clean up working memory by removing expired or excess items."""
        now = time.time()
        
        # Remove expired items
        expired_ids = [
            entity_id for entity_id, item in self._working_memory.items()
            if item.expiry is not None and item.expiry < now
        ]
        
        for entity_id in expired_ids:
            del self._working_memory[entity_id]
            
        # If still over limit, remove oldest accessed items
        if len(self._working_memory) > self._max_working_memory_items:
            # Sort by last accessed time (oldest first)
            sorted_items = sorted(
                self._working_memory.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest items
            items_to_remove = len(self._working_memory) - self._max_working_memory_items
            for entity_id, _ in sorted_items[:items_to_remove]:
                del self._working_memory[entity_id]
    
    async def clear_working_memory(self) -> None:
        """Clear all items from working memory."""
        async with self._lock:
            self._working_memory.clear()
            logger.info("Cleared working memory") 