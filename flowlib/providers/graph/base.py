"""Graph database provider base class.

This module defines the base class for graph database providers,
establishing a common interface for entity and relationship operations.
"""

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Any, Generic, TypeVar

from ..base import Provider
from ...flows.base import FlowSettings
from .models import Entity

logger = logging.getLogger(__name__)

class GraphDBProviderSettings(FlowSettings):
    """Settings for graph database providers.
    
    Attributes:
        max_retries: Maximum retries for graph operations
        retry_delay_seconds: Delay between retries in seconds
        timeout_seconds: Operation timeout in seconds
        max_batch_size: Maximum batch size for operations
    """
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    max_batch_size: int = 100

# Define Settings TypeVar based on GraphDBProviderSettings
SettingsType = TypeVar('SettingsType', bound=GraphDBProviderSettings)

class GraphDBProvider(Provider[GraphDBProviderSettings], Generic[SettingsType]):
    """Base class for graph database providers.
    
    This class defines the common interface for all graph database providers,
    with methods for entity and relationship operations.
    """
    
    def __init__(self, name: str = "graph", settings: Optional[GraphDBProviderSettings] = None):
        """Initialize graph database provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Initialize with provider_type="graph_db"
        super().__init__(name=name, settings=settings, provider_type="graph_db")
        
    @abstractmethod
    async def add_entity(self, entity: Entity) -> str:
        """Add or update an entity node.
        
        Args:
            entity: Entity to add or update
            
        Returns:
            ID of the created/updated entity
            
        Raises:
            ProviderError: If entity creation fails
        """
        raise NotImplementedError("Subclasses must implement add_entity()")
        
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            ProviderError: If entity retrieval fails
        """
        raise NotImplementedError("Subclasses must implement get_entity()")
        
    @abstractmethod
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
            ProviderError: If relationship creation fails
        """
        raise NotImplementedError("Subclasses must implement add_relationship()")
        
    @abstractmethod
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
            ProviderError: If relationship query fails
        """
        raise NotImplementedError("Subclasses must implement query_relationships()")
        
    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement traverse()")
        
    @abstractmethod
    async def query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a native query in the graph database.
        
        Args:
            query: Query string in the native query language
            params: Optional query parameters
            
        Returns:
            Query results
            
        Raises:
            ProviderError: If query execution fails
        """
        raise NotImplementedError("Subclasses must implement query()")
        
    async def get_health(self) -> Dict[str, Any]:
        """Get provider health information.
        
        Returns:
            Health metrics
            
        Raises:
            ProviderError: If health check fails
        """
        try:
            return {
                "status": "healthy",
                "provider": self.name,
                "provider_type": self.provider_type,
                "initialized": self.initialized
            }
        except Exception as e:
            logger.error(f"Error checking health for provider '{self.name}': {str(e)}")
            return {
                "status": "unhealthy",
                "provider": self.name,
                "provider_type": self.provider_type,
                "error": str(e)
            }
        
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
            
        Raises:
            ProviderError: If entity deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_entity()")
    
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
            
        Raises:
            ProviderError: If relationship deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_relationship()")
    
    async def bulk_add_entities(self, entities: List[Entity]) -> List[str]:
        """Add multiple entities in a batch.
        
        Args:
            entities: List of entities to add
            
        Returns:
            List of created/updated entity IDs
            
        Raises:
            ProviderError: If batch creation fails
        """
        # Default implementation adds each entity individually
        entity_ids = []
        for entity in entities:
            entity_id = await self.add_entity(entity)
            entity_ids.append(entity_id)
        return entity_ids
    
    @abstractmethod
    async def search_entities(
        self,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities based on criteria.
        
        Args:
            query: Optional text query to match against entity ID or attributes.
            entity_type: Optional entity type to filter by.
            tags: Optional list of tags to filter by.
            limit: Maximum number of entities to return.
            
        Returns:
            List of matching Entity objects.
            
        Raises:
            ProviderError: If the search operation fails.
        """
        raise NotImplementedError("Subclasses must implement search_entities()")

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement remove_relationship()") 