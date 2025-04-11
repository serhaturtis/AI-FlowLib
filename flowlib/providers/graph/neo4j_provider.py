"""Neo4j graph database provider implementation.

This module provides a concrete implementation of the GraphDBProvider 
for Neo4j, a popular open-source graph database.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...core.errors import ProviderError, ErrorContext
from ..decorators import provider
from ..constants import ProviderType
from .base import GraphDBProvider, GraphDBProviderSettings
from .models import Entity, EntityAttribute, EntityRelationship

logger = logging.getLogger(__name__)

# Define dummy models for type annotations when neo4j is not installed
try:
    from neo4j import GraphDatabase, Driver, Session, Transaction, Result
    from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError
    NEO4J_AVAILABLE = True
except ImportError:
    logger.warning("neo4j-python-driver package not found. Install with 'pip install neo4j'")
    NEO4J_AVAILABLE = False
    # Define dummy classes for type annotations
    class Driver:
        pass
    class Session:
        pass
    class Transaction:
        pass
    class Result:
        pass


class Neo4jProviderSettings(GraphDBProviderSettings):
    """Settings for Neo4j graph database provider.
    
    Attributes:
        uri: Neo4j connection URI (e.g., 'bolt://localhost:7687')
        username: Neo4j username for authentication
        password: Neo4j password for authentication
        database: Neo4j database name (default: 'neo4j')
        encryption: Whether to use encrypted connection
        trust: Trust level for certificates
        connection_timeout: Connection timeout in seconds
        connection_acquisition_timeout: Timeout for acquiring connection from pool
        max_connection_lifetime: Maximum lifetime of a connection
        max_connection_pool_size: Maximum number of connections in the pool
    """
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"  # Should be overridden in production
    database: str = "neo4j"
    
    # Connection settings
    encryption: bool = False
    trust: str = "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
    connection_timeout: int = 30
    connection_acquisition_timeout: int = 60
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 100


@provider(provider_type=ProviderType.GRAPH_DB, name="neo4j")
class Neo4jProvider(GraphDBProvider):
    """Neo4j graph database provider implementation.
    
    This provider interfaces with Neo4j using the official Python driver,
    mapping entities and relationships to Neo4j's property graph model.
    """
    
    def __init__(self, name: str = "neo4j", settings: Optional[Neo4jProviderSettings] = None):
        """Initialize Neo4j graph database provider.
        
        Args:
            name: Provider name
            settings: Provider settings
        """
        # Create settings explicitly if not provided to avoid TypeVar issues
        settings = settings or Neo4jProviderSettings()
        
        super().__init__(name=name, settings=settings)
        self._driver: Optional[Driver] = None
        self._initialized = False
        
    async def _initialize(self) -> None:
        """Initialize the Neo4j connection.
        
        Creates the Neo4j driver instance and verifies the connection.
        Also ensures that required indexes and constraints are created.
        
        Raises:
            ProviderError: If Neo4j driver is not available or connection fails
        """
        if not NEO4J_AVAILABLE:
            raise ProviderError(
                message="Neo4j driver is not installed. Install with 'pip install neo4j'",
                provider_name=self.name
            )
        
        settings = self.settings
        
        try:
            # Create the Neo4j driver with settings
            self._driver = GraphDatabase.driver(
                settings.uri,
                auth=(settings.username, settings.password),
                encrypted=settings.encryption,
                trust=settings.trust,
                connection_timeout=settings.connection_timeout,
                connection_acquisition_timeout=settings.connection_acquisition_timeout,
                max_connection_lifetime=settings.max_connection_lifetime,
                max_connection_pool_size=settings.max_connection_pool_size
            )
            
            # Verify connection
            await self._execute_query("RETURN 1 AS test")
            
            # Create indexes and constraints
            await self._setup_schema()
            
            self._initialized = True
            logger.info(f"Neo4j provider '{self.name}' initialized successfully")
            
        except (ServiceUnavailable, AuthError) as e:
            raise ProviderError(
                message=f"Failed to connect to Neo4j: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    uri=settings.uri,
                    username=settings.username
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to initialize Neo4j provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Shut down the Neo4j connection.
        
        Closes the Neo4j driver instance and releases resources.
        """
        if self._driver:
            self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info(f"Neo4j provider '{self.name}' shut down")
    
    async def _setup_schema(self) -> None:
        """Set up Neo4j schema (indexes and constraints).
        
        Creates indexes and constraints to optimize performance:
        - Unique constraint on Entity.id
        - Index on Entity.type
        - Index on EntityRelationship.relation_type
        """
        # Create constraint for unique entity IDs
        await self._execute_query(
            """
            CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) 
            REQUIRE e.id IS UNIQUE
            """
        )
        
        # Create index on entity type
        await self._execute_query(
            """
            CREATE INDEX IF NOT EXISTS FOR (e:Entity) 
            ON (e.type)
            """
        )
        
        # Create index on relationship types
        await self._execute_query(
            """
            CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() 
            ON (r.relation_type)
            """
        )
    
    async def _execute_query(
        self, 
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against Neo4j.
        
        Args:
            query: Cypher query to execute
            parameters: Parameters for the query
            
        Returns:
            List of records as dictionaries
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._driver:
            raise ProviderError(
                message="Neo4j provider not initialized",
                provider_name=self.name
            )
        
        try:
            settings = self.settings
            parameters = parameters or {}
            
            # Use async with statement when proper async driver is available
            # For now, we're using the synchronous driver methods
            with self._driver.session(database=settings.database) as session:
                result = session.run(query, parameters)
                records = [dict(record) for record in result]
                return records
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute Neo4j query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query,
                    parameters=parameters
                ),
                cause=e
            )
    
    async def add_entity(self, entity: Entity) -> str:
        """Add or update an entity node in Neo4j.
        
        Args:
            entity: Entity to add or update
            
        Returns:
            ID of the created/updated entity
            
        Raises:
            ProviderError: If entity creation fails
        """
        try:
            # Convert entity to Neo4j-compatible format
            entity_props = {
                "id": entity.id,
                "type": entity.type,
                "source": entity.source,
                "importance": entity.importance,
                "last_updated": entity.last_updated
            }
            
            # Convert attributes to a serializable format
            attributes = {}
            for attr_name, attr in entity.attributes.items():
                attributes[attr_name] = {
                    "name": attr.name,
                    "value": attr.value,
                    "confidence": attr.confidence,
                    "source": attr.source,
                    "timestamp": attr.timestamp
                }
            
            # Create or merge the entity node
            await self._execute_query(
                """
                MERGE (e:Entity {id: $id})
                SET e = $properties, e.attributes = $attributes
                RETURN e.id as id
                """,
                {
                    "id": entity.id,
                    "properties": entity_props,
                    "attributes": attributes
                }
            )
            
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
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity.id,
                    entity_type=entity.type
                ),
                cause=e
            )
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID from Neo4j.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            ProviderError: If retrieval fails
        """
        try:
            # Query the entity with its relationships
            result = await self._execute_query(
                """
                MATCH (e:Entity {id: $id})
                OPTIONAL MATCH (e)-[r:RELATES_TO]->(target:Entity)
                RETURN 
                    e.id as id, 
                    e.type as type, 
                    e.source as source,
                    e.importance as importance,
                    e.last_updated as last_updated,
                    e.attributes as attributes,
                    collect({
                        target_id: target.id,
                        relation_type: r.relation_type,
                        confidence: r.confidence,
                        source: r.source,
                        timestamp: r.timestamp
                    }) as relationships
                """,
                {"id": entity_id}
            )
            
            if not result:
                return None
                
            record = result[0]
            
            # Deserialize attributes
            attributes = {}
            for attr_name, attr_data in record.get("attributes", {}).items():
                attributes[attr_name] = EntityAttribute(
                    name=attr_name,
                    value=attr_data.get("value", ""),
                    confidence=attr_data.get("confidence", 0.8),
                    source=attr_data.get("source", ""),
                    timestamp=attr_data.get("timestamp", datetime.now().isoformat())
                )
            
            # Deserialize relationships
            relationships = []
            for rel_data in record.get("relationships", []):
                # Skip invalid relationship items (null target)
                if not rel_data.get("target_id"):
                    continue
                    
                relationships.append(
                    EntityRelationship(
                        relation_type=rel_data.get("relation_type", ""),
                        target_entity=rel_data.get("target_id", ""),
                        confidence=rel_data.get("confidence", 0.8),
                        source=rel_data.get("source", ""),
                        timestamp=rel_data.get("timestamp", datetime.now().isoformat())
                    )
                )
            
            # Create and return entity
            entity = Entity(
                id=record.get("id", entity_id),
                type=record.get("type", "unknown"),
                attributes=attributes,
                relationships=relationships,
                source=record.get("source", ""),
                importance=record.get("importance", 0.5),
                last_updated=record.get("last_updated", datetime.now().isoformat())
            )
            
            return entity
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
    
    async def add_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Add a relationship between two entities in Neo4j.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID or name of the target entity
            relation_type: Type of relationship
            properties: Properties for the relationship
            
        Raises:
            ProviderError: If relationship creation fails
        """
        properties = properties or {}
        
        try:
            # Check if entities exist, create target if needed
            source_exists = await self._entity_exists(source_id)
            if not source_exists:
                raise ProviderError(
                    message=f"Source entity {source_id} does not exist",
                    provider_name=self.name
                )
            
            target_exists = await self._entity_exists(target_entity)
            if not target_exists:
                # Create a placeholder entity for the target
                logger.info(f"Creating placeholder entity for relationship target: {target_entity}")
                placeholder = Entity(
                    id=target_entity,
                    type="placeholder",
                    attributes={
                        "name": EntityAttribute(
                            name="name",
                            value=target_entity,
                            confidence=0.8,
                            source="system"
                        )
                    },
                    relationships=[],
                    source="system",
                    importance=0.5,
                    last_updated=datetime.now().isoformat()
                )
                await self.add_entity(placeholder)
            
            # Add relationship with properties
            rel_props = {
                "relation_type": relation_type,
                "confidence": properties.get("confidence", 0.8),
                "source": properties.get("source", "system"),
                "timestamp": properties.get("timestamp", datetime.now().isoformat())
            }
            
            await self._execute_query(
                """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATES_TO]->(target)
                SET r = $properties
                """,
                {
                    "source_id": source_id,
                    "target_id": target_entity,
                    "properties": rel_props
                }
            )
            
        except ProviderError:
            # Re-raise provider errors
            raise
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    source_id=source_id,
                    target_entity=target_entity,
                    relation_type=relation_type
                ),
                cause=e
            )
    
    async def _entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists.
        
        Args:
            entity_id: Entity ID to check
            
        Returns:
            True if entity exists, False otherwise
        """
        result = await self._execute_query(
            "MATCH (e:Entity {id: $id}) RETURN count(e) as count",
            {"id": entity_id}
        )
        
        if result and result[0]["count"] > 0:
            return True
        return False
        
    async def query_relationships(
        self, 
        entity_id: str, 
        relation_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """Query relationships for an entity in Neo4j.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional type to filter by
            direction: 'outgoing' or 'incoming'
            
        Returns:
            List of relationship information dicts
            
        Raises:
            ProviderError: If query fails
        """
        try:
            # Construct the query based on direction
            if direction == "outgoing":
                query = """
                MATCH (source:Entity {id: $entity_id})-[r:RELATES_TO]->(target:Entity)
                WHERE $relation_type IS NULL OR r.relation_type = $relation_type
                RETURN 
                    source.id as source, 
                    target.id as target, 
                    r.relation_type as type,
                    r as properties
                """
            else:  # incoming
                query = """
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity {id: $entity_id})
                WHERE $relation_type IS NULL OR r.relation_type = $relation_type
                RETURN 
                    source.id as source, 
                    target.id as target, 
                    r.relation_type as type,
                    r as properties
                """
            
            # Execute the query
            results = await self._execute_query(
                query,
                {
                    "entity_id": entity_id,
                    "relation_type": relation_type
                }
            )
            
            # Process results
            relationships = []
            for result in results:
                relationship = {
                    "source": result["source"],
                    "target": result["target"],
                    "type": result["type"],
                    "properties": {k: v for k, v in result.get("properties", {}).items() 
                                if k not in ["relation_type"]}
                }
                relationships.append(relationship)
                
            return relationships
            
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
        """Traverse the graph starting from an entity in Neo4j.
        
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
            # Check if start entity exists
            if not await self._entity_exists(start_id):
                return []
            
            # Create relationship filter
            rel_filter = ""
            if relation_types:
                types_list = [f"r.relation_type = '{type}'" for type in relation_types]
                rel_filter = f"WHERE {' OR '.join(types_list)}"
            
            # Use variable length path for traversal
            query = f"""
            MATCH path = (start:Entity {{id: $start_id}})-[r:RELATES_TO*1..{max_depth}]->(e:Entity)
            {rel_filter}
            RETURN DISTINCT e.id as entity_id
            """
            
            # Execute the query
            results = await self._execute_query(
                query,
                {"start_id": start_id}
            )
            
            # Fetch complete entity objects
            entities = []
            for result in results:
                entity_id = result["entity_id"]
                entity = await self.get_entity(entity_id)
                if entity:
                    entities.append(entity)
                    
            # Add the start entity
            start_entity = await self.get_entity(start_id)
            if start_entity and start_entity not in entities:
                entities.insert(0, start_entity)
                
            return entities
            
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
        """Execute a simple query language for Neo4j.
        
        Supports a limited set of commands:
            - "find_entities type={entity_type}" - Find entities by type
            - "find_entities name={name}" - Find entities by name
            - "neighbors id={id} [relation={relation_type}]" - Find neighboring entities
            - "path from={id} to={id} [max_depth={n}]" - Find path between entities
            - "cypher {cypher_query}" - Execute custom Cypher query
        
        Args:
            query: Query string
            params: Optional query parameters
            
        Returns:
            Query results
            
        Raises:
            ProviderError: If query parsing fails
        """
        try:
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
                return await self._find_entities_by_type(entity_type)
                
            # Find entities by name
            elif query.startswith("find_entities") and "name" in param_dict:
                name = param_dict["name"]
                return await self._find_entities_by_name(name)
                
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
                
            # Execute custom Cypher query
            elif query.startswith("cypher") and "cypher_query" in param_dict:
                cypher_query = param_dict["cypher_query"]
                cypher_params = param_dict.get("params", {})
                return await self._execute_query(cypher_query, cypher_params)
                
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
            
    async def _find_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Find entities by type in Neo4j."""
        results = await self._execute_query(
            """
            MATCH (e:Entity)
            WHERE e.type = $type
            RETURN e.id as id, e as entity
            """,
            {"type": entity_type}
        )
        
        return [{"id": r["id"], "entity": r["entity"]} for r in results]
        
    async def _find_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find entities by name in Neo4j."""
        results = await self._execute_query(
            """
            MATCH (e:Entity)
            WHERE e.attributes.name.value CONTAINS $name
            RETURN e.id as id, e as entity
            """,
            {"name": name}
        )
        
        return [{"id": r["id"], "entity": r["entity"]} for r in results]
        
    async def _find_neighbors(self, entity_id: str, relation_type: Optional[str]) -> List[Dict[str, Any]]:
        """Find neighboring entities in Neo4j."""
        query = """
        MATCH (e:Entity {id: $id})-[r:RELATES_TO]->(neighbor:Entity)
        WHERE $relation_type IS NULL OR r.relation_type = $relation_type
        RETURN 
            neighbor.id as id, 
            r.relation_type as relation, 
            neighbor as entity
        """
        
        results = await self._execute_query(
            query,
            {
                "id": entity_id,
                "relation_type": relation_type
            }
        )
        
        return [
            {
                "id": r["id"],
                "relation": r["relation"],
                "entity": r["entity"]
            } for r in results
        ]
        
    async def _find_path(self, from_id: str, to_id: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find path between entities in Neo4j."""
        query = f"""
        MATCH path = shortestPath((source:Entity {{id: $from_id}})-[r:RELATES_TO*1..{max_depth}]->(target:Entity {{id: $to_id}}))
        UNWIND nodes(path) as node
        WITH node, index(nodes(path), node) as position
        RETURN position, node.id as id, node as entity
        ORDER BY position
        """
        
        results = await self._execute_query(
            query,
            {
                "from_id": from_id,
                "to_id": to_id
            }
        )
        
        return [
            {
                "position": r["position"],
                "id": r["id"],
                "entity": r["entity"]
            } for r in results
        ]
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships from Neo4j.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Check if entity exists
            entity_exists = await self._entity_exists(entity_id)
            if not entity_exists:
                return False
            
            # Delete entity and all relationships
            result = await self._execute_query(
                """
                MATCH (e:Entity {id: $id})
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                {"id": entity_id}
            )
            
            if result and result[0]["deleted"] > 0:
                return True
            
            return False
            
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
        target_entity: str, 
        relation_type: Optional[str] = None
    ) -> bool:
        """Delete relationship(s) between entities in Neo4j.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Optional type to filter by
            
        Returns:
            True if relationships were deleted, False if none found
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Construct relation type filter
            relation_filter = ""
            if relation_type:
                relation_filter = "AND r.relation_type = $relation_type"
            
            # Delete matching relationships
            result = await self._execute_query(
                f"""
                MATCH (source:Entity {{id: $source_id}})-[r:RELATES_TO]->(target:Entity {{id: $target_id}})
                WHERE true {relation_filter}
                WITH r
                DELETE r
                RETURN count(r) as deleted
                """,
                {
                    "source_id": source_id,
                    "target_id": target_entity,
                    "relation_type": relation_type
                }
            )
            
            if result and result[0]["deleted"] > 0:
                return True
            
            return False
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    source_id=source_id,
                    target_entity=target_entity,
                    relation_type=relation_type
                ),
                cause=e
            )
    
    async def remove_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str
    ) -> None:
        """Remove a relationship between two entities in Neo4j.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
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
    
    async def bulk_add_entities(self, entities: List[Entity]) -> List[str]:
        """Add multiple entities in bulk to Neo4j.
        
        This method optimizes bulk insertions by batching entities.
        
        Args:
            entities: List of entities to add
            
        Returns:
            List of IDs of added entities
            
        Raises:
            ProviderError: If bulk operation fails
        """
        try:
            settings = self.settings
            added_ids = []
            
            # Process in batches
            batch_size = settings.max_batch_size
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                
                # Add each entity in the batch
                for entity in batch:
                    entity_id = await self.add_entity(entity)
                    added_ids.append(entity_id)
                    
            return added_ids
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to bulk add entities: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_count=len(entities)
                ),
                cause=e
            ) 