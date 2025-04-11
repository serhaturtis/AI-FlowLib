"""JanusGraph database provider implementation.

This module provides a concrete implementation of the GraphDBProvider 
for JanusGraph, a distributed graph database based on Apache TinkerPop.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...core.errors import ProviderError, ErrorContext
from ..decorators import provider
from ..constants import ProviderType
from .base import GraphDBProvider, GraphDBProviderSettings
from .models import Entity, EntityAttribute, EntityRelationship

logger = logging.getLogger(__name__)

# Define dummy models for type annotations when gremlin-python is not installed
try:
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.driver.client import Client
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.process.graph_traversal import __
    from gremlin_python.driver.protocol.graph_binary_message_serializer import GraphBinaryMessageSerializer
    JANUS_AVAILABLE = True
except ImportError:
    logger.warning("gremlin-python package not found. Install with 'pip install gremlinpython'")
    JANUS_AVAILABLE = False
    # Define dummy classes for type annotations
    class DriverRemoteConnection:
        pass
    class Client:
        pass
    # Define a dummy object for __
    class DummyUnderscore:
        pass
    __ = DummyUnderscore()


class JanusProviderSettings(GraphDBProviderSettings):
    """Settings for JanusGraph graph database provider.
    
    Attributes:
        url: JanusGraph Gremlin server WebSocket URL (e.g., 'ws://localhost:8182/gremlin')
        username: Username for authentication (if using authentication)
        password: Password for authentication (if using authentication)
        graph_name: Name of the graph instance (default: 'g')
        traversal_source: Name of the traversal source (default: 'g')
        connection_pool_size: Size of the connection pool
        message_serializer: Message serializer to use (default: 'graphbinary-1.0')
        read_timeout: Read timeout in seconds
        write_timeout: Write timeout in seconds
        max_retry_count: Maximum number of retry attempts for operations
    """
    url: str = "ws://localhost:8182/gremlin"
    username: str = ""
    password: str = ""
    graph_name: str = "g"
    traversal_source: str = "g"
    connection_pool_size: int = 4
    message_serializer: str = "graphbinary-1.0"
    read_timeout: int = 30
    write_timeout: int = 30
    max_retry_count: int = 3


@provider(provider_type=ProviderType.GRAPH_DB, name="janusgraph")
class JanusProvider(GraphDBProvider):
    """JanusGraph graph database provider implementation.
    
    This provider interfaces with JanusGraph using the Gremlin Python client,
    mapping entities and relationships to JanusGraph's property graph model.
    """
    
    def __init__(self, name: str = "janusgraph", settings: Optional[JanusProviderSettings] = None):
        """Initialize JanusGraph graph database provider.
        
        Args:
            name: Provider name
            settings: Provider settings
        """
        # Create settings explicitly if not provided to avoid TypeVar issues
        settings = settings or JanusProviderSettings()
        
        super().__init__(name=name, settings=settings)
        self._client: Optional[Client] = None
        self._g = None  # Traversal source
        self._initialized = False
        
    async def _initialize(self) -> None:
        """Initialize the JanusGraph connection.
        
        Creates the Gremlin client instance and verifies the connection.
        Also ensures that required indexes are created.
        
        Raises:
            ProviderError: If Gremlin Python driver is not available or connection fails
        """
        if not JANUS_AVAILABLE:
            raise ProviderError(
                message="Gremlin Python driver is not installed. Install with 'pip install gremlinpython'",
                provider_name=self.name
            )
        
        settings = self.settings
        
        try:
            # Set up serializer
            message_serializer = GraphBinaryMessageSerializer()
            
            # Create Gremlin client
            self._client = Client(
                settings.url,
                'g',
                pool_size=settings.connection_pool_size,
                message_serializer=message_serializer,
                username=settings.username if settings.username else None,
                password=settings.password if settings.password else None,
                read_timeout=settings.read_timeout,
                write_timeout=settings.write_timeout
            )
            
            # Create remote connection for traversals
            connection = DriverRemoteConnection(settings.url, settings.traversal_source)
            self._g = traversal().withRemote(connection)
            
            # Verify connection by executing a simple query
            self._client.submit('g.V().limit(1).count()').all().result()
            
            # Create schema (indexes)
            await self._setup_schema()
            
            self._initialized = True
            logger.info(f"JanusGraph provider '{self.name}' initialized successfully")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to connect to JanusGraph: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    url=settings.url,
                    username=settings.username if settings.username else None
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Shut down the JanusGraph connection.
        
        Closes the Gremlin client instance and releases resources.
        """
        if self._client:
            self._client.close()
            self._client = None
            self._g = None
            self._initialized = False
            logger.info(f"JanusGraph provider '{self.name}' shut down")
    
    async def _setup_schema(self) -> None:
        """Set up JanusGraph schema (indexes).
        
        Creates indexes to optimize performance:
        - Create vertex label for Entity
        - Create edge label for RELATES_TO
        - Create property key for id, type, etc.
        - Create composite index on Entity(id)
        - Create index on Entity(type)
        """
        try:
            # JanusGraph schema management is typically done via the management API
            # This uses execute_query to send schema management commands
            
            # Create vertex label for Entity if it doesn't exist
            await self._execute_query("""
                mgmt = graph.openManagement()
                if (!mgmt.getVertexLabel('Entity')) {
                    mgmt.makeVertexLabel('Entity').make()
                }
                mgmt.commit()
            """)
            
            # Create edge label for RELATES_TO if it doesn't exist
            await self._execute_query("""
                mgmt = graph.openManagement()
                if (!mgmt.getEdgeLabel('RELATES_TO')) {
                    mgmt.makeEdgeLabel('RELATES_TO').make()
                }
                mgmt.commit()
            """)
            
            # Create property keys
            await self._execute_query("""
                mgmt = graph.openManagement()
                if (!mgmt.getPropertyKey('id')) {
                    mgmt.makePropertyKey('id').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('type')) {
                    mgmt.makePropertyKey('type').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('relation_type')) {
                    mgmt.makePropertyKey('relation_type').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('attributes')) {
                    mgmt.makePropertyKey('attributes').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('importance')) {
                    mgmt.makePropertyKey('importance').dataType(Double.class).make()
                }
                mgmt.commit()
            """)
            
            # Create composite index for Entity.id
            await self._execute_query("""
                mgmt = graph.openManagement()
                id = mgmt.getPropertyKey('id')
                if (id && !mgmt.getGraphIndex('entityById')) {
                    mgmt.buildIndex('entityById', Vertex.class).addKey(id).unique().buildCompositeIndex()
                }
                mgmt.commit()
            """)
            
            # Create index for Entity.type
            await self._execute_query("""
                mgmt = graph.openManagement()
                type = mgmt.getPropertyKey('type')
                if (type && !mgmt.getGraphIndex('entityByType')) {
                    mgmt.buildIndex('entityByType', Vertex.class).addKey(type).buildCompositeIndex()
                }
                mgmt.commit()
            """)
            
            # Create index for edge relation_type
            await self._execute_query("""
                mgmt = graph.openManagement()
                relType = mgmt.getPropertyKey('relation_type')
                if (relType && !mgmt.getGraphIndex('edgeByRelationType')) {
                    mgmt.buildIndex('edgeByRelationType', Edge.class).addKey(relType).buildCompositeIndex()
                }
                mgmt.commit()
            """)
            
        except Exception as e:
            logger.warning(f"Failed to set up JanusGraph schema: {str(e)}")
            # Continue initialization even if schema setup fails
            # as the schema might already exist or will be managed externally
    
    async def _execute_query(self, query: str, bindings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Gremlin query against JanusGraph.
        
        Args:
            query: Gremlin query to execute
            bindings: Parameter bindings for the query
            
        Returns:
            List of records as dictionaries
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._client:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
        
        try:
            # Execute query with bindings
            bindings = bindings or {}
            result = self._client.submit(query, bindings).all().result()
            
            # Convert result to dictionaries
            records = []
            for item in result:
                # Handle different result types
                if hasattr(item, 'properties'):
                    # Convert vertex/edge to dict
                    record = self._element_to_dict(item)
                    records.append(record)
                elif isinstance(item, dict):
                    records.append(item)
                elif isinstance(item, (int, float, str, bool)):
                    records.append({"value": item})
                else:
                    # Try to convert to dict
                    try:
                        records.append(dict(item))
                    except:
                        records.append({"value": str(item)})
            
            return records
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute JanusGraph query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query,
                    bindings=bindings
                ),
                cause=e
            )
    
    def _element_to_dict(self, element) -> Dict[str, Any]:
        """Convert a Gremlin graph element (vertex/edge) to a dictionary."""
        result = {}
        
        # Extract element ID
        result["_id"] = str(element.id)
        
        # Extract element label
        if hasattr(element, 'label'):
            result["_label"] = element.label
            
        # Extract properties
        if hasattr(element, 'properties'):
            for key, value in element.properties.items():
                # Handle multi-valued properties
                if isinstance(value, list):
                    if len(value) == 1:
                        result[key] = value[0].value
                    else:
                        result[key] = [v.value for v in value]
                else:
                    result[key] = value
        
        return result
    
    async def add_entity(self, entity: Entity) -> str:
        """Add or update an entity node in JanusGraph.
        
        Args:
            entity: Entity to add or update
            
        Returns:
            ID of the created/updated entity
            
        Raises:
            ProviderError: If entity creation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Convert entity to JanusGraph compatible format
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
            entity_props["attributes"] = json.dumps(attributes)
            
            # Check if entity exists (this is case-sensitive)
            existing = await self._execute_query(
                "g.V().has('Entity', 'id', id).hasNext()",
                {"id": entity.id}
            )
            
            if existing and existing[0].get("value", False):
                # Update existing entity
                await self._execute_query(
                    """
                    g.V().has('Entity', 'id', id)
                      .property('type', type)
                      .property('source', source)
                      .property('importance', importance)
                      .property('last_updated', last_updated)
                      .property('attributes', attributes)
                    """,
                    {
                        "id": entity.id,
                        "type": entity.type,
                        "source": entity.source,
                        "importance": entity.importance,
                        "last_updated": entity.last_updated,
                        "attributes": json.dumps(attributes)
                    }
                )
            else:
                # Create new entity
                await self._execute_query(
                    """
                    g.addV('Entity')
                      .property('id', id)
                      .property('type', type)
                      .property('source', source)
                      .property('importance', importance)
                      .property('last_updated', last_updated)
                      .property('attributes', attributes)
                    """,
                    {
                        "id": entity.id,
                        "type": entity.type,
                        "source": entity.source,
                        "importance": entity.importance,
                        "last_updated": entity.last_updated,
                        "attributes": json.dumps(attributes)
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
        """Get an entity by ID from JanusGraph.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            ProviderError: If retrieval fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Get entity with its properties
            result = await self._execute_query(
                """
                g.V().has('Entity', 'id', id)
                  .project('id', 'type', 'source', 'importance', 'last_updated', 'attributes')
                  .by('id')
                  .by('type')
                  .by('source')
                  .by('importance')
                  .by('last_updated')
                  .by('attributes')
                """,
                {"id": entity_id}
            )
            
            if not result:
                return None
                
            entity_data = result[0]
            
            # Parse attributes
            attributes = {}
            attr_json = entity_data.get("attributes", "{}")
            try:
                attr_data = json.loads(attr_json)
                for attr_name, attr_props in attr_data.items():
                    attributes[attr_name] = EntityAttribute(
                        name=attr_name,
                        value=attr_props.get("value", ""),
                        confidence=attr_props.get("confidence", 0.8),
                        source=attr_props.get("source", ""),
                        timestamp=attr_props.get("timestamp", datetime.now().isoformat())
                    )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse attributes for entity {entity_id}")
            
            # Get relationships
            rel_result = await self._execute_query(
                """
                g.V().has('Entity', 'id', id)
                  .outE('RELATES_TO')
                  .project('relation_type', 'confidence', 'source', 'timestamp', 'target_id')
                  .by('relation_type')
                  .by('confidence')
                  .by('source')
                  .by('timestamp')
                  .by(inV().values('id'))
                """,
                {"id": entity_id}
            )
            
            relationships = []
            for rel in rel_result:
                if not rel.get("target_id"):
                    continue
                    
                relationships.append(
                    EntityRelationship(
                        relation_type=rel.get("relation_type", ""),
                        target_entity=rel.get("target_id", ""),
                        confidence=rel.get("confidence", 0.8),
                        source=rel.get("source", ""),
                        timestamp=rel.get("timestamp", datetime.now().isoformat())
                    )
                )
            
            # Create Entity object
            entity = Entity(
                id=entity_data.get("id", entity_id),
                type=entity_data.get("type", "unknown"),
                attributes=attributes,
                relationships=relationships,
                source=entity_data.get("source", ""),
                importance=entity_data.get("importance", 0.5),
                last_updated=entity_data.get("last_updated", datetime.now().isoformat())
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
        """Add a relationship between two entities in JanusGraph.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID or name of the target entity
            relation_type: Type of relationship
            properties: Properties for the relationship
            
        Raises:
            ProviderError: If relationship creation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
        properties = properties or {}
        
        try:
            # Check if source entity exists
            source_exists = await self._entity_exists(source_id)
            if not source_exists:
                raise ProviderError(
                    message=f"Source entity {source_id} does not exist",
                    provider_name=self.name
                )
            
            # Check if target entity exists, create placeholder if not
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
            
            # Add edge properties
            props = {
                "relation_type": relation_type,
                "confidence": properties.get("confidence", 0.8),
                "source": properties.get("source", "system"),
                "timestamp": properties.get("timestamp", datetime.now().isoformat())
            }
            
            # Check if relationship already exists
            rel_exists = await self._execute_query(
                """
                g.V().has('Entity', 'id', sourceId)
                  .outE('RELATES_TO')
                  .where(inV().has('id', targetId))
                  .where(values('relation_type').is(relType))
                  .hasNext()
                """,
                {
                    "sourceId": source_id,
                    "targetId": target_entity,
                    "relType": relation_type
                }
            )
            
            if rel_exists and rel_exists[0].get("value", False):
                # Update existing relationship
                await self._execute_query(
                    """
                    g.V().has('Entity', 'id', sourceId)
                      .outE('RELATES_TO')
                      .where(inV().has('id', targetId))
                      .where(values('relation_type').is(relType))
                      .property('confidence', confidence)
                      .property('source', source)
                      .property('timestamp', timestamp)
                    """,
                    {
                        "sourceId": source_id,
                        "targetId": target_entity,
                        "relType": relation_type,
                        "confidence": props["confidence"],
                        "source": props["source"],
                        "timestamp": props["timestamp"]
                    }
                )
            else:
                # Create new relationship
                await self._execute_query(
                    """
                    g.V().has('Entity', 'id', sourceId)
                      .as('source')
                      .V().has('Entity', 'id', targetId)
                      .as('target')
                      .addE('RELATES_TO')
                      .from('source')
                      .to('target')
                      .property('relation_type', relType)
                      .property('confidence', confidence)
                      .property('source', source)
                      .property('timestamp', timestamp)
                    """,
                    {
                        "sourceId": source_id,
                        "targetId": target_entity,
                        "relType": relation_type,
                        "confidence": props["confidence"],
                        "source": props["source"],
                        "timestamp": props["timestamp"]
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
        """Check if an entity exists in JanusGraph.
        
        Args:
            entity_id: ID of the entity to check
            
        Returns:
            True if entity exists, False otherwise
        """
        result = await self._execute_query(
            "g.V().has('Entity', 'id', id).hasNext()",
            {"id": entity_id}
        )
        
        return result and result[0].get("value", False)
    
    async def query_relationships(
        self, 
        entity_id: str, 
        relation_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """Query relationships for an entity in JanusGraph.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional type to filter by
            direction: 'outgoing' or 'incoming'
            
        Returns:
            List of relationship information dicts
            
        Raises:
            ProviderError: If query fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Construct filter for relation type
            rel_filter = ""
            if relation_type:
                rel_filter = ".has('relation_type', relType)"
                
            # Construct query based on direction
            if direction == "outgoing":
                query = f"""
                g.V().has('Entity', 'id', id)
                  .outE('RELATES_TO'){rel_filter}
                  .as('rel')
                  .inV()
                  .project('source', 'target', 'type', 'properties')
                  .by(constant(id))
                  .by('id')
                  .by(select('rel').values('relation_type'))
                  .by(select('rel').valueMap())
                """
            else:  # incoming
                query = f"""
                g.V().has('Entity', 'id', id)
                  .inE('RELATES_TO'){rel_filter}
                  .as('rel')
                  .outV()
                  .project('source', 'target', 'type', 'properties')
                  .by('id')
                  .by(constant(id))
                  .by(select('rel').values('relation_type'))
                  .by(select('rel').valueMap())
                """
            
            # Execute query with parameters
            params = {
                "id": entity_id,
                "relType": relation_type
            }
            
            results = await self._execute_query(query, params)
            
            # Process results
            relationships = []
            for result in results:
                # Extract properties excluding relation_type
                properties = result.get("properties", {})
                if "relation_type" in properties:
                    del properties["relation_type"]
                    
                relationship = {
                    "source": result.get("source", ""),
                    "target": result.get("target", ""),
                    "type": result.get("type", ""),
                    "properties": properties
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
        """Traverse the graph starting from an entity in JanusGraph.
        
        Args:
            start_id: ID of the starting entity
            relation_types: Optional list of relation types to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            List of entities found in traversal
            
        Raises:
            ProviderError: If traversal fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Check if start entity exists
            if not await self._entity_exists(start_id):
                return []
                
            # Build relation type filter
            rel_filter = ""
            if relation_types:
                rel_types_str = ", ".join([f"'{rel}'" for rel in relation_types])
                rel_filter = f".has('relation_type', within({rel_types_str}))"
                
            # Perform traversal
            query = f"""
            g.V().has('Entity', 'id', startId)
              .repeat(outE('RELATES_TO'){rel_filter}.inV().dedup())
              .emit()
              .times({max_depth})
              .values('id')
            """
            
            # Execute traversal
            results = await self._execute_query(
                query,
                {"startId": start_id}
            )
            
            # Collect entity IDs
            entity_ids = set()
            for result in results:
                if isinstance(result, dict) and "value" in result:
                    entity_ids.add(result["value"])
                    
            # Add start entity ID
            entity_ids.add(start_id)
            
            # Retrieve full entity objects
            entities = []
            for entity_id in entity_ids:
                entity = await self.get_entity(entity_id)
                if entity:
                    entities.append(entity)
                    
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
        """Execute a simple query language for JanusGraph.
        
        Supports a limited set of commands:
            - "find_entities type={entity_type}" - Find entities by type
            - "find_entities name={name}" - Find entities by name
            - "neighbors id={id} [relation={relation_type}]" - Find neighboring entities
            - "path from={id} to={id} [max_depth={n}]" - Find path between entities
            - "gremlin {gremlin_query}" - Execute custom Gremlin query
        
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
                
            # Execute custom Gremlin query
            elif query.startswith("gremlin") and "gremlin_query" in param_dict:
                gremlin_query = param_dict["gremlin_query"]
                gremlin_params = param_dict.get("params", {})
                return await self._execute_query(gremlin_query, gremlin_params)
                
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
        """Find entities by type in JanusGraph."""
        results = await self._execute_query(
            """
            g.V().has('Entity', 'type', type)
              .project('id', 'entity')
              .by('id')
              .by(valueMap())
            """,
            {"type": entity_type}
        )
        
        return results
        
    async def _find_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find entities by name attribute in JanusGraph."""
        # This requires searching in the JSON attributes field
        results = await self._execute_query(
            """
            g.V().has('Entity', 'attributes', textContains('name'))
              .filter(values('attributes').is(textContains(name)))
              .project('id', 'entity')
              .by('id')
              .by(valueMap())
            """,
            {"name": name}
        )
        
        return results
        
    async def _find_neighbors(self, entity_id: str, relation_type: Optional[str]) -> List[Dict[str, Any]]:
        """Find neighboring entities in JanusGraph."""
        # Construct filter for relation type
        rel_filter = ""
        if relation_type:
            rel_filter = ".has('relation_type', relType)"
            
        query = f"""
        g.V().has('Entity', 'id', id)
          .outE('RELATES_TO'){rel_filter}
          .as('rel')
          .inV()
          .project('id', 'relation', 'entity')
          .by('id')
          .by(select('rel').values('relation_type'))
          .by(valueMap())
        """
        
        results = await self._execute_query(
            query,
            {
                "id": entity_id,
                "relType": relation_type
            }
        )
        
        return results
        
    async def _find_path(self, from_id: str, to_id: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find path between entities in JanusGraph."""
        query = f"""
        g.V().has('Entity', 'id', fromId)
          .until(has('id', toId))
          .repeat(out().simplePath())
          .limit(1)
          .path()
          .unfold()
          .project('position', 'id', 'entity')
          .by(constant(-1))  // Will update position later
          .by('id')
          .by(valueMap())
        """
        
        results = await self._execute_query(
            query,
            {
                "fromId": from_id,
                "toId": to_id
            }
        )
        
        # Update positions
        for i, result in enumerate(results):
            result["position"] = i
            
        return results
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships from JanusGraph.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Check if entity exists
            if not await self._entity_exists(entity_id):
                return False
                
            # Delete entity and its relationships
            await self._execute_query(
                """
                g.V().has('Entity', 'id', id).drop()
                """,
                {"id": entity_id}
            )
            
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
        target_entity: str, 
        relation_type: Optional[str] = None
    ) -> bool:
        """Delete relationship(s) between entities in JanusGraph.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Optional type to filter by
            
        Returns:
            True if relationships were deleted, False if none found
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Construct relation type filter
            rel_filter = ""
            if relation_type:
                rel_filter = ".has('relation_type', relType)"
                
            # Delete matching edges
            result = await self._execute_query(
                f"""
                g.V().has('Entity', 'id', sourceId)
                  .outE('RELATES_TO'){rel_filter}
                  .where(inV().has('id', targetId))
                  .drop()
                  .count()
                """,
                {
                    "sourceId": source_id,
                    "targetId": target_entity,
                    "relType": relation_type
                }
            )
            
            # Check if any edges were deleted
            deleted_count = result[0].get("value", 0) if result else 0
            return deleted_count > 0
            
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
        """Remove a relationship between two entities in JanusGraph.
        
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
        """Add multiple entities in bulk to JanusGraph.
        
        This method optimizes bulk insertions by batching entities.
        
        Args:
            entities: List of entities to add
            
        Returns:
            List of IDs of added entities
            
        Raises:
            ProviderError: If bulk operation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                provider_name=self.name
            )
            
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