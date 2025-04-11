"""ArangoDB graph database provider implementation.

This module provides a concrete implementation of the GraphDBProvider 
for ArangoDB, a multi-model database with strong graph capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...core.errors import ProviderError, ErrorContext
from ..decorators import provider
from ..constants import ProviderType
from .base import GraphDBProvider, GraphDBProviderSettings
from .models import Entity, EntityAttribute

logger = logging.getLogger(__name__)

# Define dummy models for type annotations when python-arango is not installed
try:
    from arango import ArangoClient, Database, Collection, StandardDatabase
    from arango.exceptions import ArangoError
    ARANGO_AVAILABLE = True
except ImportError:
    logger.warning("python-arango package not found. Install with 'pip install python-arango'")
    ARANGO_AVAILABLE = False
    # Define dummy classes for type annotations
    class ArangoClient:
        pass
    class Database:
        pass
    class Collection:
        pass
    class StandardDatabase:
        pass


class ArangoProviderSettings(GraphDBProviderSettings):
    """Settings for ArangoDB provider.
    
    Attributes:
        url: ArangoDB server URL (e.g., 'http://localhost:8529')
        username: Username for authentication
        password: Password for authentication
        database: Database name
        graph_name: Graph name
        entity_collection: Collection name for entities
        relation_collection: Collection name for relationships
        verify: Whether to verify SSL certificates
        max_http_pool_size: Maximum size of the HTTP connection pool
    """
    url: str = "http://localhost:8529"
    username: str = "root"
    password: str = ""  # Should be overridden in production
    database: str = "flowlib"
    graph_name: str = "memory_graph"
    entity_collection: str = "entities"
    relation_collection: str = "relationships"
    verify: bool = True
    max_http_pool_size: int = 10


@provider(provider_type=ProviderType.GRAPH_DB, name="arango")
class ArangoProvider(GraphDBProvider):
    """ArangoDB graph database provider implementation.
    
    This provider interfaces with ArangoDB using the python-arango client,
    mapping entities and relationships to ArangoDB's graph model.
    """
    
    def __init__(self, name: str = "arango", settings: Optional[ArangoProviderSettings] = None):
        """Initialize ArangoDB graph database provider.
        
        Args:
            name: Provider name
            settings: Provider settings
        """
        # Create settings explicitly if not provided to avoid TypeVar issues
        settings = settings or ArangoProviderSettings()
        
        super().__init__(name=name, settings=settings)
        self._client: Optional[ArangoClient] = None
        self._db: Optional[StandardDatabase] = None
        self._entity_collection: Optional[Collection] = None
        self._relation_collection: Optional[Collection] = None
        self._initialized = False
        
    async def _initialize(self) -> None:
        """Initialize the ArangoDB connection.
        
        Creates the ArangoDB client instance and verifies the connection.
        Also ensures that required collections, indexes, and graph are created.
        
        Raises:
            ProviderError: If ArangoDB driver is not available or connection fails
        """
        if not ARANGO_AVAILABLE:
            raise ProviderError(
                message="ArangoDB driver is not installed. Install with 'pip install python-arango'",
                provider_name=self.name
            )
        
        settings = self.settings
        
        try:
            # Create the ArangoDB client
            self._client = ArangoClient(
                hosts=settings.url,
                verify=settings.verify,
                http_pool_size=settings.max_http_pool_size
            )
            
            # Connect to system database first to ensure target database exists
            sys_db = self._client.db(
                "_system",
                username=settings.username,
                password=settings.password
            )
            
            # Create database if it doesn't exist
            if not sys_db.has_database(settings.database):
                logger.info(f"Creating database '{settings.database}'")
                sys_db.create_database(settings.database)
                
            # Connect to the target database
            self._db = self._client.db(
                settings.database,
                username=settings.username,
                password=settings.password
            )
            
            # Create collections if they don't exist
            if not self._db.has_collection(settings.entity_collection):
                logger.info(f"Creating entity collection '{settings.entity_collection}'")
                self._db.create_collection(settings.entity_collection)
                
            if not self._db.has_collection(settings.relation_collection):
                logger.info(f"Creating relationship collection '{settings.relation_collection}'")
                self._db.create_collection(settings.relation_collection, edge=True)
                
            # Get collection references
            self._entity_collection = self._db.collection(settings.entity_collection)
            self._relation_collection = self._db.collection(settings.relation_collection)
            
            # Create graph if it doesn't exist
            if not self._db.has_graph(settings.graph_name):
                logger.info(f"Creating graph '{settings.graph_name}'")
                self._db.create_graph(
                    settings.graph_name,
                    edge_definitions=[
                        {
                            "edge_collection": settings.relation_collection,
                            "from_vertex_collections": [settings.entity_collection],
                            "to_vertex_collections": [settings.entity_collection]
                        }
                    ]
                )
                
            # Create indexes
            self._setup_indexes()
            
            self._initialized = True
            logger.info(f"ArangoDB provider '{self.name}' initialized successfully")
            
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to connect to ArangoDB: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    url=settings.url,
                    database=settings.database
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to initialize ArangoDB provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
    
    def _setup_indexes(self) -> None:
        """Set up indexes for entity and relationship collections."""
        # Create index on entity ID
        self._entity_collection.add_hash_index(["id"], unique=True)
        
        # Create index on entity type
        self._entity_collection.add_hash_index(["type"], unique=False)
        
        # Create index on relationship type
        self._relation_collection.add_hash_index(["relation_type"], unique=False)
    
    async def _shutdown(self) -> None:
        """Shut down the ArangoDB connection."""
        self._client = None
        self._db = None
        self._entity_collection = None
        self._relation_collection = None
        self._initialized = False
        logger.info(f"ArangoDB provider '{self.name}' shut down")
        
    async def add_entity(self, entity: Entity) -> str:
        """Add or update an entity node in ArangoDB.
        
        Args:
            entity: Entity to add or update
            
        Returns:
            ID of the created/updated entity
            
        Raises:
            ProviderError: If entity creation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Convert entity to ArangoDB document
            doc = self._entity_to_document(entity)
            
            # Check if entity already exists (by custom id field)
            existing = None
            try:
                existing = self._entity_collection.get({"id": entity.id})
            except:
                pass
                
            if existing:
                # Update existing entity
                self._entity_collection.update(
                    {"id": entity.id},
                    doc
                )
            else:
                # Create new entity
                self._entity_collection.insert(doc)
                
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
            
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
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
    
    def _entity_to_document(self, entity: Entity) -> Dict[str, Any]:
        """Convert Entity object to ArangoDB document format."""
        # Convert attributes to serializable format
        attributes = {}
        for attr_name, attr in entity.attributes.items():
            attributes[attr_name] = {
                "name": attr.name,
                "value": attr.value,
                "confidence": attr.confidence,
                "source": attr.source,
                "timestamp": attr.timestamp
            }
            
        # Core entity properties
        doc = {
            "id": entity.id,
            "type": entity.type,
            "source": entity.source,
            "importance": entity.importance,
            "last_updated": entity.last_updated,
            "attributes": attributes
        }
        
        return doc
        
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID from ArangoDB.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            ProviderError: If retrieval fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Get entity document
            doc = self._entity_collection.get({"id": entity_id})
            if not doc:
                return None
                
            # Get relationships
            relationships = []
            
            # AQL to find outgoing relationships
            query = f"""
            FOR r IN {self.settings.relation_collection}
            FILTER r._from == @from_key
            LET target = DOCUMENT(r._to)
            RETURN {{
                relation_type: r.relation_type,
                target_id: target.id,
                confidence: r.confidence,
                source: r.source,
                timestamp: r.timestamp
            }}
            """
            
            from_key = f"{self.settings.entity_collection}/{self._get_doc_key(entity_id)}"
            cursor = self._db.aql.execute(
                query,
                bind_vars={"from_key": from_key}
            )
            
            # Convert to EntityRelationship objects
            for rel in cursor:
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
                
            # Deserialize attributes
            attributes = {}
            for attr_name, attr_data in doc.get("attributes", {}).items():
                attributes[attr_name] = EntityAttribute(
                    name=attr_name,
                    value=attr_data.get("value", ""),
                    confidence=attr_data.get("confidence", 0.8),
                    source=attr_data.get("source", ""),
                    timestamp=attr_data.get("timestamp", datetime.now().isoformat())
                )
                
            # Create entity object
            entity = Entity(
                id=doc.get("id", entity_id),
                type=doc.get("type", "unknown"),
                attributes=attributes,
                relationships=relationships,
                source=doc.get("source", ""),
                importance=doc.get("importance", 0.5),
                last_updated=doc.get("last_updated", datetime.now().isoformat())
            )
            
            return entity
            
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
    
    def _get_doc_key(self, entity_id: str) -> str:
        """Convert entity ID to a valid document key by replacing invalid characters."""
        # Replace characters that are not allowed in ArangoDB keys
        return entity_id.replace("/", "_").replace(" ", "_")
        
    async def add_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Add a relationship between two entities in ArangoDB.
        
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
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
        properties = properties or {}
        
        try:
            # Ensure source entity exists
            source_doc = self._entity_collection.get({"id": source_id})
            if not source_doc:
                raise ProviderError(
                    message=f"Source entity {source_id} does not exist",
                    provider_name=self.name
                )
                
            # Check if target entity exists - create it if it doesn't
            target_doc = self._entity_collection.get({"id": target_entity})
            if not target_doc:
                # Create a placeholder entity
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
                
                # Get the newly created document
                target_doc = self._entity_collection.get({"id": target_entity})
                
            # Create edge document
            edge = {
                "_from": f"{self.settings.entity_collection}/{source_doc['_key']}",
                "_to": f"{self.settings.entity_collection}/{target_doc['_key']}",
                "relation_type": relation_type,
                "confidence": properties.get("confidence", 0.8),
                "source": properties.get("source", "system"),
                "timestamp": properties.get("timestamp", datetime.now().isoformat())
            }
            
            # Check if relationship exists
            existing_edge = self._db.aql.execute(
                f"""
                FOR r IN {self.settings.relation_collection}
                FILTER r._from == @from AND r._to == @to AND r.relation_type == @type
                RETURN r
                """,
                bind_vars={
                    "from": edge["_from"],
                    "to": edge["_to"],
                    "type": relation_type
                }
            ).next()
            
            if existing_edge:
                # Update existing edge
                self._relation_collection.update(
                    existing_edge["_key"],
                    edge
                )
            else:
                # Create new edge
                self._relation_collection.insert(edge)
                
        except ProviderError:
            # Re-raise provider errors
            raise
        except ArangoError as e:
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
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    source_id=source_id,
                    target_entity=target_entity
                ),
                cause=e
            )
    
    async def query_relationships(
        self, 
        entity_id: str, 
        relation_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """Query relationships for an entity in ArangoDB.
        
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
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Get document key from entity ID
            doc = self._entity_collection.get({"id": entity_id})
            if not doc:
                return []
                
            entity_key = doc["_key"]
            collection_name = self.settings.entity_collection
            
            # Construct AQL query based on direction
            if direction == "outgoing":
                query = f"""
                FOR v, e IN 1..1 OUTBOUND '{collection_name}/{entity_key}' {self.settings.relation_collection}
                FILTER @relation_type IS NULL OR e.relation_type == @relation_type
                RETURN {{
                    source: '{entity_id}',
                    target: v.id,
                    type: e.relation_type,
                    properties: {{
                        confidence: e.confidence,
                        source: e.source,
                        timestamp: e.timestamp
                    }}
                }}
                """
            else:  # incoming
                query = f"""
                FOR v, e IN 1..1 INBOUND '{collection_name}/{entity_key}' {self.settings.relation_collection}
                FILTER @relation_type IS NULL OR e.relation_type == @relation_type
                RETURN {{
                    source: v.id,
                    target: '{entity_id}',
                    type: e.relation_type,
                    properties: {{
                        confidence: e.confidence,
                        source: e.source,
                        timestamp: e.timestamp
                    }}
                }}
                """
                
            # Execute the query
            cursor = self._db.aql.execute(
                query,
                bind_vars={"relation_type": relation_type}
            )
            
            # Return relationships
            return [rel for rel in cursor]
            
        except ArangoError as e:
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
        except Exception as e:
            raise ProviderError(
                message=f"Failed to query relationships: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
    
    async def traverse(
        self, 
        start_id: str, 
        relation_types: Optional[List[str]] = None, 
        max_depth: int = 2
    ) -> List[Entity]:
        """Traverse the graph starting from an entity in ArangoDB.
        
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
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Get document key from entity ID
            doc = self._entity_collection.get({"id": start_id})
            if not doc:
                return []
                
            entity_key = doc["_key"]
            collection_name = self.settings.entity_collection
            
            # Build relation type filter
            filter_clause = ""
            if relation_types:
                relation_list = [f'"{rel_type}"' for rel_type in relation_types]
                filter_clause = f"FILTER e.relation_type IN [{', '.join(relation_list)}]"
                
            # Construct AQL traversal query
            query = f"""
            FOR v, e, p IN 1..{max_depth} OUTBOUND '{collection_name}/{entity_key}' {self.settings.relation_collection}
            {filter_clause}
            RETURN DISTINCT v.id
            """
            
            # Execute the query
            cursor = self._db.aql.execute(query)
            
            # Get the full entity for each ID
            entities = []
            ids = set()
            for result in cursor:
                if result not in ids:
                    ids.add(result)
                    entity = await self.get_entity(result)
                    if entity:
                        entities.append(entity)
                        
            # Add start entity if not already included
            start_entity = await self.get_entity(start_id)
            if start_entity and start_id not in ids:
                entities.insert(0, start_entity)
                
            return entities
            
        except ArangoError as e:
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
        except Exception as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    start_id=start_id
                ),
                cause=e
            )
    
    async def query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a simple query language for ArangoDB.
        
        Supports a limited set of commands:
            - "find_entities type={entity_type}" - Find entities by type
            - "find_entities name={name}" - Find entities by name
            - "neighbors id={id} [relation={relation_type}]" - Find neighboring entities
            - "path from={id} to={id} [max_depth={n}]" - Find path between entities
            - "aql {aql_query}" - Execute custom AQL query
        
        Args:
            query: Query string
            params: Optional query parameters
            
        Returns:
            Query results
            
        Raises:
            ProviderError: If query parsing fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
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
                
            # Execute custom AQL query
            elif query.startswith("aql") and "aql_query" in param_dict:
                aql_query = param_dict["aql_query"]
                aql_params = param_dict.get("params", {})
                return self._execute_aql(aql_query, aql_params)
                
            else:
                raise ProviderError(
                    message=f"Unsupported query: {query}",
                    provider_name=self.name
                )
                
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query,
                    params=params
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query
                ),
                cause=e
            )
    
    def _execute_aql(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute an AQL query with parameters."""
        cursor = self._db.aql.execute(query, bind_vars=params or {})
        return [doc for doc in cursor]
        
    async def _find_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Find entities by type in ArangoDB."""
        query = f"""
        FOR e IN {self.settings.entity_collection}
        FILTER e.type == @type
        RETURN {{ id: e.id, entity: e }}
        """
        
        cursor = self._db.aql.execute(query, bind_vars={"type": entity_type})
        return [doc for doc in cursor]
        
    async def _find_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find entities by name in ArangoDB."""
        query = f"""
        FOR e IN {self.settings.entity_collection}
        FILTER e.attributes.name.value LIKE @name
        RETURN {{ id: e.id, entity: e }}
        """
        
        cursor = self._db.aql.execute(query, bind_vars={"name": f"%{name}%"})
        return [doc for doc in cursor]
        
    async def _find_neighbors(self, entity_id: str, relation_type: Optional[str]) -> List[Dict[str, Any]]:
        """Find neighboring entities in ArangoDB."""
        # Get document key from entity ID
        doc = self._entity_collection.get({"id": entity_id})
        if not doc:
            return []
            
        entity_key = doc["_key"]
        collection_name = self.settings.entity_collection
        
        # Build relation type filter
        filter_clause = ""
        if relation_type:
            filter_clause = f"FILTER e.relation_type == '{relation_type}'"
            
        query = f"""
        FOR v, e IN 1..1 OUTBOUND '{collection_name}/{entity_key}' {self.settings.relation_collection}
        {filter_clause}
        RETURN {{
            id: v.id,
            relation: e.relation_type,
            entity: v
        }}
        """
        
        cursor = self._db.aql.execute(query)
        return [doc for doc in cursor]
        
    async def _find_path(self, from_id: str, to_id: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find path between entities in ArangoDB."""
        # Get document keys
        from_doc = self._entity_collection.get({"id": from_id})
        to_doc = self._entity_collection.get({"id": to_id})
        
        if not from_doc or not to_doc:
            return []
            
        from_key = from_doc["_key"]
        to_key = to_doc["_key"]
        collection_name = self.settings.entity_collection
        
        query = f"""
        LET path = (
            FOR v, e, p IN 1..{max_depth} OUTBOUND 
            '{collection_name}/{from_key}' {self.settings.relation_collection}
            FILTER v._key == '{to_key}'
            LIMIT 1
            RETURN p.vertices
        )
        
        RETURN LENGTH(path) > 0 ? (
            FOR vertex IN path[0]
            LET index = POSITION(path[0], vertex)
            RETURN {{
                position: index,
                id: vertex.id,
                entity: vertex
            }}
        ) : []
        """
        
        cursor = self._db.aql.execute(query)
        result = cursor.next()
        return result if result else []
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships from ArangoDB.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Get document key from entity ID
            doc = self._entity_collection.get({"id": entity_id})
            if not doc:
                return False
                
            entity_key = doc["_key"]
            
            # Delete entity and related edges via AQL
            query = f"""
            LET vertex = DOCUMENT('{self.settings.entity_collection}/{entity_key}')
            LET edges = (
                FOR v, e IN 1..1 ANY vertex {self.settings.relation_collection}
                RETURN e._key
            )
            
            FOR edge_key IN edges
                REMOVE edge_key IN {self.settings.relation_collection}
                
            REMOVE vertex IN {self.settings.entity_collection}
            RETURN true
            """
            
            self._db.aql.execute(query)
            return True
            
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    entity_id=entity_id
                ),
                cause=e
            )
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
        """Delete relationship(s) between entities in ArangoDB.
        
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
                message="ArangoDB provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Get document keys
            source_doc = self._entity_collection.get({"id": source_id})
            target_doc = self._entity_collection.get({"id": target_entity})
            
            if not source_doc or not target_doc:
                return False
                
            source_key = source_doc["_key"]
            target_key = target_doc["_key"]
            collection_name = self.settings.entity_collection
            
            # Build relation type filter
            type_filter = ""
            if relation_type:
                type_filter = f"FILTER e.relation_type == '{relation_type}'"
                
            # Delete edges via AQL
            query = f"""
            FOR e IN {self.settings.relation_collection}
            FILTER e._from == '{collection_name}/{source_key}' AND e._to == '{collection_name}/{target_key}'
            {type_filter}
            REMOVE e IN {self.settings.relation_collection}
            COLLECT WITH COUNT INTO deleted
            RETURN deleted
            """
            
            cursor = self._db.aql.execute(query)
            deleted = cursor.next()
            
            return deleted > 0
            
        except ArangoError as e:
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
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    source_id=source_id,
                    target_entity=target_entity
                ),
                cause=e
            )
    
    async def remove_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str
    ) -> None:
        """Remove a relationship between two entities in ArangoDB.
        
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
        """Add multiple entities in bulk to ArangoDB.
        
        Args:
            entities: List of entities to add
            
        Returns:
            List of IDs of added entities
            
        Raises:
            ProviderError: If bulk operation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
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