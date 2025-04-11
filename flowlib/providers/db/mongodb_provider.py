"""MongoDB database provider implementation.

This module provides a concrete implementation of the DBProvider
for MongoDB database using motor (async driver).
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
import json
from datetime import datetime, date

logger = logging.getLogger(__name__)

# For type annotations only
ObjectId = Any

try:
    from bson import ObjectId
except ImportError:
    logger.warning("bson module not found. Install with 'pip install pymongo'")

from pydantic import Field

from ...core.errors import ProviderError, ErrorContext
from .base import DBProvider, DBProviderSettings
from ..decorators import provider
from ..constants import ProviderType

try:
    import motor.motor_asyncio
    from pymongo import ASCENDING, DESCENDING
    from pymongo.errors import PyMongoError
except ImportError:
    logger.warning("motor package not found. Install with 'pip install motor'")


class MongoDBProviderSettings(DBProviderSettings):
    """Settings for MongoDB provider.
    
    Attributes:
        database: MongoDB database name
        connection_string: MongoDB connection string (overrides host/port if provided)
        auth_source: Authentication database name
        auth_mechanism: Authentication mechanism
        connect_timeout_ms: Connection timeout in milliseconds
        server_selection_timeout_ms: Server selection timeout in milliseconds
    """
    
    # MongoDB specific settings
    database: str
    connection_string: Optional[str] = None
    auth_source: Optional[str] = "admin"
    auth_mechanism: Optional[str] = None
    connect_timeout_ms: int = 20000
    server_selection_timeout_ms: int = 20000
    
    # Default port for MongoDB if not specified
    port: int = 27017
    
    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict)


@provider(provider_type=ProviderType.DATABASE, name="mongodb")
class MongoDBProvider(DBProvider):
    """MongoDB implementation of the DBProvider.
    
    This provider implements database operations using motor,
    an asynchronous driver for MongoDB.
    """
    
    def __init__(self, name: str = "mongodb", settings: Optional[MongoDBProviderSettings] = None):
        """Initialize MongoDB provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        super().__init__(name=name, settings=settings)
        self._settings = settings or MongoDBProviderSettings(database="test")
        self._client = None
        self._db = None
        
    async def _initialize(self) -> None:
        """Initialize MongoDB connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Create client
            if self._settings.connection_string:
                # Use connection string if provided
                self._client = motor.motor_asyncio.AsyncIOMotorClient(
                    self._settings.connection_string,
                    serverSelectionTimeoutMS=self._settings.server_selection_timeout_ms,
                    connectTimeoutMS=self._settings.connect_timeout_ms,
                    **self._settings.connect_args
                )
            else:
                # Use host and port
                self._client = motor.motor_asyncio.AsyncIOMotorClient(
                    host=self._settings.host,
                    port=self._settings.port,
                    username=self._settings.username,
                    password=self._settings.password,
                    authSource=self._settings.auth_source,
                    authMechanism=self._settings.auth_mechanism,
                    serverSelectionTimeoutMS=self._settings.server_selection_timeout_ms,
                    connectTimeoutMS=self._settings.connect_timeout_ms,
                    **self._settings.connect_args
                )
            
            # Get database
            self._db = self._client[self._settings.database]
            
            # Ping database to verify connection
            await self._client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self._settings.host}:{self._settings.port}/{self._settings.database}")
            
        except Exception as e:
            self._client = None
            self._db = None
            raise ProviderError(
                message=f"Failed to connect to MongoDB: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    host=self._settings.host,
                    port=self._settings.port,
                    database=self._settings.database
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info(f"Closed MongoDB connection: {self._settings.host}:{self._settings.port}")
    
    async def execute_query(self, 
                           collection: str,
                           query: Dict[str, Any],
                           projection: Optional[Dict[str, Any]] = None,
                           sort: Optional[List[tuple]] = None,
                           limit: Optional[int] = None,
                           skip: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute a MongoDB query.
        
        Args:
            collection: Collection name
            query: MongoDB query dict
            projection: Optional fields to return
            sort: Optional sort specification
            limit: Optional limit
            skip: Optional skip
            
        Returns:
            List of documents as dictionaries
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._db:
            await self.initialize()
            
        try:
            # Get collection
            coll = self._db[collection]
            
            # Build cursor
            cursor = coll.find(query, projection)
            
            # Apply sort, limit, skip if provided
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
                
            # Get results
            results = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            for doc in results:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
                    
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute MongoDB query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection=collection,
                    query=query
                ),
                cause=e
            )
    
    async def insert_document(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a document into a collection.
        
        Args:
            collection: Collection name
            document: Document to insert
            
        Returns:
            ID of inserted document
            
        Raises:
            ProviderError: If insert fails
        """
        if not self._db:
            await self.initialize()
            
        try:
            # Get collection
            coll = self._db[collection]
            
            # Insert document
            result = await coll.insert_one(document)
            
            # Return inserted ID as string
            return str(result.inserted_id)
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to insert document: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection=collection
                ),
                cause=e
            )
    
    async def update_document(self, 
                             collection: str, 
                             query: Dict[str, Any], 
                             update: Dict[str, Any], 
                             upsert: bool = False) -> int:
        """Update documents in a collection.
        
        Args:
            collection: Collection name
            query: Query to match documents
            update: Update operations
            upsert: Whether to insert if no documents match
            
        Returns:
            Number of documents modified
            
        Raises:
            ProviderError: If update fails
        """
        if not self._db:
            await self.initialize()
            
        try:
            # Get collection
            coll = self._db[collection]
            
            # Update documents
            result = await coll.update_many(query, update, upsert=upsert)
            
            # Return modified count
            return result.modified_count
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to update documents: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection=collection,
                    query=query
                ),
                cause=e
            )
    
    async def delete_document(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from a collection.
        
        Args:
            collection: Collection name
            query: Query to match documents
            
        Returns:
            Number of documents deleted
            
        Raises:
            ProviderError: If delete fails
        """
        if not self._db:
            await self.initialize()
            
        try:
            # Get collection
            coll = self._db[collection]
            
            # Delete documents
            result = await coll.delete_many(query)
            
            # Return deleted count
            return result.deleted_count
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete documents: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection=collection,
                    query=query
                ),
                cause=e
            )
    
    async def create_index(self, 
                          collection: str, 
                          keys: List[tuple], 
                          unique: bool = False, 
                          sparse: bool = False) -> str:
        """Create an index on a collection.
        
        Args:
            collection: Collection name
            keys: List of (field, direction) tuples
            unique: Whether index should enforce uniqueness
            sparse: Whether index should be sparse
            
        Returns:
            Name of created index
            
        Raises:
            ProviderError: If index creation fails
        """
        if not self._db:
            await self.initialize()
            
        try:
            # Get collection
            coll = self._db[collection]
            
            # Create index
            result = await coll.create_index(keys, unique=unique, sparse=sparse)
            
            # Return index name
            return result
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to create index: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection=collection,
                    keys=keys
                ),
                cause=e
            )
    
    async def execute_transaction(self, operations: Callable):
        """Execute operations in a transaction.
        
        Args:
            operations: Async callable that takes the database as argument
            
        Returns:
            Transaction result
            
        Raises:
            ProviderError: If transaction fails
        """
        if not self._client:
            await self.initialize()
            
        try:
            # Start a session
            async with await self._client.start_session() as session:
                # Start a transaction
                result = await session.with_transaction(operations)
                return result
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute transaction: {str(e)}",
                provider_name=self.name,
                cause=e
            )
    
    async def count_documents(self, collection: str, query: Dict[str, Any]) -> int:
        """Count documents in a collection.
        
        Args:
            collection: Collection name
            query: Query to match documents
            
        Returns:
            Number of documents matched
            
        Raises:
            ProviderError: If count fails
        """
        if not self._db:
            await self.initialize()
            
        try:
            # Get collection
            coll = self._db[collection]
            
            # Count documents
            return await coll.count_documents(query)
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to count documents: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    collection=collection,
                    query=query
                ),
                cause=e
            ) 