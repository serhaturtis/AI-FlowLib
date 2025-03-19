"""Database provider base class and related functionality.

This module provides the base class for implementing database providers
that share common functionality for querying, updating, and managing
database operations.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union, Tuple, Sequence
import asyncio
from pydantic import BaseModel, Field

from ...core.errors import ProviderError, ErrorContext
from ...core.models.settings import ProviderSettings
from ..base import Provider

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class DBProviderSettings(ProviderSettings):
    """Base settings for database providers.
    
    Attributes:
        host: Database server host address
        port: Database server port
        database: Database name
        username: Authentication username
        password: Authentication password
        pool_size: Connection pool size
        timeout: Connection/query timeout in seconds
    """
    
    # Connection settings
    host: str = "localhost"
    port: Optional[int] = None
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Pool settings
    pool_size: int = 5
    min_size: int = 1
    max_overflow: int = 10
    
    # Timeout settings
    timeout: float = 30.0
    connect_timeout: float = 10.0
    
    # SSL settings
    use_ssl: bool = False
    ssl_ca_cert: Optional[str] = None
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    
    # Query settings
    query_timeout: float = 30.0
    max_query_size: int = 1000
    application_name: str = "flowlib"
    auto_reconnect: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0


class DBProvider(Provider, Generic[T]):
    """Base class for database providers.
    
    This class provides:
    1. Common database operations (query, execute, transaction)
    2. Type-safe operations with Pydantic models
    3. Connection pooling and lifecycle management
    4. Error handling and retries
    """
    
    def __init__(self, name: str = "db", settings: Optional[DBProviderSettings] = None):
        """Initialize database provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Pass provider_type="db" to the parent class
        super().__init__(name=name, settings=settings, provider_type="db")
        self._initialized = False
        self._pool = None
        self._settings = settings or DBProviderSettings()
        
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
        
    async def initialize(self):
        """Initialize the database provider.
        
        This method should be implemented by subclasses to establish
        connections to the database and set up connection pools.
        """
        self._initialized = True
        
    async def shutdown(self):
        """Close all connections and release resources.
        
        This method should be implemented by subclasses to properly
        close connections and clean up resources.
        """
        self._initialized = False
        self._pool = None
        
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query results
            
        Raises:
            ProviderError: If query execution fails
        """
        raise NotImplementedError("Subclasses must implement execute()")
        
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> List[Any]:
        """Execute a batch of database queries.
        
        Args:
            query: SQL query to execute
            params_list: List of query parameters
            
        Returns:
            List of query results
            
        Raises:
            ProviderError: If query execution fails
        """
        raise NotImplementedError("Subclasses must implement execute_many()")
        
    async def execute_structured(self, query: str, output_type: Type[T], 
                              params: Optional[Dict[str, Any]] = None) -> List[T]:
        """Execute a query and parse results into structured types.
        
        Args:
            query: SQL query to execute
            output_type: Pydantic model to parse results into
            params: Query parameters
            
        Returns:
            List of parsed model instances
            
        Raises:
            ProviderError: If query execution or parsing fails
        """
        try:
            # Execute the query
            results = await self.execute(query, params)
            
            # If results is empty, return empty list
            if not results:
                return []
                
            # Parse results into output type
            parsed_results = []
            for row in results:
                # If result is already a dict, use as is
                if isinstance(row, dict):
                    data = row
                # If result is a tuple/list, convert to dict
                elif isinstance(row, (tuple, list)):
                    # Get column names from the cursor description
                    # This is implementation-specific and should be handled by subclasses
                    raise NotImplementedError("Tuple/list conversion must be implemented by subclasses")
                else:
                    # Try to convert to dict if it has a method for it
                    if hasattr(row, "_asdict"):
                        data = row._asdict()
                    elif hasattr(row, "__dict__"):
                        data = row.__dict__
                    else:
                        raise TypeError(f"Cannot convert result of type {type(row)} to dict")
                
                # Parse into output type
                parsed_results.append(output_type.parse_obj(data))
                
            return parsed_results
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to execute structured query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    output_type=output_type.__name__,
                    query=query[:100] + "..." if len(query) > 100 else query
                ),
                cause=e
            )
            
    async def begin_transaction(self):
        """Begin a database transaction.
        
        Returns:
            Transaction object
            
        Raises:
            ProviderError: If transaction start fails
        """
        raise NotImplementedError("Subclasses must implement begin_transaction()")
        
    async def commit_transaction(self, transaction: Any) -> bool:
        """Commit a database transaction.
        
        Args:
            transaction: Transaction object from begin_transaction()
            
        Returns:
            True if transaction was committed successfully
            
        Raises:
            ProviderError: If transaction commit fails
        """
        raise NotImplementedError("Subclasses must implement commit_transaction()")
        
    async def rollback_transaction(self, transaction: Any) -> bool:
        """Rollback a database transaction.
        
        Args:
            transaction: Transaction object from begin_transaction()
            
        Returns:
            True if transaction was rolled back successfully
            
        Raises:
            ProviderError: If transaction rollback fails
        """
        raise NotImplementedError("Subclasses must implement rollback_transaction()")
        
    async def check_connection(self) -> bool:
        """Check if database connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_connection()")
        
    async def get_health(self) -> Dict[str, Any]:
        """Get database health information.
        
        Returns:
            Dict containing health metrics
            
        Raises:
            ProviderError: If health check fails
        """
        raise NotImplementedError("Subclasses must implement get_health()") 