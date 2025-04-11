"""PostgreSQL database provider implementation.

This module provides a concrete implementation of the DBProvider
for PostgreSQL database using asyncpg.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple, Sequence, Union
import json
from datetime import datetime, date

from pydantic import Field

from ...core.errors import ProviderError, ErrorContext
from .base import DBProvider, DBProviderSettings

logger = logging.getLogger(__name__)

# Define Connection type for type annotations
# This avoids the 'name Connection is not defined' error when asyncpg is not installed
Connection = Any
Pool = Any
Record = Any

try:
    import asyncpg
    from asyncpg import Connection, Pool, Record
except ImportError:
    logger.warning("asyncpg package not found. Install with 'pip install asyncpg'")


class PostgreSQLProviderSettings(DBProviderSettings):
    """Settings for PostgreSQL provider.
    
    Attributes:
        schema: PostgreSQL schema (default: public)
        statement_timeout: Statement timeout in milliseconds
        ssl_mode: SSL mode (disable, allow, prefer, require, verify-ca, verify-full)
        connect_args: Additional connection arguments
    """
    
    # PostgreSQL specific settings
    schema: str = "public"
    statement_timeout: Optional[int] = None
    ssl_mode: Optional[str] = None
    server_version: Optional[str] = None
    
    # Default port for PostgreSQL if not specified
    port: int = 5432
    
    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict)


class PostgreSQLProvider(DBProvider):
    """PostgreSQL implementation of the DBProvider.
    
    This provider implements database operations using asyncpg,
    an efficient asynchronous PostgreSQL driver.
    """
    
    def __init__(self, name: str = "postgres", settings: Optional[PostgreSQLProviderSettings] = None):
        """Initialize PostgreSQL provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Create settings first to avoid issues with _default_settings() method
        settings = settings or PostgreSQLProviderSettings()
        
        # Pass explicit settings to parent class
        super().__init__(name=name, settings=settings)
        
        # Store settings for local use
        self._settings = settings
        self._pool = None
        self._json_encoders = {
            # Custom JSON encoders for PostgreSQL
            datetime: lambda dt: dt.isoformat(),
            date: lambda d: d.isoformat()
        }
        
    async def initialize(self):
        """Initialize the PostgreSQL connection pool."""
        if self._initialized:
            return
            
        try:
            # Check if asyncpg is installed
            if "asyncpg" not in globals():
                raise ProviderError(
                    message="asyncpg package not installed. Install with 'pip install asyncpg'",
                    provider_name=self.name
                )
                
            # Prepare DSN (Data Source Name) for connection
            dsn = self._create_connection_string()
            
            # Prepare SSL context if needed
            ssl = None
            if self._settings.use_ssl:
                import ssl as ssl_module
                ssl_context = ssl_module.create_default_context()
                
                # Configure SSL certificates if provided
                if self._settings.ssl_ca_cert:
                    ssl_context.load_verify_locations(self._settings.ssl_ca_cert)
                if self._settings.ssl_cert and self._settings.ssl_key:
                    ssl_context.load_cert_chain(
                        self._settings.ssl_cert, 
                        self._settings.ssl_key
                    )
                    
                ssl = ssl_context
                
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=self._settings.min_size,
                max_size=self._settings.pool_size,
                max_queries=None,  # No query limit per connection
                max_inactive_connection_lifetime=300.0,  # 5 minutes
                timeout=self._settings.connect_timeout,
                command_timeout=self._settings.query_timeout,
                statement_cache_size=100,
                max_cached_statement_lifetime=300.0,  # 5 minutes
                ssl=ssl,
                server_settings={
                    'application_name': self._settings.application_name,
                    'search_path': self._settings.schema,
                    **(
                        {'statement_timeout': str(self._settings.statement_timeout)} 
                        if self._settings.statement_timeout else {}
                    )
                },
                **self._settings.connect_args
            )
            
            # Register custom type encoders and decoders
            self._register_type_codecs()
            
            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise ProviderError(
                message=f"Failed to initialize PostgreSQL provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def shutdown(self):
        """Close PostgreSQL connection pool and release resources."""
        if not self._initialized or not self._pool:
            return
            
        try:
            # Close the connection pool
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.debug(f"{self.name} provider shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during {self.name} provider shutdown: {str(e)}")
            raise ProviderError(
                message=f"Failed to shut down PostgreSQL provider: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a database query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._initialized or not self._pool:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        # Convert named parameters to positional if present
        positional_query, positional_params = self._convert_params(query, params)
        
        try:
            # Execute the query in the pool
            async with self._pool.acquire() as conn:
                # Check if it's a SELECT query or other type (INSERT, UPDATE, etc.)
                is_select = positional_query.strip().lower().startswith("select")
                
                if is_select:
                    # Execute SELECT and fetch results
                    records = await conn.fetch(positional_query, *positional_params)
                    # Convert to dict for easier consumption
                    return [dict(record) for record in records]
                else:
                    # Execute non-SELECT query
                    result = await conn.execute(positional_query, *positional_params)
                    
                    # Parse result string (e.g., "INSERT 0 1")
                    command, *rest = result.split()
                    if command in ("INSERT", "UPDATE", "DELETE"):
                        # Return affected rows count
                        count = int(rest[1]) if len(rest) > 1 else 0
                        return [{"affected_rows": count}]
                    else:
                        # Return raw result for other commands
                        return [{"result": result}]
                    
        except Exception as e:
            # Retry on connection errors if enabled
            if self._should_retry(e) and self._settings.auto_reconnect:
                return await self._retry_execute(query, params)
                
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query[:100] + "..." if len(query) > 100 else query
                ),
                cause=e
            )
            
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
        if not self._initialized or not self._pool:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        if not params_list:
            return []
            
        try:
            # Prepare positional parameters for each set of params
            all_positional_params = []
            first_param_set = params_list[0]
            
            # Extract parameter names from the first param set
            param_names = list(first_param_set.keys())
            
            # Convert query to use positional parameters
            positional_query = query
            for i, name in enumerate(param_names):
                positional_query = positional_query.replace(f":{name}", f"${i+1}")
                positional_query = positional_query.replace(f"@{name}", f"${i+1}")
                
            # Prepare positional parameters for each set
            for params in params_list:
                positional_params = [params.get(name) for name in param_names]
                all_positional_params.append(positional_params)
                
            # Execute batch with connection from pool
            async with self._pool.acquire() as conn:
                # Start a transaction
                async with conn.transaction():
                    # Execute each query in the batch
                    results = []
                    for params in all_positional_params:
                        # Execute the query
                        result = await conn.execute(positional_query, *params)
                        results.append(result)
                        
                    return results
                    
        except Exception as e:
            # Retry on connection errors if enabled
            if self._should_retry(e) and self._settings.auto_reconnect:
                return await self._retry_execute_many(query, params_list)
                
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to execute batch query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query[:100] + "..." if len(query) > 100 else query,
                    batch_size=len(params_list)
                ),
                cause=e
            )
            
    async def begin_transaction(self) -> Connection:
        """Begin a database transaction.
        
        Returns:
            Connection object with transaction
            
        Raises:
            ProviderError: If transaction start fails
        """
        if not self._initialized or not self._pool:
            raise ProviderError(
                message="Provider not initialized",
                provider_name=self.name
            )
            
        try:
            # Acquire connection from pool
            conn = await self._pool.acquire()
            
            # Start transaction
            transaction = conn.transaction()
            await transaction.start()
            
            # Save transaction info in connection
            conn._transaction = transaction
            
            return conn
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to begin transaction: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def commit_transaction(self, transaction: Connection) -> bool:
        """Commit a database transaction.
        
        Args:
            transaction: Connection object from begin_transaction()
            
        Returns:
            True if transaction was committed successfully
            
        Raises:
            ProviderError: If transaction commit fails
        """
        if not transaction or not hasattr(transaction, '_transaction'):
            raise ProviderError(
                message="Invalid transaction object",
                provider_name=self.name
            )
            
        try:
            # Commit the transaction
            await transaction._transaction.commit()
            
            # Release the connection back to the pool
            await self._pool.release(transaction)
            
            return True
            
        except Exception as e:
            # Try to release the connection
            try:
                await self._pool.release(transaction)
            except Exception:
                pass
                
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to commit transaction: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def rollback_transaction(self, transaction: Connection) -> bool:
        """Rollback a database transaction.
        
        Args:
            transaction: Connection object from begin_transaction()
            
        Returns:
            True if transaction was rolled back successfully
            
        Raises:
            ProviderError: If transaction rollback fails
        """
        if not transaction or not hasattr(transaction, '_transaction'):
            raise ProviderError(
                message="Invalid transaction object",
                provider_name=self.name
            )
            
        try:
            # Rollback the transaction
            await transaction._transaction.rollback()
            
            # Release the connection back to the pool
            await self._pool.release(transaction)
            
            return True
            
        except Exception as e:
            # Try to release the connection
            try:
                await self._pool.release(transaction)
            except Exception:
                pass
                
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to rollback transaction: {str(e)}",
                provider_name=self.name,
                cause=e
            )
            
    async def check_connection(self) -> bool:
        """Check if database connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        if not self._initialized or not self._pool:
            return False
            
        try:
            # Try to acquire a connection and execute a simple query
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception:
            return False
            
    async def get_health(self) -> Dict[str, Any]:
        """Get database health information.
        
        Returns:
            Dict containing health metrics
            
        Raises:
            ProviderError: If health check fails
        """
        if not self._initialized or not self._pool:
            return {
                "status": "not_initialized",
                "pool": None,
                "connection_active": False
            }
            
        try:
            # Check connection
            connection_active = await self.check_connection()
            
            # Get pool statistics
            pool_stats = {
                "min_size": self._settings.min_size,
                "max_size": self._settings.pool_size,
                "current_size": len(self._pool._holders),
                "free_connections": len(self._pool._queue._queue),
                "used_connections": len(self._pool._holders) - len(self._pool._queue._queue),
            }
            
            # Get PostgreSQL version
            version = None
            if connection_active:
                async with self._pool.acquire() as conn:
                    version = await conn.fetchval("SHOW server_version")
                    
            return {
                "status": "healthy" if connection_active else "unhealthy",
                "pool": pool_stats,
                "connection_active": connection_active,
                "version": version,
                "host": self._settings.host,
                "database": self._settings.database
            }
            
        except Exception as e:
            # Return error status
            return {
                "status": "error",
                "error": str(e),
                "connection_active": False
            }
            
    def _create_connection_string(self) -> str:
        """Create PostgreSQL connection string from settings.
        
        Returns:
            Connection string (DSN)
        """
        parts = []
        
        # Add host
        if self._settings.host:
            parts.append(f"host={self._settings.host}")
            
        # Add port if specified
        if self._settings.port:
            parts.append(f"port={self._settings.port}")
            
        # Add database
        if self._settings.database:
            parts.append(f"dbname={self._settings.database}")
            
        # Add credentials if specified
        if self._settings.username:
            parts.append(f"user={self._settings.username}")
        if self._settings.password:
            parts.append(f"password={self._settings.password}")
            
        # Add SSL mode if specified
        if self._settings.ssl_mode:
            parts.append(f"sslmode={self._settings.ssl_mode}")
            
        return " ".join(parts)
        
    def _convert_params(self, query: str, params: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """Convert named parameters to positional parameters.
        
        Args:
            query: SQL query with named parameters
            params: Query parameters
            
        Returns:
            Tuple of (modified query, positional parameters)
        """
        if not params:
            return query, []
            
        # Map of parameter names to their positions
        param_map = {}
        positional_params = []
        
        # Find all parameter references in the query
        modified_query = query
        for i, (name, value) in enumerate(params.items()):
            position = i + 1
            param_map[name] = position
            positional_params.append(value)
            
            # Replace named parameters with positional ones
            # Handle both :param and @param styles
            modified_query = modified_query.replace(f":{name}", f"${position}")
            modified_query = modified_query.replace(f"@{name}", f"${position}")
            
        return modified_query, positional_params
        
    def _register_type_codecs(self):
        """Register custom type encoders and decoders for PostgreSQL."""
        # This would be implemented to handle JSON, arrays, etc.
        # For example, registering JSON encoding/decoding
        pass
        
    def _should_retry(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry.
        
        Args:
            exception: Exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        # Check if it's a connection error
        if isinstance(exception, (asyncpg.ConnectionDoesNotExistError, 
                                  asyncpg.ConnectionFailureError,
                                  asyncpg.InterfaceError)):
            return True
            
        # Check for network-related errors
        if "connection" in str(exception).lower() and "closed" in str(exception).lower():
            return True
            
        return False
        
    async def _retry_execute(self, query: str, params: Optional[Dict[str, Any]] = None, 
                          attempt: int = 1) -> List[Dict[str, Any]]:
        """Retry executing a query with exponential backoff.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            attempt: Current attempt number
            
        Returns:
            Query results
            
        Raises:
            ProviderError: If all retries fail
        """
        if attempt > self._settings.retry_count:
            raise ProviderError(
                message=f"Failed to execute query after {attempt-1} retries",
                provider_name=self.name
            )
            
        # Calculate backoff delay
        delay = self._settings.retry_delay * (2 ** (attempt - 1))
        
        # Wait before retrying
        await asyncio.sleep(delay)
        
        try:
            # Try to execute the query again
            return await self.execute(query, params)
        except Exception as e:
            if self._should_retry(e):
                # Retry again
                return await self._retry_execute(query, params, attempt + 1)
            else:
                # Re-raise if not retriable
                raise
                
    async def _retry_execute_many(self, query: str, params_list: List[Dict[str, Any]], 
                               attempt: int = 1) -> List[Any]:
        """Retry executing a batch query with exponential backoff.
        
        Args:
            query: SQL query to execute
            params_list: List of query parameters
            attempt: Current attempt number
            
        Returns:
            List of query results
            
        Raises:
            ProviderError: If all retries fail
        """
        if attempt > self._settings.retry_count:
            raise ProviderError(
                message=f"Failed to execute batch query after {attempt-1} retries",
                provider_name=self.name
            )
            
        # Calculate backoff delay
        delay = self._settings.retry_delay * (2 ** (attempt - 1))
        
        # Wait before retrying
        await asyncio.sleep(delay)
        
        try:
            # Try to execute the batch query again
            return await self.execute_many(query, params_list)
        except Exception as e:
            if self._should_retry(e):
                # Retry again
                return await self._retry_execute_many(query, params_list, attempt + 1)
            else:
                # Re-raise if not retriable
                raise 