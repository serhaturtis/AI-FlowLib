"""SQLite database provider implementation.

This module provides a concrete implementation of the DBProvider
for SQLite database using aiosqlite.
"""

import logging
import asyncio
import os
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Tuple
import json
from datetime import datetime, date
from pathlib import Path

from pydantic import Field

from ...core.errors import ProviderError, ErrorContext
from ...core.models.settings import ProviderSettings
from ...core.registry.decorators import provider
from ...core.registry.constants import ProviderType
from .base import DBProvider, DBProviderSettings

logger = logging.getLogger(__name__)

try:
    import aiosqlite
    import sqlite3
except ImportError:
    logger.warning("aiosqlite package not found. Install with 'pip install aiosqlite'")


class SQLiteProviderSettings(DBProviderSettings):
    """Settings for SQLite provider.
    
    Attributes:
        database_path: Path to SQLite database file
        journal_mode: SQLite journal mode
        isolation_level: SQLite isolation level
        timeout: Connection timeout in seconds
        detect_types: SQLite type detection
        create_if_missing: Create database file if it doesn't exist
    """
    
    # SQLite specific settings
    database_path: str
    journal_mode: str = "WAL"  # WAL mode for better concurrency
    isolation_level: Optional[str] = None  # None = autocommit mode
    timeout: float = 5.0
    detect_types: int = 0  # Can use sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    create_if_missing: bool = True
    
    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict)


@provider(provider_type=ProviderType.DATABASE, name="sqlite")
class SQLiteProvider(DBProvider):
    """SQLite implementation of the DBProvider.
    
    This provider implements database operations using aiosqlite,
    an efficient asynchronous SQLite driver.
    """
    
    def __init__(self, name: str = "sqlite", settings: Optional[SQLiteProviderSettings] = None):
        """Initialize SQLite provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        super().__init__(name=name, settings=settings)
        self._settings = settings or SQLiteProviderSettings(database_path=":memory:")
        self._connection = None
        
    async def _initialize(self) -> None:
        """Initialize SQLite connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Check if database file exists
            is_memory_db = self._settings.database_path == ":memory:"
            if not is_memory_db and not os.path.exists(self._settings.database_path):
                if self._settings.create_if_missing:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(self._settings.database_path), exist_ok=True)
                    logger.info(f"Created directory for SQLite database: {os.path.dirname(self._settings.database_path)}")
                else:
                    raise ProviderError(
                        message=f"SQLite database file does not exist: {self._settings.database_path}",
                        provider_name=self.name,
                        context=ErrorContext.create(database_path=self._settings.database_path)
                    )
            
            # Connect to database
            self._connection = await aiosqlite.connect(
                database=self._settings.database_path,
                timeout=self._settings.timeout,
                isolation_level=self._settings.isolation_level,
                detect_types=self._settings.detect_types,
                **self._settings.connect_args
            )
            
            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys = ON")
            
            # Set journal mode
            await self._connection.execute(f"PRAGMA journal_mode = {self._settings.journal_mode}")
            
            # Configure JSON serialization/deserialization
            sqlite3.register_adapter(dict, json.dumps)
            sqlite3.register_adapter(list, json.dumps)
            sqlite3.register_converter("JSON", json.loads)
            
            logger.info(f"Connected to SQLite database: {self._settings.database_path}")
            
        except Exception as e:
            self._connection = None
            raise ProviderError(
                message=f"Failed to connect to SQLite database: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(database_path=self._settings.database_path),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info(f"Closed SQLite connection: {self._settings.database_path}")
    
    async def execute_query(self, query: str, params: Optional[Union[tuple, dict]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of rows as dictionaries
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Execute query
            cursor = await self._connection.execute(query, params or ())
            
            # Get column names
            columns = [column[0] for column in cursor.description] if cursor.description else []
            
            # Fetch all rows
            rows = await cursor.fetchall()
            
            # Convert rows to dictionaries
            results = []
            for row in rows:
                result = {}
                for i, column in enumerate(columns):
                    value = row[i]
                    # Handle SQLite-specific types
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            pass  # Keep as bytes if not valid UTF-8
                    result[column] = value
                results.append(result)
                
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute SQLite query: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query,
                    params=params
                ),
                cause=e
            )
    
    async def execute_update(self, query: str, params: Optional[Union[tuple, dict]] = None) -> int:
        """Execute a SQL update.
        
        Args:
            query: SQL query (INSERT, UPDATE, DELETE)
            params: Query parameters
            
        Returns:
            Number of rows affected
            
        Raises:
            ProviderError: If update execution fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Execute query
            cursor = await self._connection.execute(query, params or ())
            
            # Commit changes
            await self._connection.commit()
            
            # Return number of rows affected
            return cursor.rowcount
            
        except Exception as e:
            # Rollback transaction
            await self._connection.rollback()
            
            raise ProviderError(
                message=f"Failed to execute SQLite update: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    query=query,
                    params=params
                ),
                cause=e
            )
    
    async def execute_script(self, script: str) -> None:
        """Execute a SQL script.
        
        Args:
            script: SQL script
            
        Raises:
            ProviderError: If script execution fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Execute script
            await self._connection.executescript(script)
            
            # Commit changes
            await self._connection.commit()
            
        except Exception as e:
            # Rollback transaction
            await self._connection.rollback()
            
            raise ProviderError(
                message=f"Failed to execute SQLite script: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    script=script[:100] + "..." if len(script) > 100 else script
                ),
                cause=e
            )
    
    async def execute_transaction(self, queries: List[Tuple[str, Optional[Union[tuple, dict]]]]) -> List[Any]:
        """Execute queries in a transaction.
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            List of results
            
        Raises:
            ProviderError: If transaction fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Start transaction
            await self._connection.execute("BEGIN TRANSACTION")
            
            results = []
            for query, params in queries:
                # Execute query
                cursor = await self._connection.execute(query, params or ())
                
                # Check if query returns rows
                if cursor.description:
                    # Get column names
                    columns = [column[0] for column in cursor.description]
                    
                    # Fetch all rows
                    rows = await cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    query_results = []
                    for row in rows:
                        result = {}
                        for i, column in enumerate(columns):
                            result[column] = row[i]
                        query_results.append(result)
                    
                    results.append(query_results)
                else:
                    # For INSERT, UPDATE, DELETE queries
                    results.append(cursor.rowcount)
                
            # Commit transaction
            await self._connection.commit()
            
            return results
            
        except Exception as e:
            # Rollback transaction
            await self._connection.rollback()
            
            raise ProviderError(
                message=f"Failed to execute SQLite transaction: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    queries=[q[0] for q in queries]
                ),
                cause=e
            )
    
    async def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            List of column definitions
            
        Raises:
            ProviderError: If schema retrieval fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Get table schema
            query = f"PRAGMA table_info({table_name})"
            return await self.execute_query(query)
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get SQLite table schema: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    table_name=table_name
                ),
                cause=e
            )
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.
        
        Args:
            table_name: Table name
            
        Returns:
            True if table exists
            
        Raises:
            ProviderError: If check fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Check if table exists
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            result = await self.execute_query(query, (table_name,))
            
            return len(result) > 0
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to check if SQLite table exists: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    table_name=table_name
                ),
                cause=e
            ) 