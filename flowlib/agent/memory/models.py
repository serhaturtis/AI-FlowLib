"""
Memory models for the agent system.

This module provides Pydantic models for memory operations to ensure
consistent, type-safe interactions with memory systems.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

# Import Entity from the graph models - this is referenced by other modules
from flowlib.providers.graph.models import Entity


class MemoryItem(BaseModel):
    """Base memory item model representing stored information."""
    key: str = Field(..., description="Unique identifier for this memory item")
    value: Any = Field(..., description="The stored value/content")
    context: str = Field("default", description="Context/namespace for this memory")
    created_at: datetime = Field(default_factory=datetime.now, description="When this memory was created")
    updated_at: Optional[datetime] = Field(None, description="When this memory was last updated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this memory")
    
    def update_value(self, new_value: Any) -> None:
        """Update the value and updated_at timestamp."""
        self.value = new_value
        self.updated_at = datetime.now()


class MemoryStoreRequest(BaseModel):
    """Request model for storing items in memory."""
    key: str = Field(..., description="Key to store the memory under")
    value: Any = Field(..., description="Value to store")
    context: Optional[str] = Field(None, description="Context to store the memory in (namespace)")
    ttl: Optional[int] = Field(None, description="Time-to-live in seconds (None = no expiration)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to store")
    importance: float = Field(0.5, description="Importance of the memory (0.0-1.0)")
    
    @model_validator(mode='after')
    def validate_importance(self) -> 'MemoryStoreRequest':
        """Validate importance is between 0 and 1."""
        if self.importance < 0.0 or self.importance > 1.0:
            self.importance = max(0.0, min(1.0, self.importance))
        return self


class MemoryRetrieveRequest(BaseModel):
    """Request model for retrieving items from memory."""
    key: str = Field(..., description="Key to retrieve")
    context: Optional[str] = Field(None, description="Context to retrieve from")
    default: Optional[Any] = Field(None, description="Default value if key not found")
    metadata_only: bool = Field(False, description="Whether to return only metadata without the value")


class MemorySearchRequest(BaseModel):
    """Request model for searching memory."""
    query: str = Field(..., description="Search query (text, embedding, or hybrid)")
    context: Optional[str] = Field(None, description="Context to search in")
    limit: int = Field(10, description="Maximum number of results to return")
    threshold: Optional[float] = Field(None, description="Minimum similarity threshold (0.0-1.0)")
    sort_by: Optional[str] = Field(None, description="Field to sort results by (relevance, created_at, updated_at)")
    search_type: str = Field("hybrid", description="Type of search: 'semantic', 'keyword', or 'hybrid'")


class MemorySearchResult(BaseModel):
    """Result model for memory search operations."""
    items: List[MemoryItem] = Field(default_factory=list, description="Matching memory items")
    count: int = Field(0, description="Total number of matching items")
    query: str = Field("", description="Original search query")
    context: Optional[str] = Field(None, description="Context that was searched")


class MemoryContext(BaseModel):
    """Model representing a memory context (namespace)."""
    name: str = Field(..., description="Name of the context")
    path: str = Field(..., description="Full path of the context")
    parent: Optional[str] = Field(None, description="Parent context path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="When this context was created")
