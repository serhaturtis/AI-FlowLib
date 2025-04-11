"""
Memory models for the agent system.

This module defines the data models used for the memory system, including
memory contexts and items.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MemoryContext(BaseModel):
    """Memory context information.
    
    Contexts provide a hierarchical namespace for organizing memory items.
    """
    
    name: str = Field(description="Name of the context")
    path: str = Field(description="Full path of the context")
    parent: Optional[str] = Field(None, description="Parent context path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    def __str__(self) -> str:
        """Return string representation of the context."""
        return f"MemoryContext(path={self.path})"


class MemoryItem(BaseModel):
    """Base memory item model.
    
    Represents a single item stored in memory with associated metadata.
    """
    
    key: str = Field(description="Unique key for the memory item")
    value: Any = Field(description="Value stored in memory")
    context: str = Field(description="Context path for the memory item")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    importance: float = Field(0.5, description="Importance score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional item metadata")
    
    @property
    def is_expired(self) -> bool:
        """Check if the memory item is expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def __str__(self) -> str:
        """Return string representation of the memory item."""
        return f"MemoryItem(key={self.key}, context={self.context})"


class MemorySearchResult(BaseModel):
    """Search result from memory.
    
    Contains the memory item and additional search metadata.
    """
    
    item: MemoryItem = Field(description="Memory item")
    score: float = Field(description="Relevance score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Search result metadata")

    def __str__(self) -> str:
        """Return string representation of the search result."""
        return f"MemorySearchResult(key={self.item.key}, score={self.score:.2f})"

