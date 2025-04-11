from typing import List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class RecallStrategy(str, Enum):
    """Strategies for memory recall"""
    CONTEXTUAL = "contextual"  # Recall based on current context
    ENTITY = "entity"         # Recall specific entity information
    TEMPORAL = "temporal"     # Recall based on time/sequence
    SEMANTIC = "semantic"     # Recall based on semantic similarity

class RecallRequest(BaseModel):
    """Request for memory recall"""
    query: str = Field(..., description="The query to search for in memory")
    strategy: RecallStrategy = Field(..., description="The recall strategy to use")
    context: Optional[str] = Field(None, description="Additional context for recall")
    entity_id: Optional[str] = Field(None, description="Entity ID for entity-based recall")
    limit: int = Field(10, description="Maximum number of memories to return")
    memory_types: List[str] = Field(default_factory=list, description="Types of memory to search")

class MemoryMatch(BaseModel):
    """A single memory match from recall"""
    memory_id: str = Field(..., description="Unique identifier of the memory")
    content: str = Field(..., description="Content of the memory")
    memory_type: str = Field(..., description="Type of memory")
    relevance_score: float = Field(..., description="Relevance score of the match")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

class RecallResponse(BaseModel):
    """Response from memory recall"""
    matches: List[MemoryMatch] = Field(..., description="List of memory matches")
    strategy_used: RecallStrategy = Field(..., description="Strategy that was used")
    total_matches: int = Field(..., description="Total number of matches found")
    query_analysis: Optional[dict] = Field(None, description="Analysis of the query and strategy selection") 