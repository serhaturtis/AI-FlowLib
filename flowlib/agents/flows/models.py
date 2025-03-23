"""Models for flow operations.

This module provides common model classes used by flows in the agent system.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from flowlib.agents.memory.models import Entity


class ExtractedEntities(BaseModel):
    """Output model for entity extraction flow.
    
    Attributes:
        entities: List of extracted entities
        summary: Text summary of what was extracted
    """
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of extracted entities"
    )
    summary: str = Field(
        default="",
        description="Text summary of what was extracted"
    )


class RetrievedMemories(BaseModel):
    """Output model for memory retrieval flow.
    
    Attributes:
        entities: List of retrieved entities
        context: Formatted context for prompt injection
        relevance_scores: Relevance scores for retrieved entities
    """
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of retrieved entities"
    )
    context: str = Field(
        default="",
        description="Formatted context for prompt injection"
    )
    relevance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores for retrieved entities"
    ) 