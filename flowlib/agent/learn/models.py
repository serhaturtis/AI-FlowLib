from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class LearningStrategy(str, Enum):
    """Strategies for knowledge acquisition"""
    ENTITY_EXTRACTION = "entity_extraction"  # Extract entities and properties
    RELATIONSHIP_LEARNING = "relationship_learning"  # Learn relationships between entities
    KNOWLEDGE_INTEGRATION = "knowledge_integration"  # Integrate new knowledge with existing
    CONCEPT_FORMATION = "concept_formation"  # Form new concepts from observations

class Entity(BaseModel):
    """Represents a learned entity"""
    id: str = Field(..., description="Unique identifier for the entity")
    type: str = Field(..., description="Type/category of the entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    confidence: float = Field(..., description="Confidence score for the entity")
    source: str = Field(..., description="Source of the entity information")

class Relationship(BaseModel):
    """Represents a learned relationship between entities"""
    id: str = Field(..., description="Unique identifier for the relationship")
    type: str = Field(..., description="Type of relationship")
    source_entity_id: str = Field(..., description="ID of the source entity")
    target_entity_id: str = Field(..., description="ID of the target entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    confidence: float = Field(..., description="Confidence score for the relationship")
    source: str = Field(..., description="Source of the relationship information")

class LearningRequest(BaseModel):
    """Request for knowledge acquisition"""
    content: str = Field(..., description="Content to learn from")
    strategy: LearningStrategy = Field(..., description="Learning strategy to use")
    context: Optional[str] = Field(None, description="Additional context for learning")
    existing_entities: Optional[List[str]] = Field(None, description="List of existing entity IDs to consider")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold for learned items")

class LearningResponse(BaseModel):
    """Response from knowledge acquisition"""
    entities: List[Entity] = Field(default_factory=list, description="Learned entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Learned relationships")
    strategy_used: LearningStrategy = Field(..., description="Strategy that was used")
    analysis: Dict[str, Any] = Field(..., description="Analysis of the learning process")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for learned items") 