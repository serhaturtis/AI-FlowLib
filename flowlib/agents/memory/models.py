"""Entity-centric memory models.

This module defines the data models for the entity-centric memory system,
including entity attributes, relationships, and extracted information.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

class EntityAttribute(BaseModel):
    """An attribute of an entity."""
    name: str = Field(..., description="Name of this attribute (e.g., 'full_name', 'age')")
    value: str = Field(..., description="Value of this attribute")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Confidence score (0-1)")
    source: str = Field("conversation", description="Source of this information")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                          description="When this attribute was recorded")

class EntityRelationship(BaseModel):
    """A relationship between entities."""
    relation_type: str = Field(..., description="Type of relationship (e.g., 'friend_of')")
    target_id: str = Field(..., description="ID of the target entity")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Confidence in this relationship")
    source: str = Field("conversation", description="Source of this relationship")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                          description="When this relationship was established")

class Entity(BaseModel):
    """An entity in the knowledge graph."""
    id: str = Field(..., description="Unique identifier for this entity")
    type: str = Field(..., description="Type of entity (person, location, event, etc.)")
    attributes: Dict[str, EntityAttribute] = Field(default_factory=dict)
    relationships: List[EntityRelationship] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    importance: float = Field(0.7, ge=0.0, le=1.0, description="Overall importance of this entity")
    vector_id: Optional[str] = Field(None, description="ID in vector store if applicable")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                             description="When this entity was last updated")
    
    def to_memory_item(self, attribute_name: Optional[str] = None) -> Dict[str, Any]:
        """Convert entity to a memory item for storage.
        
        Args:
            attribute_name: Optional specific attribute to convert
            
        Returns:
            Dictionary representation for memory storage
        """
        if attribute_name and attribute_name in self.attributes:
            # Return specific attribute as memory item
            attr = self.attributes[attribute_name]
            return {
                "entity_id": self.id,
                "entity_type": self.type,
                "attribute": attribute_name,
                "value": attr.value,
                "confidence": attr.confidence,
                "importance": self.importance,
                "source": attr.source,
                "tags": self.tags,
                "relationships": [{"type": r.relation_type, "target": r.target_id} for r in self.relationships],
                "timestamp": attr.timestamp
            }
        else:
            # Return entity overview
            return {
                "entity_id": self.id,
                "entity_type": self.type,
                "attribute": "summary",
                "value": f"{self.type} with {len(self.attributes)} attributes and {len(self.relationships)} relationships",
                "confidence": 1.0,
                "importance": self.importance,
                "source": "system",
                "tags": self.tags,
                "relationships": [{"type": r.relation_type, "target": r.target_id} for r in self.relationships],
                "timestamp": self.last_updated
            }
    
    def get_formatted_view(self) -> str:
        """Get a human-readable formatted view of the entity.
        
        Returns:
            String representation with attributes and relationships
        """
        lines = [f"Entity: {self.id} (Type: {self.type})"]
        
        # Add attributes
        if self.attributes:
            lines.append("Attributes:")
            for name, attr in self.attributes.items():
                lines.append(f"  {name}: {attr.value} (confidence: {attr.confidence:.2f})")
        
        # Add relationships
        if self.relationships:
            lines.append("Relationships:")
            for rel in self.relationships:
                lines.append(f"  {rel.relation_type} {rel.target_id} (confidence: {rel.confidence:.2f})")
        
        # Add tags
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        
        return "\n".join(lines)

class RelationshipUpdate(BaseModel):
    """A relationship to be added or updated."""
    type: str = Field(..., description="Type of relationship")
    target: str = Field(..., description="ID of the target entity")

class ExtractedEntityInfo(BaseModel):
    """Information extracted from conversation to be stored in memory."""
    entity_id: str = Field(..., description="Unique identifier for this entity")
    entity_type: str = Field(..., description="Type of entity")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Attribute name to value mapping")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Confidence in this information")
    relationships: List[Dict[str, str]] = Field(default_factory=list, 
        description="Relationships to other entities (e.g., [{'type': 'friend_of', 'target': 'serhat'}])")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    importance: float = Field(0.7, ge=0.0, le=1.0, description="Importance of this entity")
    source: str = Field("conversation", description="Source of this information")
    context: Optional[str] = Field(None, description="Human-readable context about this entity")

class EntityExtractionResult(BaseModel):
    """Result of entity extraction from conversation."""
    entities: List[ExtractedEntityInfo] = Field(default_factory=list, description="Extracted entities")
    reasoning: str = Field("", description="Reasoning for extraction decisions")

class MemoryQueryRequest(BaseModel):
    """Request for querying memory."""
    should_query_memory: bool = Field(..., description="Whether memory should be queried")
    entity_ids: List[str] = Field(default_factory=list, description="Specific entity IDs to retrieve")
    include_related: bool = Field(False, description="Whether to include related entities")
    semantic_query: Optional[str] = Field(None, description="Query for semantic search")

class MemoryRetrievalInput(BaseModel):
    """Input for memory retrieval flow."""
    message: str = Field(..., description="User message to analyze")
    memory_manager: Any = Field(..., description="Memory manager instance")
    model_name: str = Field("default", description="Model to use for analysis")

class MemoryRetrievalOutput(BaseModel):
    """Output from memory retrieval flow."""
    retrieved_entities: List[Entity] = Field(default_factory=list, description="Retrieved entities")
    memory_context: str = Field("", description="Human-readable memory context")

class MemoryExtractionInput(BaseModel):
    """Input for memory extraction flow."""
    user_message: str = Field(..., description="User message from conversation")
    agent_response: str = Field(..., description="Agent response from conversation")
    known_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Known entity information")
    memory_manager: Any = Field(..., description="Memory manager instance")
    model_name: str = Field("default", description="Model to use for extraction")

class MemoryExtractionOutput(BaseModel):
    """Output from memory extraction flow."""
    extracted_entities: List[Entity] = Field(default_factory=list, description="Stored entities")
    extraction_summary: str = Field("", description="Summary of extraction operation") 