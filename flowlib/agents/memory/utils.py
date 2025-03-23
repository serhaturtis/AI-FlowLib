"""Utility functions for memory operations.

This module provides helper functions for working with entity-centric memory,
including conversion between different memory representations, formatting,
and validation utilities.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Set, Union, Tuple, TypeVar
from datetime import datetime
import hashlib
import uuid

from pydantic import ValidationError as PydanticValidationError

from flowlib.core.errors import ValidationError
from flowlib.agents.memory.models import Entity, EntityAttribute, EntityRelationship

logger = logging.getLogger(__name__)

def normalize_entity_id(name: str) -> str:
    """Normalize a string to be used as an entity ID.
    
    Converts to lowercase, replaces spaces and special characters with 
    underscores, and ensures the result is a valid identifier.
    
    Args:
        name: The string to normalize
        
    Returns:
        Normalized string suitable for use as an entity ID
    """
    # Convert to lowercase
    normalized = name.lower()
    
    # Replace spaces and special characters with underscores
    normalized = re.sub(r'[^\w\s]', '_', normalized)
    normalized = re.sub(r'\s+', '_', normalized)
    
    # Remove consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    # If empty after normalization, return a UUID
    if not normalized:
        return f"entity_{uuid.uuid4().hex[:8]}"
        
    return normalized

def generate_entity_id(entity_type: str, name: Optional[str] = None) -> str:
    """Generate a unique entity ID based on entity type and optional name.
    
    Args:
        entity_type: The type of entity
        name: Optional name to incorporate into the ID
        
    Returns:
        Normalized entity ID
    """
    base = f"{entity_type.lower()}"
    
    if name:
        normalized_name = normalize_entity_id(name)
        base = f"{base}_{normalized_name}"
        
    # Add a short UUID segment to ensure uniqueness
    uuid_segment = str(uuid.uuid4())[:8]
    return f"{base}_{uuid_segment}"

def merge_entities(entity1: Entity, entity2: Entity) -> Entity:
    """Merge two entities with the same ID.
    
    Combines attributes and relationships from both entities.
    
    Args:
        entity1: First entity
        entity2: Second entity
        
    Returns:
        Merged entity
    """
    # Ensure entities have the same ID
    if entity1.id != entity2.id:
        raise ValueError(f"Cannot merge entities with different IDs: {entity1.id} vs {entity2.id}")
        
    # Create a new entity with the same ID and type
    merged = Entity(
        id=entity1.id,
        type=entity1.type,  # Use type from entity1
        attributes={},
        relationships=[],
        importance=max(entity1.importance, entity2.importance),  # Take the higher importance
        last_updated=datetime.now().isoformat()
    )
    
    # Merge attributes
    all_attributes = {}
    # Add attributes from entity1
    for name, attr in entity1.attributes.items():
        all_attributes[name] = attr
        
    # Add or update attributes from entity2
    for name, attr in entity2.attributes.items():
        if name not in all_attributes or attr.confidence > all_attributes[name].confidence:
            all_attributes[name] = attr
            
    merged.attributes = all_attributes
    
    # Merge relationships (avoiding duplicates)
    all_relationships = []
    # Start with all relationships from entity1
    all_relationships.extend(entity1.relationships)
    
    # Track existing relationships by (type, target)
    existing_rels = {(rel.relation_type, rel.target_entity) for rel in merged.relationships}
    
    # Add relationships from entity2 if not already present or if higher confidence
    for rel in entity2.relationships:
        key = (rel.relation_type, rel.target_entity)
        if key not in existing_rels:
            all_relationships.append(rel)
        else:
            # Find the existing relationship with the same key
            for i, existing_rel in enumerate(all_relationships):
                if (existing_rel.relation_type, existing_rel.target_entity) == key:
                    # Replace if confidence is higher
                    if rel.confidence > existing_rel.confidence:
                        all_relationships[i] = rel
                    break
    
    merged.relationships = all_relationships
    
    # Merge tags
    merged.tags = list(set(entity1.tags + entity2.tags))
    
    return merged

def entity_to_dict(entity: Entity, include_relationships: bool = True) -> Dict[str, Any]:
    """Convert an entity to a simple dictionary representation.
    
    This creates a flattened dictionary view of an entity with its attributes,
    suitable for display or serialization.
    
    Args:
        entity: The entity to convert
        include_relationships: Whether to include relationships
        
    Returns:
        Dictionary representation of the entity
    """
    result = {
        "id": entity.id,
        "type": entity.type,
        "last_updated": entity.last_updated,
    }
    
    # Add attributes as top-level keys
    for name, attr in entity.attributes.items():
        result[name] = attr.value
    
    # Add relationships if requested
    if include_relationships and entity.relationships:
        relationships = {}
        for rel in entity.relationships:
            rel_type = rel.relation_type
            if rel_type not in relationships:
                relationships[rel_type] = []
            relationships[rel_type].append(rel.target_entity)
        
        result["relationships"] = relationships
        
    # Add other metadata
    if entity.tags:
        result["tags"] = entity.tags
        
    if entity.importance is not None:
        result["importance"] = entity.importance
        
    if entity.source:
        result["source"] = entity.source
        
    return result

def dict_to_entity(data: Dict[str, Any]) -> Entity:
    """Convert a dictionary to an entity.
    
    This function attempts to convert a dictionary into an Entity object,
    interpreting fields according to entity model expectations.
    
    Args:
        data: Dictionary containing entity data
        
    Returns:
        Entity object
        
    Raises:
        EntityValidationError: If the dictionary cannot be converted to a valid entity
    """
    try:
        # Required fields
        entity_id = data.get("id")
        entity_type = data.get("type")
        
        if not entity_id or not entity_type:
            raise ValidationError(
                message="Dictionary must contain 'id' and 'type' fields"
            )
        
        # Create entity with basic fields
        entity = Entity(
            id=entity_id,
            type=entity_type,
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            source=data.get("source"),
            importance=data.get("importance"),
            tags=data.get("tags", []),
        )
        
        # Process attributes (all non-special fields are treated as attributes)
        special_fields = {
            "id", "type", "last_updated", "source", "importance", 
            "tags", "relationships", "vector_id"
        }
        
        for key, value in data.items():
            if key not in special_fields and value is not None:
                # Create an attribute
                entity.attributes[key] = EntityAttribute(
                    name=key,
                    value=value,
                    source=data.get("source"),
                    confidence=1.0,  # Default confidence for direct attributes
                    timestamp=data.get("last_updated")
                )
                
        # Process relationships if present
        if "relationships" in data and isinstance(data["relationships"], dict):
            for rel_type, targets in data["relationships"].items():
                if isinstance(targets, list):
                    for target in targets:
                        entity.relationships.append(
                            EntityRelationship(
                                relation_type=rel_type,
                                target_entity=target,
                                source=data.get("source"),
                                confidence=1.0,  # Default confidence for direct relationships
                                timestamp=data.get("last_updated")
                            )
                        )
                elif isinstance(targets, str):
                    # Handle single target as string
                    entity.relationships.append(
                        EntityRelationship(
                            relation_type=rel_type,
                            target_entity=targets,
                            source=data.get("source"),
                            confidence=1.0,
                            timestamp=data.get("last_updated")
                        )
                    )
                    
        return entity
        
    except PydanticValidationError as e:
        raise ValidationError(
            message=f"Failed to convert dictionary to entity: {str(e)}",
            cause=e
        )
    except Exception as e:
        raise ValidationError(
            message=f"Unexpected error converting dictionary to entity: {str(e)}",
            cause=e
        )

def format_entity_for_display(entity: Entity, detailed: bool = False) -> str:
    """Format an entity for human-readable display.
    
    Args:
        entity: The entity to format
        detailed: Whether to include all details
        
    Returns:
        Formatted string representation
    """
    lines = [f"Entity: {entity.type.title()} - {entity.id}"]
    
    # Add attributes
    if entity.attributes:
        lines.append("Attributes:")
        for name, attr in sorted(entity.attributes.items()):
            attr_line = f"  {name}: {attr.value}"
            if detailed and attr.source:
                attr_line += f" (from {attr.source}, confidence: {attr.confidence:.2f})"
            lines.append(attr_line)
    
    # Add relationships
    if entity.relationships:
        lines.append("Relationships:")
        # Group by relation type for cleaner display
        rel_by_type = {}
        for rel in entity.relationships:
            if rel.relation_type not in rel_by_type:
                rel_by_type[rel.relation_type] = []
            rel_by_type[rel.relation_type].append(rel)
            
        for rel_type, rels in sorted(rel_by_type.items()):
            targets = [rel.target_entity for rel in rels]
            rel_line = f"  {rel_type}: {', '.join(targets)}"
            lines.append(rel_line)
    
    # Add metadata if detailed
    if detailed:
        if entity.tags:
            lines.append(f"Tags: {', '.join(entity.tags)}")
            
        if entity.importance is not None:
            lines.append(f"Importance: {entity.importance:.2f}")
            
        if entity.source:
            lines.append(f"Source: {entity.source}")
            
        if entity.last_updated:
            lines.append(f"Last Updated: {entity.last_updated}")
    
    return "\n".join(lines)

def format_entities_as_context(entities: List[Entity], include_relationships: bool = True) -> str:
    """Format multiple entities as context for prompt injection.
    
    Args:
        entities: List of entities to format
        include_relationships: Whether to include relationship information
        
    Returns:
        Formatted context string
    """
    if not entities:
        return ""
        
    parts = ["Relevant memory information:"]
    
    for entity in entities:
        entity_part = format_entity_for_display(
            entity,
            detailed=False  # Less verbose for context
        )
        parts.append(entity_part)
    
    return "\n\n".join(parts)

def extract_entity_id_from_text(text: str, entity_type: Optional[str] = None) -> Optional[str]:
    """Extract an entity ID from text description.
    
    This attempts to identify and normalize an entity identifier from text.
    
    Args:
        text: Text to extract from
        entity_type: Optional entity type to prefix the ID
        
    Returns:
        Extracted entity ID or None if no clear identifier can be found
    """
    # Remove common filler words
    fillers = ["the", "a", "an", "this", "that", "these", "those"]
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in fillers]
    
    if not filtered_words:
        return None
        
    # Try to find a name-like pattern (capitalized words)
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    matches = re.findall(name_pattern, text)
    
    if matches:
        # Use the longest match as the name
        name = max(matches, key=len)
        normalized = normalize_entity_id(name)
        
        if entity_type:
            return f"{entity_type}_{normalized}"
        return normalized
        
    # If no name pattern found, use the first few significant words
    significant = " ".join(filtered_words[:3])  # Limit to 3 words
    normalized = normalize_entity_id(significant)
    
    if not normalized:
        return None
        
    if entity_type:
        return f"{entity_type}_{normalized}"
    return normalized

def calculate_entity_hash(entity: Entity) -> str:
    """Calculate a hash representing the entity's content.
    
    This is useful for detecting changes to an entity.
    
    Args:
        entity: The entity to hash
        
    Returns:
        Hash string
    """
    # Create a simplified representation for hashing
    hash_dict = {
        "id": entity.id,
        "type": entity.type,
        "attributes": {
            name: attr.value for name, attr in entity.attributes.items()
        },
        "relationships": [
            {"type": rel.relation_type, "target": rel.target_entity}
            for rel in entity.relationships
        ]
    }
    
    # Convert to a stable string representation and hash
    json_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()

def validate_entity(entity: Entity) -> Tuple[bool, List[str]]:
    """Validate an entity for data quality and completeness.
    
    Checks for required fields, valid relationships, etc.
    
    Args:
        entity: The entity to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check ID and type
    if not entity.id:
        errors.append("Entity ID is required")
        
    if not entity.type:
        errors.append("Entity type is required")
    
    # Check attributes for valid values
    for name, attr in entity.attributes.items():
        if attr.value is None:
            errors.append(f"Attribute '{name}' has None value")
            
        if attr.confidence is not None and (attr.confidence < 0 or attr.confidence > 1):
            errors.append(f"Attribute '{name}' has invalid confidence {attr.confidence}")
    
    # Check relationships
    for i, rel in enumerate(entity.relationships):
        if not rel.relation_type:
            errors.append(f"Relationship {i} is missing relation_type")
            
        if not rel.target_entity:
            errors.append(f"Relationship {i} is missing target_entity")
            
        if rel.confidence is not None and (rel.confidence < 0 or rel.confidence > 1):
            errors.append(f"Relationship {i} has invalid confidence {rel.confidence}")
    
    # Check other fields
    if entity.importance is not None and (entity.importance < 0 or entity.importance > 1):
        errors.append(f"Entity has invalid importance {entity.importance}")
    
    return (len(errors) == 0, errors)

def get_entity_relationships(entity: Entity) -> Dict[str, List[str]]:
    """Extract relationships from an entity, grouped by relation type.
    
    Args:
        entity: The entity to process
        
    Returns:
        Dictionary mapping relation types to lists of target entity IDs
    """
    relationships = {}
    
    for rel in entity.relationships:
        rel_type = rel.relation_type
        if rel_type not in relationships:
            relationships[rel_type] = []
            
        relationships[rel_type].append(rel.target_entity)
        
    return relationships

def create_entity_relationship(source: Entity, relation_type: str, target: str) -> None:
    """Create a relationship between source entity and target.
    
    Args:
        source: Source entity
        relation_type: Type of relationship
        target: Target entity ID
    """
    # Ensure the relationship doesn't already exist
    for rel in source.relationships:
        if rel.relation_type == relation_type and rel.target_entity == target:
            return  # Relationship already exists
            
    # Add the relationship
    source.relationships.append(
        EntityRelationship(
            relation_type=relation_type,
            target_entity=target,
            confidence=0.9,
            source="system"
        )
    )

def create_entity_relationships(source: Entity, relation_type: str, targets: List[str]) -> None:
    """Create relationships between source entity and multiple targets.
    
    Args:
        source: Source entity
        relation_type: Type of relationship 
        targets: List of target entity IDs
    """
    for target in targets:
        create_entity_relationship(source, relation_type, target)

def get_related_entities(entities: List[Entity], relation_type: Optional[str] = None) -> List[str]:
    """Get the IDs of entities related to the given entities.
    
    Args:
        entities: List of entities to check for relationships
        relation_type: Optional relation type to filter by
        
    Returns:
        List of related entity IDs
    """
    related = set()
    
    for entity in entities:
        # Filter relationships by type if specified
        rels = entity.relationships
        if relation_type:
            rels = [r for r in rels if r.relation_type == relation_type]
            
        # Add target IDs to related set
        targets = [rel.target_entity for rel in rels]
        related.update(targets)
        
    # Remove original entities from result
    original_ids = {e.id for e in entities}
    related = related - original_ids
    
    return list(related) 