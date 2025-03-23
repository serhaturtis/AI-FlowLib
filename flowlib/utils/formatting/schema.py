"""Schema formatting utilities.

This module provides utilities for formatting schema models such as
Pydantic models for display in prompts or logs.
"""

from typing import Any, Dict, List, Optional, Type, Union


def format_schema_model(model, error_context: str = "Error formatting schema") -> Optional[str]:
    """Format a schema model into a readable string representation.
    
    This helper handles both Pydantic v1 and v2 field formats to extract
    type information safely.
    
    Args:
        model: The Pydantic model class to format
        error_context: Context string for error logging
            
    Returns:
        A formatted string representation of the model schema
    """
    if not model:
        return None
        
    try:
        model_name = model.__name__ if hasattr(model, "__name__") else str(model)
        
        # Try Pydantic v2 style first (model_fields)
        if hasattr(model, "model_fields"):
            fields = model.model_fields
            field_strs = []
            
            for name, field in fields.items():
                # Handle different ways field types might be stored
                field_type = None
                
                # Try to get annotation directly
                if hasattr(field, "annotation"):
                    annotation = field.annotation
                    if hasattr(annotation, "__name__"):
                        field_type = annotation.__name__
                    elif hasattr(annotation, "_name"):
                        field_type = annotation._name
                    else:
                        field_type = str(annotation)
                
                if field_type:
                    field_strs.append(f"{name}: {field_type}")
                else:
                    field_strs.append(name)
                    
            if field_strs:
                return f"{model_name} ({', '.join(field_strs)})"
            
        # Try Pydantic v1 style (__fields__)
        if hasattr(model, "__fields__"):
            fields = model.__fields__
            field_strs = []
            
            for name, field in fields.items():
                # Handle different ways field types might be stored
                field_type = None
                
                # Try type_ attribute (older Pydantic versions)
                if hasattr(field, "type_") and hasattr(field.type_, "__name__"):
                    field_type = field.type_.__name__
                # Try outer_type_ attribute
                elif hasattr(field, "outer_type_") and hasattr(field.outer_type_, "__name__"):
                    field_type = field.outer_type_.__name__
                # Try annotation attribute
                elif hasattr(field, "annotation") and hasattr(field.annotation, "__name__"):
                    field_type = field.annotation.__name__
                
                if field_type:
                    field_strs.append(f"{name}: {field_type}")
                else:
                    field_strs.append(name)
                    
            if field_strs:
                return f"{model_name} ({', '.join(field_strs)})"
        
        # Special handling for RootModel
        if hasattr(model, "__origin__") and getattr(model.__origin__, "__name__", "") == "RootModel":
            # Get root type annotation if possible
            if hasattr(model, "__annotations__") and "root" in model.__annotations__:
                root_type = model.__annotations__["root"]
                root_type_name = getattr(root_type, "__name__", str(root_type))
                return f"{model_name} (root: {root_type_name})"
        
        # If we have __annotations__ but not model_fields or __fields__
        if hasattr(model, "__annotations__"):
            field_strs = []
            for name, annotation in model.__annotations__.items():
                if hasattr(annotation, "__name__"):
                    field_strs.append(f"{name}: {annotation.__name__}")
                else:
                    field_strs.append(f"{name}: {str(annotation)}")
                    
            if field_strs:
                return f"{model_name} ({', '.join(field_strs)})"
        
        # Fallback to just the model name if we can't extract fields
        return model_name
    except Exception as e:
        # Log error but don't halt execution
        return f"{error_context}: {str(e)}"


def format_schema_description(model) -> str:
    """Extract schema description from model.
    
    Args:
        model: The Pydantic model class
            
    Returns:
        Schema description string
    """
    if not model:
        return "No schema description available"
        
    # Try to get docstring
    doc = getattr(model, "__doc__", None)
    if doc:
        # Clean up docstring (remove indentation and extra newlines)
        lines = [line.strip() for line in doc.split("\n")]
        # Filter out empty lines at the beginning and end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop(-1)
        
        if lines:
            return "\n".join(lines)
    
    # Fallback to model name
    return getattr(model, "__name__", str(model)) 