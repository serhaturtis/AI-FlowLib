"""
Consolidated validation module using Pydantic as the foundation.

This module provides a unified, Pydantic-based approach to validation
that replaces the multiple validation mechanisms previously used.
"""

from typing import Type, TypeVar, Dict, Any, Optional, Union, List, get_type_hints, Callable
import inspect
import warnings
from functools import wraps
from pydantic import BaseModel, ValidationError as PydanticValidationError, create_model, Field

from ..errors import ValidationError, ErrorContext

T = TypeVar('T', bound=BaseModel)

def validate_data(
    data: Any,
    model_type: Type[T],
    location: str = "data",
    strict: bool = False
) -> T:
    """
    Validate data against a Pydantic model.
    
    Args:
        data: Data to validate
        model_type: Pydantic model class
        location: Location identifier for error reporting
        strict: Whether to use strict validation
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(data, model_type):
            return data
        
        if strict:
            return model_type.model_validate(data, strict=True)
        return model_type.model_validate(data)
            
    except PydanticValidationError as e:
        # Convert Pydantic validation errors to framework format
        validation_errors = []
        for error in e.errors():
            # Format the location path
            loc_path = ".".join(str(loc) for loc in error["loc"])
            validation_errors.append({
                "location": f"{location}.{loc_path}" if loc_path else location,
                "message": error["msg"],
                "type": error["type"]
            })
        
        # Create and raise our validation error
        raise ValidationError(
            message=f"Validation failed for {model_type.__name__}",
            validation_errors=validation_errors,
            context=ErrorContext.create(
                model=model_type.__name__,
                location=location
            ),
            cause=e
        )

def create_dynamic_model(
    schema: Dict[str, Any],
    model_name: str = "DynamicModel"
) -> Type[BaseModel]:
    """
    Create a Pydantic model dynamically from a schema-like dictionary.
    
    Args:
        schema: Schema dictionary defining fields
        model_name: Name for the dynamic model
        
    Returns:
        Pydantic model class
    """
    field_definitions = {}
    
    # Process properties from the schema
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    
    for field_name, field_def in properties.items():
        # Determine if field is required
        is_required = field_name in required_fields
        
        # Get field type
        field_type = _convert_type(field_def.get("type", "any"))
        
        # Extract constraints for Field
        field_constraints = _extract_constraints(field_def)
        
        # Set default if field is not required
        if not is_required:
            field_definitions[field_name] = (Optional[field_type], Field(default=None, **field_constraints))
        else:
            field_definitions[field_name] = (field_type, Field(**field_constraints))
    
    return create_model(model_name, **field_definitions)

def validate_with_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    location: str = "data"
) -> Dict[str, Any]:
    """
    Validate data using a schema dictionary by converting it to a Pydantic model.
    
    Args:
        data: Data to validate
        schema: Schema dictionary
        location: Location identifier for error reporting
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Create a dynamic model from the schema
        model = create_dynamic_model(schema)
        
        # Validate the data using the model
        validated = validate_data(data, model, location)
        
        # Return as dictionary
        return validated.model_dump()
    except Exception as e:
        if isinstance(e, ValidationError):
            # Re-raise validation errors
            raise
        
        # Handle other errors
        raise ValidationError(
            message="Schema validation failed",
            context=ErrorContext.create(
                location=location,
                schema=str(schema)
            ),
            cause=e
        )

def validate_function(func: Callable) -> Callable:
    """
    Decorator to validate function arguments and return values using Pydantic models.
    
    This decorator validates:
    1. Any argument annotated with a Pydantic model
    2. The return value if it's annotated with a Pydantic model
    
    Args:
        func: Function to validate
        
    Returns:
        Decorated function with validation
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to parameters
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate arguments
        for param_name, param_value in bound_args.arguments.items():
            param_type = hints.get(param_name)
            
            # Skip validation for parameters without type hints
            if not param_type:
                continue
                
            # Validate if parameter type is a Pydantic model
            if isinstance(param_type, type) and issubclass(param_type, BaseModel):
                try:
                    # Replace the argument with validated model
                    if not isinstance(param_value, param_type):
                        bound_args.arguments[param_name] = validate_data(
                            param_value, 
                            param_type,
                            location=f"args.{param_name}"
                        )
                except ValidationError as e:
                    # Add function context to error
                    e.context = e.context.add(
                        function=func.__name__,
                        parameter=param_name
                    )
                    raise e
        
        # Call the function with validated arguments
        result = func(*bound_args.args, **bound_args.kwargs)
        
        # Validate return value if needed
        return_type = hints.get('return')
        if return_type and isinstance(return_type, type) and issubclass(return_type, BaseModel):
            try:
                return validate_data(result, return_type, location="return")
            except ValidationError as e:
                # Add function context to error
                e.context = e.context.add(
                    function=func.__name__,
                    return_type=return_type.__name__
                )
                raise e
                
        return result
    
    return wrapper

# Helper functions for model creation
def _convert_type(type_name: str) -> Type:
    """Convert schema type to Python type."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "any": Any
    }
    return type_mapping.get(type_name, Any)

def _extract_constraints(field_def: Dict[str, Any]) -> Dict[str, Any]:
    """Extract validation constraints from field definition."""
    constraints = {}
    
    # Number validations
    if "minimum" in field_def:
        constraints["ge"] = field_def["minimum"]
    if "maximum" in field_def:
        constraints["le"] = field_def["maximum"]
    
    # String validations
    if "minLength" in field_def:
        constraints["min_length"] = field_def["minLength"]
    if "maxLength" in field_def:
        constraints["max_length"] = field_def["maxLength"]
    if "pattern" in field_def:
        constraints["pattern"] = field_def["pattern"]
    
    # Enum validation
    if "enum" in field_def:
        constraints["enum"] = field_def["enum"]
    
    # Title and description
    if "title" in field_def:
        constraints["title"] = field_def["title"]
    if "description" in field_def:
        constraints["description"] = field_def["description"]
        
    return constraints 