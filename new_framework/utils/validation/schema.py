"""Schema validation utilities."""

from typing import Dict, Any, Type
from pydantic import BaseModel

from ...core.errors.base import ValidationError, ErrorContext

class FlowValidation:
    """Centralized validation utilities for flows."""
    
    @staticmethod
    def validate_schema(
        data: Dict[str, Any],
        schema: Type[BaseModel],
        flow_name: str,
        location: str = "data"
    ) -> Dict[str, Any]:
        """Validate data against a schema.
        
        Args:
            data: Data to validate
            schema: Pydantic model for validation
            flow_name: Name of flow for error context
            location: Location of validation (input/output)
            
        Returns:
            Validated and converted data
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(data, schema):
                return data.model_dump()
            validated = schema.model_validate(data)
            return validated.model_dump()
        except Exception as e:
            raise ValidationError(
                message=f"{location.title()} validation failed",
                validation_errors=[{
                    "location": location,
                    "message": str(e)
                }],
                context=ErrorContext.create(
                    flow_name=flow_name,
                    data=data,
                    schema=schema.model_json_schema()
                ),
                cause=e
            ) 