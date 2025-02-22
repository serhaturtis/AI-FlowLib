"""Flow-specific validation utilities."""

from typing import Dict, Any, Type
from pydantic import BaseModel

from ...core.errors.base import ValidationError, ErrorContext

def validate_flow_schema(
    data: Dict[str, Any],
    schema: Type[BaseModel],
    flow_name: str
) -> Dict[str, Any]:
    """Validate flow data against a schema.
    
    Args:
        data: Flow data to validate
        schema: Expected schema
        flow_name: Name of flow for error context
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = schema.model_validate(data)
        return validated.model_dump()
    except Exception as e:
        raise ValidationError(
            message="Flow validation failed",
            context=ErrorContext.create(
                flow_name=flow_name,
                data=data,
                schema=schema.model_json_schema()
            ),
            cause=e
        )

def validate_composite_schema(
    flows: Dict[str, Any],
    output_schema: Type[BaseModel],
    flow_name: str
) -> None:
    """Validate composite flow output schemas compatibility.
    
    Args:
        flows: Dictionary of flow name to flow instance
        output_schema: Expected output schema
        flow_name: Name of composite flow
        
    Raises:
        ValidationError: If schemas are incompatible
    """
    output_fields = output_schema.model_fields
    
    for flow_name, flow in flows.items():
        if not flow.output_schema:
            continue
            
        # Direct schema match
        if flow.output_schema == output_schema:
            continue
            
        # Field type match
        field_match = False
        for field_info in output_fields.values():
            if (isinstance(field_info.annotation, type) and 
                issubclass(flow.output_schema, field_info.annotation)):
                field_match = True
                break
        
        if not field_match:
            raise ValidationError(
                message=f"Flow '{flow_name}' output schema is not compatible",
                context=ErrorContext.create(
                    flow_name=flow_name,
                    flow_schema=flow.output_schema.model_json_schema(),
                    output_schema=output_schema.model_json_schema()
                )
            ) 