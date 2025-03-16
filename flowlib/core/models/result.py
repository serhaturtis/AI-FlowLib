"""Enhanced result models for flow execution.

This module provides improved result handling with attribute-based access
and better type preservation throughout the flow execution. It consolidates
functionality from previous result implementations.
"""

from datetime import datetime
import enum
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Union, get_origin, get_args, cast
from pydantic import BaseModel, Field, create_model, root_validator

T = TypeVar('T')

class FlowStatus(str, enum.Enum):
    """Enumeration of possible flow execution statuses."""
    
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    
    def is_terminal(self) -> bool:
        """Check if status is terminal."""
        return self in (
            self.SUCCESS,
            self.ERROR,
            self.CANCELED,
            self.TIMEOUT,
            self.SKIPPED
        )
    
    def is_error(self) -> bool:
        """Check if status indicates an error."""
        return self in (
            self.ERROR,
            self.TIMEOUT,
        )
    
    def __str__(self) -> str:
        """Return string representation."""
        return self.value

class FlowResult(BaseModel, Generic[T]):
    """Enhanced result model with attribute-based access.
    
    This class provides:
    1. Attribute-based access to result data
    2. Structured error information
    3. Type-safe data retrieval
    4. Rich metadata for monitoring and debugging
    """
    
    data: Union[Dict[str, Any], BaseModel] = Field(default_factory=dict)
    flow_name: str = ""
    status: FlowStatus = FlowStatus.SUCCESS
    error: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    duration: Optional[float] = None
    original_type: Optional[Type] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
    
    @root_validator(pre=True)
    def set_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set default values based on other fields."""
        # Set error status if error is present
        if values.get('error') and values.get('status') == FlowStatus.SUCCESS:
            values['status'] = FlowStatus.ERROR
        return values
    
    def __getattr__(self, name: str) -> Any:
        """Enable attribute-based access to result data.
        
        Args:
            name: Attribute name to access
            
        Returns:
            Attribute value from data dict
            
        Raises:
            AttributeError: If attribute not found
        """
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get_typed(self, model_cls: Type[T]) -> T:
        """Convert result data to a typed model.
        
        Args:
            model_cls: Model class to convert to
            
        Returns:
            Typed model instance
            
        Raises:
            ValueError: If conversion fails
        """
        try:
            if self.original_type == model_cls:
                # This result was originally from this model type, safe to convert back
                return cast(T, model_cls(**self.data))
            return cast(T, model_cls(**self.data))
        except Exception as e:
            raise ValueError(f"Failed to convert result to {model_cls.__name__}: {str(e)}")
    
    def as_dict(self) -> Dict[str, Any]:
        """Get result as a dictionary.
        
        Returns:
            Dictionary containing all result data and metadata
        """
        return {
            "data": self.data,
            "flow_name": self.flow_name,
            "status": self.status.value,
            "error": self.error,
            "error_details": self.error_details,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration
        }
    
    def is_success(self) -> bool:
        """Check if result indicates success.
        
        Returns:
            True if status is SUCCESS, False otherwise
        """
        return self.status == FlowStatus.SUCCESS
    
    def is_error(self) -> bool:
        """Check if result indicates error.
        
        Returns:
            True if status is ERROR or TIMEOUT, False otherwise
        """
        return self.status.is_error()
    
    def raise_if_error(self) -> 'FlowResult[T]':
        """Raise exception if result indicates error.
        
        Returns:
            Self if no error
            
        Raises:
            Exception: If result indicates error
        """
        if self.is_error():
            error_msg = self.error or f"Flow '{self.flow_name}' failed with status {self.status}"
            raise Exception(error_msg)
        return self
    
    def __str__(self) -> str:
        """String representation."""
        if self.is_error():
            return f"FlowResult(flow='{self.flow_name}', status={self.status}, error='{self.error}')"
        return f"FlowResult(flow='{self.flow_name}', status={self.status}, data_keys={list(self.data.keys())})"

# Type helpers for specific result types
TResult = FlowResult[T]

def result_from_value(value: Any, flow_name: str = "unnamed_flow") -> FlowResult:
    """Create a success result from a simple value.
    
    Args:
        value: Value to wrap in result
        flow_name: Optional flow name
        
    Returns:
        FlowResult containing the value
    """
    return FlowResult(
        data=value,
        flow_name=flow_name,
        status=FlowStatus.SUCCESS
    )

def error_result(
    error: str,
    flow_name: str = "unnamed_flow",
    error_details: Optional[Dict[str, Any]] = None
) -> FlowResult:
    """Create an error result.
    
    Args:
        error: Error message
        flow_name: Optional flow name
        error_details: Optional error details
        
    Returns:
        FlowResult representing the error
    """
    return FlowResult(
        data={},
        flow_name=flow_name,
        status=FlowStatus.ERROR,
        error=error,
        error_details=error_details or {}
    ) 