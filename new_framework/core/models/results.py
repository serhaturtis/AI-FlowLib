# src/core/models/results.py

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class FlowStatus(str, Enum):
    """Possible states of a flow execution.
    
    States:
        PENDING: Flow has not started execution
        RUNNING: Flow is currently executing
        SUCCESS: Flow completed successfully
        ERROR: Flow encountered an error during execution
        FAILED: Flow failed to complete its task
        SKIPPED: Flow was skipped (e.g., in conditional execution)
        CANCELLED: Flow execution was cancelled
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"      # For execution/system errors
    FAILED = "failed"    # For task/business logic failures
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (
            self.SUCCESS,
            self.ERROR,
            self.FAILED,
            self.CANCELLED
        )
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error state."""
        return self in (self.ERROR, self.FAILED)

class FlowResult(BaseModel):
    """Result of a flow execution.
    
    This class represents the outcome of a flow execution, including:
    - Status of the execution
    - Output data produced
    - Error information if applicable
    - Execution metadata
    - Timing information
    """
    
    flow_name: str = Field(description="Name of the flow that produced this result")
    status: FlowStatus = Field(description="Status of the flow execution")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output data from the flow"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this result was produced"
    )
    duration: Optional[float] = Field(
        default=None,
        description="Execution duration in seconds"
    )

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == FlowStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if execution resulted in error."""
        return self.status.is_error

    def is_terminal(self) -> bool:
        """Check if status is terminal."""
        return self.status.is_terminal

    def with_error(
        self,
        error: str,
        details: Optional[Dict[str, Any]] = None,
        status: FlowStatus = FlowStatus.ERROR
    ) -> 'FlowResult':
        """
        Create a new result with error information.
        
        Args:
            error: Error message
            details: Optional error details
            status: Error status (ERROR or FAILED)
            
        Returns:
            New result instance with error info
        """
        return FlowResult(
            flow_name=self.flow_name,
            status=status,
            data=self.data,
            error=error,
            error_details=details,
            metadata=self.metadata,
            timestamp=datetime.now(),
            duration=self.duration
        )

    def with_data(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'FlowResult':
        """
        Create a new result with updated data.
        
        Args:
            data: New data dictionary
            metadata: Optional metadata to merge
            
        Returns:
            New result instance with updated data
        """
        new_metadata = dict(self.metadata)
        if metadata:
            new_metadata.update(metadata)
            
        return FlowResult(
            flow_name=self.flow_name,
            status=self.status,
            data=data,
            error=self.error,
            error_details=self.error_details,
            metadata=new_metadata,
            timestamp=datetime.now(),
            duration=self.duration
        )

    def __str__(self) -> str:
        """String representation."""
        status_str = f"[{self.status.value}]"
        if self.error:
            return f"{status_str} {self.flow_name}: {self.error}"
        return f"{status_str} {self.flow_name}: {len(self.data)} data items"

