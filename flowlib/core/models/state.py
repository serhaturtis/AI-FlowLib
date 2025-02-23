# src/core/models/state.py

from datetime import datetime
from typing import Optional, Dict, Any, List
from .results import FlowStatus
from ..errors.base import StateError

class ExecutionRecord:
    """Record of a single flow execution."""
    
    def __init__(self):
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        self.status: FlowStatus = FlowStatus.RUNNING
        self.error: Optional[Exception] = None
        self.metadata: Dict[str, Any] = {}

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    def complete(
        self,
        status: FlowStatus,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Complete the execution record.
        
        Args:
            status: Final execution status
            error: Optional error that occurred
            metadata: Optional execution metadata
        """
        self.end_time = datetime.now()
        self.status = status
        self.error = error
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "duration": self.duration,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata
        }

class FlowState:
    """
    Manages the state of a flow throughout its lifecycle.
    Tracks execution history and statistics.
    """
    
    def __init__(self):
        """Initialize flow state."""
        self.status: FlowStatus = FlowStatus.PENDING
        self.current_execution: Optional[ExecutionRecord] = None
        self.execution_history: List[ExecutionRecord] = []
        self.last_error: Optional[Exception] = None
        self.execution_count: int = 0
        self.success_count: int = 0
        self.failure_count: int = 0
        self.total_duration: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def start_execution(self) -> None:
        """
        Start a new execution.
        
        Raises:
            StateError: If flow is already running
        """
        if self.status == FlowStatus.RUNNING:
            raise StateError(
                message="Cannot start execution while flow is already running",
                flow_name="unknown"  # Flow name not available in state
            )
            
        self.status = FlowStatus.RUNNING
        self.current_execution = ExecutionRecord()
        self.execution_count += 1

    def mark_success(
        self,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark current execution as successful.
        
        Args:
            metadata: Optional execution metadata
            
        Raises:
            StateError: If no execution is running
        """
        if not self.current_execution:
            raise StateError(
                message="No execution in progress to mark as success",
                flow_name="unknown"
            )
            
        self.current_execution.complete(
            status=FlowStatus.SUCCESS,
            metadata=metadata
        )
        self.execution_history.append(self.current_execution)
        
        self.status = FlowStatus.SUCCESS
        self.success_count += 1
        self.total_duration += self.current_execution.duration
        self.current_execution = None

    def mark_failure(
        self,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark current execution as failed.
        
        Args:
            error: Error that caused failure
            metadata: Optional execution metadata
            
        Raises:
            StateError: If no execution is running
        """
        if not self.current_execution:
            raise StateError(
                message="No execution in progress to mark as failure",
                flow_name="unknown"
            )
            
        self.current_execution.complete(
            status=FlowStatus.FAILED,
            error=error,
            metadata=metadata
        )
        self.execution_history.append(self.current_execution)
        
        self.status = FlowStatus.FAILED
        self.last_error = error
        self.failure_count += 1
        self.total_duration += self.current_execution.duration
        self.current_execution = None

    def mark_cancelled(
        self,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark current execution as cancelled.
        
        Args:
            reason: Reason for cancellation
            metadata: Optional execution metadata
            
        Raises:
            StateError: If no execution is running
        """
        if not self.current_execution:
            raise StateError(
                message="No execution in progress to mark as cancelled",
                flow_name="unknown"
            )
            
        metadata = metadata or {}
        metadata["cancellation_reason"] = reason
        
        self.current_execution.complete(
            status=FlowStatus.CANCELLED,
            metadata=metadata
        )
        self.execution_history.append(self.current_execution)
        
        self.status = FlowStatus.CANCELLED
        self.total_duration += self.current_execution.duration
        self.current_execution = None

    def is_running(self) -> bool:
        """Check if flow is currently running."""
        return self.status == FlowStatus.RUNNING

    def can_execute(self) -> bool:
        """Check if flow can start new execution."""
        return self.status != FlowStatus.RUNNING

    def get_average_duration(self) -> float:
        """Get average execution duration in seconds."""
        if self.execution_count == 0:
            return 0.0
        return self.total_duration / self.execution_count

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100

    def get_last_execution(self) -> Optional[ExecutionRecord]:
        """Get most recent execution record."""
        if not self.execution_history:
            return None
        return self.execution_history[-1]

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution history."""
        return {
            "total_executions": self.execution_count,
            "successful_executions": self.success_count,
            "failed_executions": self.failure_count,
            "success_rate": self.get_success_rate(),
            "average_duration": self.get_average_duration(),
            "current_status": self.status.value,
            "last_error": str(self.last_error) if self.last_error else None
        }

    def reset(self) -> None:
        """Reset state to initial values."""
        self.status = FlowStatus.PENDING
        self.current_execution = None
        self.execution_history.clear()
        self.last_error = None
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_duration = 0.0
        self.metadata.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "status": self.status.value,
            "current_execution": (
                self.current_execution.to_dict()
                if self.current_execution else None
            ),
            "execution_history": [
                record.to_dict() for record in self.execution_history
            ],
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_duration": self.total_duration,
            "metadata": self.metadata,
            "last_error": str(self.last_error) if self.last_error else None
        }