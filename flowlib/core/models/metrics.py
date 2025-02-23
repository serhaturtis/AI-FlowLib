# src/core/models/metrics.py

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ExecutionMetrics(BaseModel):
    """Metrics about flow execution."""
    
    flow_name: str = Field(description="Name of the flow")
    start_time: datetime = Field(description="When execution started")
    end_time: datetime = Field(description="When execution completed")
    duration: float = Field(description="Time taken for execution in seconds")
    memory_usage: Optional[int] = Field(
        default=None,
        description="Peak memory usage in bytes"
    )
    error_count: int = Field(
        default=0,
        description="Number of errors encountered"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata"
    )