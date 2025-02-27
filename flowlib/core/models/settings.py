# src/core/models/config.py

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator

class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts"
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Base delay between retries in seconds"
    )
    max_delay: float = Field(
        default=60.0,
        ge=0.0,
        description="Maximum delay between retries in seconds"
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.0,
        description="Base for exponential backoff"
    )
    jitter: bool = Field(
        default=True,
        description="Whether to add random jitter to delays"
    )
    fallback_model_path: Optional[str] = Field(
        default=None,
        description="Path to fallback model"
    )
    fallback_model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for fallback model"
    )

    @field_validator('max_delay')
    def validate_max_delay(cls, v, values):
        """Validate max_delay is greater than base_delay."""
        if 'base_delay' in values and v < values['base_delay']:
            raise ValueError('max_delay must be greater than base_delay')
        return v

class ResourceRequirements(BaseModel):
    """Resource requirements for a flow."""
    
    min_memory_mb: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum required memory in MB"
    )
    max_memory_mb: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum memory limit in MB"
    )
    min_cpu_cores: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum required CPU cores"
    )
    max_cpu_cores: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum CPU cores to use"
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Execution timeout in seconds"
    )
    priority: int = Field(
        default=0,
        description="Execution priority (higher = more important)"
    )

    @field_validator('max_memory_mb')
    def validate_max_memory(cls, v, values):
        """Validate max_memory is greater than min_memory."""
        if v is not None and 'min_memory_mb' in values and values['min_memory_mb'] is not None:
            if v < values['min_memory_mb']:
                raise ValueError('max_memory_mb must be greater than min_memory_mb')
        return v

    @field_validator('max_cpu_cores')
    def validate_max_cpu(cls, v, values):
        """Validate max_cpu is greater than min_cpu."""
        if v is not None and 'min_cpu_cores' in values and values['min_cpu_cores'] is not None:
            if v < values['min_cpu_cores']:
                raise ValueError('max_cpu_cores must be greater than min_cpu_cores')
        return v

