"""Base models for the Integrated Agent System."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

class DomainType(str, Enum):
    """Base class for domain type enums."""
    pass

class StateType(str, Enum):
    """Types of state changes."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"
    LOCKED = "locked"
    UNLOCKED = "unlocked"

class EventType(str, Enum):
    """Types of events in the system."""
    STATE_CHANGED = "state_changed"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    ERROR_OCCURRED = "error_occurred"

class ValidationLevel(str, Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationResult(BaseModel):
    """Result of a validation check."""
    is_valid: bool = Field(..., description="Whether the validation passed")
    level: ValidationLevel = Field(..., description="Severity level of the validation")
    message: str = Field(..., description="Validation message")
    domain: str = Field(..., description="Domain that performed the validation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the validation occurred")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional validation details")

class StateMetadata(BaseModel):
    """Metadata about a state object."""
    version: int = Field(default=1, description="State version number")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the state was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the state was last updated")
    created_by: str = Field(..., description="Domain that created the state")
    updated_by: str = Field(..., description="Domain that last updated the state")
    is_locked: bool = Field(default=False, description="Whether the state is locked for updates")
    lock_holder: Optional[str] = Field(default=None, description="Domain holding the lock")
    dependencies: Set[UUID] = Field(default_factory=set, description="IDs of dependent states")
    validations: List[ValidationResult] = Field(default_factory=list, description="Validation results")

class DomainState(BaseModel):
    """Base class for domain-specific state."""
    id: UUID = Field(default_factory=uuid4, description="Unique state identifier")
    type: DomainType = Field(..., description="Type of domain state")
    metadata: StateMetadata = Field(..., description="State metadata")
    data: Dict[str, Any] = Field(..., description="Domain-specific state data")

class Event(BaseModel):
    """System event."""
    id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    type: EventType = Field(..., description="Type of event")
    domain: str = Field(..., description="Domain that generated the event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the event occurred")
    data: Dict[str, Any] = Field(..., description="Event data")
    correlation_id: Optional[UUID] = Field(default=None, description="ID for correlating related events")
    causation_id: Optional[UUID] = Field(default=None, description="ID of the event that caused this one")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")

class WorkflowStep(BaseModel):
    """Step in a workflow."""
    id: UUID = Field(default_factory=uuid4, description="Unique step identifier")
    domain: str = Field(..., description="Domain responsible for this step")
    action: str = Field(..., description="Action to perform")
    dependencies: Set[UUID] = Field(default_factory=set, description="IDs of dependent steps")
    timeout: Optional[float] = Field(default=None, description="Step timeout in seconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")

class Workflow(BaseModel):
    """Workflow definition."""
    id: UUID = Field(default_factory=uuid4, description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    version: str = Field(default="1.0", description="Workflow version")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the workflow was created")
    timeout: Optional[float] = Field(default=None, description="Overall workflow timeout in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional workflow metadata") 