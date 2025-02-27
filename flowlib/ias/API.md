# Integrated Agent System (IAS) API Reference

## Table of Contents
- [Models](#models)
  - [Base Models](#base-models)
  - [Enums](#enums)
- [Core Components](#core-components)
  - [State Management](#state-management)
  - [Event System](#event-system)
  - [Validation Framework](#validation-framework)
  - [Workflow Engine](#workflow-engine)
- [Error Handling](#error-handling)

## Models

### Base Models

#### DomainState
Base class for domain-specific state.

**Fields:**
- `id: UUID` - Unique state identifier
- `type: DomainType` - Type of domain state
- `metadata: StateMetadata` - State metadata
- `data: Dict[str, Any]` - Domain-specific state data

#### StateMetadata
Metadata about a state object.

**Fields:**
- `version: int` - State version number (default: 1)
- `created_at: datetime` - Creation timestamp
- `updated_at: datetime` - Last update timestamp
- `created_by: str` - Domain that created the state
- `updated_by: str` - Domain that last updated the state
- `is_locked: bool` - Whether the state is locked (default: False)
- `lock_holder: Optional[str]` - Domain holding the lock
- `dependencies: Set[UUID]` - IDs of dependent states
- `validations: List[ValidationResult]` - Validation results

#### Event
System event model.

**Fields:**
- `id: UUID` - Unique event identifier
- `type: EventType` - Type of event
- `domain: str` - Domain that generated the event
- `timestamp: datetime` - Event timestamp
- `data: Dict[str, Any]` - Event data
- `correlation_id: Optional[UUID]` - ID for correlating related events
- `causation_id: Optional[UUID]` - ID of the event that caused this one
- `metadata: Dict[str, Any]` - Additional event metadata

#### ValidationResult
Result of a validation check.

**Fields:**
- `is_valid: bool` - Whether the validation passed
- `level: ValidationLevel` - Severity level
- `message: str` - Validation message
- `domain: str` - Domain that performed the validation
- `timestamp: datetime` - Validation timestamp
- `details: Dict[str, Any]` - Additional validation details

#### WorkflowStep
Step in a workflow.

**Fields:**
- `id: UUID` - Unique step identifier
- `domain: str` - Domain responsible for this step
- `action: str` - Action to perform
- `dependencies: Set[UUID]` - IDs of dependent steps
- `timeout: Optional[float]` - Step timeout in seconds
- `retry_count: int` - Number of retries attempted (default: 0)
- `max_retries: int` - Maximum number of retries (default: 3)
- `metadata: Dict[str, Any]` - Additional step metadata

#### Workflow
Workflow definition.

**Fields:**
- `id: UUID` - Unique workflow identifier
- `name: str` - Workflow name
- `version: str` - Workflow version (default: "1.0")
- `steps: List[WorkflowStep]` - Workflow steps
- `created_at: datetime` - Creation timestamp
- `timeout: Optional[float]` - Overall workflow timeout in seconds
- `metadata: Dict[str, Any]` - Additional workflow metadata

### Enums

#### DomainType
Base class for domain type enums.

#### StateType
Types of state changes:
- `CREATED` - State created
- `UPDATED` - State updated
- `DELETED` - State deleted
- `VALIDATED` - State validated
- `INVALIDATED` - State invalidated
- `LOCKED` - State locked
- `UNLOCKED` - State unlocked

#### EventType
Types of events:
- `STATE_CHANGED` - State change event
- `VALIDATION_STARTED` - Validation started
- `VALIDATION_COMPLETED` - Validation completed
- `VALIDATION_FAILED` - Validation failed
- `WORKFLOW_STARTED` - Workflow started
- `WORKFLOW_COMPLETED` - Workflow completed
- `WORKFLOW_FAILED` - Workflow failed
- `AGENT_REGISTERED` - Agent registered
- `AGENT_UNREGISTERED` - Agent unregistered
- `ERROR_OCCURRED` - Error occurred

#### ValidationLevel
Validation severity levels:
- `INFO` - Informational
- `WARNING` - Warning
- `ERROR` - Error
- `CRITICAL` - Critical

## Core Components

### State Management

#### StateManager
Manages state for all domains in the system.

**Methods:**
```python
async def create_state(
    self,
    domain: str,
    state_type: Type[DomainState],
    data: Dict[str, Any]
) -> DomainState
```
Creates a new state object.

```python
async def get_state(
    self,
    state_id: UUID
) -> DomainState
```
Retrieves a state by ID.

```python
async def update_state(
    self,
    domain: str,
    state_id: UUID,
    data: Dict[str, Any],
    expected_version: Optional[int] = None
) -> DomainState
```
Updates an existing state.

```python
async def validate_state(
    self,
    domain: str,
    state_id: UUID,
    validation: ValidationResult
) -> DomainState
```
Adds validation results to a state.

```python
async def lock_state(
    self,
    domain: str,
    state_id: UUID
) -> DomainState
```
Locks a state for exclusive access.

```python
async def unlock_state(
    self,
    domain: str,
    state_id: UUID
) -> DomainState
```
Unlocks a previously locked state.

```python
def subscribe(
    self,
    domain: str,
    callback: callable
) -> None
```
Subscribes to state changes in a domain.

```python
def unsubscribe(
    self,
    domain: str,
    callback: callable
) -> None
```
Unsubscribes from state changes in a domain.

### Event System

#### EventBus
Manages event publishing and subscriptions.

**Methods:**
```python
async def publish(
    self,
    event: Event,
    retry_count: int = 0
) -> None
```
Publishes an event to subscribers.

```python
def subscribe_to_type(
    self,
    event_type: EventType,
    callback: callable
) -> None
```
Subscribes to events of a specific type.

```python
def subscribe_to_domain(
    self,
    domain: str,
    callback: callable
) -> None
```
Subscribes to events from a specific domain.

```python
def subscribe_to_correlation(
    self,
    correlation_id: UUID,
    callback: callable
) -> None
```
Subscribes to events with a specific correlation ID.

```python
def get_dead_letter_queue(self) -> List[Event]
```
Retrieves failed event deliveries.

```python
def clear_dead_letter_queue(self) -> None
```
Clears the dead letter queue.

```python
async def retry_dead_letter_queue(self) -> None
```
Retries failed event deliveries.

### Validation Framework

#### Validator
Abstract base class for validators.

**Methods:**
```python
@abstractmethod
async def validate(
    self,
    state: DomainState
) -> List[ValidationResult]
```
Validates a state object.

#### ValidationRegistry
Registry for domain validators.

**Methods:**
```python
def register_validator(
    self,
    domain: str,
    validator: Validator
) -> None
```
Registers a validator for a domain.

```python
def unregister_validator(
    self,
    domain: str,
    validator: Validator
) -> None
```
Unregisters a validator from a domain.

```python
def get_validators(
    self,
    domain: str
) -> Set[Validator]
```
Gets all validators for a domain.

#### ValidationManager
Manages validation execution.

**Methods:**
```python
async def validate_state(
    self,
    state: DomainState,
    raise_on_error: bool = False
) -> List[ValidationResult]
```
Validates a state using registered validators.

#### CrossDomainValidator
Abstract base class for cross-domain validators.

**Methods:**
```python
@abstractmethod
async def validate_relationship(
    self,
    source_state: DomainState,
    target_state: DomainState
) -> List[ValidationResult]
```
Validates relationships between states from different domains.

### Workflow Engine

#### WorkflowEngine
Manages workflow execution.

**Methods:**
```python
async def start_workflow(
    self,
    workflow: Workflow
) -> None
```
Starts executing a workflow.

```python
def get_workflow_status(
    self,
    workflow_id: UUID
) -> Optional[Dict[str, Any]]
```
Gets the status of a workflow.

```python
def list_active_workflows(self) -> List[Dict[str, Any]]
```
Lists all active workflows.

## Error Handling

### Base Exceptions

#### IASError
Base class for all IAS errors.

**Attributes:**
- `message: str` - Error message
- `details: Dict[str, Any]` - Additional error details

#### StateError
Error raised when state operations fail.

#### EventError
Error raised when event operations fail.

#### ValidationError
Error raised when validation fails.

**Additional Attributes:**
- `results: List[ValidationResult]` - Failed validation results

#### WorkflowError
Error raised when workflow operations fail.

#### DomainError
Error raised when domain-specific operations fail.

**Additional Attributes:**
- `domain: str` - Domain where the error occurred 