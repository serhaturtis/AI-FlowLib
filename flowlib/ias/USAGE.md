# Using the Integrated Agent System (IAS)

This guide demonstrates how to use the Integrated Agent System (IAS) to create and manage integrated agent workflows.

## Table of Contents
- [Basic Concepts](#basic-concepts)
- [Getting Started](#getting-started)
- [Creating Domain Agents](#creating-domain-agents)
- [State Management](#state-management)
- [Event Handling](#event-handling)
- [Validation](#validation)
- [Workflow Orchestration](#workflow-orchestration)
- [Error Handling](#error-handling)
- [Complete Example](#complete-example)

## Basic Concepts

IAS provides a framework for coordinating multiple specialized agents through:
- Shared state management
- Event-driven communication
- Cross-domain validation
- Workflow orchestration

## Getting Started

1. Install the package:
```bash
pip install flowlib-ias
```

2. Import required components:
```python
from flowlib.ias.models.base import DomainState, Event, Workflow
from flowlib.ias.core.state import StateManager
from flowlib.ias.core.events import EventBus
from flowlib.ias.core.validation import ValidationManager, ValidationRegistry
from flowlib.ias.core.workflow import WorkflowEngine
```

## Creating Domain Agents

1. Define domain-specific state:
```python
from enum import Enum
from pydantic import BaseModel
from flowlib.ias.models.base import DomainState, DomainType

class DesignDomainType(DomainType):
    SYSTEM = "system"
    HARDWARE = "hardware"
    SOFTWARE = "software"

class DesignState(DomainState):
    type: DesignDomainType
    data: Dict[str, Any]  # Domain-specific data
```

2. Create a domain agent:
```python
class DesignAgent:
    def __init__(
        self,
        state_manager: StateManager,
        event_bus: EventBus,
        validation_manager: ValidationManager
    ):
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.validation_manager = validation_manager
        
    async def create_design(self, design_data: Dict[str, Any]) -> UUID:
        # Create new design state
        state = await self.state_manager.create_state(
            domain="design",
            state_type=DesignState,
            data=design_data
        )
        return state.id
        
    async def update_design(self, design_id: UUID, updates: Dict[str, Any]) -> None:
        # Update existing design
        await self.state_manager.update_state(
            domain="design",
            state_id=design_id,
            data=updates
        )
```

## State Management

1. Initialize state manager:
```python
state_manager = StateManager()
```

2. Create and manage state:
```python
# Create state
state_id = await state_manager.create_state(
    domain="design",
    state_type=DesignState,
    data={"name": "Project X", "version": "1.0"}
)

# Get state
state = await state_manager.get_state(state_id)

# Update state
await state_manager.update_state(
    domain="design",
    state_id=state_id,
    data={"version": "1.1"}
)

# Lock state for exclusive access
await state_manager.lock_state("design", state_id)
try:
    # Perform exclusive operations
    pass
finally:
    await state_manager.unlock_state("design", state_id)
```

## Event Handling

1. Initialize event bus:
```python
event_bus = EventBus()
```

2. Subscribe to events:
```python
# Subscribe to specific event type
async def handle_state_change(event: Event) -> None:
    print(f"State changed: {event.data}")
    
event_bus.subscribe_to_type(EventType.STATE_CHANGED, handle_state_change)

# Subscribe to domain events
async def handle_design_events(event: Event) -> None:
    print(f"Design event: {event.type}")
    
event_bus.subscribe_to_domain("design", handle_design_events)
```

3. Publish events:
```python
await event_bus.publish(Event(
    type=EventType.STATE_CHANGED,
    domain="design",
    data={"state_id": state_id, "changes": {"version": "1.1"}}
))
```

## Validation

1. Create domain validator:
```python
class DesignValidator(Validator):
    async def validate(self, state: DomainState) -> List[ValidationResult]:
        results = []
        
        # Perform validation checks
        if "name" not in state.data:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Design name is required",
                domain="design"
            ))
            
        return results
```

2. Set up validation:
```python
# Create registry and register validators
registry = ValidationRegistry()
registry.register_validator("design", DesignValidator())

# Create validation manager
validation_manager = ValidationManager(registry)

# Validate state
results = await validation_manager.validate_state(state)
```

## Workflow Orchestration

1. Define workflow:
```python
workflow = Workflow(
    name="design_review",
    steps=[
        WorkflowStep(
            domain="design",
            action="validate_design",
            id=uuid4()
        ),
        WorkflowStep(
            domain="review",
            action="collect_feedback",
            id=uuid4()
        ),
        WorkflowStep(
            domain="design",
            action="update_design",
            id=uuid4(),
            dependencies={previous_step_id}
        )
    ]
)
```

2. Execute workflow:
```python
# Initialize workflow engine
workflow_engine = WorkflowEngine(event_bus)

# Start workflow
await workflow_engine.start_workflow(workflow)

# Monitor status
status = workflow_engine.get_workflow_status(workflow.id)
```

## Error Handling

Handle errors appropriately:
```python
from flowlib.ias.utils.errors import (
    StateError, EventError, ValidationError, WorkflowError
)

try:
    await state_manager.update_state(...)
except StateError as e:
    print(f"State error: {e.message}")
    if e.details:
        print(f"Details: {e.details}")
```

## Complete Example

Here's a complete example integrating all components:

```python
import asyncio
from uuid import uuid4
from typing import Dict, Any

from flowlib.ias.models.base import (
    DomainState, Event, EventType, Workflow, WorkflowStep
)
from flowlib.ias.core.state import StateManager
from flowlib.ias.core.events import EventBus
from flowlib.ias.core.validation import (
    ValidationManager, ValidationRegistry, Validator
)
from flowlib.ias.core.workflow import WorkflowEngine

# Initialize components
state_manager = StateManager()
event_bus = EventBus()
validation_registry = ValidationRegistry()
validation_manager = ValidationManager(validation_registry)
workflow_engine = WorkflowEngine(event_bus)

# Create domain agent
class DesignAgent:
    def __init__(
        self,
        state_manager: StateManager,
        event_bus: EventBus,
        validation_manager: ValidationManager
    ):
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.validation_manager = validation_manager
    
    async def create_design(self, data: Dict[str, Any]) -> UUID:
        state = await self.state_manager.create_state(
            domain="design",
            state_type=DesignState,
            data=data
        )
        return state.id
    
    async def validate_design(self, design_id: UUID) -> bool:
        state = await self.state_manager.get_state(design_id)
        results = await self.validation_manager.validate_state(state)
        return all(r.is_valid for r in results)

# Create workflow
async def run_design_workflow():
    # Create agent
    agent = DesignAgent(state_manager, event_bus, validation_manager)
    
    # Create initial design
    design_id = await agent.create_design({
        "name": "Project X",
        "version": "1.0",
        "description": "Example design"
    })
    
    # Create and start workflow
    workflow = Workflow(
        name="design_review",
        steps=[
            WorkflowStep(
                domain="design",
                action="validate_design",
                id=uuid4()
            )
        ]
    )
    
    await workflow_engine.start_workflow(workflow)
    
    # Wait for completion
    while True:
        status = workflow_engine.get_workflow_status(workflow.id)
        if not status:
            break
        await asyncio.sleep(1)

# Run example
if __name__ == "__main__":
    asyncio.run(run_design_workflow())
```

This example demonstrates:
- Setting up the IAS components
- Creating a domain agent
- Managing state and validation
- Creating and executing a workflow
- Proper error handling

For more examples and detailed API documentation, refer to the [API Reference](API.md). 