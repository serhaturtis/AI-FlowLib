"""Workflow orchestration system for the Integrated Agent System."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from ..models.base import Event, EventType, Workflow, WorkflowStep
from ..utils.errors import WorkflowError
from .events import EventBus

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """Engine for executing and managing workflows."""
    
    def __init__(
        self,
        event_bus: EventBus,
        max_concurrent_workflows: int = 10
    ):
        """Initialize workflow engine.
        
        Args:
            event_bus: Event bus for workflow events
            max_concurrent_workflows: Maximum number of concurrent workflows
        """
        self._event_bus = event_bus
        self._max_concurrent_workflows = max_concurrent_workflows
        self._active_workflows: Dict[UUID, Workflow] = {}
        self._step_results: Dict[UUID, Dict[str, Any]] = {}
        self._workflow_semaphore = asyncio.Semaphore(max_concurrent_workflows)
        
    async def start_workflow(
        self,
        workflow: Workflow
    ) -> None:
        """Start executing a workflow.
        
        Args:
            workflow: Workflow to execute
            
        Raises:
            WorkflowError: If workflow execution fails
        """
        if workflow.id in self._active_workflows:
            raise WorkflowError(f"Workflow {workflow.id} is already running")
            
        async with self._workflow_semaphore:
            try:
                self._active_workflows[workflow.id] = workflow
                self._step_results[workflow.id] = {}
                
                # Emit workflow started event
                await self._event_bus.publish(
                    Event(
                        id=uuid4(),
                        type=EventType.WORKFLOW_STARTED,
                        domain=workflow.domain,
                        timestamp=datetime.utcnow(),
                        data={
                            "workflow_id": workflow.id,
                            "workflow_name": workflow.name,
                            "workflow_version": workflow.version
                        }
                    )
                )
                
                # Execute workflow steps
                await self._execute_workflow_steps(workflow)
                
                # Emit workflow completed event
                await self._event_bus.publish(
                    Event(
                        id=uuid4(),
                        type=EventType.WORKFLOW_COMPLETED,
                        domain=workflow.domain,
                        timestamp=datetime.utcnow(),
                        data={
                            "workflow_id": workflow.id,
                            "workflow_name": workflow.name,
                            "workflow_version": workflow.version,
                            "results": self._step_results[workflow.id]
                        }
                    )
                )
                
            except Exception as e:
                # Emit workflow failed event
                await self._event_bus.publish(
                    Event(
                        id=uuid4(),
                        type=EventType.WORKFLOW_FAILED,
                        domain=workflow.domain,
                        timestamp=datetime.utcnow(),
                        data={
                            "workflow_id": workflow.id,
                            "workflow_name": workflow.name,
                            "workflow_version": workflow.version,
                            "error": str(e)
                        }
                    )
                )
                raise WorkflowError(f"Workflow {workflow.id} failed: {str(e)}") from e
                
            finally:
                # Clean up workflow state
                del self._active_workflows[workflow.id]
                del self._step_results[workflow.id]
                
    async def _execute_workflow_steps(
        self,
        workflow: Workflow
    ) -> None:
        """Execute steps in a workflow.
        
        Args:
            workflow: Workflow to execute steps for
            
        Raises:
            WorkflowError: If step execution fails
        """
        # Build dependency graph
        dependencies: Dict[str, Set[str]] = {}
        for step in workflow.steps:
            dependencies[step.id] = set(step.dependencies or [])
            
        # Track completed steps
        completed_steps: Set[str] = set()
        
        # Execute until all steps are complete
        while len(completed_steps) < len(workflow.steps):
            # Find ready steps
            ready_steps = [
                step for step in workflow.steps
                if step.id not in completed_steps
                and dependencies[step.id].issubset(completed_steps)
            ]
            
            if not ready_steps:
                raise WorkflowError(
                    f"Workflow {workflow.id} has cyclic dependencies"
                )
                
            # Execute ready steps in parallel
            tasks = [
                self._execute_workflow_step(workflow, step)
                for step in ready_steps
            ]
            await asyncio.gather(*tasks)
            
            # Mark steps as completed
            completed_steps.update(step.id for step in ready_steps)
            
    async def _execute_workflow_step(
        self,
        workflow: Workflow,
        step: WorkflowStep
    ) -> None:
        """Execute a single workflow step.
        
        Args:
            workflow: Parent workflow
            step: Step to execute
            
        Raises:
            WorkflowError: If step execution fails
        """
        try:
            # Emit step started event
            await self._event_bus.publish(
                Event(
                    id=uuid4(),
                    type=EventType.WORKFLOW_STEP_STARTED,
                    domain=workflow.domain,
                    timestamp=datetime.utcnow(),
                    data={
                        "workflow_id": workflow.id,
                        "step_id": step.id,
                        "step_action": step.action
                    }
                )
            )
            
            # Set up timeout
            timeout = step.timeout or workflow.timeout
            if timeout:
                try:
                    async with asyncio.timeout(timeout):
                        result = await self._execute_step_action(workflow, step)
                except asyncio.TimeoutError:
                    raise WorkflowError(
                        f"Step {step.id} timed out after {timeout} seconds"
                    )
            else:
                result = await self._execute_step_action(workflow, step)
                
            # Store step result
            self._step_results[workflow.id][step.id] = result
            
            # Emit step completed event
            await self._event_bus.publish(
                Event(
                    id=uuid4(),
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    domain=workflow.domain,
                    timestamp=datetime.utcnow(),
                    data={
                        "workflow_id": workflow.id,
                        "step_id": step.id,
                        "step_action": step.action,
                        "result": result
                    }
                )
            )
            
        except Exception as e:
            # Handle retry logic
            if step.retry_count and step.retry_count > 0:
                try:
                    for attempt in range(step.retry_count):
                        logger.warning(
                            f"Retrying step {step.id} (attempt {attempt + 1})"
                        )
                        try:
                            result = await self._execute_step_action(workflow, step)
                            self._step_results[workflow.id][step.id] = result
                            return
                        except Exception as retry_e:
                            if attempt == step.retry_count - 1:
                                raise retry_e
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as retry_e:
                    e = retry_e
                    
            # Emit step failed event
            await self._event_bus.publish(
                Event(
                    id=uuid4(),
                    type=EventType.WORKFLOW_STEP_FAILED,
                    domain=workflow.domain,
                    timestamp=datetime.utcnow(),
                    data={
                        "workflow_id": workflow.id,
                        "step_id": step.id,
                        "step_action": step.action,
                        "error": str(e)
                    }
                )
            )
            raise WorkflowError(f"Step {step.id} failed: {str(e)}") from e
            
    async def _execute_step_action(
        self,
        workflow: Workflow,
        step: WorkflowStep
    ) -> Any:
        """Execute the action for a workflow step.
        
        This method should be implemented by subclasses to:
        1. Parse the step action
        2. Execute the appropriate domain-specific logic
        3. Return the result
        
        Args:
            workflow: Parent workflow
            step: Step to execute
            
        Returns:
            Result of the step action
            
        Raises:
            WorkflowError: If step action execution fails
        """
        raise NotImplementedError(
            "WorkflowEngine._execute_step_action must be implemented by subclasses"
        )
        
    def get_workflow_status(
        self,
        workflow_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Get the current status of a workflow.
        
        Args:
            workflow_id: ID of workflow to get status for
            
        Returns:
            Workflow status information or None if not found
        """
        if workflow_id not in self._active_workflows:
            return None
            
        workflow = self._active_workflows[workflow_id]
        return {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "workflow_version": workflow.version,
            "domain": workflow.domain,
            "start_time": workflow.created_at,
            "step_results": self._step_results[workflow_id].copy()
        }
        
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all currently active workflows.
        
        Returns:
            List of active workflow status information
        """
        return [
            self.get_workflow_status(workflow_id)
            for workflow_id in self._active_workflows
        ] 