"""
Base engine implementation for the agent system.

This module provides the central execution engine that coordinates
planning, execution, and reflection cycles.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from ..core.base import BaseComponent
from ..core.errors import ExecutionError, PlanningError, ReflectionError, NotInitializedError, StatePersistenceError
from .interfaces import EngineInterface
from ..models.config import EngineConfig
from ..models.state import AgentState
from ...flows.results import FlowResult
from ...flows.registry import stage_registry
from ..planning.models import PlanningResult
from ..reflection.models import ReflectionResult
from ..planning.interfaces import PlanningInterface
from ..memory.interfaces import MemoryInterface
from ..reflection.interfaces import ReflectionInterface
from ...core.context import Context

logger = logging.getLogger(__name__)


class AgentEngine(BaseComponent, EngineInterface):
    """Execution engine for the agent system.
    
    Responsibilities:
    1. Coordinating the planning-execution-reflection cycle
    2. Managing flow execution
    3. Handling error recovery
    4. Managing state persistence
    """
    
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        memory: Optional[MemoryInterface] = None,
        planner: Optional[PlanningInterface] = None,
        reflection: Optional[ReflectionInterface] = None,
        name: str = "engine"
    ):
        """Initialize the engine.
        
        Args:
            config: Engine configuration
            memory: Memory component implementing MemoryInterface
            planner: Planner component implementing PlanningInterface
            reflection: Reflection component implementing ReflectionInterface
            name: Component name
        """
        super().__init__(name)
        
        # Configuration
        self._config = config or EngineConfig()
        
        # Components (these should be provided by AgentCore)
        self._memory = memory
        self._planner = planner
        self._reflection = reflection
        
        # Execution state
        self._iteration = 0
        self._last_execution_result = None
    
    async def _initialize_impl(self) -> None:
        """
        Initialize the agent engine.
        
        Raises:
            ExecutionError: If required components are missing
        """
        # Check for required components
        if not self._memory:
            raise ExecutionError("Memory component is required")
        
        if not self._planner:
            raise ExecutionError("Planner component is required")
        
        if not self._reflection:
            raise ExecutionError("Reflection component is required")
        
        # Reset iteration counter
        self._iteration = 0
        
        logger.info(f"Initialized agent engine with {self._config}")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the engine."""
        logger.info("Shutting down agent engine")
    
    async def execute_cycle(
        self,
        state: AgentState,
        memory_context: str = "agent",
        no_flow_is_error: bool = False
    ) -> bool:
        """Execute a single execution cycle of the agent.
        
        An execution cycle involves:
        1. Planning the next action
        2. Executing the selected flow
        3. Reflecting on the results
        4. Updating the agent state
        
        Args:
            state: Agent state to use for this cycle
            memory_context: Parent memory context
            no_flow_is_error: Whether to treat no flow selection as an error
            
        Returns:
            True if execution should continue, False if execution is complete
            
        Raises:
            ExecutionError: If execution fails for any reason
        """
        try:
            # Increment iteration counter
            self._iteration += 1
            
            # Create memory context for this cycle
            cycle_context = await self._create_cycle_context(memory_context, self._iteration)
            
            # Update cycles counter in state
            state.increment_cycle()
            
            # Log cycle start
            logger.info(f"Starting execution cycle {state.cycles}")
            
            # Plan next action
            result = await self._plan_next_action(state, cycle_context)            
            # Get selected flow
            selected_flow = result.selected_flow
            
            # Check if a flow was selected
            if not selected_flow or selected_flow == "none":
                if no_flow_is_error:
                    state.add_error("No flow selected by planner")
                    logger.warning("No flow selected by planner")
                    await self._save_state(state)
                    return False
                else:
                    # No flow needed, just continue
                    logger.info("No flow selected by planner, continuing to next cycle")
                    return True
            
            # Generate inputs for the flow
            inputs = await self._generate_inputs(state, cycle_context, selected_flow, result)
            
            # Execute the flow
            flow_result = await self.execute_flow(selected_flow, inputs, state)
            
            # Reflect on results
            reflection_result = await self._reflect_on_results(state, cycle_context, selected_flow, inputs, flow_result)
            
            # Check if reflection indicated completion
            if reflection_result.is_complete:
                state.set_complete(reflection_result.completion_reason)
                state.progress = reflection_result.progress
                logger.info(f"Reflection indicated task completion: {state.completion_reason}")
                await self._save_state(state)
                return False
            
            # Update progress
            state.progress = reflection_result.progress
            
            # Check if we'll reach max iterations on the next cycle
            if self._iteration >= self._config.max_iterations:
                state.set_complete("Maximum iterations reached")
                logger.info(f"Reached max iterations ({self._config.max_iterations})")
                await self._save_state(state)
                return False
            
            # Save state after successful cycle
            await self._save_state(state)
            
            # Continue with more cycles
            return True
            
        except Exception as e:
            # Record the error in the state
            error_message = f"Error during execution cycle {state.cycles}: {str(e)}"
            state.add_error(error_message)
            logger.error(error_message, exc_info=True)
            
            # Save state after error
            await self._save_state(state)
            
            # Determine whether to continue
            should_continue = not self._config.stop_on_error
            
            if should_continue:
                logger.info(f"Continuing execution despite error (stop_on_error={self._config.stop_on_error})")
            else:
                logger.info(f"Stopping execution due to error (stop_on_error={self._config.stop_on_error})")
                
            return should_continue
    
    async def _save_state(self, state: AgentState) -> None:
        """Attempt to save the agent state using parent's state persister.
        
        Args:
            state: Agent state to save
        """
        parent = self.parent
        if parent and hasattr(parent, "save_state") and callable(parent.save_state):
            try:
                await parent.save_state()
                logger.debug(f"Successfully saved agent state for task: {state.task_id}")
            except StatePersistenceError as e:
                logger.warning(f"Failed to save agent state for task {state.task_id}: {str(e)}")
            except Exception as e:
                logger.warning(
                    f"Unexpected error saving agent state for task {state.task_id}: {str(e)}. "
                    f"Current state has {len(state.execution_history)} execution history entries and "
                    f"progress: {state.progress}%"
                )
    
    def _get_serializable_result(self, result: FlowResult) -> Dict[str, Any]:
        """Convert a FlowResult to a serializable dictionary.
        
        Args:
            result: Flow result to convert
            
        Returns:
            Serializable dictionary
        """
        if not result:
            return {"status": "unknown", "data": None}
        
        # Try to convert to dict using model_dump (the standard method)
        if hasattr(result, "model_dump") and callable(result.model_dump):
            return result.model_dump()
        
        # Fallback for non-Pydantic objects
        if hasattr(result, "to_dict") and callable(result.to_dict):
            return result.to_dict()
        elif hasattr(result, "__dict__"):
            return result.__dict__
        else:
            return {"status": str(result.status), "data": result.data}
    
    async def _plan_next_action(self, state: AgentState, context: str) -> PlanningResult:
        """Plan the next action to take.
        
        Args:
            state: Current agent state
            context: Memory context for this cycle
            
        Returns:
            PlanningResult with selected flow and rationale
        
        Raises:
            PlanningError: If planning fails
        """
        try:
            logger.debug("Planning next action")
            
            # Create execution context from state
            from ..planning.models import ExecutionContext, TaskState, MessageHistory, ErrorLog
            
            # Create task state
            task_state = TaskState(
                task_id=state.task_id,
                task_description=state.task_description,
                cycle=state.cycles,
                progress=state.progress,
                is_complete=state.is_complete,
                completion_reason=state.completion_reason,
                last_updated=datetime.now()
            )
            
            # Create message history
            messages = MessageHistory()
            for msg in state.user_messages:
                messages.user_messages.append(MessageHistory.Message(content=msg))
            for msg in state.system_messages:
                messages.system_messages.append(MessageHistory.Message(content=msg))
            
            # Create error log
            errors = ErrorLog()
            for err in state.errors:
                errors.errors.append(ErrorLog.ErrorEntry(message=err, source="agent"))
            
            # Create execution context
            execution_context = ExecutionContext(
                state=task_state,
                messages=messages,
                errors=errors
            )
            
            # Call planner using standard interface
            planning_result = await self._planner.plan(context=execution_context)
            
            logger.debug(f"Planning result: {planning_result}")
            
            # Store planning result in memory using interface
            from ..memory.models import MemoryStoreRequest
            store_request = MemoryStoreRequest(
                key="planning_result",
                value=planning_result.model_dump(),
                context=context,
                importance=0.7,
                metadata={
                    "type": "planning",
                    "timestamp": datetime.now().isoformat()
                }
            )
            await self._memory.store_with_model(store_request)
            
            # Return the PlanningResult
            return planning_result
            
        except Exception as e:
            raise PlanningError(
                message=f"Planning failed: {str(e)}",
                planning_type="planning",
                cause=e
            )
    
    async def _generate_inputs(
        self,
        state: AgentState,
        context: str,
        flow_name: str,
        planning_result: PlanningResult
    ) -> Dict[str, Any]:
        """Generate inputs for a flow.
        
        Args:
            state: Current agent state
            context: Memory context for this cycle
            flow_name: Name of the flow to generate inputs for
            planning_result: Result from the planning phase
            
        Returns:
            Generated inputs for the flow
            
        Raises:
            PlanningError: If input generation fails
        """
        try:
            logger.debug(f"Generating inputs for flow '{flow_name}'")
            
            # Get flow from core
            parent = self.parent
            flow = None
            
            if parent and hasattr(parent, "flows"):
                flow = parent.flows.get(flow_name)
                
            # Call input generator with the interface method
            inputs = await self._planner.generate_inputs(
                state=state,
                flow_name=flow_name,
                planning_result=planning_result.model_dump(),
                memory_context=context,
                flow=flow
            )
            
            logger.debug(f"Generated inputs: {inputs}")
            
            # Store inputs in memory using interface
            from ..memory.models import MemoryStoreRequest
            store_request = MemoryStoreRequest(
                key="flow_inputs",
                value=inputs,
                context=context,
                importance=0.7,
                metadata={
                    "type": "inputs",
                    "flow_name": flow_name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            await self._memory.store_with_model(store_request)
            
            return inputs
            
        except Exception as e:
            raise PlanningError(
                message=f"Input generation failed for flow '{flow_name}': {str(e)}",
                planning_type="input_generation",
                flow=flow_name,
                cause=e
            )
    
    async def execute_flow(
        self,
        flow_name: str,
        inputs: Any,
        state: Optional[AgentState] = None
    ) -> FlowResult:
        """
        Execute a flow with given inputs.
        
        Args:
            flow_name: Name of the flow to execute
            inputs: A Pydantic model instance matching the flow's input model
            state: Current agent state
            
        Returns:
            Result from flow execution
            
        Raises:
            ExecutionError: If flow execution fails or is not found
            NotInitializedError: If engine is not initialized
            ValueError: If inputs is not the correct Pydantic model
        """
        self._check_initialized()
        
        start_time = time.time()
        logger.debug(f"Executing flow '{flow_name}'")
        
        # Get flow from parent
        if not self.parent:
            raise ExecutionError("Cannot execute flow without parent agent")
            
        if not hasattr(self.parent, "flows"):
            raise ExecutionError("Parent agent does not have a flows registry")
            
        flow = self.parent.flows.get(flow_name)
        if not flow:
            raise ExecutionError(f"Flow '{flow_name}' not found in agent's flow registry")
        
        try:
            # Get flow metadata to determine input model
            flow_metadata = stage_registry.get_flow_metadata(flow_name)
            if not flow_metadata:
                raise ExecutionError(f"Flow '{flow_name}' has no metadata in registry")
                
            # Get the input model class from metadata
            input_model_cls = flow_metadata.input_model
            
            # Convert inputs to proper Pydantic model instance if it's a dict
            if isinstance(inputs, dict):
                try:
                    inputs = input_model_cls(**inputs)
                except Exception as e:
                    raise ExecutionError(
                        message=f"Failed to convert inputs to {input_model_cls.__name__}: {str(e)}",
                        flow=flow_name,
                        cause=e
                    ) from e
            elif not isinstance(inputs, input_model_cls):
                raise ExecutionError(
                    message=f"Inputs must be an instance of {input_model_cls.__name__}, got {type(inputs).__name__}",
                    flow=flow_name
                )
                
            # Create a Context object to wrap the inputs
            flow_context = Context(data=inputs)
            
            # Execute flow with proper Context object
            result = await flow.execute(flow_context)
            
            # Update state with result
            if state:
                elapsed_time = time.time() - start_time
                self._update_state_with_result(
                    state=state,
                    flow_name=flow_name,
                    inputs=inputs.model_dump() if hasattr(inputs, "model_dump") else inputs,
                    flow_result=result,
                    elapsed_time=elapsed_time
                )
                
                # Save state after flow execution
                await self._save_state(state)
            
            # Store result
            self._last_execution_result = result
            
            # Log success
            elapsed_time = time.time() - start_time
            logger.info(
                f"Flow '{flow_name}' completed successfully in {elapsed_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Flow '{flow_name}' failed after {elapsed_time:.2f}s: {str(e)}")
            
            # Update state with error and save
            if state:
                state.add_error(f"Flow '{flow_name}' execution failed: {str(e)}")
                await self._save_state(state)
            
            if not isinstance(e, ExecutionError):
                raise ExecutionError(
                    message=f"Flow execution failed: {str(e)}",
                    flow=flow_name,
                    cause=e
                ) from e
            else:
                raise e from e.__cause__ if e.__cause__ else e
    
    def _update_state_with_result(
        self, 
        state: AgentState, 
        flow_name: str, 
        inputs: Dict[str, Any], 
        flow_result: FlowResult,
        elapsed_time: float
    ) -> None:
        """Update agent state with flow execution result.
        
        Args:
            state: Agent state to update
            flow_name: Name of the executed flow
            inputs: Inputs used for the flow
            flow_result: Result from flow execution
            elapsed_time: Execution time in seconds
            
        Raises:
            ExecutionError: If state object doesn't have required methods
        """
        try:
            # Serialize result for state storage
            serialized_result = self._get_serializable_result(flow_result)
            
            # Update state with standard method
            state.add_to_history(
                flow_name=flow_name,
                inputs=inputs,
                result=serialized_result,
                elapsed_time=elapsed_time
            )
            
        except Exception as e:
            # Wrap exceptions
            raise ExecutionError(
                message=f"Failed to update state with execution result: {str(e)}",
                flow=flow_name,
                cause=e
            ) from e
    
    async def _create_cycle_context(self, memory_context: str, iteration: int) -> str:
        """Create a memory context for this execution cycle.
        
        Args:
            memory_context: Parent memory context
            iteration: Current iteration number
            
        Returns:
            Created context path
        """
        # Create context using memory interface
        context_name = f"cycle_{iteration}"
        context_path = self._memory.create_context(
            context_name=context_name,
            parent=memory_context,
            metadata={
                "iteration": iteration,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.debug(f"Created cycle context: {context_path}")
        return context_path
    
    async def _reflect_on_results(
        self,
        state: AgentState,
        context: str,
        flow_name: str,
        flow_inputs: Dict[str, Any],
        flow_result: FlowResult
    ) -> ReflectionResult:
        """Reflect on the results of flow execution.
        
        Args:
            state: Current agent state
            context: Memory context for this cycle
            flow_name: Name of the executed flow
            flow_inputs: Inputs that were used
            flow_result: Result from flow execution
            
        Returns:
            Reflection result with analysis and state updates
            
        Raises:
            ReflectionError: If reflection fails
        """
        try:
            logger.debug("Reflecting on results")
            
            # Call reflection component using interface
            reflection_result = await self._reflection.reflect(
                state=state,
                flow_name=flow_name,
                flow_inputs=flow_inputs,
                flow_result=flow_result,
                memory_context=context
            )
            
            logger.debug(f"Reflection result: {reflection_result}")
            
            # Store reflection result in memory using interface
            from ..memory.models import MemoryStoreRequest
            store_request = MemoryStoreRequest(
                key="reflection_result",
                value=reflection_result,
                context=context,
                importance=0.7,
                metadata={
                    "type": "reflection",
                    "flow_name": flow_name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            await self._memory.store_with_model(store_request)
            
            return reflection_result
            
        except Exception as e:
            raise ReflectionError(
                message=f"Reflection failed: {str(e)}",
                flow=flow_name,
                cause=e
            )
    
    def _check_initialized(self) -> None:
        """Check if the engine is initialized.
        
        Raises:
            NotInitializedError: If the engine is not initialized
        """
        if not self._initialized:
            raise NotInitializedError(
                component_name=self._name,
                operation="execute"
            ) 