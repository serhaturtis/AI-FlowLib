"""
Agent core implementation.

This module defines the central component for the agent system that coordinates
all other components, manages state, and provides the high-level API.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ...providers.constants import ProviderType

from .base import BaseComponent
from .errors import ConfigurationError, ExecutionError, StatePersistenceError, NotInitializedError, ComponentError
from ..engine.base import AgentEngine
from ..planning.base import AgentPlanner
from ..reflection.base import AgentReflection
from ..learn.flows import BaseLearningFlow
from ..models.config import AgentConfig
from ..models.state import AgentState
from ..persistence.base import BaseStatePersister
from ...flows.base import Flow
from ...flows.metadata import FlowMetadata
from ...flows.registry import stage_registry, StageRegistry
from ...flows.results import FlowResult
from ..discovery.flow_discovery import FlowDiscovery
from ..memory.models import MemoryStoreRequest
from ..memory.models import MemoryRetrieveRequest
from ..memory.models import MemorySearchRequest

from ..learn.entity_extraction import EntityExtractionFlow
from ..learn.relationship_learning import RelationshipLearningFlow
from ..learn.knowledge_integration import KnowledgeIntegrationFlow
from ..learn.concept_formation import ConceptFormationFlow

from flowlib.agent.learn.models import LearningRequest, LearningResponse, LearningStrategy, Entity, Relationship

from ..persistence.factory import create_state_persister

# Import the new memory components
from ..memory.comprehensive import ComprehensiveMemory
from ..memory.working import WorkingMemory
from ..memory.vector import VectorMemory
from ..memory.knowledge import KnowledgeBaseMemory

# Import provider registry and provider types
from ...providers.registry import provider_registry
from ...providers.constants import ProviderType

# Import create_and_initialize_provider
from ...providers.factory import create_and_initialize_provider

# Import Neo4jProvider
from ...providers.graph.neo4j_provider import Neo4jProvider

# Configure logger
logger = logging.getLogger(__name__)


class AgentCore(BaseComponent):
    """Central component for the agent system.
    
    Attributes:
        config: Agent configuration
        state: Agent state
        flows: Dictionary of flows registered with the agent
        flow_descriptions: Dictionary of flow descriptions for planning
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
        task_description: str = "",
        name: str = None,
        state_persister: Optional[BaseStatePersister] = None
    ):
        """Initialize the agent.
        
        Args:
            config: Agent configuration
            task_description: Task description for the agent
            name: Name for the agent
            state_persister: State persister for agent
        """
        # Set up configuration first
        self.config = self._prepare_config(config)

        # Store persona (already validated in _prepare_config or AgentConfig model)
        self.persona = self.config.persona

        # Set name from config or override
        super().__init__(name or self.config.name)
        
        # Store initial task description, state will be created/loaded in initialize
        self._initial_task_description = task_description
        self.state = None # Initialize state as None
        
        # Set up components
        self._planner = None
        self._reflection = None
        self._memory = None
        self._engine = None
        self._llm_provider = None
        
        # Set up state persister
        self._state_persister = state_persister
        
        # Store flows
        self.flows = {}
        
        # Store last result
        self.last_result = None

    
    def _prepare_config(self, config: Optional[Union[Dict[str, Any], AgentConfig]] = None) -> AgentConfig:
        """Prepare configuration for the agent using a simplified approach.
        
        Args:
            config: Configuration dictionary or AgentConfig instance
            
        Returns:
            Prepared AgentConfig instance
        """
        prepared_config: AgentConfig

        # If config is provided as dict, convert to AgentConfig
        if isinstance(config, dict):
            prepared_config = AgentConfig(**config)
        # If config is already an AgentConfig, use it
        elif isinstance(config, AgentConfig):
            prepared_config = config
        # If no config is provided, use defaults
        else:
            # Need a default name and persona if no config is given *at all*
            # This case might need more robust handling depending on required fields
            default_name = getattr(self, "_name", "default_agent")
            default_persona = "Default helpful assistant"
            logger.warning(f"No config provided, creating default AgentConfig for {default_name}")
            prepared_config = AgentConfig(
                name=default_name,
                persona=default_persona 
            )

        # Ensure critical fields like name and persona are present after potential defaults
        if not hasattr(prepared_config, 'name') or not prepared_config.name:
             prepared_config.name = "default_agent"
        if not hasattr(prepared_config, 'persona') or not prepared_config.persona:
             # This should ideally be caught by Pydantic if made mandatory non-optional
             # but adding a fallback just in case.
             prepared_config.persona = "Default helpful assistant"

        return prepared_config
    
    @property
    def llm_provider(self) -> Optional[Any]:
        """Get the model provider.
        
        Returns:
            Model provider component
        """
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, provider: Any):
        """Set the model provider.
        
        Args:
            provider: Model provider component
        """
        self._llm_provider = provider
    
    async def _initialize_impl(self) -> None:
        """Initialize AgentCore, including state loading/creation and components."""
        try:
            # Set up state persistence if configured and not already provided
            if self.config.state_config and not self._state_persister:
                self._state_persister = create_state_persister(
                    persister_type=self.config.state_config.persistence_type,
                    **self.config.state_config.model_dump(exclude={"persistence_type"})
                )
            
            state_loaded_or_created = False
            # Initialize the state persister if provided
            if self._state_persister:
                await self._state_persister.initialize()
                
                # Determine if we should attempt to load state
                should_auto_load = self.config.state_config and self.config.state_config.auto_load
                task_id_to_load = self.config.task_id # Get task_id from config
                
                if should_auto_load and task_id_to_load:
                    logger.info(f"Attempting to load state for task_id: {task_id_to_load}")
                    try:
                        await self.load_state(task_id=task_id_to_load) # Pass specific task_id
                        state_loaded_or_created = True # Flag that state is now set
                    except StatePersistenceError as e:
                        # Log the error but continue to create a new state if loading failed
                        logger.warning(f"Failed to load state for task {task_id_to_load}, creating new state: {e}")
                else:
                    logger.info("Auto-load disabled or no task_id provided, will create new state if needed.")
            
            # If state wasn't loaded, create a new one now
            if not state_loaded_or_created:
                logger.info(f"Creating new agent state with description: '{self._initial_task_description}'")
                self.state = AgentState(task_description=self._initial_task_description)
                # The new state will have a new task_id generated by AgentState
                state_loaded_or_created = True

            # Ensure state is actually set before proceeding (should always be true here)
            if not self.state:
                 raise ConfigurationError("Agent state was not created or loaded during initialization.")

            # --- Instantiate Memory Components --- 
            if not self._memory:
                mem_config = self.config.memory_config # ComprehensiveMemoryConfig
                
                # --- Create Providers (fetch or create & initialize) ---
                # Embedding Provider (created via factory by registry.get)
                embedding_provider = await provider_registry.get(
                    ProviderType.EMBEDDING,
                    mem_config.vector_memory.embedding_provider_name
                )

                # Vector Provider (created via factory by registry.get)
                vector_provider = await provider_registry.get(
                    ProviderType.VECTOR_DB,
                    mem_config.vector_memory.vector_provider_name
                )
                # Note: Vector provider init might depend on embedding provider,
                # registry.get should handle initialization order if factories are used.
                # If providers are registered directly, ensure init order is correct.

                # Graph Provider (create directly using settings)
                # Import factory and constants
                from ...providers.factory import create_provider 
                # Import specific provider class needed (adjust if supporting others)
                from ...providers.graph.neo4j_provider import Neo4jProvider 

                # Ensure we have settings
                graph_provider_name = mem_config.knowledge_memory.graph_provider_name
                graph_settings = mem_config.knowledge_memory.provider_settings
                if not graph_settings:
                    raise ConfigurationError("provider_settings missing for knowledge memory")

                # Create and initialize the graph provider instance
                graph_provider = await create_and_initialize_provider(
                    provider_type=ProviderType.GRAPH_DB, 
                    name=graph_provider_name, 
                    implementation=graph_provider_name, # Assume name matches implementation key
                    register=True, # Register instance
                    **graph_settings
                )
                # -------------------------------------------------------
                
                # Instantiate specialized memories, passing provider instances
                working_mem = WorkingMemory(
                    default_ttl_seconds=mem_config.working_memory.default_ttl_seconds
                )
                vector_mem = VectorMemory(
                    # VectorMemory now gets provider from registry
                    provider_name=vector_provider.name, 
                    embedding_provider_name=embedding_provider.name
                )
                knowledge_mem = KnowledgeBaseMemory(
                    graph_provider=graph_provider # Pass instance
                )
                
                # Instantiate comprehensive memory
                self._memory = ComprehensiveMemory(
                    vector_memory=vector_mem,
                    knowledge_memory=knowledge_mem,
                    working_memory=working_mem,
                    fusion_provider_name=mem_config.fusion_provider_name,
                    fusion_model_name=mem_config.fusion_model_name
                )
                self._memory.set_parent(self) # Register for lifecycle management
            # -------------------------------------
            
            self._planner = self._planner or AgentPlanner(config=self.config.planner_config)
            self._planner.set_parent(self)
            
            self._reflection = self._reflection or AgentReflection(config=self.config.reflection_config)
            self._reflection.set_parent(self)
            
            self._engine = self._engine or AgentEngine(
                config=self.config.engine_config,
                memory=self._memory,
                planner=self._planner,
                reflection=self._reflection
            )
            self._engine.set_parent(self)
            
            # Initialize components
            await self._memory.initialize()
            await self._planner.initialize()
            await self._reflection.initialize()
            await self._engine.initialize()
            
            # Discover and register flows using stage_registry
            await self._discover_flows()
                
        except Exception as e:
            logger.error(f"Failed to initialize AgentCore: {str(e)}", exc_info=True)
            raise ConfigurationError(
                message=f"Failed to initialize AgentCore: {str(e)}",
                operation="initialize",
                cause=e
            )
    
    async def _shutdown_impl(self) -> None:
        """Shutdown AgentCore and its components."""
        try:
            # Shutdown components in reverse order
            if self._engine and self._engine.initialized:
                await self._engine.shutdown()
                
            if self._reflection and self._reflection.initialized:
                await self._reflection.shutdown()
                
            if self._planner and self._planner.initialized:
                await self._planner.shutdown()
                
            if self._memory and self._memory.initialized:
                await self._memory.shutdown()
                
            # Save state if persistence is enabled and configured
            if self._state_persister and self.config.state_config and self.config.state_config.auto_save:
                await self.save_state()
                
            # Shutdown state persister
            if self._state_persister:
                await self._state_persister.shutdown()
                
        except Exception as e:
            logger.error(f"Error during AgentCore shutdown: {str(e)}")
            raise ComponentError(
                message=f"Failed to shutdown AgentCore: {str(e)}",
                component_name=self.name,
                operation="shutdown",
                cause=e
            )
    
    async def save_state(self) -> None:
        """Save the current agent state.
        
        Raises:
            StatePersistenceError: If state persister is not configured or persistence fails
        """
        if not self._state_persister:
            raise StatePersistenceError(
                message="No state persister configured",
                operation="save",
                task_id=self.state.task_id
            )
            
        try:
            # Save state with metadata
            metadata = {
                "task_id": self.state.task_id,
                "task_description": self.state.task_description,
                "is_complete": str(self.state.is_complete),
                "completion_reason": self.state.completion_reason or "",
                "progress": str(self.state.progress),
                "cycles": str(self.state.cycles),
                "errors": str(len(self.state.errors)),
                "timestamp": datetime.now().isoformat()
            }
            
            await self._state_persister.save_state(
                state=self.state,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Failed to save agent state: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="save",
                task_id=self.state.task_id,
                cause=e
            )
    
    async def load_state(self, task_id: str) -> None:
        """Load agent state.
        
        Args:
            task_id: The specific task ID to load state for.
            
        Raises:
            StatePersistenceError: If state persister is not configured, state is not found, or persistence fails
        """
        if not self._state_persister:
            raise StatePersistenceError(
                message="No state persister configured",
                operation="load"
            )
            
        try:
            # Use provided task ID or current one
            if not task_id:
                raise StatePersistenceError(
                    message="No task ID provided for state loading",
                    operation="load"
                )
            
            # Load state
            loaded_state = await self._state_persister.load_state(task_id=task_id)
            if not loaded_state:
                raise StatePersistenceError(
                    message=f"No state found for task ID: {task_id}",
                    operation="load",
                    task_id=task_id
                )
            
            # Update current state
            self.state = loaded_state
            logger.info(f"Successfully loaded state for task: {task_id}")
            
        except Exception as e:
            if isinstance(e, StatePersistenceError):
                raise
                
            error_msg = f"Failed to load agent state: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="load",
                task_id=task_id,
                cause=e
            )
    
    async def delete_state(self, task_id: Optional[str] = None) -> None:
        """Delete agent state.
        
        Args:
            task_id: Optional task ID to delete. If not provided, uses current task ID.
            
        Raises:
            StatePersistenceError: If state persister is not configured or persistence fails
        """
        if not self._state_persister:
            raise StatePersistenceError(
                message="No state persister configured",
                operation="delete"
            )
            
        try:
            # Use provided task ID or current one
            target_task_id = task_id or self.state.task_id
            if not target_task_id:
                raise StatePersistenceError(
                    message="No task ID provided for state deletion",
                    operation="delete"
                )
            
            # Delete state
            deleted = await self._state_persister.delete_state(task_id=target_task_id)
            if not deleted:
                raise StatePersistenceError(
                    message=f"Failed to delete state with task ID: {target_task_id}",
                    operation="delete",
                    task_id=target_task_id
                )
            
        except Exception as e:
            if isinstance(e, StatePersistenceError):
                raise
                
            error_msg = f"Failed to delete agent state: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="delete",
                task_id=task_id,
                cause=e
            )
    
    async def list_states(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """List available agent states.
        
        Args:
            filter_criteria: Optional criteria to filter states by
            
        Returns:
            List of state metadata dictionaries
            
        Raises:
            StatePersistenceError: If state persister is not configured or persistence fails
        """
        if not self._state_persister:
            raise StatePersistenceError(
                message="No state persister configured",
                operation="list"
            )
            
        try:
            return await self._state_persister.list_states(filter_criteria=filter_criteria)
            
        except Exception as e:
            error_msg = f"Failed to list agent states: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="list",
                cause=e
            )
    
    async def _discover_flows(self) -> None:
        """Discover and register flows from the stage registry.
        
        Raises:
            ConfigurationError: If flow discovery fails
        """
        if not stage_registry:
            raise ConfigurationError(
                message="Stage registry is not available for flow discovery",
                operation="discover_flows"
            )
        
        # Create flow discovery
        discovery = FlowDiscovery()
        
        try:
            # Get agent-compatible flows from registry
            flow_classes = discovery.discover_agent_flows()
            
            # Track registration results
            registered_flows = []
            failed_flows = []
            
            # Register flows
            for flow_class in flow_classes:
                try:
                    flow_name = getattr(flow_class, "__name__", "unknown")
                    
                    # Create flow instance according to its API
                    if hasattr(flow_class, 'create') and callable(flow_class.create):
                        instance = flow_class.create()
                    else:
                        instance = flow_class()
                    
                    # Register the flow instance
                    self.register_flow(instance)
                    registered_flows.append(flow_name)
                    
                except Exception as e:
                    flow_name = getattr(flow_class, "__name__", "unknown")
                    logger.warning(f"Failed to register flow {flow_name}: {str(e)}")
                    failed_flows.append(flow_name)
            
            # Log registration results
            if registered_flows:
                logger.info(f"Registered {len(registered_flows)} agent-compatible flows")
                
            if failed_flows:
                logger.warning(f"Failed to register {len(failed_flows)} flows: {', '.join(failed_flows)}")
                
        except Exception as e:
            error_msg = f"Flow discovery failed: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(
                message=error_msg,
                operation="discover_flows",
                cause=e
            )
    
    def register_flow(self, flow: Any) -> None:
        """Register a flow with the agent.
        
        Args:
            flow: Flow instance to register
            
        Raises:
            ValueError: If flow is invalid or missing required attributes
            ConfigurationError: If flow registration fails
        """
        # Get flow name
        if not hasattr(flow, "name"):
            if hasattr(flow, "__class__") and hasattr(flow.__class__, "__name__"):
                flow_name = flow.__class__.__name__
            else:
                raise ValueError("Flow must have a 'name' attribute or be a class with __name__")
        else:
            flow_name = flow.name
        
        # Store flow instance in the agent's flow dict for easy access
        self.flows[flow_name] = flow
        
        # Register the flow with the stage_registry for discovery
        if not stage_registry:
            raise ConfigurationError(
                message="Stage registry not available for flow registration",
                operation="register_flow",
                flow_name=flow_name
            )
            
        # Register flow with stage_registry
        stage_registry.register_flow(flow_name, flow)
        logger.debug(f"Registered flow: {flow_name} with stage_registry")
    
    def get_flow_registry(self) -> StageRegistry:
        """Get the flow registry with all registered flows.
        
        Returns:
            Stage registry containing flow instances and metadata
        """
        return stage_registry
    
    async def register_flow_async(self, flow: Flow) -> None:
        """Register a flow with the agent (async version).
        
        Args:
            flow: Flow instance to register
            
        Raises:
            ValueError: If flow is invalid or missing required attributes
            ConfigurationError: If flow registration fails
        """
        # Use the synchronous version directly
        self.register_flow(flow)
    
    def unregister_flow(self, flow_name: str) -> None:
        """Unregister a flow from the agent.
        
        Args:
            flow_name: Name of the flow to unregister
            
        Raises:
            ValueError: If flow is not registered
        """
        # Check if flow exists in the agent's flows
        if flow_name not in self.flows:
            raise ValueError(f"Flow '{flow_name}' is not registered with this agent")
            
        # Remove from agent's flows dictionary
        del self.flows[flow_name]
        
        # Unregister from stage_registry if available
        if stage_registry:
            stage_registry.unregister_flow(flow_name)
            logger.debug(f"Unregistered flow: {flow_name} from stage_registry")
        else:
            logger.debug(f"Removed flow: {flow_name} from agent (stage_registry not available)")
    
    def get_flow_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all registered flows.
        
        Returns:
            List of flow metadata dictionaries
            
        Raises:
            ConfigurationError: If flow metadata cannot be accessed
        """
        
        if not stage_registry:
            raise ConfigurationError(
                message="Stage registry not available for flow descriptions",
                operation="get_flow_descriptions"
            )
            
        descriptions = []
        
        # Process each flow
        for flow_name, flow_instance in self.flows.items():
            try:
                # Create metadata for this flow
                metadata = FlowMetadata.from_flow(flow_instance, flow_name)
                descriptions.append(metadata.dict())
            except Exception as e:
                logger.warning(f"Failed to get metadata for flow {flow_name}: {e}")
        
        return descriptions
    
    # Methods that delegate to components
    async def store_memory(self, key: str, value: Any, **kwargs) -> None:
        """Store a value in memory.
        
        Args:
            key: Memory key
            value: Value to store
            **kwargs: Additional arguments for memory storage
            
        Raises:
            NotInitializedError: If the agent is not initialized
            MemoryError: If memory storage fails
        """
        if not self.initialized:
            raise NotInitializedError(
                component_name=self.name,
                operation="store_memory"
            )
        
        request = MemoryStoreRequest(
            key=key,
            value=value
        )
        
        # Add any additional parameters
        for k, v in kwargs.items():
            if hasattr(request, k):
                setattr(request, k, v)
        
        # Use model-based implementation
        await self._memory.store_with_model(request)
    
    async def retrieve_memory(self, key: str, **kwargs) -> Any:
        """Retrieve a value from memory.
        
        Args:
            key: Memory key
            **kwargs: Additional arguments for memory retrieval
            
        Returns:
            Retrieved value
            
        Raises:
            NotInitializedError: If the agent is not initialized
            MemoryError: If memory retrieval fails
        """
        if not self.initialized:
            raise NotInitializedError(
                component_name=self.name,
                operation="retrieve_memory"
            )
        
        # Create retrieve request from parameters
        request = MemoryRetrieveRequest(
            key=key
        )
        
        # Add any additional parameters
        for k, v in kwargs.items():
            if hasattr(request, k):
                setattr(request, k, v)
        
        # Use model-based implementation
        return await self._memory.retrieve_with_model(request)
    
    async def search_memory(self, query: str, **kwargs) -> List[Any]:
        """Search memory for relevant information.
        
        Args:
            query: Search query
            **kwargs: Additional arguments for memory search
            
        Returns:
            List of relevant memories
            
        Raises:
            NotInitializedError: If the agent is not initialized
            MemoryError: If memory search fails
        """
        if not self.initialized:
            raise NotInitializedError(
                component_name=self.name,
                operation="search_memory"
            )
        
        # Create search request from parameters        
        request = MemorySearchRequest(
            query=query
        )
        
        # Add any additional parameters
        for k, v in kwargs.items():
            if hasattr(request, k):
                setattr(request, k, v)
        
        # Use model-based implementation
        result = await self._memory.search_with_model(request)
        
        # Return the properly typed result directly
        return result
    
    async def execute_flow(
        self,
        flow_name: str,
        inputs: Any,
        **kwargs
    ) -> FlowResult:
        """Execute a specific flow.
        
        Args:
            flow_name: Name of the flow to execute
            inputs: Inputs for the flow - must be a proper Pydantic model instance
                   matching the flow's input model
            **kwargs: Additional execution arguments
            
        Returns:
            Flow result
            
        Raises:
            NotInitializedError: If the agent is not initialized
            ExecutionError: If flow not found or execution fails
            ValueError: If inputs is not the correct Pydantic model
        """
        if not self.initialized:
            raise NotInitializedError(
                component_name=self.name,
                operation="execute_flow"
            )
        
        # Get the flow to validate inputs type
        if flow_name not in self.flows:
            raise ExecutionError(f"Flow '{flow_name}' not found")
            
        flow = self.flows[flow_name]
        
        # Get pipeline method and expected input model
        pipeline_method = getattr(flow, 'get_pipeline_method', None)
        if not pipeline_method or not callable(pipeline_method):
            raise ExecutionError(f"Flow '{flow_name}' does not have a valid pipeline method")
            
        pipeline = pipeline_method()
        if not pipeline or not hasattr(pipeline, '__input_model__'):
            raise ExecutionError(f"Flow '{flow_name}' pipeline does not have an input model")
            
        expected_model = pipeline.__input_model__
        if not expected_model:
            raise ExecutionError(f"Flow '{flow_name}' does not have an input model defined")
        
        # Strictly validate input type - fail fast if not correct
        if not isinstance(inputs, expected_model):
            raise ValueError(
                f"Flow '{flow_name}' expects inputs to be a {expected_model.__name__} instance. "
                f"Got {type(inputs).__name__} instead."
            )
        
        # Update state with flow execution
        self.state.increment_cycle()
        
        # Execute flow
        result = await self._engine.execute_flow(
            flow_name=flow_name,
            inputs=inputs,
            state=self.state
        )
        
        # Store result
        self.last_result = result
        
        return result
    
    async def execute_cycle(self, **kwargs) -> bool:
        """Execute a single planning-execution-reflection cycle.
        
        Args:
            **kwargs: Additional cycle execution arguments
            
        Returns:
            True if agent should continue, False if task is complete
            
        Raises:
            NotInitializedError: If the agent is not initialized
            ExecutionError: If cycle execution fails
        """
        if not self.initialized:
            raise NotInitializedError(
                component_name=self.name,
                operation="execute_cycle"
            )
        
        # Execute cycle
        continue_execution = await self._engine.execute_cycle(
            state=self.state,
            memory_context=kwargs.get("context", "agent"),
            no_flow_is_error=kwargs.get("no_flow_is_error", False)
        )
        
        # Auto-save state if configured
        if (self._state_persister and 
            self.config.state_config and 
            self.config.state_config.auto_save and
            self.config.state_config.save_frequency == "cycle"):
            await self.save_state()
        
        return continue_execution
    
    async def learn(self, request: LearningRequest) -> LearningResponse:
        """Execute a learning flow based on the requested strategy.
        
        Args:
            request: Learning request with strategy and content
            
        Returns:
            Learning response with results
            
        Raises:
            NotInitializedError: If the agent is not initialized
            ValueError: If no flow exists for the requested strategy
            TypeError: If the flow is not a proper learning flow
            ExecutionError: If flow execution fails
        """
        if not self.initialized:
            raise NotInitializedError(
                component_name=self.name,
                operation="learn"
            )
            
        flow_name = f"{request.strategy.value}Flow"
        flow = self.flows.get(flow_name)
        
        if not flow:
            raise ValueError(f"No learning flow found for strategy: {request.strategy}")
            
        if not isinstance(flow, BaseLearningFlow):
            raise TypeError(f"Flow {flow_name} is not a learning flow")
            
        return await flow.execute(request)
    
    async def extract_entities(self, content: str, context: Optional[str] = None) -> List[Entity]:
        """Extract entities from content.
        
        This is a convenience method that creates a learning request and executes
        the entity extraction flow.
        
        Args:
            content: The content to extract entities from
            context: Optional context for extraction
            
        Returns:
            List of extracted entities
            
        Raises:
            NotInitializedError: If the agent is not initialized
            ValueError: If the entity extraction flow is not available
            ExecutionError: If entity extraction fails
        """
        request = LearningRequest(
            content=content,
            strategy=LearningStrategy.ENTITY_EXTRACTION,
            context=context
        )
        response = await self.learn(request)
        return response.entities
    
    async def learn_relationships(self, content: str, entity_ids: List[str]) -> List[Relationship]:
        """Learn relationships between entities.
        
        This is a convenience method that creates a learning request and executes
        the relationship learning flow.
        
        Args:
            content: The content containing relationship information
            entity_ids: List of entity IDs to find relationships between
            
        Returns:
            List of discovered relationships
            
        Raises:
            NotInitializedError: If the agent is not initialized
            ValueError: If the relationship learning flow is not available
            ExecutionError: If relationship learning fails
        """
        request = LearningRequest(
            content=content,
            strategy=LearningStrategy.RELATIONSHIP_LEARNING,
            existing_entities=entity_ids
        )
        response = await self.learn(request)
        return response.relationships
    
    async def integrate_knowledge(self, content: str, entity_ids: List[str]) -> LearningResponse:
        """Integrate new knowledge with existing knowledge.
        
        This is a convenience method that creates a learning request and executes
        the knowledge integration flow.
        
        Args:
            content: The content to integrate
            entity_ids: List of entity IDs to integrate with
            
        Returns:
            Learning response with integration results
            
        Raises:
            NotInitializedError: If the agent is not initialized
            ValueError: If the knowledge integration flow is not available
            ExecutionError: If knowledge integration fails
        """
        request = LearningRequest(
            content=content,
            strategy=LearningStrategy.KNOWLEDGE_INTEGRATION,
            existing_entities=entity_ids
        )
        return await self.learn(request)
    
    async def form_concepts(self, content: str) -> List[Entity]:
        """Form new concepts from observations.
        
        This is a convenience method that creates a learning request and executes
        the concept formation flow.
        
        Args:
            content: The content to extract concepts from
            
        Returns:
            List of formed concept entities
            
        Raises:
            NotInitializedError: If the agent is not initialized
            ValueError: If the concept formation flow is not available
            ExecutionError: If concept formation fails
        """
        request = LearningRequest(
            content=content,
            strategy=LearningStrategy.CONCEPT_FORMATION
        )
        response = await self.learn(request)
        return response.entities 