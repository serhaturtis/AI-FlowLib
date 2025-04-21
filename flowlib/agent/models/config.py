"""
Configuration models for agent components.

This module contains the configuration models used for
the agent engine and other components.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """
    Configuration for the agent engine.
    
    Attributes:
        max_iterations: Maximum number of planning-execution cycles
        stop_on_error: Whether to stop execution on errors
        log_level: Level of detail for logging
        execution_timeout: Maximum time in seconds for a flow execution
        planning_timeout: Maximum time in seconds for planning
        reflection_timeout: Maximum time in seconds for reflection
    """
    max_iterations: int = Field(
        default=10,
        description="Maximum number of planning-execution cycles"
    )
    stop_on_error: bool = Field(
        default=False,
        description="Whether to stop execution on errors"
    )
    log_level: str = Field(
        default="INFO",
        description="Level of detail for logging"
    )
    execution_timeout: Optional[float] = Field(
        default=60.0,
        description="Maximum time in seconds for a flow execution"
    )
    planning_timeout: Optional[float] = Field(
        default=30.0,
        description="Maximum time in seconds for planning"
    )
    reflection_timeout: Optional[float] = Field(
        default=30.0,
        description="Maximum time in seconds for reflection"
    )
    
    
class PlannerConfig(BaseModel):
    """Configuration for the agent planner.

    This configuration provides parameters for the planning component.
    
    Note: These configuration values are used to load the appropriate prompts and models,
    but parameters like max_tokens and temperature should be defined in the prompt template's 
    config attribute, not passed directly to the LLM provider.
    """
    model_name: str = Field(default="default", description="Name of the model to use")
    provider_name: Optional[str] = Field(default="llamacpp", description="Name of the provider to use")
    planning_max_tokens: int = Field(default=1024, description="Maximum number of tokens for planning")
    planning_temperature: float = Field(default=0.2, description="Temperature for planning")
    input_generation_max_tokens: int = Field(default=1024, description="Maximum number of tokens for input generation")
    input_generation_temperature: float = Field(default=0.7, description="Temperature for input generation")


class ReflectionConfig(BaseModel):
    """Configuration for the agent reflection component.
    
    This configuration provides parameters for the reflection component.
    
    Note: These configuration values are used to load the appropriate prompts and models,
    but parameters like max_tokens and temperature should be defined in the prompt template's 
    config attribute, not passed directly to the LLM provider.
    """
    model_name: str = Field(default="default", description="Name of the model to use")
    provider_name: str = Field(default="llamacpp", description="Name of the provider to use")
    max_tokens: int = Field(default=1024, description="Maximum tokens for reflection")
    temperature: float = Field(default=0.7, description="Temperature for reflection")


class WorkingMemoryConfig(BaseModel):
    """Configuration for WorkingMemory."""
    default_ttl_seconds: Optional[int] = Field(
        default=3600, 
        description="Default time-to-live for items in seconds (None means no expiry)"
    )
    # Add other WorkingMemory specific settings if needed (e.g., cleanup interval)


class VectorMemoryConfig(BaseModel):
    """Configuration for VectorMemory."""
    vector_provider_name: str = Field(
        default="chroma", # Changed default to chroma
        description="Name of the vector database provider (e.g., qdrant, chroma)" 
    )
    embedding_provider_name: str = Field(
        default="default_embedding", # Example default - needs configuration elsewhere
        description="Name of the embedding model provider"
    )
    # Add provider-specific configs if necessary, or handle them via provider registry


class KnowledgeMemoryConfig(BaseModel):
    """Configuration for KnowledgeBaseMemory."""
    graph_provider_name: str = Field(
        default="neo4j",
        description="Name of the graph database provider (e.g., neo4j, in_memory_graph)"
    )
    # Add nested settings for the chosen provider
    provider_settings: Dict[str, Any] = Field(
        default_factory=lambda: { # Default to matching docker-compose
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "pleaseChangeThisPassword"
        },
        description="Settings specific to the chosen graph_provider_name."
    )


class ComprehensiveMemoryConfig(BaseModel):
    """Configuration for the ComprehensiveMemory system."""
    working_memory: WorkingMemoryConfig = Field(
        default_factory=WorkingMemoryConfig,
        description="Configuration for the working memory component."
    )
    vector_memory: VectorMemoryConfig = Field(
        default_factory=VectorMemoryConfig,
        description="Configuration for the vector memory component."
    )
    knowledge_memory: KnowledgeMemoryConfig = Field(
        default_factory=KnowledgeMemoryConfig,
        description="Configuration for the knowledge base memory component."
    )
    fusion_provider_name: str = Field(
        default="llamacpp", 
        description="LLM Provider name for memory fusion/synthesis."
    )
    fusion_model_name: str = Field(
        default="default", 
        description="LLM Model name for memory fusion/synthesis."
    )
    store_execution_history: bool = Field(
        default=True,
        description="Whether to store flow execution history in memory (typically working memory)."
    )


class StatePersistenceConfig(BaseModel):
    """
    Enhanced configuration for agent state persistence.
    
    This class provides a more detailed configuration with additional options for controlling
    how and when states are persisted.
    
    Attributes:
        persistence_type: Type of persistence to use ('file', 'provider', or 'none')
        base_path: Base directory to store state files (for 'file' persistence)
        provider_id: ID of provider to use (for 'provider' persistence)
        provider_config: Additional configuration for the provider
        auto_save: Whether to automatically save state after changes
        auto_load: Whether to automatically load state on initialization
        save_frequency: When to automatically save state ('cycle', 'action', 'change', or 'never')
        max_states: Maximum number of states to keep for a given agent
        compress: Whether to compress state data
    """
    persistence_type: str = Field(
        default="file",
        description="Type of persistence to use ('file', 'provider', or 'none')"
    )
    base_path: Optional[str] = Field(
        default="./states",
        description="Base directory to store state files (for 'file' persistence)"
    )
    provider_id: Optional[str] = Field(
        default=None,
        description="ID of provider to use (for 'provider' persistence)"
    )
    provider_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration for the provider"
    )
    auto_save: bool = Field(
        default=True,
        description="Whether to automatically save state after changes"
    )
    auto_load: bool = Field(
        default=True,
        description="Whether to automatically load state on initialization"
    )
    save_frequency: str = Field(
        default="change",
        description="When to automatically save state ('cycle', 'action', 'change', or 'never')"
    )
    max_states: Optional[int] = Field(
        default=None,
        description="Maximum number of states to keep for a given agent"
    )
    compress: bool = Field(
        default=False,
        description="Whether to compress state data"
    )


class AgentConfig(BaseModel):
    """
    Master configuration for the agent.
    
    Attributes:
        name: Name of the agent
        persona: Description of the agent's personality/style.
        task_id: Unique identifier for the agent task
        task_description: Description of the agent task
        engine_config: Configuration for the agent engine
        planner_config: Configuration for the agent planner
        reflection_config: Configuration for the agent reflection
        memory_config: Configuration for the agent memory
        state_config: Configuration for state persistence
        provider_config: Configuration for the model provider
        components: Configuration for custom components
    """
    name: str = Field(description="Name of the agent")
    persona: str = Field(description="Description of the agent's personality/style.")
    task_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the agent task"
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Description of the agent task"
    )
    engine_config: EngineConfig = Field(
        default_factory=EngineConfig,
        description="Configuration for the agent engine"
    )
    planner_config: PlannerConfig = Field(
        default_factory=lambda: PlannerConfig(model_name="default"),
        description="Configuration for the agent planner"
    )
    reflection_config: ReflectionConfig = Field(
        default_factory=lambda: ReflectionConfig(model_name="default", provider_name="llamacpp"),
        description="Configuration for the agent reflection"
    )
    memory_config: ComprehensiveMemoryConfig = Field(
        default_factory=ComprehensiveMemoryConfig,
        description="Configuration for the agent memory"
    )
    state_config: Optional[StatePersistenceConfig] = Field(
        default=None,
        description="Configuration for state persistence"
    )
    provider_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the model provider"
    )
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for custom components"
    )
