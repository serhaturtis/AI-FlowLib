"""Configuration system for agents.

This module provides configuration classes and utilities for customizing
agent behavior, memory settings, and execution parameters.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    
    # Base model settings
    name: str = "default"
    path: str = ""
    model_type: str = "chatml"
    
    # Context settings
    n_ctx: int = 8192
    n_threads: int = 4
    n_batch: int = 512
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    
    # Hardware settings
    use_gpu: bool = True
    n_gpu_layers: int = -1
    
    def __post_init__(self):
        """Initialize with environment variables if available."""
        if not self.path and "MODEL_PATH" in os.environ:
            self.path = os.environ["MODEL_PATH"]

@dataclass
class MemoryConfig:
    """Configuration for agent memory."""
    
    # Memory types
    use_working_memory: bool = True
    use_short_term_memory: bool = True
    use_long_term_memory: bool = True
    
    # Memory providers
    working_memory_provider: str = "memory-cache"
    short_term_memory_provider: str = "memory-cache"
    long_term_memory_provider: str = "chroma"
    
    # Memory settings
    max_working_memory_items: int = 50
    max_short_term_memory_items: int = 200
    default_ttl: int = 3600  # 1 hour in seconds
    importance_threshold: float = 0.5
    retrieval_count: int = 10

@dataclass
class ExecutionConfig:
    """Configuration for agent execution."""
    
    # Flow settings
    flow_paths: List[str] = field(default_factory=lambda: ["./flows"])
    disabled_flows: Set[str] = field(default_factory=set)
    planning_flow: str = "agent_planning"
    reflection_flow: str = "agent_reflection"
    input_generation_flow: str = "agent_input_generation"
    
    # Execution control
    max_iterations: int = 10
    timeout_seconds: int = 60
    enable_reflection: bool = True
    reflection_frequency: int = 3  # Every N iterations
    logging_level: str = "INFO"

@dataclass
class AgentConfig:
    """Master configuration for an agent."""
    
    # Basic settings
    name: str = "default_agent"
    description: str = "A conversational agent"
    version: str = "1.0.0"
    
    # Component configs
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Provider settings
    provider_name: str = "llamacpp"
    planner_model: str = "default"
    input_generator_model: str = "default"
    reflection_model: str = "default"
    
    def __post_init__(self):
        """Initialize with default models if not provided."""
        if not self.models:
            self.models = {
                "default": ModelConfig()
            }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create an AgentConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            AgentConfig instance
        """
        # Create base config with simple values
        config = cls(
            name=config_dict.get("name", "default_agent"),
            description=config_dict.get("description", "A conversational agent"),
            version=config_dict.get("version", "1.0.0"),
            provider_name=config_dict.get("provider_name", "llamacpp"),
            planner_model=config_dict.get("planner_model", "default"),
            input_generator_model=config_dict.get("input_generator_model", "default"),
            reflection_model=config_dict.get("reflection_model", "default"),
        )
        
        # Process models
        models_dict = config_dict.get("models", {})
        for model_name, model_config in models_dict.items():
            config.models[model_name] = ModelConfig(
                name=model_name,
                **{k: v for k, v in model_config.items() if k != "name"}
            )
        
        # Process memory config
        if "memory" in config_dict:
            config.memory = MemoryConfig(**config_dict["memory"])
            
        # Process execution config
        if "execution" in config_dict:
            exec_config = config_dict["execution"]
            
            # Handle special cases for execution config
            flow_paths = exec_config.get("flow_paths", ["./flows"])
            disabled_flows = set(exec_config.get("disabled_flows", []))
            
            # Create execution config
            config.execution = ExecutionConfig(
                flow_paths=flow_paths,
                disabled_flows=disabled_flows,
                **{k: v for k, v in exec_config.items() 
                   if k not in ["flow_paths", "disabled_flows"]}
            )
            
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'AgentConfig':
        """Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
            
        Returns:
            AgentConfig instance
        """
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "provider_name": self.provider_name,
            "planner_model": self.planner_model,
            "input_generator_model": self.input_generator_model,
            "reflection_model": self.reflection_model,
            "models": {},
            "memory": asdict(self.memory),
            "execution": {}
        }
        
        # Convert models to dict
        for model_name, model_config in self.models.items():
            config_dict["models"][model_name] = asdict(model_config)
            
        # Handle special cases for execution config
        exec_dict = asdict(self.execution)
        exec_dict["disabled_flows"] = list(exec_dict["disabled_flows"])
        config_dict["execution"] = exec_dict
        
        return config_dict
    
    def save_to_file(self, file_path: str) -> bool:
        """Save the configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
            return False
        
def get_default_config() -> AgentConfig:
    """Get a default agent configuration.
    
    Returns:
        Default AgentConfig instance
    """
    return AgentConfig() 