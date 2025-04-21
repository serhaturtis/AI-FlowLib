"""
Utilities for loading and validating agent configurations.
"""

import yaml
import os
import logging
from typing import Dict, Any

from .models.config import AgentConfig
from .core.errors import ConfigurationError

logger = logging.getLogger(__name__)

def load_agent_config(filepath: str) -> AgentConfig:
    """Loads agent configuration from a YAML file.

    Reads the specified YAML file, parses its content, and validates it
    against the AgentConfig Pydantic model.

    Args:
        filepath: The path to the YAML configuration file.

    Returns:
        A validated AgentConfig instance.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ConfigurationError: If the file cannot be parsed as YAML or if the
                            content does not conform to the AgentConfig model.
    """
    logger.info(f"Attempting to load agent configuration from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found at: {filepath}")

    try:
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ConfigurationError(f"Failed to parse YAML or file is empty: {filepath}")

        # Validate and instantiate AgentConfig using Pydantic
        # Pydantic will raise validation errors if the structure/types are wrong
        loaded_config = AgentConfig(**config_dict)
        logger.info(f"Successfully loaded and validated AgentConfig from {filepath}")
        return loaded_config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{filepath}': {e}", exc_info=True)
        raise ConfigurationError(
            message=f"Error parsing YAML configuration file: {filepath}",
            cause=e
        )
    except Exception as e:
        # Catch Pydantic ValidationErrors and other potential issues
        logger.error(f"Error validating configuration from '{filepath}': {e}", exc_info=True)
        raise ConfigurationError(
            message=f"Error validating configuration loaded from file: {filepath}",
            cause=e
        ) 