"""Flow settings model for configuring execution behavior.

This module provides a FlowSettings class for configuring flow execution
behavior, including timeouts, retries, and logging options.
"""

from typing import Any, Dict, Optional, Union, List, Type, TypeVar, cast
from pydantic import BaseModel, Field, field_validator

T = TypeVar('T', bound='BaseModel')

class FlowSettings(BaseModel):
    """Settings for configuring flow execution behavior.
    
    This class provides:
    1. Timeout configuration for flow execution
    2. Retry behavior for handling transient errors
    3. Logging and debugging options
    4. Resource management settings
    """
    
    # Execution settings
    timeout_seconds: Optional[float] = None
    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    
    # Validation settings
    validate_inputs: bool = True
    validate_outputs: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Resource settings
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[int] = None
    
    # Advanced settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("max_retries")
    def validate_max_retries(cls, v: int) -> int:
        """Validate max retries."""
        if v < 0:
            raise ValueError("Max retries must be non-negative")
        return v
    
    @field_validator("retry_delay_seconds")
    def validate_retry_delay(cls, v: float) -> float:
        """Validate retry delay."""
        if v < 0:
            raise ValueError("Retry delay must be non-negative")
        return v
    
    def merge(self, other: Union['FlowSettings', Dict[str, Any]]) -> 'FlowSettings':
        """Merge with another settings object.
        
        Args:
            other: Settings to merge with
            
        Returns:
            New settings instance with merged values
        """
        if isinstance(other, dict):
            # Convert dict to settings
            other_settings = FlowSettings(**other)
        else:
            other_settings = other
            
        # Start with current settings
        merged_dict = self.model_dump()
        
        # Update with other settings (only non-None values)
        for key, value in other_settings.model_dump().items():
            if value is not None:
                if key == "custom_settings":
                    # Merge custom settings
                    merged_dict["custom_settings"].update(value)
                else:
                    merged_dict[key] = value
        
        return FlowSettings(**merged_dict)
    
    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> 'FlowSettings':
        """Create settings from dictionary.
        
        Args:
            settings_dict: Dictionary of settings
            
        Returns:
            FlowSettings instance
        """
        return cls(**settings_dict)
    
    def with_overrides(self, **kwargs: Any) -> 'FlowSettings':
        """Create new settings with overrides.
        
        Args:
            **kwargs: Settings to override
            
        Returns:
            New settings instance with overrides
        """
        settings_dict = self.model_dump()
        settings_dict.update(kwargs)
        return FlowSettings(**settings_dict)
    
    def __str__(self) -> str:
        """String representation."""
        timeout_str = f"{self.timeout_seconds}s" if self.timeout_seconds else "None"
        return f"FlowSettings(timeout={timeout_str}, retries={self.max_retries}, debug={self.debug_mode})"

class ProviderSettings(BaseModel):
    """Base settings for providers.
    
    This class provides:
    1. Common configuration for all providers
    2. Authentication settings
    3. Rate limiting and throttling options
    """
    
    # Authentication settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    
    # Timeout settings
    timeout_seconds: float = 60.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Logging settings
    log_requests: bool = False
    log_responses: bool = False
    
    # Advanced settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    def merge(self, other: Union['ProviderSettings', Dict[str, Any]]) -> 'ProviderSettings':
        """Merge with another settings object.
        
        Args:
            other: Settings to merge with
            
        Returns:
            New settings instance with merged values
        """
        if isinstance(other, dict):
            # Convert dict to settings
            other_settings = self.__class__(**other)
        else:
            other_settings = other
            
        # Start with current settings
        merged_dict = self.model_dump()
        
        # Update with other settings (only non-None values)
        for key, value in other_settings.model_dump().items():
            if value is not None:
                if key == "custom_settings":
                    # Merge custom settings
                    merged_dict["custom_settings"].update(value)
                else:
                    merged_dict[key] = value
        
        return self.__class__(**merged_dict)
    
    def with_overrides(self, **kwargs: Any) -> 'ProviderSettings':
        """Create new settings with overrides.
        
        Args:
            **kwargs: Settings to override
            
        Returns:
            New settings instance with overrides
        """
        settings_dict = self.model_dump()
        settings_dict.update(kwargs)
        return self.__class__(**settings_dict)

class AgentSettings(BaseModel):
    """Base settings for agents.
    
    This class provides:
    1. Common configuration for all agents
    2. Execution parameters
    3. Logging options
    """
    
    # Execution settings
    timeout_seconds: Optional[float] = 60.0
    max_iterations: int = 10
    
    # Logging settings
    log_level: str = "INFO"
    debug_mode: bool = False
    log_iterations: bool = True
    
    # Advanced settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("max_iterations")
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max iterations."""
        if v < 1:
            raise ValueError("Max iterations must be at least 1")
        return v
    
    def merge(self, other: Union['AgentSettings', Dict[str, Any]]) -> 'AgentSettings':
        """Merge with another settings object.
        
        Args:
            other: Settings to merge with
            
        Returns:
            New settings instance with merged values
        """
        if isinstance(other, dict):
            # Convert dict to settings
            other_settings = self.__class__(**other)
        else:
            other_settings = other
            
        # Start with current settings
        merged_dict = self.model_dump()
        
        # Update with other settings (only non-None values)
        for key, value in other_settings.model_dump().items():
            if value is not None:
                if key == "custom_settings":
                    # Merge custom settings
                    merged_dict["custom_settings"].update(value)
                else:
                    merged_dict[key] = value
        
        return self.__class__(**merged_dict)
    
    def with_overrides(self, **kwargs: Any) -> 'AgentSettings':
        """Create new settings with overrides.
        
        Args:
            **kwargs: Settings to override
            
        Returns:
            New settings instance with overrides
        """
        settings_dict = self.model_dump()
        settings_dict.update(kwargs)
        return self.__class__(**settings_dict)

class LLMProviderSettings(ProviderSettings):
    """Settings for LLM providers.
    
    This class provides:
    1. Model configuration
    2. Generation parameters
    3. Token management
    """
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Token management
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    
    # Streaming settings
    stream: bool = False
    
    # Advanced settings
    stop_sequences: List[str] = Field(default_factory=list)
    
    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature."""
        if v < 0 or v > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @field_validator("top_p")
    def validate_top_p(cls, v: float) -> float:
        """Validate top_p."""
        if v < 0 or v > 1:
            raise ValueError("Top_p must be between 0 and 1")
        return v

def create_settings(settings_class: Type[T], **kwargs: Any) -> T:
    """Create settings instance with provided values.
    
    Args:
        settings_class: Settings class to instantiate
        **kwargs: Settings values
        
    Returns:
        Settings instance
    """
    return cast(T, settings_class(**kwargs))
