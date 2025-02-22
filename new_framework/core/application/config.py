import os
from typing import Dict, Any, Type, TypeVar, Generic, Callable, get_type_hints
from pydantic import BaseModel, Field, create_model

from ..errors.base import ConfigurationError, ErrorContext

T = TypeVar('T', bound=BaseModel)

def _create_nested_model(cls: Type) -> Type[BaseModel]:
    """Create a Pydantic model from a nested class.
    
    Args:
        cls: Class to convert
        
    Returns:
        Pydantic model class
    """
    fields = {}
    
    # Process nested classes first
    for key, value in cls.__dict__.items():
        if key.startswith('_'):
            continue
            
        if isinstance(value, type):
            nested_model = _create_nested_model(value)
            fields[key] = (nested_model, Field(default_factory=nested_model))
    
    # Then process regular attributes
    annotations = get_type_hints(cls)
    for key, type_hint in annotations.items():
        if key.startswith('_') or key in fields:
            continue
            
        # Get default value
        default = getattr(cls, key, None)
        
        # Get value from environment
        env_key = key.upper()
        env_value = os.environ.get(env_key)
        
        if env_value is not None:
            if type_hint == bool:
                default = env_value.lower() in ('true', '1', 'yes')
            else:
                default = type_hint(env_value)
                
        fields[key] = (type_hint, Field(default=default))
    
    # Create model
    model_name = f"{cls.__name__}Model"
    return create_model(model_name, **fields)

class ConfigBase:
    """Base class for configuration objects."""
    
    @classmethod
    def load(cls) -> 'ConfigBase':
        """Load configuration from environment."""
        return cls()
    
    def __init__(self):
        # Create Pydantic model
        model_class = self.__class__._pydantic_model()
        self._model = model_class()
        
    def __getattr__(self, name: str) -> Any:
        """Get attribute from model."""
        return getattr(self._model, name)
    
    def model_dump(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._model.model_dump()

def config(cls: Type[T]) -> Type[ConfigBase]:
    """Decorator to create a configuration class.
    
    The decorator converts a class with class variables into a configuration
    object that can be overridden by environment variables.
    
    Example:
        @config
        class AppConfig:
            HOST: str = "localhost"
            PORT: int = 8080
            DEBUG: bool = False
            
            class Database:
                URL: str = "sqlite:///db.sqlite3"
                POOL_SIZE: int = 5
    
    Args:
        cls: Class to convert to config
        
    Returns:
        Configuration class
    """
    # Create Pydantic model
    pydantic_model = _create_nested_model(cls)
    
    # Create new class
    class Config(ConfigBase):
        pass
    
    # Add Pydantic model
    Config._pydantic_model = staticmethod(lambda: pydantic_model)
    
    # Copy class attributes
    Config.__name__ = cls.__name__
    Config.__doc__ = cls.__doc__
    Config.__module__ = cls.__module__
    
    return Config 

class ConfigurationManager(Generic[T]):
    """Manages configuration validation and resource creation."""
    
    def __init__(
        self,
        config_model: Type[T],
        resource_factories: Dict[str, Callable[[Dict[str, Any]], Any]] = None
    ):
        """Initialize configuration manager.
        
        Args:
            config_model: Pydantic model for configuration validation
            resource_factories: Optional mapping of section names to factory functions
        """
        self.config_model = config_model
        self.resource_factories = resource_factories or {}
    
    def validate_config(self, config_data: Dict[str, Any]) -> T:
        """Validate configuration data.
        
        Args:
            config_data: Raw configuration data
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigurationError: If validation fails
        """
        try:
            return self.config_model.model_validate(config_data)
        except Exception as e:
            raise ConfigurationError(
                "Configuration validation failed",
                ErrorContext.create(
                    error=str(e),
                    config=str(config_data)
                )
            )
    
    def create_resource(
        self,
        section: str,
        config: Dict[str, Any]
    ) -> Any:
        """Create a resource from configuration section.
        
        Args:
            section: Configuration section name
            config: Configuration data for the section
            
        Returns:
            Created resource
            
        Raises:
            ConfigurationError: If resource creation fails
        """
        if section not in self.resource_factories:
            raise ConfigurationError(
                f"No factory found for section: {section}",
                ErrorContext.create(
                    section=section,
                    available_sections=list(self.resource_factories.keys())
                )
            )
            
        try:
            return self.resource_factories[section](config)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create resource for section: {section}",
                ErrorContext.create(
                    section=section,
                    error=str(e),
                    config=str(config)
                )
            ) 