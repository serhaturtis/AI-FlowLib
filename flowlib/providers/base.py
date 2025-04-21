"""Provider base implementation with enhanced configuration and lifecycle management.

This module provides the foundation for all providers with improved
configuration, initialization, and error handling.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic, List, Union, Callable
from pydantic import BaseModel, Field
import logging

from ..core.errors import ResourceError, ErrorContext
from ..flows.base import FlowSettings

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=FlowSettings)

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

class Provider(ABC, Generic[T]):
    """Base class for all providers with enhanced lifecycle management.
    
    This class provides:
    1. Consistent initialization and cleanup pattern
    2. Configuration via settings models
    3. Clean error handling
    4. Asynchronous execution with retry and timeout capabilities
    """
    
    def __init__(
        self,
        name: str,
        settings: Optional[T] = None,
        provider_type: Optional[str] = None
    ):
        """Initialize provider.
        
        Args:
            name: Unique provider name
            settings: Provider settings
            provider_type: Optional provider type for categorization
        """
        self.name = name
        self.provider_type = provider_type or self.__class__.__name__
        self._initialized = False
        self._setup_lock = asyncio.Lock()
        
        # --- New Settings Handling --- 
        try:
            settings_type = self.__class__.__orig_bases__[0].__args__[0]
            if settings is None:
                self.settings = settings_type() # Use default if none provided
            elif isinstance(settings, dict):
                self.settings = settings_type(**settings) # Parse dict into model
            elif isinstance(settings, settings_type):
                self.settings = settings # Use directly if already correct type
            else:
                 raise TypeError(f"Invalid settings type provided. Expected dict or {settings_type.__name__}, got {type(settings).__name__}")
        except (AttributeError, IndexError, TypeError) as e:
            # Reraise type error if generic hint is missing
            if isinstance(e, (AttributeError, IndexError)):
                raise TypeError(
                    f"Provider class {self.__class__.__name__} must specify settings type as a generic parameter. "
                    f"Example: class MyProvider(Provider[MySettings]): ..."
                ) from e
            else:
                # Reraise other type errors (e.g. from parsing)
                raise e
        # --------------------------

        # No self-registration
        logger.debug(f"Created provider: {name} ({self.provider_type}) with settings: {self.settings}")
    
    @property
    def initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized
    
    def _default_settings(self) -> T:
        """Create default settings instance.
        
        Returns:
            Default settings for this provider
            
        Raises:
            TypeError: If the provider class doesn't have proper generic type information
        """
        try:
            # Get settings type from Generic[T] parameter
            settings_type = self.__class__.__orig_bases__[0].__args__[0]
            return settings_type()
        except (AttributeError, IndexError):
            # No type fallbacks - providers must have proper type parameters
            raise TypeError(
                f"Provider class {self.__class__.__name__} must specify settings type as a generic parameter. "
                f"Example: class MyProvider(Provider[MySettings]): ..."
            )
    
    async def initialize(self) -> None:
        """Initialize the provider.
        
        This method:
        1. Ensures the provider is only initialized once
        2. Provides thread safety with a lock
        3. Standardizes the initialization pattern
        
        Raises:
            ResourceError: If initialization fails
        """
        if self._initialized:
            return
            
        async with self._setup_lock:
            if self._initialized:
                return
                
            try:
                await self._initialize()
                self._initialized = True
                logger.info(f"Provider '{self.name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize provider '{self.name}': {str(e)}")
                raise ResourceError(
                    message=f"Failed to initialize provider: {str(e)}",
                    resource_id=self.name,
                    resource_type=self.provider_type,
                    cause=e
                )
    
    async def shutdown(self) -> None:
        """Close provider resources.
        
        This method:
        1. Ensures clean shutdown of provider resources
        2. Only attempts shutdown if previously initialized
        3. Handles shutdown errors gracefully
        """
        if not self._initialized:
            return
            
        try:
            await self._shutdown()
            self._initialized = False
            logger.info(f"Provider '{self.name}' shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down provider '{self.name}': {str(e)}")
            # We don't re-raise the error to allow graceful shutdown
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Concrete initialization logic implemented by subclasses."""
        pass
    
    async def _shutdown(self) -> None:
        """Concrete shutdown logic implemented by subclasses.
        
        Default implementation does nothing.
        """
        pass
        
    async def execute_with_retry(
        self,
        operation: Callable,
        *args: Any,
        retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> Any:
        """Execute an operation with retry and timeout handling.
        
        Args:
            operation: Async callable to execute
            *args: Arguments for the operation
            retries: Number of retries (defaults to settings)
            retry_delay: Delay between retries in seconds (defaults to settings)
            timeout: Timeout in seconds (defaults to settings)
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result
            
        Raises:
            ResourceError: If operation fails after retries or times out
        """
        # Use provided values or defaults from settings
        max_retries = retries if retries is not None else getattr(self.settings, 'max_retries', 3)
        delay = retry_delay if retry_delay is not None else getattr(self.settings, 'retry_delay_seconds', 1.0)
        timeout_seconds = timeout if timeout is not None else getattr(self.settings, 'timeout_seconds', 30.0)
        
        # Ensure provider is initialized
        if not self._initialized:
            await self.initialize()
        
        # Execute with retries
        attempt = 0
        last_error = None
        
        while attempt <= max_retries:
            try:
                # Execute with timeout if specified
                if timeout_seconds:
                    return await asyncio.wait_for(
                        operation(*args, **kwargs),
                        timeout=timeout_seconds
                    )
                else:
                    return await operation(*args, **kwargs)
                    
            except asyncio.TimeoutError as e:
                logger.warning(f"Provider {self.name} operation timed out after {timeout_seconds}s")
                last_error = e
                break  # Don't retry on timeout
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                if attempt <= max_retries:
                    logger.warning(
                        f"Provider {self.name} operation failed (attempt {attempt}/{max_retries}): {str(e)}"
                    )
                    # Wait before retrying
                    await asyncio.sleep(delay)
                else:
                    # Max retries reached
                    break
        
        # If we get here, all retries failed or timed out
        error_msg = f"Provider operation failed after {attempt} attempt(s)"
        logger.error(f"{error_msg}: {str(last_error)}")
        
        raise ResourceError(
            message=error_msg,
            resource_id=self.name,
            resource_type=self.provider_type,
            context=ErrorContext.create(
                provider_name=self.name,
                provider_type=self.provider_type,
                max_retries=max_retries,
                timeout=timeout_seconds
            ),
            cause=last_error
        )