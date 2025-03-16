"""Provider base implementation with enhanced configuration and lifecycle management.

This module provides the foundation for all providers with improved
configuration, initialization, and error handling.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Generic, List, Union
import logging

from ..core.models.settings import FlowSettings
from ..core.errors import ResourceError, ErrorContext

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=FlowSettings)

class Provider(ABC, Generic[T]):
    """Base class for all providers with enhanced lifecycle management.
    
    This class provides:
    1. Consistent initialization and cleanup pattern
    2. Configuration via settings models
    3. Clean error handling
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
        self.settings = settings or self._default_settings()
        self.provider_type = provider_type or self.__class__.__name__
        self._initialized = False
        self._setup_lock = asyncio.Lock()
        
        # No self-registration
        logger.debug(f"Created provider: {name} ({self.provider_type})")
    
    @property
    def initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized
    
    def _default_settings(self) -> T:
        """Create default settings instance.
        
        Returns:
            Default settings for this provider
        """
        # Get settings type from Generic
        settings_type = self.__class__.__orig_bases__[0].__args__[0]
        return settings_type()
    
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
        """Clean up provider resources.
        
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


class AsyncProvider(Provider[T]):
    """Enhanced provider base with asynchronous execution support.
    
    This class adds support for asynchronous execution with timeout and retry
    handling, useful for providers that interact with external services.
    """
    
    async def execute_with_retry(
        self,
        operation: callable,
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
        max_retries = retries if retries is not None else self.settings.max_retries
        delay = retry_delay if retry_delay is not None else self.settings.retry_delay_seconds
        timeout_seconds = timeout if timeout is not None else self.settings.timeout_seconds
        
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