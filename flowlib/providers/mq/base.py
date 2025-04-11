"""Message Queue provider base class and related functionality.

This module provides the base class for implementing message queue providers
that share common functionality for publishing and consuming messages.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Generic, Callable
from pydantic import BaseModel

from ...core.errors import ProviderError, ErrorContext
from ..base import Provider
from ..base import ProviderSettings

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class MQProviderSettings(ProviderSettings):
    """Base settings for message queue providers.
    
    Attributes:
        host: Message queue host address
        port: Message queue port
        username: Authentication username
        password: Authentication password (should use SecretStr in implementations)
        virtual_host: Virtual host/namespace
        timeout: Connection timeout in seconds
        heartbeat: Heartbeat interval in seconds
        ssl_enabled: Whether to use SSL for connections
    """
    
    # Connection settings
    host: str
    port: int
    username: str
    password: str  # Use SecretStr in implementations
    virtual_host: str = "/"
    
    # Performance settings
    timeout: float = 30.0
    heartbeat: int = 60
    ssl_enabled: bool = False
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Delivery settings
    prefetch_count: int = 10
    auto_ack: bool = False


class MessageMetadata(BaseModel):
    """Metadata for message queue messages.
    
    Attributes:
        message_id: Unique message identifier
        correlation_id: Correlation identifier for related messages
        timestamp: Message creation timestamp
        expiration: Message expiration time (if any)
        priority: Message priority (0-9, higher is more important)
        content_type: MIME type of message content
        headers: Custom message headers
    """
    
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: Optional[int] = None
    expiration: Optional[int] = None
    priority: Optional[int] = None
    content_type: Optional[str] = None
    headers: Dict[str, Any] = {}


class MQProvider(Provider, Generic[T]):
    """Base class for message queue providers.
    
    This class provides:
    1. Connection management
    2. Publishing messages with structured data
    3. Consuming messages with callback handlers
    4. Type-safe message serialization/deserialization
    """
    
    def __init__(self, name: str = "mq", settings: Optional[MQProviderSettings] = None):
        """Initialize message queue provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Pass provider_type="mq" to the parent class
        super().__init__(name=name, settings=settings, provider_type="mq")
        self._initialized = False
        self._connection = None
        
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
        
    async def initialize(self):
        """Initialize the message queue connection.
        
        This method should be implemented by subclasses to establish
        connections to the message queue system.
        """
        self._initialized = True
        
    async def shutdown(self):
        """Close all connections and release resources.
        
        This method should be implemented by subclasses to properly
        close connections and clean up resources.
        """
        self._initialized = False
        self._connection = None
        
    async def publish(self, 
                     exchange: str, 
                     routing_key: str, 
                     message: Any,
                     metadata: Optional[MessageMetadata] = None) -> bool:
        """Publish a message to a queue.
        
        Args:
            exchange: Exchange name
            routing_key: Routing key or queue name
            message: Message content (can be dict, string, bytes, etc.)
            metadata: Optional message metadata
            
        Returns:
            True if message was published successfully
            
        Raises:
            ProviderError: If publishing fails
        """
        raise NotImplementedError("Subclasses must implement publish()")
        
    async def publish_structured(self,
                                exchange: str,
                                routing_key: str,
                                message: BaseModel,
                                metadata: Optional[MessageMetadata] = None) -> bool:
        """Publish a structured message to a queue.
        
        Args:
            exchange: Exchange name
            routing_key: Routing key or queue name
            message: Pydantic model instance
            metadata: Optional message metadata
            
        Returns:
            True if message was published successfully
            
        Raises:
            ProviderError: If publishing fails
        """
        try:
            # Convert model to JSON-compatible dict
            message_dict = message.dict()
            
            # Set content type in metadata if not provided
            if metadata is None:
                metadata = MessageMetadata(content_type="application/json")
            elif metadata.content_type is None:
                metadata.content_type = "application/json"
                
            # Publish the message
            return await self.publish(exchange, routing_key, message_dict, metadata)
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to publish structured message: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    exchange=exchange,
                    routing_key=routing_key,
                    model_type=type(message).__name__
                ),
                cause=e
            )
            
    async def consume(self,
                     queue: str,
                     callback: Callable[[Any, MessageMetadata], Any],
                     consumer_tag: Optional[str] = None) -> Any:
        """Start consuming messages from a queue.
        
        Args:
            queue: Queue name to consume from
            callback: Function to call for each message
            consumer_tag: Optional consumer identifier
            
        Returns:
            Consumer instance (implementation-specific)
            
        Raises:
            ProviderError: If consumer creation fails
        """
        raise NotImplementedError("Subclasses must implement consume()")
        
    async def consume_structured(self,
                               queue: str,
                               output_type: Type[T],
                               callback: Callable[[T, MessageMetadata], Any],
                               consumer_tag: Optional[str] = None) -> Any:
        """Start consuming structured messages from a queue.
        
        Args:
            queue: Queue name to consume from
            output_type: Pydantic model for message structure
            callback: Function to call for each parsed message
            consumer_tag: Optional consumer identifier
            
        Returns:
            Consumer instance (implementation-specific)
            
        Raises:
            ProviderError: If consumer creation fails
        """
        try:
            # Create a wrapper callback that parses messages
            async def wrapper_callback(message: Any, metadata: MessageMetadata):
                try:
                    # Parse the message into the output type
                    parsed_message = output_type.parse_obj(message)
                    
                    # Call the original callback with the parsed message
                    return await callback(parsed_message, metadata)
                    
                except Exception as e:
                    logger.error(f"Failed to parse message as {output_type.__name__}: {str(e)}")
                    # Handle parsing errors according to implementation
                    
            # Start consuming with the wrapper callback
            return await self.consume(queue, wrapper_callback, consumer_tag)
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to create structured consumer: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    queue=queue,
                    output_type=output_type.__name__
                ),
                cause=e
            )
            
    async def acknowledge(self, delivery_tag: Any):
        """Acknowledge a message.
        
        Args:
            delivery_tag: Delivery tag from consumed message
            
        Raises:
            ProviderError: If acknowledgement fails
        """
        raise NotImplementedError("Subclasses must implement acknowledge()")
        
    async def reject(self, delivery_tag: Any, requeue: bool = True):
        """Reject a message.
        
        Args:
            delivery_tag: Delivery tag from consumed message
            requeue: Whether to requeue the message
            
        Raises:
            ProviderError: If rejection fails
        """
        raise NotImplementedError("Subclasses must implement reject()")
        
    async def check_connection(self) -> bool:
        """Check if message queue connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_connection()") 