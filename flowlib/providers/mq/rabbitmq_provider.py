"""RabbitMQ message queue provider implementation.

This module provides a concrete implementation of the MQProvider
for RabbitMQ messaging using aio-pika.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Callable
import uuid
from datetime import datetime

from pydantic import Field

from ...core.errors import ProviderError, ErrorContext
from .base import MQProvider, MQProviderSettings, MessageMetadata
from ..decorators import provider
from ..constants import ProviderType

logger = logging.getLogger(__name__)

try:
    import aio_pika
    from aio_pika import Message, Connection, Channel, Queue, Exchange
    from aio_pika.abc import AbstractIncomingMessage
except ImportError:
    logger.warning("aio_pika package not found. Install with 'pip install aio-pika'")


class RabbitMQProviderSettings(MQProviderSettings):
    """Settings for RabbitMQ provider.
    
    Attributes:
        connection_string: Connection string (overrides host/port if provided)
        virtual_host: RabbitMQ virtual host
        heartbeat: Heartbeat interval in seconds
        connection_timeout: Connection timeout in seconds
        ssl: Whether to use SSL
        ssl_options: SSL options
        exchange_type: Exchange type (direct, fanout, topic, headers)
        exchange_durable: Whether exchange is durable
        queue_durable: Whether queues are durable
        queue_auto_delete: Whether queues are auto-deleted
        delivery_mode: Delivery mode (1 = non-persistent, 2 = persistent)
    """
    
    # RabbitMQ specific settings
    connection_string: Optional[str] = None
    virtual_host: str = "/"
    heartbeat: int = 60
    connection_timeout: float = 10.0
    ssl: bool = False
    ssl_options: Optional[Dict[str, Any]] = None
    
    # Exchange and queue settings
    exchange_type: str = "topic"
    exchange_durable: bool = True
    queue_durable: bool = True
    queue_auto_delete: bool = False
    delivery_mode: int = 2  # 2 = persistent
    
    # Default port for RabbitMQ if not specified
    port: int = 5672
    
    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict)


@provider(provider_type=ProviderType.MESSAGE_QUEUE, name="rabbitmq")
class RabbitMQProvider(MQProvider):
    """RabbitMQ implementation of the MQProvider.
    
    This provider implements message queue operations using aio_pika,
    an asynchronous client for RabbitMQ.
    """
    
    def __init__(self, name: str = "rabbitmq", settings: Optional[RabbitMQProviderSettings] = None):
        """Initialize RabbitMQ provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Create settings first to avoid issues with _default_settings() method
        settings = settings or RabbitMQProviderSettings(host="localhost")
        
        # Pass explicit settings to parent class
        super().__init__(name=name, settings=settings)
        
        # Store settings for local use
        self._settings = settings
        self._connection = None
        self._channel = None
        self._exchange = None
        self._queues = {}
        self._consumer_tags = set()
        
    async def _initialize(self) -> None:
        """Initialize RabbitMQ connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Create connection
            if self._settings.connection_string:
                # Use connection string if provided
                self._connection = await aio_pika.connect_robust(
                    self._settings.connection_string,
                    timeout=self._settings.connection_timeout,
                    **self._settings.connect_args
                )
            else:
                # Construct connection URL
                connection_url = f"amqp://{self._settings.username}:{self._settings.password}@{self._settings.host}:{self._settings.port}/{self._settings.virtual_host}"
                
                # Connect to RabbitMQ
                self._connection = await aio_pika.connect_robust(
                    connection_url,
                    timeout=self._settings.connection_timeout,
                    **self._settings.connect_args
                )
            
            # Create channel
            self._channel = await self._connection.channel()
            
            # Create exchange
            self._exchange = await self._channel.declare_exchange(
                name=self._settings.exchange_name or self.name,
                type=self._settings.exchange_type,
                durable=self._settings.exchange_durable
            )
            
            logger.info(f"Connected to RabbitMQ: {self._settings.host}:{self._settings.port}/{self._settings.virtual_host}")
            
        except Exception as e:
            self._connection = None
            self._channel = None
            self._exchange = None
            raise ProviderError(
                message=f"Failed to connect to RabbitMQ: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    host=self._settings.host,
                    port=self._settings.port,
                    virtual_host=self._settings.virtual_host
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close RabbitMQ connection."""
        try:
            # Cancel all consumers
            for consumer_tag in self._consumer_tags:
                try:
                    await self._channel.cancel(consumer_tag)
                except Exception as e:
                    logger.warning(f"Error canceling consumer {consumer_tag}: {str(e)}")
            
            # Close connection
            if self._connection:
                await self._connection.close()
                self._connection = None
                self._channel = None
                self._exchange = None
                self._queues = {}
                self._consumer_tags = set()
                logger.info(f"Closed RabbitMQ connection: {self._settings.host}:{self._settings.port}")
        except Exception as e:
            logger.error(f"Error during RabbitMQ connection shutdown: {str(e)}")
    
    async def publish(self, 
                     routing_key: str, 
                     message: Any, 
                     metadata: Optional[MessageMetadata] = None, 
                     content_type: str = "application/json",
                     expiration: Optional[int] = None) -> None:
        """Publish a message to RabbitMQ.
        
        Args:
            routing_key: Routing key
            message: Message to publish
            metadata: Optional message metadata
            content_type: Content type
            expiration: Message expiration in milliseconds
            
        Raises:
            ProviderError: If publish fails
        """
        if not self._connection or not self._channel or not self._exchange:
            await self.initialize()
            
        try:
            # Serialize message
            if isinstance(message, (dict, list)):
                body = json.dumps(message).encode()
            elif isinstance(message, str):
                body = message.encode()
            elif isinstance(message, bytes):
                body = message
            else:
                body = str(message).encode()
            
            # Prepare metadata
            headers = {}
            if metadata:
                headers = metadata.model_dump()
            
            # Create message
            message = Message(
                body=body,
                content_type=content_type,
                delivery_mode=self._settings.delivery_mode,
                message_id=str(uuid.uuid4()),
                timestamp=int(datetime.now().timestamp()),
                expiration=str(expiration) if expiration else None,
                headers=headers
            )
            
            # Publish message
            await self._exchange.publish(
                message=message,
                routing_key=routing_key
            )
            
            logger.debug(f"Published message to {routing_key}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to publish message: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    routing_key=routing_key,
                    exchange=self._exchange.name
                ),
                cause=e
            )
    
    async def subscribe(self, 
                       queue_name: str, 
                       routing_keys: List[str], 
                       callback: Callable[[Any, MessageMetadata], Any],
                       auto_ack: bool = False) -> None:
        """Subscribe to messages from RabbitMQ.
        
        Args:
            queue_name: Queue name
            routing_keys: List of routing keys to bind
            callback: Callback function
            auto_ack: Whether to auto-acknowledge messages
            
        Raises:
            ProviderError: If subscribe fails
        """
        if not self._connection or not self._channel or not self._exchange:
            await self.initialize()
            
        try:
            # Declare queue
            queue = await self._channel.declare_queue(
                name=queue_name,
                durable=self._settings.queue_durable,
                auto_delete=self._settings.queue_auto_delete
            )
            
            # Bind queue to exchange with routing keys
            for routing_key in routing_keys:
                await queue.bind(self._exchange, routing_key=routing_key)
            
            # Store queue
            self._queues[queue_name] = queue
            
            # Define message handler
            async def message_handler(message: AbstractIncomingMessage) -> None:
                async with message.process(auto_ack=auto_ack):
                    try:
                        # Get message body
                        body = message.body
                        
                        # Parse message based on content type
                        if message.content_type == "application/json":
                            try:
                                data = json.loads(body.decode())
                            except json.JSONDecodeError:
                                data = body.decode()
                        else:
                            data = body
                        
                        # Extract metadata
                        meta = MessageMetadata(
                            routing_key=message.routing_key,
                            content_type=message.content_type,
                            message_id=message.message_id,
                            timestamp=message.timestamp
                        )
                        
                        # Add headers to metadata
                        if message.headers:
                            for key, value in message.headers.items():
                                if hasattr(meta, key):
                                    setattr(meta, key, value)
                        
                        # Call callback
                        await callback(data, meta)
                        
                        # Acknowledge message if not auto_ack
                        if not auto_ack:
                            await message.ack()
                            
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        # Reject message if not auto_ack
                        if not auto_ack:
                            await message.reject(requeue=False)
            
            # Start consuming
            consumer_tag = await queue.consume(message_handler)
            self._consumer_tags.add(consumer_tag)
            
            logger.info(f"Subscribed to queue {queue_name} with routing keys {routing_keys}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to subscribe to queue: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    queue_name=queue_name,
                    routing_keys=routing_keys
                ),
                cause=e
            )
    
    async def unsubscribe(self, queue_name: str) -> None:
        """Unsubscribe from a queue.
        
        Args:
            queue_name: Queue name
            
        Raises:
            ProviderError: If unsubscribe fails
        """
        if not self._connection or not self._channel:
            raise ProviderError(
                message="Not connected to RabbitMQ",
                provider_name=self.name
            )
            
        try:
            # Get queue
            queue = self._queues.get(queue_name)
            if not queue:
                raise ProviderError(
                    message=f"Queue {queue_name} not found",
                    provider_name=self.name
                )
            
            # Cancel consumers
            await queue.cancel()
            
            # Remove queue
            del self._queues[queue_name]
            
            logger.info(f"Unsubscribed from queue {queue_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to unsubscribe from queue: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    queue_name=queue_name
                ),
                cause=e
            )
    
    async def create_queue(self, 
                          queue_name: str, 
                          routing_keys: Optional[List[str]] = None,
                          durable: Optional[bool] = None,
                          auto_delete: Optional[bool] = None) -> None:
        """Create a queue and optionally bind it to routing keys.
        
        Args:
            queue_name: Queue name
            routing_keys: Optional list of routing keys to bind
            durable: Whether queue is durable (overrides settings)
            auto_delete: Whether queue is auto-deleted (overrides settings)
            
        Raises:
            ProviderError: If queue creation fails
        """
        if not self._connection or not self._channel or not self._exchange:
            await self.initialize()
            
        try:
            # Use provided values or defaults from settings
            queue_durable = durable if durable is not None else self._settings.queue_durable
            queue_auto_delete = auto_delete if auto_delete is not None else self._settings.queue_auto_delete
            
            # Declare queue
            queue = await self._channel.declare_queue(
                name=queue_name,
                durable=queue_durable,
                auto_delete=queue_auto_delete
            )
            
            # Bind queue to exchange with routing keys
            if routing_keys:
                for routing_key in routing_keys:
                    await queue.bind(self._exchange, routing_key=routing_key)
            
            # Store queue
            self._queues[queue_name] = queue
            
            logger.info(f"Created queue {queue_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to create queue: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    queue_name=queue_name,
                    routing_keys=routing_keys
                ),
                cause=e
            )
    
    async def delete_queue(self, queue_name: str) -> None:
        """Delete a queue.
        
        Args:
            queue_name: Queue name
            
        Raises:
            ProviderError: If queue deletion fails
        """
        if not self._connection or not self._channel:
            await self.initialize()
            
        try:
            # Delete queue
            await self._channel.queue_delete(queue_name)
            
            # Remove queue from tracked queues
            if queue_name in self._queues:
                del self._queues[queue_name]
            
            logger.info(f"Deleted queue {queue_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete queue: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    queue_name=queue_name
                ),
                cause=e
            ) 