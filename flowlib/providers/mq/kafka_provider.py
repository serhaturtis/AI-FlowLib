"""Kafka message queue provider implementation.

This module provides a concrete implementation of the MQProvider
for Apache Kafka messaging using aiokafka.
"""

import logging
import asyncio
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable, Set
import uuid
from datetime import datetime
import io

from pydantic import Field

from ...core.errors import ProviderError, ErrorContext
from ...core.models.settings import ProviderSettings
from ...core.registry.decorators import provider
from ...core.registry.constants import ProviderType
from .base import MQProvider, MQProviderSettings, MessageMetadata

logger = logging.getLogger(__name__)

try:
    import aiokafka
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError
except ImportError:
    logger.warning("aiokafka package not found. Install with 'pip install aiokafka'")


class KafkaProviderSettings(MQProviderSettings):
    """Settings for Kafka provider.
    
    Attributes:
        bootstrap_servers: Comma-separated Kafka bootstrap servers (overrides host/port)
        client_id: Kafka client ID
        group_id: Consumer group ID
        auto_offset_reset: Offset reset strategy ('earliest', 'latest', 'none')
        enable_auto_commit: Whether to auto-commit offsets
        auto_commit_interval_ms: Auto-commit interval in milliseconds
        security_protocol: Security protocol ('PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL')
        sasl_mechanism: SASL mechanism (PLAIN, GSSAPI, SCRAM-SHA-256, SCRAM-SHA-512)
        sasl_username: SASL username
        sasl_password: SASL password
        ssl_context: SSL context
        ssl_check_hostname: Whether to check SSL hostname
        acks: Producer acks setting (0, 1, 'all')
        compression_type: Compression type ('gzip', 'snappy', 'lz4', None)
    """
    
    # Kafka connection settings
    bootstrap_servers: Optional[str] = None
    client_id: Optional[str] = None
    group_id: str = "flowlib_consumer_group"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    
    # Security settings
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_context: Optional[Any] = None
    ssl_check_hostname: bool = True
    
    # Producer settings
    acks: Union[str, int] = "all"  # all, 1, 0
    compression_type: Optional[str] = None  # gzip, snappy, lz4, None
    
    # Default port for Kafka if not specified
    port: int = 9092
    
    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict)


@provider(provider_type=ProviderType.MESSAGE_QUEUE, name="kafka")
class KafkaProvider(MQProvider):
    """Kafka implementation of the MQProvider.
    
    This provider implements message queue operations using aiokafka,
    an asynchronous client for Apache Kafka.
    """
    
    def __init__(self, name: str = "kafka", settings: Optional[KafkaProviderSettings] = None):
        """Initialize Kafka provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        super().__init__(name=name, settings=settings)
        self._settings = settings or KafkaProviderSettings(host="localhost")
        self._producer = None
        self._consumers = {}
        self._consumer_tasks = {}
        
    async def _initialize(self) -> None:
        """Initialize Kafka connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Get bootstrap servers
            bootstrap_servers = self._settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self._settings.host}:{self._settings.port}"
            
            # Prepare producer config
            producer_config = {
                "bootstrap_servers": bootstrap_servers,
                "acks": self._settings.acks,
                "compression_type": self._settings.compression_type,
                "security_protocol": self._settings.security_protocol,
                **self._settings.connect_args
            }
            
            # Add client ID if provided
            if self._settings.client_id:
                producer_config["client_id"] = self._settings.client_id
                
            # Add SASL settings if provided
            if self._settings.sasl_mechanism and self._settings.sasl_username and self._settings.sasl_password:
                producer_config["sasl_mechanism"] = self._settings.sasl_mechanism
                producer_config["sasl_plain_username"] = self._settings.sasl_username
                producer_config["sasl_plain_password"] = self._settings.sasl_password
                
            # Add SSL settings if provided
            if self._settings.ssl_context:
                producer_config["ssl_context"] = self._settings.ssl_context
                producer_config["ssl_check_hostname"] = self._settings.ssl_check_hostname
            
            # Create producer
            self._producer = AIOKafkaProducer(**producer_config)
            
            # Start producer
            await self._producer.start()
            
            logger.info(f"Connected to Kafka: {bootstrap_servers}")
            
        except Exception as e:
            self._producer = None
            raise ProviderError(
                message=f"Failed to connect to Kafka: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    bootstrap_servers=bootstrap_servers,
                    security_protocol=self._settings.security_protocol
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close Kafka connection."""
        try:
            # Stop all consumers
            for topic, consumer_task in self._consumer_tasks.items():
                if not consumer_task.done():
                    consumer_task.cancel()
                    
            for topic, consumer in self._consumers.items():
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.warning(f"Error stopping consumer for topic {topic}: {str(e)}")
            
            # Stop producer
            if self._producer:
                await self._producer.stop()
                self._producer = None
                logger.info("Stopped Kafka producer")
                
            # Clear consumers
            self._consumers = {}
            self._consumer_tasks = {}
                
        except Exception as e:
            logger.error(f"Error during Kafka shutdown: {str(e)}")
    
    async def publish(self, 
                     topic: str, 
                     message: Any, 
                     metadata: Optional[MessageMetadata] = None, 
                     key: Optional[str] = None) -> None:
        """Publish a message to Kafka.
        
        Args:
            topic: Kafka topic
            message: Message to publish
            metadata: Optional message metadata
            key: Optional message key
            
        Raises:
            ProviderError: If publish fails
        """
        if not self._producer:
            await self.initialize()
            
        try:
            # Serialize message
            if isinstance(message, (dict, list)):
                value = json.dumps(message).encode()
            elif isinstance(message, str):
                value = message.encode()
            elif isinstance(message, bytes):
                value = message
            else:
                value = str(message).encode()
            
            # Prepare key
            if key is None and metadata and metadata.message_id:
                key = metadata.message_id.encode()
            elif key is not None and isinstance(key, str):
                key = key.encode()
            
            # Prepare headers
            headers = []
            if metadata:
                metadata_dict = metadata.model_dump()
                for k, v in metadata_dict.items():
                    if v is not None:
                        if isinstance(v, (str, int, float, bool)):
                            headers.append((k, str(v).encode()))
            
            # Add timestamp as header
            timestamp = int(datetime.now().timestamp() * 1000)
            headers.append(("timestamp", str(timestamp).encode()))
            
            # Publish message
            await self._producer.send_and_wait(
                topic=topic,
                value=value,
                key=key,
                headers=headers
            )
            
            logger.debug(f"Published message to topic {topic}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to publish message: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    topic=topic,
                    headers=headers if 'headers' in locals() else None
                ),
                cause=e
            )
    
    async def subscribe(self, 
                       topic: str, 
                       callback: Callable[[Any, MessageMetadata], Any],
                       group_id: Optional[str] = None,
                       auto_offset_reset: Optional[str] = None) -> None:
        """Subscribe to messages from Kafka.
        
        Args:
            topic: Topic to subscribe to
            callback: Callback function
            group_id: Optional consumer group ID (overrides settings)
            auto_offset_reset: Optional offset reset strategy (overrides settings)
            
        Raises:
            ProviderError: If subscribe fails
        """
        if not self._producer:  # We just need any connection
            await self.initialize()
            
        try:
            # Check if already subscribed
            if topic in self._consumers:
                raise ProviderError(
                    message=f"Already subscribed to topic {topic}",
                    provider_name=self.name,
                    context=ErrorContext.create(topic=topic)
                )
            
            # Get bootstrap servers
            bootstrap_servers = self._settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self._settings.host}:{self._settings.port}"
            
            # Use provided values or defaults from settings
            consumer_group_id = group_id or self._settings.group_id
            consumer_auto_offset_reset = auto_offset_reset or self._settings.auto_offset_reset
            
            # Prepare consumer config
            consumer_config = {
                "bootstrap_servers": bootstrap_servers,
                "group_id": consumer_group_id,
                "auto_offset_reset": consumer_auto_offset_reset,
                "enable_auto_commit": self._settings.enable_auto_commit,
                "auto_commit_interval_ms": self._settings.auto_commit_interval_ms,
                "security_protocol": self._settings.security_protocol,
                **self._settings.connect_args
            }
            
            # Add client ID if provided
            if self._settings.client_id:
                consumer_config["client_id"] = f"{self._settings.client_id}_consumer"
                
            # Add SASL settings if provided
            if self._settings.sasl_mechanism and self._settings.sasl_username and self._settings.sasl_password:
                consumer_config["sasl_mechanism"] = self._settings.sasl_mechanism
                consumer_config["sasl_plain_username"] = self._settings.sasl_username
                consumer_config["sasl_plain_password"] = self._settings.sasl_password
                
            # Add SSL settings if provided
            if self._settings.ssl_context:
                consumer_config["ssl_context"] = self._settings.ssl_context
                consumer_config["ssl_check_hostname"] = self._settings.ssl_check_hostname
            
            # Create consumer
            consumer = AIOKafkaConsumer(topic, **consumer_config)
            
            # Store consumer
            self._consumers[topic] = consumer
            
            # Define consumer task
            async def consume_task():
                try:
                    await consumer.start()
                    logger.info(f"Started consuming from topic {topic} with group ID {consumer_group_id}")
                    
                    async for message in consumer:
                        try:
                            # Get message value
                            value = message.value
                            
                            # Parse message based on content
                            try:
                                data = json.loads(value.decode())
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                data = value
                            
                            # Extract metadata from headers
                            meta_dict = {
                                "topic": message.topic,
                                "partition": message.partition,
                                "offset": message.offset,
                                "timestamp": message.timestamp,
                                "message_id": message.key.decode() if message.key else None,
                                "routing_key": message.topic
                            }
                            
                            # Add headers to metadata
                            if message.headers:
                                for key, value in message.headers:
                                    key_str = key.decode() if isinstance(key, bytes) else key
                                    try:
                                        value_str = value.decode() if isinstance(value, bytes) else value
                                        meta_dict[key_str] = value_str
                                    except UnicodeDecodeError:
                                        # Skip binary headers
                                        pass
                            
                            # Create metadata object
                            meta = MessageMetadata(**meta_dict)
                            
                            # Call callback
                            await callback(data, meta)
                            
                        except Exception as e:
                            logger.error(f"Error processing Kafka message: {str(e)}")
                            # Continue consuming
                            continue
                            
                except asyncio.CancelledError:
                    logger.info(f"Stopped consuming from topic {topic}")
                    await consumer.stop()
                    
                except Exception as e:
                    logger.error(f"Error in Kafka consumer task: {str(e)}")
                    await consumer.stop()
                    # Remove from consumers
                    if topic in self._consumers:
                        del self._consumers[topic]
                    if topic in self._consumer_tasks:
                        del self._consumer_tasks[topic]
            
            # Start consumer task
            task = asyncio.create_task(consume_task())
            self._consumer_tasks[topic] = task
            
        except Exception as e:
            # Cleanup on error
            if topic in self._consumers:
                del self._consumers[topic]
            if topic in self._consumer_tasks:
                del self._consumer_tasks[topic]
                
            raise ProviderError(
                message=f"Failed to subscribe to topic: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    topic=topic,
                    group_id=consumer_group_id if 'consumer_group_id' in locals() else None
                ),
                cause=e
            )
    
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            
        Raises:
            ProviderError: If unsubscribe fails
        """
        try:
            # Check if subscribed
            if topic not in self._consumers:
                raise ProviderError(
                    message=f"Not subscribed to topic {topic}",
                    provider_name=self.name,
                    context=ErrorContext.create(topic=topic)
                )
            
            # Get consumer and task
            consumer = self._consumers[topic]
            task = self._consumer_tasks.get(topic)
            
            # Cancel task
            if task and not task.done():
                task.cancel()
                
            # Stop consumer
            if consumer:
                await consumer.stop()
            
            # Remove from maps
            del self._consumers[topic]
            if topic in self._consumer_tasks:
                del self._consumer_tasks[topic]
                
            logger.info(f"Unsubscribed from topic {topic}")
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise ProviderError(
                message=f"Failed to unsubscribe from topic: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(topic=topic),
                cause=e
            )
    
    async def create_topic(self, 
                          topic: str, 
                          num_partitions: int = 1,
                          replication_factor: int = 1) -> None:
        """Create a Kafka topic.
        
        Note: This requires admin privileges and may not work in all Kafka deployments.
        
        Args:
            topic: Topic name
            num_partitions: Number of partitions
            replication_factor: Replication factor
            
        Raises:
            ProviderError: If topic creation fails
        """
        if not self._producer:
            await self.initialize()
            
        try:
            # Import kafka-python for admin operations
            try:
                from kafka.admin import KafkaAdminClient, NewTopic
                from kafka.errors import TopicAlreadyExistsError
            except ImportError:
                raise ProviderError(
                    message="kafka-python package not found. Install with 'pip install kafka-python'",
                    provider_name=self.name,
                    context=ErrorContext.create(topic=topic)
                )
            
            # Get bootstrap servers
            bootstrap_servers = self._settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self._settings.host}:{self._settings.port}"
                
            # Create admin client
            admin_config = {
                "bootstrap_servers": bootstrap_servers.split(",") if "," in bootstrap_servers else bootstrap_servers,
                "security_protocol": self._settings.security_protocol
            }
            
            # Add client ID if provided
            if self._settings.client_id:
                admin_config["client_id"] = f"{self._settings.client_id}_admin"
                
            # Add SASL settings if provided
            if self._settings.sasl_mechanism and self._settings.sasl_username and self._settings.sasl_password:
                admin_config["sasl_mechanism"] = self._settings.sasl_mechanism
                admin_config["sasl_plain_username"] = self._settings.sasl_username
                admin_config["sasl_plain_password"] = self._settings.sasl_password
            
            admin_client = KafkaAdminClient(**admin_config)
            
            # Create topic
            topic_list = [
                NewTopic(
                    name=topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor
                )
            ]
            
            # Create topics
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            
            # Close admin client
            admin_client.close()
            
            logger.info(f"Created Kafka topic {topic} with {num_partitions} partitions and replication factor {replication_factor}")
            
        except Exception as e:
            # Check if topic already exists
            if hasattr(e, "__class__") and e.__class__.__name__ == "TopicAlreadyExistsError":
                logger.info(f"Kafka topic {topic} already exists")
                return
                
            raise ProviderError(
                message=f"Failed to create Kafka topic: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(
                    topic=topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor
                ),
                cause=e
            )
    
    async def delete_topic(self, topic: str) -> None:
        """Delete a Kafka topic.
        
        Note: This requires admin privileges and may not work in all Kafka deployments.
        
        Args:
            topic: Topic name
            
        Raises:
            ProviderError: If topic deletion fails
        """
        if not self._producer:
            await self.initialize()
            
        try:
            # Import kafka-python for admin operations
            try:
                from kafka.admin import KafkaAdminClient
                from kafka.errors import UnknownTopicOrPartitionError
            except ImportError:
                raise ProviderError(
                    message="kafka-python package not found. Install with 'pip install kafka-python'",
                    provider_name=self.name,
                    context=ErrorContext.create(topic=topic)
                )
            
            # Get bootstrap servers
            bootstrap_servers = self._settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self._settings.host}:{self._settings.port}"
                
            # Create admin client
            admin_config = {
                "bootstrap_servers": bootstrap_servers.split(",") if "," in bootstrap_servers else bootstrap_servers,
                "security_protocol": self._settings.security_protocol
            }
            
            # Add client ID if provided
            if self._settings.client_id:
                admin_config["client_id"] = f"{self._settings.client_id}_admin"
                
            # Add SASL settings if provided
            if self._settings.sasl_mechanism and self._settings.sasl_username and self._settings.sasl_password:
                admin_config["sasl_mechanism"] = self._settings.sasl_mechanism
                admin_config["sasl_plain_username"] = self._settings.sasl_username
                admin_config["sasl_plain_password"] = self._settings.sasl_password
            
            admin_client = KafkaAdminClient(**admin_config)
            
            # Unsubscribe if subscribed
            if topic in self._consumers:
                await self.unsubscribe(topic)
            
            # Delete topic
            admin_client.delete_topics([topic])
            
            # Close admin client
            admin_client.close()
            
            logger.info(f"Deleted Kafka topic {topic}")
            
        except Exception as e:
            # Check if topic doesn't exist
            if hasattr(e, "__class__") and e.__class__.__name__ == "UnknownTopicOrPartitionError":
                logger.info(f"Kafka topic {topic} does not exist")
                return
                
            raise ProviderError(
                message=f"Failed to delete Kafka topic: {str(e)}",
                provider_name=self.name,
                context=ErrorContext.create(topic=topic),
                cause=e
            ) 