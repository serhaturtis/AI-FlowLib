"""Event system for the Integrated Agent System."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from ..models.base import Event, EventType
from ..utils.errors import EventError

logger = logging.getLogger(__name__)

class EventBus:
    """Event bus for publishing and subscribing to events."""
    
    def __init__(self, max_retry_count: int = 3):
        """Initialize event bus.
        
        Args:
            max_retry_count: Maximum number of retries for failed event delivery
        """
        self._subscribers: Dict[EventType, Set[callable]] = {}
        self._domain_subscribers: Dict[str, Set[callable]] = {}
        self._correlation_subscribers: Dict[UUID, Set[callable]] = {}
        self._dead_letter_queue: List[Event] = []
        self._max_retry_count = max_retry_count
        self._retry_delays = [1, 5, 15]  # Retry delays in seconds
        
    async def publish(
        self,
        event: Event,
        retry_count: int = 0
    ) -> None:
        """Publish an event to subscribers.
        
        Args:
            event: Event to publish
            retry_count: Current retry attempt number
            
        Raises:
            EventError: If event delivery fails after max retries
        """
        try:
            # Get all relevant subscribers
            subscribers = set()
            
            # Type subscribers
            if event.type in self._subscribers:
                subscribers.update(self._subscribers[event.type])
                
            # Domain subscribers
            if event.domain in self._domain_subscribers:
                subscribers.update(self._domain_subscribers[event.domain])
                
            # Correlation subscribers
            if event.correlation_id and event.correlation_id in self._correlation_subscribers:
                subscribers.update(self._correlation_subscribers[event.correlation_id])
                
            # Deliver event to subscribers
            failed_deliveries = []
            for subscriber in subscribers:
                try:
                    await subscriber(event)
                except Exception as e:
                    logger.error(
                        f"Failed to deliver event {event.id} to subscriber: {str(e)}"
                    )
                    failed_deliveries.append((subscriber, e))
                    
            # Handle failed deliveries
            if failed_deliveries:
                if retry_count < self._max_retry_count:
                    # Wait before retry
                    delay = self._retry_delays[min(retry_count, len(self._retry_delays) - 1)]
                    await asyncio.sleep(delay)
                    
                    # Retry failed deliveries
                    for subscriber, error in failed_deliveries:
                        try:
                            await subscriber(event)
                        except Exception as e:
                            logger.error(
                                f"Retry {retry_count + 1} failed for event {event.id}: {str(e)}"
                            )
                            if retry_count + 1 == self._max_retry_count:
                                # Add to dead letter queue
                                self._dead_letter_queue.append(event)
                                logger.warning(
                                    f"Event {event.id} added to dead letter queue after "
                                    f"{self._max_retry_count} failed delivery attempts"
                                )
                            else:
                                # Try again
                                await self.publish(event, retry_count + 1)
                                
        except Exception as e:
            raise EventError(f"Failed to publish event: {str(e)}") from e
            
    def subscribe_to_type(
        self,
        event_type: EventType,
        callback: callable
    ) -> None:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Callback function for events
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)
        
    def subscribe_to_domain(
        self,
        domain: str,
        callback: callable
    ) -> None:
        """Subscribe to events from a specific domain.
        
        Args:
            domain: Domain to subscribe to
            callback: Callback function for events
        """
        if domain not in self._domain_subscribers:
            self._domain_subscribers[domain] = set()
        self._domain_subscribers[domain].add(callback)
        
    def subscribe_to_correlation(
        self,
        correlation_id: UUID,
        callback: callable
    ) -> None:
        """Subscribe to events with a specific correlation ID.
        
        Args:
            correlation_id: Correlation ID to subscribe to
            callback: Callback function for events
        """
        if correlation_id not in self._correlation_subscribers:
            self._correlation_subscribers[correlation_id] = set()
        self._correlation_subscribers[correlation_id].add(callback)
        
    def unsubscribe_from_type(
        self,
        event_type: EventType,
        callback: callable
    ) -> None:
        """Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
            
    def unsubscribe_from_domain(
        self,
        domain: str,
        callback: callable
    ) -> None:
        """Unsubscribe from events from a specific domain.
        
        Args:
            domain: Domain to unsubscribe from
            callback: Callback function to remove
        """
        if domain in self._domain_subscribers:
            self._domain_subscribers[domain].discard(callback)
            
    def unsubscribe_from_correlation(
        self,
        correlation_id: UUID,
        callback: callable
    ) -> None:
        """Unsubscribe from events with a specific correlation ID.
        
        Args:
            correlation_id: Correlation ID to unsubscribe from
            callback: Callback function to remove
        """
        if correlation_id in self._correlation_subscribers:
            self._correlation_subscribers[correlation_id].discard(callback)
            
    def get_dead_letter_queue(self) -> List[Event]:
        """Get events in the dead letter queue.
        
        Returns:
            List of events that failed delivery
        """
        return self._dead_letter_queue.copy()
        
    def clear_dead_letter_queue(self) -> None:
        """Clear the dead letter queue."""
        self._dead_letter_queue.clear()
        
    async def retry_dead_letter_queue(self) -> None:
        """Attempt to redeliver events in the dead letter queue."""
        events = self._dead_letter_queue.copy()
        self._dead_letter_queue.clear()
        
        for event in events:
            try:
                await self.publish(event)
            except Exception as e:
                logger.error(
                    f"Failed to redeliver event {event.id} from dead letter queue: {str(e)}"
                )
                self._dead_letter_queue.append(event) 