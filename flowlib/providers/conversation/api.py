"""API conversation provider implementation.

This module provides a RESTful API interface for agent conversations,
enabling integration with external applications.
"""

import logging
import asyncio
import json
import os
from typing import Optional, Dict, Any, List, Set, Deque
from collections import deque
import uuid
from aiohttp import web

from ...core.registry.decorators import conversation_provider
from .base import ConversationProvider, ConversationProviderSettings

logger = logging.getLogger(__name__)

class Conversation:
    """Represents an active conversation with an agent.
    
    Each conversation has a unique ID, a queue for incoming messages,
    and a history of exchanged messages.
    """
    
    def __init__(self, conversation_id: str):
        """Initialize a conversation.
        
        Args:
            conversation_id: Unique ID for the conversation
        """
        self.id = conversation_id
        self.messages = []
        self.queue = asyncio.Queue()
        self.last_activity = asyncio.get_event_loop().time()
        
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = asyncio.get_event_loop().time()
        
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.
        
        Args:
            role: Role of the sender ("user" or "assistant")
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        })
        self.update_activity()


class APIConversationProviderSettings(ConversationProviderSettings):
    """Settings for API conversation provider.
    
    Attributes:
        host: Host address to bind to
        port: Port to listen on
        inactive_timeout: Timeout for inactive conversations in seconds
        max_history_size: Maximum number of messages to keep in history
    """
    host: str = "localhost"
    port: int = 8081
    inactive_timeout: int = 3600  # 1 hour
    max_history_size: int = 100


@conversation_provider("api")
class APIConversationProvider(ConversationProvider[APIConversationProviderSettings]):
    """API-based conversation provider.
    
    This provider enables conversations through a RESTful API interface,
    making it easy to integrate with external applications.
    """
    
    def __init__(
        self, 
        name: str = "api", 
        settings: Optional[APIConversationProviderSettings] = None,
        provider_type: str = "conversation"
    ):
        """Initialize the API conversation provider.
        
        Args:
            name: Provider name
            settings: Provider settings
            provider_type: Provider type
        """
        # Use default settings if none provided
        if settings is None:
            settings = APIConversationProviderSettings()
            
        super().__init__(name=name, settings=settings, provider_type=provider_type)
        
        # Initialize app and conversations
        self.app = web.Application()
        self.app.add_routes([
            web.post('/api/conversations', self.create_conversation),
            web.get('/api/conversations', self.list_conversations),
            web.get('/api/conversations/{conversation_id}', self.get_conversation),
            web.post('/api/conversations/{conversation_id}/messages', self.send_message),
            web.get('/api/conversations/{conversation_id}/messages', self.get_messages),
            web.delete('/api/conversations/{conversation_id}', self.delete_conversation),
        ])
        
        # Initialize conversations
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation: Optional[Conversation] = None
        
        # Start cleanup task for inactive conversations
        self.cleanup_task = None
        
    async def initialize(self):
        """Initialize the API server."""
        await super().initialize()
        
        # Start the web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.settings.host, self.settings.port)
        await self.site.start()
        logger.info(f"API interface started at http://{self.settings.host}:{self.settings.port}")
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_inactive_conversations())
    
    async def get_next_input(self) -> Optional[str]:
        """Get input from API.
        
        This method waits for messages from the current conversation.
        """
        try:
            # If we don't have a current conversation, wait for one
            if not self.current_conversation or not self.conversations:
                for _ in range(10):  # Try for a short time
                    if self.conversations:
                        # Choose the conversation with the most recent activity
                        self.current_conversation = max(
                            self.conversations.values(),
                            key=lambda c: c.last_activity
                        )
                        break
                    await asyncio.sleep(0.1)
                
                if not self.current_conversation:
                    return None
            
            # Get the next message from the current conversation queue
            user_input = await self.current_conversation.queue.get()
            self.current_conversation.add_message("user", user_input)
            return user_input
            
        except asyncio.CancelledError:
            logger.info("API interface input task cancelled")
            return None
        except Exception as e:
            logger.error(f"Error getting next input: {e}")
            return None
    
    async def send_response(self, response: str):
        """Store the response in the current conversation."""
        if not self.current_conversation:
            logger.warning("No current conversation to send response to")
            return
            
        # Store the response
        self.current_conversation.last_response = response
        self.current_conversation.add_message("agent", response)
    
    async def show_details(self, details: Dict[str, Any]):
        """Store execution details in the current conversation."""
        if not self.current_conversation:
            return
            
        # Store details
        self.current_conversation.details = details
    
    async def handle_error(self, error: Exception) -> str:
        """Handle errors during conversation."""
        error_message = await super().handle_error(error)
        
        if self.current_conversation:
            # Store error message
            self.current_conversation.last_error = error_message
            self.current_conversation.add_message("system", f"Error: {error_message}")
        
        return error_message
    
    async def create_conversation(self, request):
        """Handle requests to create a new conversation."""
        # Check authorization if token is set
        if self.auth_token and request.headers.get('Authorization') != f"Bearer {self.auth_token}":
            return web.json_response({"error": "Unauthorized"}, status=401)
            
        # Create a new conversation
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = Conversation(conversation_id)
        logger.info(f"Created new conversation: {conversation_id}")
        
        # If this is the first conversation, make it the current one
        if len(self.conversations) == 1:
            self.current_conversation = self.conversations[conversation_id]
        
        return web.json_response({
            "conversation_id": conversation_id,
            "created_at": asyncio.get_event_loop().time()
        })
    
    async def list_conversations(self, request):
        """Handle requests to list all conversations."""
        # Check authorization if token is set
        if self.auth_token and request.headers.get('Authorization') != f"Bearer {self.auth_token}":
            return web.json_response({"error": "Unauthorized"}, status=401)
            
        # Return list of conversations
        conversations = []
        for conv_id, conv in self.conversations.items():
            conversations.append({
                "id": conv_id,
                "message_count": len(conv.messages),
                "last_activity": conv.last_activity
            })
        
        return web.json_response({
            "conversations": conversations,
            "count": len(conversations)
        })
    
    async def get_conversation(self, request):
        """Handle requests to get conversation details."""
        # Check authorization if token is set
        if self.auth_token and request.headers.get('Authorization') != f"Bearer {self.auth_token}":
            return web.json_response({"error": "Unauthorized"}, status=401)
            
        # Get conversation ID from URL
        conversation_id = request.match_info.get('conversation_id')
        if conversation_id not in self.conversations:
            return web.json_response({"error": "Conversation not found"}, status=404)
        
        # Get conversation
        conv = self.conversations[conversation_id]
        
        return web.json_response({
            "id": conv.id,
            "message_count": len(conv.messages),
            "last_activity": conv.last_activity,
            "last_response": conv.last_response,
            "last_error": conv.last_error
        })
    
    async def send_message(self, request):
        """Handle requests to send a message to a conversation."""
        # Check authorization if token is set
        if self.auth_token and request.headers.get('Authorization') != f"Bearer {self.auth_token}":
            return web.json_response({"error": "Unauthorized"}, status=401)
            
        # Get conversation ID from URL
        conversation_id = request.match_info.get('conversation_id')
        if conversation_id not in self.conversations:
            return web.json_response({"error": "Conversation not found"}, status=404)
        
        # Get conversation
        conv = self.conversations[conversation_id]
        
        # Get message from request body
        try:
            data = await request.json()
            if 'message' not in data:
                return web.json_response({"error": "Missing 'message' field"}, status=400)
            
            # Put message in the queue
            await conv.queue.put(data['message'])
            
            # Make this the current conversation
            self.current_conversation = conv
            
            # Update last activity
            conv.update_activity()
            
            return web.json_response({
                "status": "message_queued",
                "conversation_id": conversation_id
            })
            
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
    
    async def get_messages(self, request):
        """Handle requests to get messages from a conversation."""
        # Check authorization if token is set
        if self.auth_token and request.headers.get('Authorization') != f"Bearer {self.auth_token}":
            return web.json_response({"error": "Unauthorized"}, status=401)
            
        # Get conversation ID from URL
        conversation_id = request.match_info.get('conversation_id')
        if conversation_id not in self.conversations:
            return web.json_response({"error": "Conversation not found"}, status=404)
        
        # Get conversation
        conv = self.conversations[conversation_id]
        
        # Get messages
        messages = list(conv.messages)
        
        # Update last activity
        conv.update_activity()
        
        # Prepare details - ensure they are JSON serializable
        details = None
        if hasattr(conv, "details") and conv.details:
            try:
                # Try to serialize details to JSON to verify they're serializable
                details_json = json.dumps(conv.details)
                details = conv.details
            except (TypeError, ValueError) as e:
                logger.warning(f"Error serializing details: {e}")
                details = {
                    "error": "Details contain non-serializable objects and cannot be displayed"
                }
        
        try:
            # Try to create the response
            response_data = {
                "conversation_id": conversation_id,
                "messages": messages,
                "details": details
            }
            
            # Verify the whole response is serializable
            json.dumps(response_data)
            
            return web.json_response(response_data)
        except (TypeError, ValueError) as e:
            # If serialization fails, return a simplified response
            logger.error(f"Error creating JSON response: {e}")
            safe_response = {
                "conversation_id": conversation_id, 
                "error": "Some data could not be serialized to JSON",
                "message_count": len(messages)
            }
            return web.json_response(safe_response)
    
    async def delete_conversation(self, request):
        """Handle requests to delete a conversation."""
        # Check authorization if token is set
        if self.auth_token and request.headers.get('Authorization') != f"Bearer {self.auth_token}":
            return web.json_response({"error": "Unauthorized"}, status=401)
            
        # Get conversation ID from URL
        conversation_id = request.match_info.get('conversation_id')
        if conversation_id not in self.conversations:
            return web.json_response({"error": "Conversation not found"}, status=404)
        
        # Delete conversation
        if self.current_conversation and self.current_conversation.id == conversation_id:
            self.current_conversation = None
        
        del self.conversations[conversation_id]
        
        return web.json_response({
            "status": "conversation_deleted",
            "conversation_id": conversation_id
        })
    
    async def _cleanup_inactive_conversations(self):
        """Periodically clean up inactive conversations."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = asyncio.get_event_loop().time()
                to_delete = []
                
                for conv_id, conv in self.conversations.items():
                    if current_time - conv.last_activity > self.settings.inactive_timeout:
                        to_delete.append(conv_id)
                
                for conv_id in to_delete:
                    logger.info(f"Cleaning up inactive conversation: {conv_id}")
                    if self.current_conversation and self.current_conversation.id == conv_id:
                        self.current_conversation = None
                    del self.conversations[conv_id]
        
        except asyncio.CancelledError:
            logger.info("Conversation cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in conversation cleanup: {e}")

    async def shutdown(self):
        """Cleanup the API server resources."""
        logger.info("Shutting down API server...")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Cleanup task cancelled")
        
        # Stop web server
        if hasattr(self, 'site') and self.site:
            await self.site.stop()
            logger.info("API site stopped")
        
        if hasattr(self, 'runner') and self.runner:
            await self.runner.cleanup()
            logger.info("API runner cleaned up")
            
        # Clear conversations
        self.conversations.clear()
        logger.info("Cleared all conversations") 