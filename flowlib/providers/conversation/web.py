"""Web conversation provider implementation.

This module provides a web-based interface for agent conversations,
allowing users to interact with agents through a browser.
"""

import logging
import asyncio
import json
import os
import aiohttp
from aiohttp import web
from typing import Optional, Dict, Any, Set

from ..decorators import conversation_provider
from .base import ConversationProvider, ConversationProviderSettings
from ...utils.formatting import format_execution_details

logger = logging.getLogger(__name__)

class WebConversationProviderSettings(ConversationProviderSettings):
    """Settings for web conversation provider.
    
    Attributes:
        host: Host address to bind to
        port: Port to listen on
        static_path: Path to static files for the web UI
        debug: Whether to run in debug mode
    """
    host: str = "localhost"
    port: int = 8080
    static_path: Optional[str] = None
    debug: bool = False

@conversation_provider("web")
class WebConversationProvider(ConversationProvider[WebConversationProviderSettings]):
    """Web-based conversation provider.
    
    This provider facilitates conversations through a web interface,
    using websockets for real-time communication and a simple HTML/JS frontend.
    """
    
    def __init__(
        self, 
        name: str = "web", 
        settings: Optional[WebConversationProviderSettings] = None,
        provider_type: str = "conversation"
    ):
        """Initialize the web conversation provider.
        
        Args:
            name: Provider name
            settings: Provider settings
            provider_type: Provider type
        """
        # Use default settings if none provided
        if settings is None:
            settings = WebConversationProviderSettings()
            
        super().__init__(name=name, settings=settings, provider_type=provider_type)
        
        # Get settings with defaults
        self.host = self.settings.host
        self.port = self.settings.port
        self.static_path = self.settings.static_path
        self.show_execution_details = self.settings.debug
        
        # Create the static directory if it doesn't exist
        if not os.path.exists(self.static_path):
            os.makedirs(self.static_path)
            
        # Initialize app and websocket connections
        self.app = web.Application()
        self.websockets: Set[web.WebSocketResponse] = set()
        self.app.add_routes([
            web.get('/', self.handle_index),
            web.get('/ws', self.handle_websocket),
            web.static('/static', self.static_path)
        ])
        
        # Initialize input queue for receiving messages
        self.input_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize the web server."""
        await super().initialize()
        
        # Create static files if they don't exist
        static_files = {
            "index.html": self._get_default_html(),
            "chat.js": self._get_default_js(),
            "style.css": self._get_default_css()
        }
        
        # Create each file if it doesn't exist
        for filename, content in static_files.items():
            file_path = os.path.join(self.static_path, filename)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write(content)
                logger.info(f"Created file: {file_path}")
                
        # Start the web server in a background task
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        logger.info(f"Web interface started at http://{self.host}:{self.port}")
    
    async def get_next_input(self) -> Optional[str]:
        """Get input from the web interface.
        
        This method waits for messages from websocket connections.
        """
        try:
            # Get the next message from the queue
            user_input = await self.input_queue.get()
            return user_input
        except asyncio.CancelledError:
            logger.info("Web interface input task cancelled")
            return None
    
    async def send_response(self, response: str):
        """Send a response to all connected clients."""
        if not self.websockets:
            logger.warning("No websocket connections to send response to")
            return
            
        # Send response to all connected clients
        message = {
            "type": "response",
            "content": response
        }
        await self._broadcast(message)
    
    async def show_details(self, details: Dict[str, Any]):
        """Send execution details to all connected clients."""
        if not self.show_execution_details or not self.websockets:
            return
            
        # Extract and format execution details for client display
        formatted_details = self._format_execution_details(details)
        message = {
            "type": "details",
            "content": formatted_details
        }
        await self._broadcast(message)
        
    async def handle_error(self, error: Exception) -> str:
        """Handle errors during conversation."""
        error_message = await super().handle_error(error)
        
        # Send error message to all connected clients
        message = {
            "type": "error",
            "content": error_message
        }
        await self._broadcast(message)
        
        return error_message
        
    async def handle_index(self, request):
        """Handle requests to the index page."""
        return web.FileResponse(os.path.join(self.static_path, "index.html"))
        
    async def handle_websocket(self, request):
        """Handle websocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        logger.info(f"New websocket connection, total: {len(self.websockets)}")
        
        try:
            # Send welcome message
            await ws.send_json({
                "type": "system",
                "content": "Connected to agent. Type a message to begin."
            })
            
            # Handle messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "message":
                        # Put the message in the queue for processing
                        await self.input_queue.put(data.get("content"))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Websocket connection closed with exception {ws.exception()}")
        finally:
            self.websockets.remove(ws)
            logger.info(f"Websocket connection closed, remaining: {len(self.websockets)}")
            
        return ws
        
    async def _broadcast(self, message: Dict[str, Any]):
        """Send a message to all connected clients."""
        disconnected = set()
        for ws in self.websockets:
            try:
                # Ensure message is JSON serializable
                try:
                    # First try to serialize to check for any issues
                    json.dumps(message)
                    await ws.send_json(message)
                except TypeError as e:
                    # If there's a serialization error, make a safer version of the message
                    logger.warning(f"JSON serialization error: {e}. Converting to safe format.")
                    
                    # Create a sanitized copy with simple string representation for complex objects
                    if message.get("type") == "details" and "content" in message:
                        safe_message = {
                            "type": message["type"],
                            "content": {
                                "progress": message["content"].get("progress", 0),
                                "complete": message["content"].get("complete", False),
                                "error": f"Some details couldn't be displayed due to serialization error: {e}"
                            }
                        }
                    else:
                        # For other message types, just convert content to string if needed
                        safe_message = {
                            "type": message["type"],
                            "content": str(message.get("content", ""))
                        }
                    
                    await ws.send_json(safe_message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected.add(ws)
                
        # Remove disconnected clients
        for ws in disconnected:
            self.websockets.remove(ws)
            
    def _format_execution_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Format execution details for display in the web interface."""
        # Use the shared formatting utility
        return format_execution_details(details)
        
    def _get_default_html(self) -> str:
        """Get default HTML for the chat interface."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Chat Interface</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>Agent Chat Interface</h1>
        </div>
        <div class="chat-messages" id="chatMessages"></div>
        <div class="execution-details" id="executionDetails"></div>
        <div class="chat-input-container">
            <input type="text" id="messageInput" placeholder="Type your message..." />
            <button id="sendButton">Send</button>
        </div>
    </div>
    <script src="/static/chat.js"></script>
</body>
</html>"""

    def _get_default_js(self) -> str:
        """Get default JavaScript for the chat interface."""
        return """document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const chatMessages = document.getElementById('chatMessages');
    const executionDetails = document.getElementById('executionDetails');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    // WebSocket connection
    let socket = new WebSocket(`ws://${location.host}/ws`);
    
    // Connection event handlers
    socket.onopen = function(e) {
        console.log('WebSocket connection established');
    };
    
    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        switch(data.type) {
            case 'system':
                addSystemMessage(data.content);
                break;
            case 'response':
                addAgentMessage(data.content);
                break;
            case 'error':
                addErrorMessage(data.content);
                break;
            case 'details':
                updateExecutionDetails(data.content);
                break;
        }
    };
    
    socket.onclose = function(event) {
        if (event.wasClean) {
            addSystemMessage(`Connection closed: ${event.reason}`);
        } else {
            addSystemMessage('Connection lost. Please refresh the page.');
        }
    };
    
    socket.onerror = function(error) {
        addErrorMessage('WebSocket error');
        console.error('WebSocket error:', error);
    };
    
    // Send message function
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addUserMessage(message);
        
        // Send to server
        socket.send(JSON.stringify({
            type: 'message',
            content: message
        }));
        
        // Clear input
        messageInput.value = '';
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Message display functions
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addAgentMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message agent-message';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addSystemMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message system-message';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addErrorMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message error-message';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function updateExecutionDetails(details) {
        // Clear previous details
        executionDetails.innerHTML = '';
        
        // Create header
        const header = document.createElement('div');
        header.className = 'details-header';
        header.textContent = 'Execution Details';
        executionDetails.appendChild(header);
        
        // Create progress section
        const progress = document.createElement('div');
        progress.className = 'details-section';
        progress.innerHTML = `<span class="details-label">Progress:</span> ${details.progress}%, <span class="details-label">Complete:</span> ${details.complete}`;
        executionDetails.appendChild(progress);
        
        // Create planning section if available
        if (details.planning) {
            const planning = document.createElement('div');
            planning.className = 'details-section';
            planning.innerHTML = `
                <div class="details-subtitle">Planning</div>
                <div><span class="details-label">Reasoning:</span> ${details.planning.reasoning}</div>
                <div><span class="details-label">Flow:</span> ${details.planning.flow}</div>
            `;
            executionDetails.appendChild(planning);
        }
        
        // Create executions section if available
        if (details.executions && details.executions.length > 0) {
            const executions = document.createElement('div');
            executions.className = 'details-section';
            executions.innerHTML = `<div class="details-subtitle">Recent Executions</div>`;
            
            const list = document.createElement('ul');
            details.executions.forEach(execution => {
                const item = document.createElement('li');
                item.textContent = `${execution.action} - ${execution.flow}`;
                list.appendChild(item);
            });
            
            executions.appendChild(list);
            executionDetails.appendChild(executions);
        }
        
        // Create reflection section if available
        if (details.reflection) {
            const reflection = document.createElement('div');
            reflection.className = 'details-section';
            reflection.innerHTML = `
                <div class="details-subtitle">Latest Reflection</div>
                <div>${details.reflection.text}</div>
            `;
            
            executionDetails.appendChild(reflection);
        }
    }
});"""

    def _get_default_css(self) -> str:
        """Get default CSS for the chat interface."""
        return """body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.chat-container {
    width: 90%;
    max-width: 1200px;
    height: 90vh;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
}

.chat-header {
    padding: 15px;
    border-bottom: 1px solid #eaeaea;
    text-align: center;
}

.chat-header h1 {
    margin: 0;
    font-size: 1.5rem;
    color: #333;
}

.chat-messages {
    flex: 3;
    overflow-y: auto;
    padding: 15px;
    display: flex;
    flex-direction: column;
}

.execution-details {
    flex: 2;
    overflow-y: auto;
    padding: 15px;
    background-color: #f9f9f9;
    border-top: 1px solid #eaeaea;
}

.chat-input-container {
    padding: 15px;
    border-top: 1px solid #eaeaea;
    display: flex;
}

.chat-input-container input {
    flex: 1;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1rem;
}

.chat-input-container button {
    margin-left: 10px;
    padding: 10px 20px;
    background-color: #4a69bd;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
}

.chat-input-container button:hover {
    background-color: #3c56a5;
}

.message {
    margin-bottom: 10px;
    padding: 10px 15px;
    border-radius: 18px;
    max-width: 70%;
    word-wrap: break-word;
}

.user-message {
    align-self: flex-end;
    background-color: #4a69bd;
    color: white;
    border-bottom-right-radius: 4px;
}

.agent-message {
    align-self: flex-start;
    background-color: #e9e9e9;
    color: #333;
    border-bottom-left-radius: 4px;
}

.system-message, .error-message {
    align-self: center;
    background-color: #f8f8f8;
    color: #666;
    font-style: italic;
    border-radius: 4px;
    max-width: 90%;
    padding: 5px 10px;
    font-size: 0.9rem;
}

.error-message {
    background-color: #ffebee;
    color: #c62828;
}

.details-header {
    font-weight: bold;
    font-size: 1.1rem;
    margin-bottom: 10px;
    color: #333;
}

.details-section {
    margin-bottom: 15px;
    padding: 10px;
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.details-subtitle {
    font-weight: bold;
    margin-bottom: 5px;
    color: #4a69bd;
}

.details-label {
    font-weight: bold;
    color: #666;
}

.details-section ul {
    margin: 5px 0;
    padding-left: 20px;
}""" 

    async def shutdown(self):
        """Cleanup the web server resources."""
        logger.info("Shutting down web server...")
        if hasattr(self, 'site') and self.site:
            await self.site.stop()
            logger.info("Web site stopped")
        
        if hasattr(self, 'runner') and self.runner:
            await self.runner.cleanup()
            logger.info("Web runner cleaned up")
            
        # Close websocket connections
        if self.websockets:
            for ws in list(self.websockets):
                await ws.close(code=1000, message=b'Server shutdown')
            self.websockets.clear()
            logger.info("Closed all websocket connections") 