"""
UserInputFlow for agent-user interaction.

This module provides a flow that allows agents to request input from users,
display their current progress, and wait for user responses.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field

from ...flows.decorators import stage, pipeline
from ...flows.base import Flow
from ..decorators.base import agent_flow


class UserInputRequest(BaseModel):
    """Model for requesting user input."""
    message: str = Field(
        ..., 
        description="The message to show to the user"
    )
    prompt: str = Field(
        "Please respond:", 
        description="The prompt for user input"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Context information to show to the user"
    )
    show_progress: bool = Field(
        True, 
        description="Whether to show progress information"
    )
    options: List[str] = Field(
        default_factory=list, 
        description="Optional list of predefined options to present to the user"
    )


class UserInputResponse(BaseModel):
    """Model for user input responses."""
    input: str = Field(..., description="The input provided by the user")
    timestamp: Any = Field(..., description="When the input was provided")
    additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class FormattedRequest(BaseModel):
    """Model for formatted user input request."""
    formatted_message: str = Field(..., description="The formatted message to display to the user")
    prompt: str = Field(..., description="The prompt for user input")
    show_progress: bool = Field(..., description="Whether to show progress information")
    options: List[str] = Field(default_factory=list, description="Optional list of predefined options")
    original_context: Dict[str, Any] = Field(default_factory=dict, description="Original context information")


@agent_flow(
    name="UserInput",
    description="Request input from the user and wait for a response"
)
class UserInputFlow(Flow):
    """A flow for interacting with users during agent execution.
    
    This flow enables an agent to:
    1. Display information about its current progress
    2. Request specific input from the user
    3. Provide context for the user's decision
    4. Pause execution until user input is received
    """
    
    def __init__(self):
        """Initialize the UserInputFlow."""
        super().__init__("UserInput")
        self._input_callback = None
        
    def get_description(self) -> str:
        """Return the description of this flow."""
        return "Request input from the user and wait for a response"
    
    @stage(input_model=UserInputRequest, output_model=FormattedRequest)
    async def prepare_request(self, request: UserInputRequest) -> FormattedRequest:
        """Format the user request for display.
        
        Args:
            request: The input request containing the message and context
            
        Returns:
            Formatted request data
        """
        # Format the message with appropriate context
        formatted_message = request.message
        
        # Add context information if available
        if request.context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in request.context.items()])
            formatted_message = f"{formatted_message}\n\nContext:\n{context_str}"
            
        # Add options if specified
        options_text = ""
        if request.options:
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in request.options])
            formatted_message = f"{formatted_message}{options_text}"
        
        return FormattedRequest(
            formatted_message=formatted_message,
            prompt=request.prompt,
            show_progress=request.show_progress,
            options=request.options,
            original_context=request.context
        )
    
    @stage(input_model=FormattedRequest, output_model=UserInputResponse)
    async def get_user_input(self, data: FormattedRequest) -> UserInputResponse:
        """Wait for and process user input.
        
        Args:
            data: Formatted request data
            
        Returns:
            The user's response
            
        Raises:
            ValueError: If no user input callback is registered
        """
        # Check if we have a callback for user input
        if not self._input_callback:
            raise ValueError(
                "No user input callback registered. You must provide a callback function "
                "that handles user interaction when running the agent."
            )
        
        # Convert the pydantic model to a dictionary for compatibility with callbacks
        data_dict = data.model_dump()
        
        # Display the message to the user and get their input
        response = await self._input_callback(data_dict)
        
        # Return structured response
        return response
    
    @pipeline(input_model=UserInputRequest, output_model=UserInputResponse)
    async def run_pipeline(self, request: UserInputRequest) -> UserInputResponse:
        """Process a user interaction request.
        
        Args:
            request: The user input request
            
        Returns:
            The user's response
        """
        prepared = await self.prepare_request(request)
        return await self.get_user_input(prepared)
    
    def set_input_callback(self, callback):
        """Set the callback function for receiving user input.
        
        Args:
            callback: Async function that displays information to the user
                     and returns their input
        """
        self._input_callback = callback
        
    def get_input_callback(self):
        """Get the current input callback function.
        
        Returns:
            The current callback function, or None if not set
        """
        return self._input_callback
        
    @classmethod
    def create(cls) -> Flow:
        """Create a new instance of this flow.
        
        Returns:
            A Flow instance
        """
        return cls() 