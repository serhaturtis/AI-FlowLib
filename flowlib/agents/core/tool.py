"""Base tool implementation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
import json

class Tool(ABC):
    """Base class for all agent tools."""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None
    ):
        """Initialize tool.
        
        Args:
            name: Tool name
            description: Tool description
            input_model: Optional input validation model
            output_model: Optional output validation model
        """
        self.name = name
        self.description = description
        self.input_model = input_model
        self.output_model = output_model
    
    def describe_interface(self) -> str:
        """Get a human-readable description of the tool interface.
        
        Returns:
            Tool interface description
        """
        description = f"Tool: {self.name}\n"
        description += f"Description: {self.description}\n\n"
        
        def format_schema(schema: dict) -> str:
            """Format schema as readable string."""
            formatted = ""
            if "properties" in schema:
                for prop_name, prop_info in schema["properties"].items():
                    formatted += f"  {prop_name}:\n"
                    formatted += f"    type: {prop_info.get('type', 'any')}\n"
                    if "description" in prop_info:
                        formatted += f"    description: {prop_info['description']}\n"
            return formatted
        
        if self.input_model:
            description += "Input Schema:\n"
            description += format_schema(self.input_model.model_json_schema())
            description += "\n"
            
        if self.output_model:
            description += "Output Schema:\n"
            description += format_schema(self.output_model.model_json_schema())
            
        return description
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with given input data.
        
        Args:
            input_data: Input data for tool execution
            
        Returns:
            Tool execution result
        """
        pass 