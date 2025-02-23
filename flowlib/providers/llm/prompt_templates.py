"""Model-specific prompt template configurations."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class PromptTemplate:
    """Configuration for model-specific prompt formatting."""
    pre_prompt: str = ""  # Text to add before the prompt
    post_prompt: str = ""  # Text to add after the prompt
    pre_response: str = ""  # Text to add before expected response
    post_response: str = ""  # Text to add after expected response
    role_user: str = ""  # User role identifier if needed
    role_assistant: str = ""  # Assistant role identifier if needed
    role_system: str = ""  # System role identifier if needed
    format_type: str = "default"  # Type of formatting (default, chatml, etc.)

# Model-specific template configurations
TEMPLATES = {
    "default": PromptTemplate(),
    
    "llama2": PromptTemplate(
        pre_prompt="<s>[INST] ",
        post_prompt=" [/INST]",
        role_system="<<SYS>>\n{system}\n<</SYS>>\n\n"
    ),
    
    "chatml": PromptTemplate(
        pre_prompt="<|im_start|>user\n",
        post_prompt="\n<|im_end|>",
        pre_response="<|im_start|>assistant\n",
        post_response="\n<|im_end|>",
        format_type="chatml"
    ),
    
    "phi2": PromptTemplate(
        pre_prompt="Instruct: ",
        post_prompt="\nOutput: ",
        format_type="instruction"
    ),
    
    "phi4": PromptTemplate(
        pre_prompt="<|im_start|>user<|im_sep|>",
        post_prompt="<|im_end|>",
        pre_response="<|im_start|>assistant<|im_sep|>",
        post_response="<|im_end|>",
        format_type="chatml"
    ),
    
    "r1-llama": PromptTemplate(
        pre_prompt="<｜User｜>",
        post_prompt="<｜Assistant｜>",
        format_type="chat"
    ),
    
    "starcoder2-instruct": PromptTemplate(
        pre_prompt="<|endoftext|>You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n### Instruction\n",
        post_prompt="\n\n### Response\n<|endoftext|>",
        format_type="instruction"
    ),
    
    "mistral": PromptTemplate(
        pre_prompt="<|user|>\n",
        post_prompt="\n</s>",
        pre_response="<|assistant|>\n",
        post_response="\n</s>",
        format_type="chat"
    ),
    
    "zephyr": PromptTemplate(
        pre_prompt="<|user|>\n",
        post_prompt="\n</s>",
        pre_response="<|assistant|>\n",
        post_response="\n</s>",
        format_type="chat"
    )
}

def get_template(model_type: str) -> PromptTemplate:
    """Get the prompt template for a specific model type.
    
    Args:
        model_type: The type/name of the model
        
    Returns:
        PromptTemplate for the specified model
        
    Raises:
        ValueError: If model_type is not recognized
    """
    template = TEMPLATES.get(model_type.lower(), TEMPLATES["default"])
    return template

def format_prompt(prompt: str, model_type: str, system_prompt: Optional[str] = None) -> str:
    """Format a prompt according to model-specific requirements.
    
    Args:
        prompt: The main prompt text
        model_type: The type/name of the model
        system_prompt: Optional system prompt/instruction
        
    Returns:
        Formatted prompt string
    """
    template = get_template(model_type)
    
    # Handle system prompt if provided
    if system_prompt and template.role_system:
        system_section = template.role_system.format(system=system_prompt)
        prompt = system_section + prompt
    
    # Apply template formatting
    formatted = (
        template.pre_prompt +
        prompt +
        template.post_prompt
    )
    
    return formatted

def format_chat(messages: list[Dict[str, str]], model_type: str) -> str:
    """Format a chat conversation according to model-specific requirements.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model_type: The type/name of the model
        
    Returns:
        Formatted chat string
    """
    template = get_template(model_type)
    formatted_messages = []
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'system' and template.role_system:
            formatted_messages.append(template.role_system.format(system=content))
        elif role == 'user':
            formatted_messages.append(f"{template.role_user}{content}")
        elif role == 'assistant':
            formatted_messages.append(f"{template.role_assistant}{content}")
    
    return "\n".join(formatted_messages) 