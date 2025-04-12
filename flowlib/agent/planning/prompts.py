"""
Standard planning prompt templates for the agent system.

This module provides default prompt templates for planning and input generation
that can be used by agents out of the box.
"""

from ...resources.decorators import prompt


@prompt("planning_default")
class DefaultPlanningPrompt:
    """Default planning prompt for the agent to select the appropriate flow."""
    
    template = """
    You are a planning assistant for an autonomous agent.
    
    Your task is to select the most appropriate flow to execute next based on the current state and task description.
    
    Task description: {{task_description}}
    
    Available flows:
    {{available_flows_text}}
    
    Current state:
    - Task: {{task_description}}
    - Cycle: {{cycle}}
    
    Execution history:
    {{execution_history_text}}
    
    {{memory_context}}
    
    CRITICAL REQUIREMENTS:
    - You MUST select a flow from the available flows listed above.
    - Do NOT return null or None for the selected_flow.
    - Always respond to user messages when appropriate.
    - ONLY select from flows that are actually available and listed above.
    - If the task is complete, set is_complete to true.
    """
    config = {
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("conversational_planning")
class ConversationalPlanningPrompt:
    """Planning prompt optimized for conversational agents."""
    
    template = """
    You are a planning assistant for a conversational agent.
    
    Your task is to select the most appropriate flow to execute next based on the user's message and available flows.
    
    User message: {{task_description}}
    Current cycle: {{cycle}}
    
    Available flows:
    {{available_flows_text}}
    
    Conversation history:
    {{execution_history_text}}
    
    {{memory_context}}
    
    CRITICAL REQUIREMENTS:
    - You MUST select a flow from the available flows listed above.
    - ONLY select from flows that are actually available and listed above.
    """
    config = {
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("input_generation_default")
class DefaultInputGenerationPrompt:
    """Default prompt template for generating inputs for flows."""
    
    template = """
    You are an input generation assistant for an autonomous agent.
    
    Your task is to generate appropriate inputs for a flow based on the current state and task description.
    
    Task description: {{task_description}}
    
    Selected flow: {{flow_name}}
    Flow description: {{flow_description}}
    Input schema: {{input_schema}}
    
    Planning rationale: {{planning_rationale}}
    
    Execution history: 
    {{execution_history_text}}
    
    Relevant memories:
    {{relevant_memories_text}}
    
    GUIDELINES:
    - Generate inputs that follow the specified input schema
    - Ensure inputs are appropriate for the selected flow
    - Use relevant context from execution history and memories
    - For conversation flows, ensure the message is natural and appropriate
    """
    config = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("conversational_input_generation")
class ConversationalInputGenerationPrompt:
    """Input generation prompt optimized for conversational flows."""
    
    template = """
    You are an input generation assistant for a conversational agent.
    
    Your task is to generate appropriate inputs for the selected conversation flow.
    
    User message: {{task_description}}
    
    Selected flow: {{flow_name}}
    Flow description: {{flow_description}}
    Input schema: {{input_schema}}
    
    Planning rationale: {{planning_rationale}}
    
    Conversation history: 
    {{execution_history_text}}
    
    Relevant memories:
    {{relevant_memories_text}}
    
    GUIDELINES:
    - Add any relevant context from memories or conversation history
    - Format the inputs according to the input schema
    """
    config = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40
    } 