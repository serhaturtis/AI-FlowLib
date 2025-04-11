"""
Standard reflection prompt templates for the agent system.

This module provides default prompt templates for reflection that can be
used by agents out of the box to evaluate execution results and improve performance.
"""

from ...resources.decorators import prompt


@prompt("reflection_default")
class DefaultReflectionPrompt:
    """Default reflection prompt for evaluating agent execution results."""
    
    template = """
    You are a reflection system for an autonomous agent.
    
    Your task is to analyze the execution results and provide insights to improve future planning.
    
    Task description: {{task_description}}
    Flow name: {{flow_name}}
    
    Planning rationale: {{planning_rationale}}
    
    Flow inputs:
    {{flow_inputs}}
    
    Flow execution result:
    {{flow_result}}
    
    Execution history:
    {{execution_history_text}}
    
    Analyze the execution results and reflect on the following:
    1. Was the execution successful? Why or why not?
    2. Was the selected flow appropriate for the task?
    3. Were the inputs well-formed and appropriate?
    4. Is the task complete or are additional steps needed?
    5. What could be improved in future planning cycles?
    """
    config = {
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("conversational_reflection")
class ConversationalReflectionPrompt:
    """Reflection prompt optimized for conversational agents."""
    
    template = """
    You are a reflection system for a conversational agent.
    
    Your task is to analyze the conversation response and provide insights to improve future interactions.
    
    User message: {{task_description}}
    Flow name: {{flow_name}}
    
    Planning rationale: {{planning_rationale}}
    
    Response inputs:
    {{flow_inputs}}
    
    Generated response:
    {{flow_result}}
    
    Conversation history:
    {{execution_history_text}}
    
    Analyze the conversation and reflect on the following:
    1. Was the response relevant and helpful to the user's message?
    2. Did the response address all aspects of the user's question or request?
    3. Was the conversation flow natural and engaging?
    4. Is further clarification or additional information needed?
    5. How could future responses be improved?
    """
    config = {
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("task_completion_reflection")
class TaskCompletionReflectionPrompt:
    """Reflection prompt focused on determining task completion."""
    
    template = """
    You are a task evaluation system for an autonomous agent.
    
    Your task is to determine if the current task has been completed based on execution results.
    
    Task description: {{task_description}}
    Flow name: {{flow_name}}
    
    Flow execution result:
    {{flow_result}}
    
    Execution history summary:
    {{execution_history_text}}
    
    Evaluate whether the task described has been completed.
    
    GUIDELINES:
    - Be strict in your evaluation - only mark a task complete if there's clear evidence
    - For multi-step tasks, ensure all steps have been addressed
    - For information requests, ensure the information has been provided accurately
    - For action tasks, ensure the actions have been executed successfully
    """
    config = {
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 30
    } 