"""
Prompts for message classification.

This module provides prompt templates for message classification.
"""

from ...resources.decorators import prompt

@prompt("message_classifier_prompt")
class MessageClassifierPrompt:
    """Prompt for classifying user messages"""
    
    template = """You are a message classifier for an agent system.

Your task is to classify this user message based on whether it can be handled through simple conversation or requires executing a complex task.

User message: {{message}}

Conversation history:
{{conversation_history}}

Classification rules:
- CONVERSATION: Greetings, simple questions answerable from general knowledge or conversation history, social responses, acknowledgements, confirmations. The key is that you can respond fully and accurately *without* needing external tools, real-time data, or complex computation.
- TASK: Instructions to perform actions (e.g., run commands, manage files), complex questions needing research or external/up-to-date information (e.g., current prices, weather, news, specific technical details not commonly known), multi-step requests, requests requiring planning, computation, coding, or analysis.

## Memory Context Summary
{{memory_context_summary}}

Analyze the message carefully based on complexity, intent, and available memory context. Does fulfilling the request require capabilities beyond simple conversational recall and generation, or accessing information not present in memory?

Output a classification with:
1. execute_task: Set to `true` if fulfilling the user's request requires performing an action, planning, computation, retrieving external/up-to-date information, or accessing information *not found* in the Memory Context Summary. Set to `false` otherwise.
2. confidence: How confident you are in the classification (0.0-1.0).
3. category: Specific category (e.g., greeting, question, instruction, research_query, planning_request).
4. task_description: If execute_task=true, provide a clear, concise task description reformulating the user's request for the agent. Leave empty or null for conversation messages.

"""
    
    config = {
        "max_tokens": 150,  # Minimal tokens for efficiency
        "temperature": 0.1,  # Low temperature for consistent classifications
    } 