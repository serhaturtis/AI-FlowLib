"""Agent prompt templates for LLM interactions.

This module provides prompt templates for agent planning, input generation,
and reflection operations.
"""

from ..core.registry.decorators import prompt


@prompt("agent-planning")
class AgentPlanningPrompt:
    """Prompt template for agent planning decisions."""
    template = """
You are a planning system for an AI agent that completes tasks by selecting appropriate flows to execute.

Available flows:
{available_flows}

TASK: {task_description}

CURRENT STATE:
{current_state}

RECENT EXECUTION HISTORY:
{execution_history}

Your job is to decide which flow to execute next to make progress on the task.

Respond with:
- A detailed explanation of your reasoning process
- The name of the flow to execute next (MUST BE EXACTLY as listed in the 'Flow: ' entries above, including the quotes)
- Whether the task is complete (true/false)
- If the task is complete, provide a reason for completion

IMPORTANT: You must use the EXACT flow name as it appears in quotes after 'Flow:' in the Available flows list.
For example, if you see "Flow: 'conversation-flow'" above, you must respond with "conversation-flow", not "ConversationFlow".
"""

@prompt("agent-input-generation")
class AgentInputGenerationPrompt:
    """Prompt template for generating flow inputs."""
    template = """
You need to generate appropriate inputs for the selected flow.

FLOW: {flow_name}

INPUT SCHEMA:
{input_schema}

TASK: {task_description}

CURRENT STATE:
{current_state}

PLANNING REASONING:
{planning_reasoning}

Your job is to generate valid inputs for this flow that will help accomplish the task.
The inputs must conform to the schema definition provided.

Generate the inputs in JSON format that matches the input schema exactly.
"""

@prompt("agent-reflection")
class AgentReflectionPrompt:
    """Prompt template for reflecting on flow results."""
    template = """
Analyze the results of the flow execution and provide insights.

FLOW: {flow_name}

INPUTS:
{inputs}

OUTPUTS:
{outputs}

TASK: {task_description}

CURRENT STATE:
{current_state}

Your job is to:
1. Provide a detailed reflection on what was learned from this flow execution
2. Extract ONLY important factual entities mentioned in the conversation
3. Estimate the current progress toward task completion (0-100%)
4. Determine if the task is now complete

IMPORTANT: For extracting information, FOCUS EXCLUSIVELY on specific factual entities such as:
- People's names, relationships, contact information (phone numbers, emails, etc.)
- Places, locations, addresses
- Dates, times, deadlines, appointments
- Preferences, interests, requirements
- Specific objects, products, or services mentioned
- Numerical values, quantities, prices

DO NOT extract meta-information about the conversation itself, such as:
- "User asked about X"
- "Agent provided information about Y"
- "User expressed interest in Z"
- "Agent explained how to do X"

For each piece of factual information, create a structured memory item with:
- key: A unique identifier for this type of information (e.g., "person_name", "contact_phone", "meeting_time")
- value: The actual factual information (e.g., "John Smith", "555-123-4567", "March 15 at 3pm")
- relevant_keys: Array of related memory keys this might be connected to
- importance: A score from 0.0 to 1.0 indicating how important this information is
- source: Where this information came from (default is "reflection")
- context: Brief context explaining what this fact refers to

Respond with:
- A reflection on what was learned and what progress was made
- A list of structured memory items ONLY containing important factual entities (not conversation details)
- A progress estimate as a percentage (0-100)
- Whether the task is complete (true/false)
- If the task is complete, a reason for completion
- Be concise and to the point
"""

@prompt("agent-conversation")
class AgentConversationPrompt:
    """Prompt template for agent conversations."""
    template = """
You are a helpful assistant with access to various flows that can execute tasks.

USER MESSAGE: {message}

CONVERSATION HISTORY:
{conversation_history}

CURRENT AGENT STATE:
{agent_state}


"""