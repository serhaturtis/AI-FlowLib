"""Agent prompt templates for LLM interactions.

This module provides prompt templates for agent planning, input generation,
and reflection operations.
"""

from flowlib.core.registry.decorators import prompt


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
2. Evaluate how the flow execution contributes to the overall task
3. Estimate the current progress toward task completion (0-100%)
4. Determine if the task is now complete

Think through the following questions:
- What information was gained or what actions were completed in this flow execution?
- How does this advance the overall task?
- What remaining steps (if any) are needed to complete the task?
- Has the user's original request been fully addressed?

Respond with:
- A thorough reflection on what was learned and what progress was made
- A progress estimate as a percentage (0-100)
- Whether the task is complete (true/false)
- If the task is complete, a reason for completion
- If the task is not complete, what needs to be done next
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