"""Conversation formatting utilities.

This module provides utilities for formatting conversation data,
including message history, state information, and flow representations.
"""

from typing import List, Dict, Any, Optional


def format_conversation(conversation: List[Dict[str, str]]) -> str:
    """Format conversation history into a string for prompts.
    
    Args:
        conversation: List of message dictionaries with 'speaker' and 'content' keys
            
    Returns:
        Formatted conversation string
    """
    if not conversation:
        return ""
        
    formatted = []
    for message in conversation:
        speaker = message.get("speaker", "Unknown")
        content = message.get("content", "")
        formatted.append(f"{speaker}: {content}")
            
    return "\n".join(formatted)


def format_state(state: Dict[str, Any]) -> str:
    """Format agent state for prompt.
    
    Args:
        state: Dictionary of state variables
            
    Returns:
        Formatted state string
    """
    if not state:
        return "No state information."
        
    result = []
    for key, value in state.items():
        result.append(f"{key}: {value}")
    
    return "\n".join(result)


def format_history(history: List[Dict[str, Any]]) -> str:
    """Format execution history for prompt.
    
    Args:
        history: List of execution steps
            
    Returns:
        Formatted history string
    """
    if not history:
        return "No execution history yet."
        
    result = []
    for i, step in enumerate(history, 1):
        if step.get("action") == "execute_flow":
            result.append(f"""
Step {i}:
  Flow: {step.get('flow')}
  Reasoning: {step.get('reasoning')}
  Reflection: {step.get('reflection', 'None')}
""")
        elif step.get("action") == "error":
            result.append(f"""
Step {i}:
  Error: {step.get('error')}
""")
    
    return "\n".join(result)


def format_flows(flows: List[Dict[str, Any]]) -> str:
    """Format available flows for prompt.
    
    Args:
        flows: List of flow information dictionaries
            
    Returns:
        Formatted flows string
    """
    if not flows:
        return "No flows available."
        
    result = ["Available flows:"]
    
    for flow in sorted(flows, key=lambda f: f.get("name", "")):
        name = flow.get("name", "Unknown")
        description = flow.get("description", "No description available.")
        
        # Add schema info if available
        schema_str = ""
        schema = flow.get("schema")
        if schema:
            input_schema = schema.get("input")
            if input_schema:
                schema_str = f"\n  Input: {input_schema}"
                
            output_schema = schema.get("output")
            if output_schema:
                schema_str += f"\n  Output: {output_schema}"
        
        result.append(f"\n{name}: {description}{schema_str}")
    
    # Add a note about flow names
    result.append("\nNOTE: When selecting a flow, use the exact flow name without any quotes.")
    
    return "\n".join(result)


def format_agent_execution_details(details: Dict[str, Any]) -> str:
    """Format agent execution details for CLI display.
    
    Args:
        details: Dictionary of execution details
            
    Returns:
        Formatted execution details string
    """
    if not details:
        return "No detailed agent execution information available."
        
    result = ["--- Agent Execution Details ---"]
    
    state = details.get("state")
    if state:
        # Get progress value - normalize to percentage if needed
        progress = getattr(state, "progress", 0)
        # Check if progress is already a percentage (0-100) or a fraction (0-1)
        if progress <= 1.0:  # If it's a fraction, convert to percentage
            progress_display = f"{progress * 100:.0f}%"
        else:  # It's already a percentage
            progress_display = f"{progress:.0f}%"
            
        is_complete = getattr(state, "is_complete", False)
        
        result.append(f"Progress: {progress_display}")
        result.append(f"Complete: {'Yes' if is_complete else 'No'}")
    
    # Format latest plan
    latest_plan = details.get("latest_plan")
    if latest_plan:
        result.append("\nLatest plan:")
        reasoning = latest_plan.get("reasoning", "No reasoning")
        flow = latest_plan.get("flow", "No flow selected")
        result.append(f"  Reasoning: {reasoning[:100]}...")
        result.append(f"  Selected flow: {flow}")
        
    # Format last execution
    execution = details.get("latest_execution")
    if execution:
        result.append("\nLatest execution:")
        action = execution.get("action", "unknown")
        flow = execution.get("flow", "unknown")
        result.append(f"  Action: {action}")
        result.append(f"  Flow: {flow}")
        
    # Format latest reflection
    latest_reflection = details.get("latest_reflection")
    if latest_reflection:
        result.append("\nLatest reflection:")
        reflection = latest_reflection.get("reflection", "No reflection available")
        result.append(f"  {reflection[:200]}...")
        
    result.append("-----------------------------")
    
    return "\n".join(result) 