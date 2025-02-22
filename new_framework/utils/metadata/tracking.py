"""Metadata tracking utilities."""

from typing import Dict, Any, List
from datetime import datetime

from ...core.models.results import FlowResult

def create_execution_metadata(
    flow_name: str,
    duration: float,
    input_keys: List[str],
    output_keys: List[str],
    **additional: Any
) -> Dict[str, Any]:
    """Create standard execution metadata.
    
    Args:
        flow_name: Name of flow
        duration: Execution duration in seconds
        input_keys: Keys present in input data
        output_keys: Keys present in output data
        **additional: Additional metadata fields
        
    Returns:
        Metadata dictionary
    """
    return {
        "flow_name": flow_name,
        "duration": duration,
        "input_keys": input_keys,
        "output_keys": output_keys,
        "timestamp": datetime.now().isoformat(),
        **additional
    }

def create_composite_metadata(
    flow_name: str,
    executed_flows: List[str],
    duration: float,
    **additional: Any
) -> Dict[str, Any]:
    """Create metadata for composite flow execution.
    
    Args:
        flow_name: Name of composite flow
        executed_flows: List of executed flow names
        duration: Total execution duration
        **additional: Additional metadata fields
        
    Returns:
        Metadata dictionary
    """
    return {
        "flow_name": flow_name,
        "flow_type": "composite",
        "executed_flows": executed_flows,
        "flow_count": len(executed_flows),
        "duration": duration,
        "timestamp": datetime.now().isoformat(),
        **additional
    }

def create_conditional_metadata(
    flow_name: str,
    condition_result: bool,
    selected_path: str,
    passthrough: bool,
    duration: float,
    **additional: Any
) -> Dict[str, Any]:
    """Create metadata for conditional flow execution.
    
    Args:
        flow_name: Name of conditional flow
        condition_result: Result of condition evaluation
        selected_path: Path taken (success/failure)
        passthrough: Whether input was passed through
        duration: Execution duration
        **additional: Additional metadata fields
        
    Returns:
        Metadata dictionary
    """
    return {
        "flow_name": flow_name,
        "flow_type": "conditional",
        "condition_result": condition_result,
        "selected_path": selected_path,
        "passthrough": passthrough,
        "duration": duration,
        "timestamp": datetime.now().isoformat(),
        **additional
    }

def update_result_metadata(
    result: FlowResult,
    **metadata: Any
) -> FlowResult:
    """Update result metadata with new fields.
    
    Args:
        result: Flow result to update
        **metadata: New metadata fields
        
    Returns:
        Updated flow result
    """
    result.metadata.update(metadata)
    return result 