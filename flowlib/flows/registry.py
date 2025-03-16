"""Stage registry for tracking and accessing flow stages.

This module provides a registry for stages defined with the @stage decorator,
enabling easy access to stages within flows.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Set, TypeVar, Generic, Type, cast

logger = logging.getLogger(__name__)

class StageRegistry:
    """
    Registry for flow stages.
    
    This registry stores information about stages, including their input/output models,
    processing functions, and metadata. It supports both flow-specific stages and standalone stages.
    """
    
    def __init__(self):
        """Initialize the stage registry with empty collections."""
        # Dictionary mapping flow names to sets of stage names
        self._flow_stages: Dict[str, Set[str]] = {}
        
        # Dictionary mapping (flow_name, stage_name) to stage info
        self._stage_info: Dict[tuple[str, str], Dict[str, Any]] = {}
        
        # Set of standalone stage names (not associated with a specific flow)
        self._standalone_stages: Set[str] = set()
    
    def register_flow(self, flow_name: str) -> None:
        """
        Register a flow in the registry.
        
        Args:
            flow_name: The name of the flow to register.
        """
        if flow_name not in self._flow_stages:
            self._flow_stages[flow_name] = set()
            logger.debug(f"Registered flow: {flow_name}")
    
    def register_stage(
        self,
        stage_name: str,
        flow_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Register a stage in the registry.
        
        Args:
            stage_name: The name of the stage to register.
            flow_name: The name of the flow to register the stage with (optional).
            **kwargs: Additional stage information to store.
        """
        # Determine if this is a standalone stage
        is_standalone = flow_name is None
        
        if is_standalone:
            # Register standalone stage
            self._standalone_stages.add(stage_name)
            # Use empty string as flow_name placeholder for standalone stages
            key = ("", stage_name)
            logger.debug(f"Registered standalone stage: {stage_name}")
        else:
            # Register flow-specific stage
            if flow_name not in self._flow_stages:
                self.register_flow(flow_name)
            
            self._flow_stages[flow_name].add(stage_name)
            key = (flow_name, stage_name)
            logger.debug(f"Registered stage '{stage_name}' for flow '{flow_name}'")
        
        # Store stage info with is_standalone flag
        self._stage_info[key] = {
            "name": stage_name,
            "is_standalone": is_standalone,
            **kwargs
        }
    
    def get_stage(self, stage_name: str, flow_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stage information from the registry.
        
        Args:
            stage_name: The name of the stage to retrieve.
            flow_name: The name of the flow to retrieve the stage from (optional).
            
        Returns:
            Dict[str, Any]: The stage information.
            
        Raises:
            KeyError: If the stage is not found in the registry.
        """
        logger.debug(f"Looking for stage '{stage_name}' in flow '{flow_name if flow_name else 'STANDALONE'}'")
        
        # First check if the stage exists in the specified flow
        if flow_name and flow_name in self._flow_stages and stage_name in self._flow_stages[flow_name]:
            logger.debug(f"Found stage '{stage_name}' in flow '{flow_name}'")
            return self._stage_info[(flow_name, stage_name)]
        
        # Check if it's a standalone stage
        if stage_name in self._standalone_stages:
            logger.debug(f"Found standalone stage '{stage_name}'")
            return self._stage_info[("", stage_name)]
        
        # If flow_name is provided but stage not found, only check that flow
        if flow_name:
            available_stages = self._flow_stages.get(flow_name, set())
            raise KeyError(
                f"Stage '{stage_name}' not found in flow '{flow_name}'. "
                f"Available stages: {', '.join(sorted(available_stages)) if available_stages else 'None'}"
            )
        
        # If no flow_name, check all flows
        for flow_name, stages in self._flow_stages.items():
            if stage_name in stages:
                logger.debug(f"Found stage '{stage_name}' in flow '{flow_name}'")
                return self._stage_info[(flow_name, stage_name)]
        
        # Stage not found anywhere
        available_standalone = sorted(self._standalone_stages)
        all_flows = sorted(self._flow_stages.keys())
        
        raise KeyError(
            f"Stage '{stage_name}' not found in any flow or as standalone stage. "
            f"Available flows: {', '.join(all_flows) if all_flows else 'None'}. "
            f"Available standalone stages: {', '.join(available_standalone) if available_standalone else 'None'}"
        )
    
    def get_stages(self, flow_name: Optional[str] = None) -> List[str]:
        """
        Get all stages registered for a flow or all standalone stages.
        
        Args:
            flow_name: The name of the flow to get stages for. If None, returns standalone stages.
            
        Returns:
            List[str]: The list of stage names.
        """
        if flow_name is None:
            return sorted(self._standalone_stages)
        
        if flow_name in self._flow_stages:
            return sorted(self._flow_stages[flow_name])
        
        return []
    
    def get_flows(self) -> List[str]:
        """
        Get all registered flow names.
        
        Returns:
            List[str]: The list of flow names.
        """
        return sorted(self._flow_stages.keys())
    
    def clear(self) -> None:
        """Clear all registered stages and flows."""
        self._flow_stages.clear()
        self._stage_info.clear()
        self._standalone_stages.clear()
        logger.debug("Cleared stage registry")


# Global stage registry instance
stage_registry = StageRegistry() 