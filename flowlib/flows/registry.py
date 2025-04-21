"""Stage registry for tracking and accessing flow stages.

This module provides a registry for stages defined with the @stage decorator,
enabling easy access to stages within flows.
"""

import logging
from typing import Dict, List, Optional, Any, Set

from .metadata import FlowMetadata

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
        
        # Store flow instances
        self._flow_instances: Dict[str, Any] = {}
        
        # Store flow metadata
        self._flow_metadata: Dict[str, FlowMetadata] = {}
    
    def register_flow(self, flow_name: str, flow_class_or_instance = None) -> None:
        """Register a flow with the registry.
        
        Args:
            flow_name: Flow name
            flow_class_or_instance: Flow class or instance (optional)
            
        Raises:
            ValueError: If flow already exists with this name
        """
        if flow_name in self._flow_stages and flow_name not in self._standalone_stages:
            logger.warning(f"Flow '{flow_name}' already exists in registry. Skipping...")
            return
        
        # Register flow name
        if flow_class_or_instance is None:
            self._flow_stages[flow_name] = set()
            self._stage_info[(flow_name, "")] = {
                "name": "",
                "is_standalone": True,
                "metadata": {
                    "is_infrastructure": False
                },
                "stages": {}
            }
            self._standalone_stages.add("")
            logger.debug(f"Registered standalone stage: {flow_name}")
        else:
            # Determine if this is a class or instance
            if isinstance(flow_class_or_instance, type):
                flow_class = flow_class_or_instance
                try:
                    # Try to create an instance
                    flow_instance = flow_class()
                except Exception as e:
                    logger.warning(f"Could not create instance of flow '{flow_name}': {e}")
                    flow_instance = None
            else:
                flow_instance = flow_class_or_instance
                flow_class = flow_instance.__class__
            
            # Get metadata from the instance if possible
            metadata = {}
            if hasattr(flow_instance, "__flow_metadata__"):
                metadata = flow_instance.__flow_metadata__
            elif hasattr(flow_class, "__flow_metadata__"):
                metadata = flow_class.__flow_metadata__
            
            # Register the flow
            self._flow_stages[flow_name] = set()
            self._stage_info[(flow_name, flow_name)] = {
                "name": flow_name,
                "is_standalone": False,
                "metadata": metadata,
                "stages": {}
            }
            self._flow_instances[flow_name] = flow_instance
            
            # Create and store FlowMetadata for the flow
            if flow_instance:
                try:
                    from .metadata import FlowMetadata
                    flow_metadata = FlowMetadata.from_flow(flow_instance, flow_name)
                    self._flow_metadata[flow_name] = flow_metadata
                    logger.debug(f"Created and stored metadata for flow: {flow_name}")
                except Exception as e:
                    logger.warning(f"Failed to create metadata for flow '{flow_name}': {e}")
            
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
    
    def get_flow_metadata(self, flow_name: str) -> Optional[FlowMetadata]:
        """
        Get flow metadata by name.
        
        Args:
            flow_name: Name of the flow
            
        Returns:
            Flow metadata or None if not found
        """
        return self._flow_metadata.get(flow_name)
    
    def get_all_flow_metadata(self) -> Dict[str, FlowMetadata]:
        """
        Get metadata for all registered flows.
        
        Returns:
            Dictionary mapping flow names to their metadata
        """
        return self._flow_metadata.copy()
    
    def get_flow(self, flow_name: str) -> Optional[Any]:
        """
        Get a flow instance by name.
        
        Args:
            flow_name: Name of the flow to retrieve.
            
        Returns:
            The flow instance if found, None otherwise.
        """
        flow_instance = self._flow_instances.get(flow_name)
        if flow_instance:
            logger.debug(f"Retrieved flow instance for: {flow_name}")
        else:
            logger.debug(f"No flow instance found for: {flow_name}")
        return flow_instance
    
    def get_flow_instances(self) -> Dict[str, Any]:
        """
        Get all registered flow instances.
        
        Returns:
            Dict[str, Any]: Dictionary mapping flow names to flow instances.
        """
        return self._flow_instances.copy()
    
    def get_agent_selectable_flows(self) -> Dict[str, Any]:
        """
        Get flow instances that can be selected by the agent.
        
        This method filters out infrastructure flows that shouldn't be directly 
        selectable by the agent during planning.
        
        Returns:
            Dict[str, Any]: Dictionary mapping flow names to flow instances where is_infrastructure=False.
        """
        selectable_flows = {}
        for flow_name, flow_instance in self._flow_instances.items():
            # Check if the flow_instance has is_infrastructure attribute or metadata
            is_infrastructure = False
            
            # Check the direct attribute first
            if hasattr(flow_instance, "is_infrastructure"):
                is_infrastructure = flow_instance.is_infrastructure
            # Then check in flow metadata if available
            elif hasattr(flow_instance, "__flow_metadata__"):
                metadata = flow_instance.__flow_metadata__
                is_infrastructure = metadata.get("is_infrastructure", False)
                
            # Add to result if not an infrastructure flow
            if not is_infrastructure:
                selectable_flows[flow_name] = flow_instance
                
        logger.debug(f"Found {len(selectable_flows)} agent-selectable flows")
        return selectable_flows
    
    def clear(self) -> None:
        """Clear all registered stages and flows."""
        self._flow_stages.clear()
        self._stage_info.clear()
        self._standalone_stages.clear()
        self._flow_instances.clear()
        self._flow_metadata.clear()
        logger.debug("Cleared stage registry")


# Global stage registry instance
stage_registry = StageRegistry() 