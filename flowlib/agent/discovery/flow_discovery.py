"""
Flow discovery for agent systems.

This module provides functionality for discovering agent-compatible flows
from the registered stages and converting them to a format compatible with
our planning system.
"""

import logging
from typing import Dict, Any, Optional, List, Type

from ...flows.base import Flow
from ...flows.registry import stage_registry
from ...flows.metadata import FlowMetadata
from ..core.errors import FlowDiscoveryError, DiscoveryError
from ..core.base import BaseComponent
from .interfaces import FlowDiscoveryInterface

logger = logging.getLogger(__name__)


class FlowDiscovery(BaseComponent, FlowDiscoveryInterface):
    """Service for discovering agent-compatible flows.
    
    This class helps discover flows that have been decorated with the
    @agent_flow decorator and provides access to them.
    """
    
    def __init__(self, name: str = "flow_discovery"):
        """Initialize flow discovery.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self._flows: Dict[str, Type[Flow]] = {}
    
    async def _initialize_impl(self) -> None:
        """Initialize the flow discovery system.
        
        This automatically refreshes flows on initialization.
        """
        await self.refresh_flows()
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the flow discovery system."""
        pass
    
    async def refresh_flows(self) -> None:
        """Refresh the discovered flows.
        
        This scans the stage registry for available flows.
        
        Raises:
            FlowDiscoveryError: If discovery fails
        """
        self._logger.info("Refreshing flows from registry")
        
        try:
            # Clear existing flow references
            self._flows.clear()
            
            # Find agent-compatible flows
            flow_classes = self.discover_agent_flows()
            
            # Register each discovered flow
            for flow_class in flow_classes:
                try:
                    # Create instance to register
                    if hasattr(flow_class, 'create') and callable(flow_class.create):
                        instance = flow_class.create()
                    else:
                        instance = flow_class()
                    
                    # Register with the discovery system
                    self.register_flow(instance)
                except Exception as e:
                    self._logger.warning(f"Failed to register flow {flow_class.__name__}: {e}")
            
            self._logger.info(f"Refreshed flows: {len(self._flows)} agent flows available")
            
        except Exception as e:
            raise FlowDiscoveryError(f"Failed to refresh flows: {str(e)}") from e
    
    def register_flow(self, flow: Any) -> None:
        """Register a flow with the discovery system.
        
        Args:
            flow: Flow to register
        """
        if not hasattr(flow, "name"):
            self._logger.warning(f"Cannot register flow without name: {flow}")
            return
            
        flow_name = flow.name
        
        # Store a reference to the flow
        self._flows[flow_name] = flow
        
        # Register with stage_registry
        if stage_registry:
            try:
                stage_registry.register_flow(flow_name, flow)
                self._logger.debug(f"Registered flow: {flow_name}")
            except Exception as e:
                self._logger.warning(f"Failed to register flow with stage_registry: {e}")
    
    def get_flow(self, name: str) -> Optional[Any]:
        """Get a flow by name.
        
        Args:
            name: Flow name
            
        Returns:
            Flow object or None if not found
        """
        return self._flows.get(name)
    
    def get_flow_registry(self):
        """Get the flow registry.
        
        Returns:
            Stage registry containing all flows
        """
        return stage_registry
    
    def get_flow_metadata(self, name: str) -> Optional[FlowMetadata]:
        """Get metadata for a flow by name.
        
        Args:
            name: Flow name
            
        Returns:
            Flow metadata or None if not found
        """
        if not stage_registry:
            return None
            
        return stage_registry.get_flow_metadata(name)
    
    def discover_agent_flows(self) -> List[Any]:
        """Discover flows that are compatible with agents.
        
        This method finds flows that have been decorated with the @agent_flow decorator.
        
        Returns:
            List of flow classes compatible with agents
            
        Raises:
            DiscoveryError: If stage registry is not available or discovery fails
        """
        self._logger.debug("Discovering agent-compatible flows")
        
        # Check if stage registry is available
        if not stage_registry:
            raise DiscoveryError(
                message="Stage registry not available for flow discovery",
                operation="discover_agent_flows"
            )
        
        # Get all flow instances from the stage registry
        flow_instances = stage_registry.get_flow_instances()
        
        # Filter to only agent-compatible flows
        flow_classes = []
        for flow_name, flow_instance in flow_instances.items():
            # Check if flow has agent_flow metadata
            if hasattr(flow_instance, "__flow_metadata__") and flow_instance.__flow_metadata__.get("agent_flow", False):
                flow_classes.append(flow_instance.__class__)
                self._logger.debug(f"Found agent flow: {flow_name}")
        
        self._logger.info(f"Discovered {len(flow_classes)} agent-compatible flows")
        return flow_classes 