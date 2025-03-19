"""Flow discovery system for runtime flow detection.

This module provides functionality for discovering flows at runtime, enabling
agents to use newly added flows without requiring a restart.
"""

import os
import time
import importlib.util
import inspect
import logging
from typing import Dict, List, Any, Optional, Union, Set

import flowlib as fl
from ..flows import Flow

logger = logging.getLogger(__name__)

class FlowDiscovery:
    """Discovers flows in specified directories at runtime.
    
    This class provides:
    1. Scanning of directories for flow classes
    2. Dynamic loading of discovered flows
    3. Tracking of flow changes over time
    """
    
    def __init__(self, flow_paths: List[str] = ["./flows"]):
        """Initialize with paths to flow directories.
        
        Args:
            flow_paths: List of directory paths to scan for flows
        """
        self.flow_paths = flow_paths
        self.discovered_flows = {}
        self.last_scan_time = 0
        self.file_timestamps = {}
    
    async def refresh_flows(self) -> Dict[str, Flow]:
        """Discover new flows and return all available flows.
        
        This method:
        1. Scans the flow directories for Python files
        2. Checks for changes since the last scan
        3. Loads new or modified flows
        
        Returns:
            Dictionary of flow names to flow instances
            
        Raises:
            Exception: If flow loading fails
        """
        current_time = time.time()
        
        # Only scan if it's been more than 30 seconds since last scan
        if current_time - self.last_scan_time < 30:
            return self.discovered_flows
            
        self.last_scan_time = current_time
        
        try:
            # Scan for new or modified flows
            new_flows = await self._scan_for_flows()
            
            # Update discovered flows
            self.discovered_flows.update(new_flows)
            
            logger.debug(f"Refreshed flows: found {len(new_flows)} new flows")
            return self.discovered_flows
            
        except Exception as e:
            logger.error(f"Error refreshing flows: {str(e)}")
            # Still return existing flows even if refresh fails
            return self.discovered_flows
    
    async def _scan_for_flows(self) -> Dict[str, Flow]:
        """Scan directories for flow classes.
        
        Returns:
            Dictionary of newly discovered flows
            
        Raises:
            Exception: If flow loading fails
        """
        new_flows = {}
        
        for path in self.flow_paths:
            if not os.path.exists(path):
                logger.warning(f"Flow path does not exist: {path}")
                continue
                
            # Scan all python files in the directory
            for file_name in os.listdir(path):
                if not file_name.endswith('.py'):
                    continue
                    
                # Get full file path
                file_path = os.path.join(path, file_name)
                
                # Check if file has been modified since last scan
                file_timestamp = os.path.getmtime(file_path)
                if file_path in self.file_timestamps and file_timestamp <= self.file_timestamps[file_path]:
                    # File hasn't changed, skip it
                    continue
                
                # Update file timestamp
                self.file_timestamps[file_path] = file_timestamp
                
                # Import module
                module_name = file_name[:-3]  # Remove .py extension
                
                try:
                    # Import module dynamically
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find flow classes in the module
                    flows_found = 0
                    for name, obj in inspect.getmembers(module):
                        # Check for flow classes
                        # Look for decorator-applied attributes
                        if (inspect.isclass(obj) and 
                            (hasattr(obj, '__flow_name__') or 
                             (hasattr(obj, '__dict__') and '__flow_name__' in obj.__dict__))):
                            
                            flow_name = obj.__flow_name__
                            
                            # Check if flow is already discovered
                            if flow_name in self.discovered_flows:
                                continue
                                
                            # Instantiate the flow
                            try:
                                flow_instance = obj()
                                new_flows[flow_name] = flow_instance
                                flows_found += 1
                                logger.info(f"Discovered new flow: {flow_name}")
                            except Exception as e:
                                logger.error(f"Error instantiating flow '{flow_name}': {str(e)}")
                    
                    logger.debug(f"Found {flows_found} flows in {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading module {module_name}: {str(e)}")
        
        return new_flows