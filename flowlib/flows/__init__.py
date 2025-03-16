"""Enhanced flow framework with improved usability, type safety, and error handling.

This package provides a comprehensive set of tools for building, executing,
and monitoring flows with:
- Attribute-based access to flow results
- Enhanced error handling and validation
- Improved composability of flows
- Simplified interface for creating and executing flows
"""

from .base import Flow
from .stage import Stage
from .standalone import StandaloneStage
from .composite import CompositeFlow
from .builder import FlowBuilder
from .decorators import flow, stage, standalone, pipeline
from .registry import stage_registry

__all__ = [
    "Flow",
    "Stage",
    "StandaloneStage",
    "CompositeFlow",
    "FlowBuilder",
    "flow",
    "stage",
    "standalone",
    "pipeline",
    "stage_registry",
]
