"""Framework for building flow-based applications."""

from .flows.decorators import flow, stage, pipeline
from .core.application.config import config
from .flows.builder import FlowBuilder
from .flows.base import Flow
from .core.models.context import Context
from .core.models.results import FlowResult, FlowStatus
from .core.errors.base import ValidationError, ErrorContext

__version__ = "0.1.0"
__all__ = [
    # Decorators for flow creation
    "flow",
    "stage",
    "pipeline",
    
    # Configuration management
    "config",
    
    # Flow building
    "FlowBuilder",
    "Flow",
    
    # Core models
    "Context",
    "FlowResult",
    "FlowStatus",
    
    # Error handling
    "ValidationError",
    "ErrorContext"
]