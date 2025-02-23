from typing import Optional, Dict, Any, Generic, TypeVar
from abc import abstractmethod
from pydantic import BaseModel

from ..models.context import Context
from ..models.results import FlowResult, FlowStatus
from ..errors.base import ExecutionError, ErrorContext
from ...flows.base import Flow
from .base import ManagedResource

C = TypeVar('C', bound=BaseModel)
R = TypeVar('R', bound=BaseModel)

class FlowApplication(ManagedResource[C], Generic[C, R]):
    """Base class for flow-based applications."""
    
    def __init__(self):
        """Initialize flow application."""
        super().__init__()
        self.flow: Optional[Flow] = None
    
    @abstractmethod
    def build_pipeline(self, config: C) -> Flow:
        """Build the flow pipeline.
        
        Args:
            config: Application configuration
            
        Returns:
            Constructed flow pipeline
        """
        pass
    
    async def initialize(self, config: C) -> None:
        """Initialize the application.
        
        Args:
            config: Application configuration
            
        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            # Build and set flow
            self.flow = self.build_pipeline(config)
            self._is_initialized = True
        except Exception as e:
            await self.cleanup()
            raise
    
    async def process(self, data: Dict[str, Any]) -> R:
        """Process data through the flow pipeline.
        
        Args:
            data: Input data
            
        Returns:
            Processing result
            
        Raises:
            StateError: If application is not initialized
            ExecutionError: If processing fails
        """
        self.check_initialized()
        
        context = Context(data=data)
        result = await self.flow._execute(context)
        
        if result.status != FlowStatus.SUCCESS:
            raise ExecutionError(
                "Flow execution failed",
                ErrorContext.create(
                    error=result.error,
                    metadata=result.metadata,
                    flow_name=result.flow_name,
                    status=result.status
                )
            )
        
        return self._validate_result(result)
    
    @abstractmethod
    def _validate_result(self, result: FlowResult) -> R:
        """Validate and convert flow result.
        
        Args:
            result: Flow execution result
            
        Returns:
            Validated result object
        """
        pass
    
    async def cleanup(self) -> None:
        """Clean up application resources."""
        if self.flow:
            self.flow.cleanup()
            self.flow = None
        await super().cleanup() 