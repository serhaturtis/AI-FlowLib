"""
Agent error classes.

This module defines the error classes used throughout the agent system.
"""

from typing import Any, Dict, Optional, Type


class AgentError(Exception):
    """Base error class for agent errors.
    
    All errors in the agent system should inherit from this class.
    """
    
    def __init__(
        self, 
        message: str,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize agent error.
        
        Args:
            message: Error message
            cause: Original exception that caused this error
            **context: Additional context information
        """
        self.message = message
        self.cause = cause
        self.context = context
        
        # Build the full message
        full_message = message
        if cause:
            full_message += f" | Caused by: {str(cause)}"
        
        # Pass to base class
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
        }
        
        if self.cause:
            if hasattr(self.cause, "to_dict") and callable(self.cause.to_dict):
                result["cause"] = self.cause.to_dict()
            else:
                result["cause"] = {
                    "error_type": self.cause.__class__.__name__,
                    "message": str(self.cause),
                }
        
        return result


class NotInitializedError(AgentError):
    """Error raised when a component is used before initialization.
    
    Components must be initialized before they can be used.
    """
    
    def __init__(
        self,
        message: str = None,
        component_name: str = None,
        operation: Optional[str] = None,
        **context
    ):
        """Initialize not initialized error.
        
        Args:
            message: Optional custom error message
            component_name: Name of the component that was not initialized
            operation: Optional operation that was attempted
            **context: Additional context information
        """
        if message is None:
            message = f"Component '{component_name}' must be initialized before use"
            if operation:
                message += f" (attempted operation: {operation})"
                
        if component_name:
            context["component_name"] = component_name
        if operation:
            context["operation"] = operation
            
        super().__init__(message, **context)


class ComponentError(AgentError):
    """Error in component operation.
    
    Raised when a component fails to initialize, operate, or shutdown.
    """
    
    def __init__(
        self, 
        message: str,
        component_name: str,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize component error.
        
        Args:
            message: Error message
            component_name: Name of the component that failed
            operation: Optional name of the operation that failed
            cause: Original exception that caused this error
            **context: Additional context information
        """
        context["component_name"] = component_name
        if operation:
            context["operation"] = operation
            
        super().__init__(message, cause, **context)


class ConfigurationError(AgentError):
    """Error in agent configuration.
    
    Raised when there is an invalid configuration value or missing required config.
    """
    
    def __init__(
        self, 
        message: str,
        config_key: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        required_type: Optional[Type] = None,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Key of the problematic configuration
            invalid_value: Invalid value that caused the error
            required_type: Type that was expected
            cause: Original exception that caused this error
            **context: Additional context information
        """
        if config_key:
            context["config_key"] = config_key
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
        if required_type:
            context["required_type"] = str(required_type)
            
        super().__init__(message, cause, **context)


class ExecutionError(AgentError):
    """Error during agent execution.
    
    Raised when there is a failure during the execution cycle.
    """
    
    def __init__(
        self, 
        message: str,
        agent: Optional[str] = None,
        state: Optional[Any] = None,
        flow: Optional[str] = None,
        stage: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize execution error.
        
        Args:
            message: Error message
            agent: Name of the agent that encountered the error
            state: State at the time of the error
            flow: Flow that was being executed
            stage: Stage in the execution cycle
            cause: Original exception that caused this error
            **context: Additional context information
        """
        if agent:
            context["agent"] = agent
        if flow:
            context["flow"] = flow
        if stage:
            context["stage"] = stage
        if state:
            # Add key state attributes but avoid huge serialization
            try:
                if hasattr(state, "task_id"):
                    context["task_id"] = state.task_id
                if hasattr(state, "is_complete"):
                    context["is_complete"] = state.is_complete
                if hasattr(state, "progress"):
                    context["progress"] = state.progress
            except Exception:
                # Ignore any errors in state extraction
                pass
            
        super().__init__(message, cause, **context)


class PlanningError(ExecutionError):
    """Error during agent planning.
    
    Raised when there is a failure during the planning phase.
    """
    
    def __init__(
        self,
        message: str,
        planning_type: str = "planning",  # "planning" or "input_generation"
        **kwargs
    ):
        """Initialize planning error.
        
        Args:
            message: Error message
            planning_type: Type of planning that failed
            **kwargs: Additional arguments for ExecutionError
        """
        # Add planning type to context
        kwargs.setdefault("context", {})
        kwargs["context"]["planning_type"] = planning_type
        kwargs["stage"] = planning_type
        
        super().__init__(message, **kwargs)


class ReflectionError(ExecutionError):
    """Error during agent reflection.
    
    Raised when there is a failure during the reflection phase.
    """
    
    def __init__(
        self,
        message: str,
        **kwargs
    ):
        """Initialize reflection error.
        
        Args:
            message: Error message
            **kwargs: Additional arguments for ExecutionError
        """
        kwargs["stage"] = "reflection"
        super().__init__(message, **kwargs)


class MemoryError(AgentError):
    """Error in memory operations.
    
    Raised when there is a failure during memory store, retrieve, or search.
    """
    
    def __init__(
        self, 
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None,
        context: Optional[str] = None,
        cause: Optional[Exception] = None,
        **kwargs
    ):
        """Initialize memory error.
        
        Args:
            message: Error message
            operation: Memory operation that failed
            key: Key being accessed
            context: Memory context
            cause: Original exception that caused this error
            **kwargs: Additional context information
        """
        if operation:
            kwargs["operation"] = operation
        if key:
            kwargs["key"] = key
        if context:
            kwargs["memory_context"] = context
            
        super().__init__(message, cause, **kwargs)


class FlowDiscoveryError(AgentError):
    """Error during flow discovery.
    
    Raised when a flow cannot be discovered or processed.
    """
    
    def __init__(
        self, 
        message: str,
        flow_name: Optional[str] = None,
        flow_path: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize flow discovery error.
        
        Args:
            message: Error message
            flow_name: Name of the flow that caused the error
            flow_path: Path to the flow that caused the error
            cause: Original exception that caused this error
            **context: Additional context information
        """
        if flow_name:
            context["flow_name"] = flow_name
        if flow_path:
            context["flow_path"] = flow_path
            
        super().__init__(message, cause, **context)


class DiscoveryError(AgentError):
    """Error during resource discovery.
    
    Raised when a discovery operation fails.
    """
    
    def __init__(
        self, 
        message: str,
        operation: Optional[str] = None,
        resource_type: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize discovery error.
        
        Args:
            message: Error message
            operation: The discovery operation that failed
            resource_type: Type of resource being discovered
            cause: Original exception that caused this error
            **context: Additional context information
        """
        if operation:
            context["operation"] = operation
        if resource_type:
            context["resource_type"] = resource_type
            
        super().__init__(message, cause, **context)


class StatePersistenceError(AgentError):
    """Error in state persistence.
    
    Raised when there is a failure during state saving or loading.
    """
    
    def __init__(
        self, 
        message: str,
        operation: str,  # "save", "load", "delete", "list"
        task_id: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context
    ):
        """Initialize state persistence error.
        
        Args:
            message: Error message
            operation: Persistence operation that failed
            task_id: ID of the task being persisted
            cause: Original exception that caused this error
            **context: Additional context information
        """
        context["operation"] = operation
        if task_id:
            context["task_id"] = task_id
            
        super().__init__(message, cause, **context)


class ProviderError(AgentError):
    """Error in provider operations.
    
    Raised when there is a failure in a provider operation.
    """
    
    def __init__(
        self, 
        message: str,
        provider_name: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
        **kwargs
    ):
        """Initialize provider error.
        
        Args:
            message: Error message
            provider_name: Name of the provider that failed
            operation: Provider operation that failed
            cause: Original exception that caused this error
            **kwargs: Additional context information
        """
        if provider_name:
            kwargs["provider_name"] = provider_name
        if operation:
            kwargs["operation"] = operation
            
        super().__init__(message, cause, **kwargs) 