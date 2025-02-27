"""Decorators for simplified flow creation."""

from typing import Any, Callable, Optional, Type, TypeVar, Union, Dict
from functools import wraps
import inspect
from pydantic import BaseModel

from .stage import Stage
from .composite import CompositeFlow
from .base import Flow
from ..core.models.context import Context
from ..core.models.results import FlowResult, FlowStatus
from ..core.errors.base import ValidationError, ErrorContext
from ..core.resources import ResourceRegistry

T = TypeVar('T')
F = TypeVar('F', bound=Callable)

class StageMethod:
    """Descriptor for stage methods that handles registration."""
    
    def __init__(self, func: Callable, stage: Stage):
        self.func = func
        self.stage = stage
        self.__doc__ = func.__doc__
        
    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Union[Callable, 'StageMethod']:
        if obj is None:
            return self
        
        # Register stage with class if not already registered
        if not hasattr(obj.__class__, '_stages'):
            obj.__class__._stages = {}
        if self.stage.name not in obj.__class__._stages:
            obj.__class__._stages[self.stage.name] = self.stage
            
        # Return bound method
        return self.func.__get__(obj, objtype)

def flow(name: Optional[str] = None, **metadata: Any) -> Callable[[Type[T]], Type[T]]:
    """Decorator to mark a class as a flow container.
    
    Args:
        name: Optional flow name
        **metadata: Additional flow metadata
        
    Returns:
        Decorated class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Store flow metadata
        cls._flow_name = name or cls.__name__
        cls._flow_metadata = metadata
        cls._stages = {}
        cls._pipeline = None
        
        # Create async context manager methods if not defined
        if not hasattr(cls, '__aenter__'):
            async def __aenter__(self):
                return self
            cls.__aenter__ = __aenter__
            
        if not hasattr(cls, '__aexit__'):
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            cls.__aexit__ = __aexit__
        
        return cls
    return decorator

def stage(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None
) -> Union[F, Callable[[F], F]]:
    """Decorator to create a flow stage from a method.
    
    Args:
        func: Function to decorate
        name: Optional stage name
        input_model: Optional input validation model
        output_model: Optional output validation model
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> StageMethod:
        # Get function signature
        sig = inspect.signature(func)
        stage_name = name or func.__name__
        
        # Create stage process function
        @wraps(func)
        async def process_fn(data: Dict[str, Any]) -> Dict[str, Any]:
            # Get instance from context
            instance = data.get('_instance')
            if not instance:
                raise ValidationError(
                    "No instance found in context",
                    ErrorContext.create(stage_name=stage_name)
                )
            
            # Extract arguments from data
            kwargs = {}
            for param_name in sig.parameters:
                if param_name == 'self':
                    continue
                if param_name in data:
                    kwargs[param_name] = data[param_name]
            
            # Call original function
            result = func(instance, **kwargs)
            if inspect.iscoroutine(result):
                result = await result
                
            # Convert to dict if needed
            if isinstance(result, BaseModel):
                result = result.model_dump()
            elif not isinstance(result, dict):
                result = {'result': result}
                
            return result
        
        # Create stage
        stage = Stage(
            name=stage_name,
            process_fn=process_fn,
            input_schema=input_model,
            output_schema=output_model
        )
        
        # Return descriptor that will handle stage registration
        return StageMethod(func, stage)
    
    if func is None:
        return decorator
    return decorator(func)

def pipeline(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None
) -> Union[F, Callable[[F], F]]:
    """Decorator to create a flow pipeline from a method.
    
    Args:
        func: Function to decorate
        name: Optional pipeline name
        input_model: Optional input validation model
        output_model: Optional output validation model
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        pipeline_name = name or func.__name__
        
        @wraps(func)
        async def wrapper(self: Any, *args, **kwargs) -> Any:
            try:
                # Execute the actual pipeline function
                result = await func(self, *args, **kwargs)
                
                # Validate result if output model specified
                if output_model and not isinstance(result, output_model):
                    result = output_model.model_validate(result)
                
                return result
                
            except Exception as e:
                # Wrap any errors in ValidationError
                if not isinstance(e, ValidationError):
                    raise ValidationError(
                        "Pipeline execution failed",
                        ErrorContext.create(
                            pipeline_name=pipeline_name,
                            error=str(e)
                        )
                    ) from e
                raise
        
        # Store pipeline attributes
        wrapper._pipeline = True
        wrapper.input_model = input_model
        wrapper.output_model = output_model
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)