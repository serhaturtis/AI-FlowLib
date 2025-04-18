"""Enhanced decorators for flow and stage creation.

This module provides a set of decorators that simplify the creation of flows,
stages, and pipelines with improved type safety and error handling.
"""

import functools
import logging
import datetime
import types
from typing import Any, Callable, Dict, Optional, Type, TypeVar, List
from pydantic import BaseModel

from .registry import stage_registry
from .stage import Stage
from .standalone import StandaloneStage
from .base import Flow

F = TypeVar('F', bound=Callable)
C = TypeVar('C', bound=type)

# Setup logging
logger = logging.getLogger(__name__)

def flow(cls=None, *, name: str = None, description: str, is_infrastructure: bool = False):
    """
    Decorator to mark a class as a flow.
    
    This decorator registers a class as a flow in the stage registry and enforces
    that the class implements the required Flow interface.
    
    Args:
        cls: The class to decorate
        name: Optional custom name for the flow
        description: Description of the flow's purpose (required)
        is_infrastructure: Whether this is an infrastructure flow
        
    Returns:
        The decorated flow class
        
    Raises:
        ValueError: If description is not provided
    """
    def wrap(cls):
        # Create a get_description method that returns the provided description
        def get_description(self):
            return description
        
        # Add the method to the class
        if not hasattr(cls, 'get_description'):
            setattr(cls, 'get_description', get_description)
            logger.debug(f"Added get_description method to flow class '{cls.__name__}'")
        
        # If the class already has a get_description method, we keep it
            
        if not issubclass(cls, Flow):
            # Create a new class that inherits from both the original class and Flow
            original_cls = cls
            original_name = cls.__name__
            original_dict = dict(cls.__dict__)
            
            # Remove items that would cause conflicts
            for key in ['__dict__', '__weakref__']:
                if key in original_dict:
                    del original_dict[key]
            
            # Create the new class with multiple inheritance
            cls = type(
                original_name,
                (original_cls, Flow),
                original_dict
            )
            
            # Initialize Flow with default parameters
            original_init = cls.__init__
            
            def new_init(self, *args, **kwargs):
                # Call Flow's __init__ with appropriate parameters
                flow_name = name or original_name
                Flow.__init__(
                    self, 
                    flow_name,  # Positional argument
                    input_schema=None,  # Will be set from pipeline method
                    output_schema=None,  # Will be set from pipeline method
                    metadata={"is_infrastructure": is_infrastructure}
                )
                
                # Only call original_init if it's different from Flow.__init__
                if original_init is not Flow.__init__:
                    try:
                        original_cls.__init__(self, *args, **kwargs)
                    except TypeError:
                        try:
                            original_cls.__init__(self)
                        except Exception as e:
                            logger.warning(f"Failed to call original __init__: {e}")
            
            cls.__init__ = new_init
        
        # Set flow metadata
        flow_name = name or cls.__name__
        cls.__flow_metadata__ = {
            "name": flow_name,
            "is_infrastructure": is_infrastructure
        }
        
        # Set the is_infrastructure flag directly on the class
        cls.is_infrastructure = is_infrastructure
        
        # Create a flow instance to store in the registry
        try:
            flow_instance = cls()
            # Set the name attribute directly on the flow instance
            setattr(flow_instance, "name", flow_name)
            
            # Find pipeline method to extract schemas and set them on the flow instance
            for method_name in dir(cls):
                method = getattr(cls, method_name)
                if hasattr(method, '__pipeline__') and method.__pipeline__:
                    # Get schemas from pipeline method
                    input_schema = getattr(method, 'input_model', None)
                    output_schema = getattr(method, 'output_model', None)
                    
                    # Set schemas on flow instance so they're available during registration
                    if input_schema:
                        setattr(flow_instance, "input_schema", input_schema)
                        logger.debug(f"Set input_schema={input_schema.__name__} on flow '{flow_name}'")
                    
                    if output_schema:
                        setattr(flow_instance, "output_schema", output_schema)
                        logger.debug(f"Set output_schema={output_schema.__name__} on flow '{flow_name}'")
                    
                    break
            
            # Register the flow with the registry, including the instance
            stage_registry.register_flow(flow_name, flow_instance)
            logger.debug(f"Registered flow class and instance: {flow_name}")
        except Exception as e:
            # If instantiation fails, still register the flow name
            stage_registry.register_flow(flow_name)
            logger.warning(f"Failed to create instance for flow '{flow_name}': {e}")
        
        # Count pipeline methods and collect stages
        pipeline_methods = []
        stage_methods = []
        
        for attr_name in dir(cls):
            # Skip special methods and attributes
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
                
            attr = getattr(cls, attr_name)
            
            # Check if this is a pipeline method
            if hasattr(attr, '__pipeline__') and attr.__pipeline__:
                pipeline_methods.append(attr_name)
            
            # Check if this is a stage method
            if hasattr(attr, "__register_stage__"):
                stage_methods.append(attr_name)
                # Register the stage with the flow
                register_stage = getattr(attr, "__register_stage__")
                register_stage(cls)
        
        # Enforce exactly one pipeline method
        if len(pipeline_methods) == 0:
            raise ValueError(f"Flow class '{flow_name}' must define exactly one pipeline method using @pipeline decorator")
        elif len(pipeline_methods) > 1:
            raise ValueError(f"Flow class '{flow_name}' has multiple pipeline methods: {', '.join(pipeline_methods)}. Only one is allowed.")
        
        # Store the pipeline method name
        cls.__pipeline_method__ = pipeline_methods[0]
        
        # Make non-execute stage methods private
        for attr_name in stage_methods:
            # Skip the pipeline method
            if attr_name == cls.__pipeline_method__:
                continue
                
            # Get the original method
            original_method = getattr(cls, attr_name)
            
            # Create a private method name (using single underscore for less intrusive approach)
            private_name = f"_{attr_name}"
            
            # Set the private method
            setattr(cls, private_name, original_method)
            
            # Remove the public method if it's not already private
            if not attr_name.startswith('_'):
                delattr(cls, attr_name)
        
        # Add flow methods if they don't already exist
        if not hasattr(cls, "get_stage"):
            setattr(cls, "get_stage", get_stage)
        
        if not hasattr(cls, "get_stages"):
            setattr(cls, "get_stages", get_stages)
            
        # Store flow class name for easier debugging
        cls.__flow_name__ = flow_name
            
        return cls
    
    if cls is None:
        return wrap
    return wrap(cls)


def stage(
    name: Optional[str] = None,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None,
    **metadata
) -> Callable:
    """
    Decorator to mark a method as a flow stage.
    
    This decorator registers the method as a stage in the stage registry and adds
    metadata to the method for later retrieval. It can be used on methods within a flow
    class or on standalone functions.
    
    Args:
        name: Optional custom name for the stage. If not provided, the method name is used.
        input_model: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
        output_model: Optional Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
        **metadata: Additional metadata to attach to the stage.
        
    Returns:
        The decorated method or function.
        
    Raises:
        ValueError: If input_model or output_model is provided but not a Pydantic BaseModel subclass.
    """
    # Validate input_model and output_model are Pydantic models if provided
    if input_model is not None and not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
        raise ValueError(f"Stage input_model must be a Pydantic BaseModel subclass, got {input_model}")
    
    if output_model is not None and not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
        raise ValueError(f"Stage output_model must be a Pydantic BaseModel subclass, got {output_model}")
    
    def decorator(func: Callable) -> Callable:
        # Get the real function behind any wrappers
        stage_func = func
        while hasattr(stage_func, "__wrapped__"):
            stage_func = getattr(stage_func, "__wrapped__")
        
        # Determine stage name
        stage_name = name or func.__name__
        
        # Add stage attributes to the function
        func.__stage_name__ = stage_name
        func.__input_model__ = input_model
        func.__output_model__ = output_model
        func.__stage_metadata__ = metadata
        
        # Register the stage based on context
        def register_stage_for_flow(flow_cls):
            flow_name = flow_cls.__name__
            logger.debug(f"Registering stage '{stage_name}' for flow '{flow_name}'")
            
            stage_registry.register_stage(
                stage_name=stage_name,
                flow_name=flow_name,
                name=func.__name__,
                input_model=input_model,
                output_model=output_model,
                metadata=metadata
            )
            
            return func
        
        # Add a special attribute to the function for flow registration
        func.__register_stage__ = register_stage_for_flow
        
        # Check if this is a standalone stage (not in a flow class)
        # Determine if this is a standalone stage or a flow method
        is_standalone = True
        if func.__qualname__ != func.__name__:
            # Format: ClassName.method_name - this is a method on a class
            class_name = func.__qualname__.split('.')[0]
            method_name = func.__name__
            
            # Check if this appears to be a method on a flow class
            # We consider it a flow method if its qualname has a specific structure
            if class_name and '.<locals>.' not in func.__qualname__:
                is_standalone = False
        
        # Register as standalone stage if not in a flow class
        if is_standalone:
            logger.debug(f"Registering standalone stage: {stage_name}")
            stage_registry.register_stage(
                stage_name=stage_name,
                flow_name=None,  # None indicates standalone
                func=func,
                input_model=input_model,
                output_model=output_model,
                metadata=metadata
            )
        
        return func
    
    return decorator


def standalone(
    name: Optional[str] = None,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], StandaloneStage]:
    """Create a standalone stage from a function.
    
    This decorator:
    1. Wraps a function as a standalone, reusable stage
    2. Provides input and output validation
    3. Preserves function usability while adding stage capabilities
    4. Registers the stage with the stage registry
    
    Args:
        name: Optional name for the stage (defaults to function name)
        input_model: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
        output_model: Optional Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
        metadata: Optional metadata about the stage
        
    Returns:
        Decorator function
        
    Raises:
        ValueError: If input_model or output_model is provided but not a Pydantic BaseModel subclass.
    """
    # Validate input_model and output_model are Pydantic models if provided
    if input_model is not None and not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
        raise ValueError(f"Standalone stage input_model must be a Pydantic BaseModel subclass, got {input_model}")
    
    if output_model is not None and not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
        raise ValueError(f"Standalone stage output_model must be a Pydantic BaseModel subclass, got {output_model}")
        
    return StandaloneStage.create(
        name=name,
        input_model=input_model,
        output_model=output_model,
        metadata=metadata
    )


def pipeline(func: Optional[Callable] = None, **pipeline_kwargs):
    """Mark a method as a flow pipeline.
    
    This decorator wraps a method to provide pipeline execution capabilities:
    1. Manages execution context
    2. Tracks pipeline metadata and execution status
    3. Initializes stages if needed
    4. Validates output conforms to declared output model
    
    Args:
        func: The method to decorate
        **pipeline_kwargs: Additional pipeline options including:
            - input_model: Pydantic model for input validation (must be a BaseModel subclass)
            - output_model: Pydantic model for output validation (must be a BaseModel subclass)
        
    Returns:
        Decorated pipeline method
        
    Raises:
        ValueError: If input_model or output_model is provided but not a Pydantic BaseModel subclass.
    """
    # Get pipeline metadata from kwargs
    input_model = pipeline_kwargs.get("input_model")
    output_model = pipeline_kwargs.get("output_model")
    
    # Validate input_model and output_model are Pydantic models if provided
    if input_model is not None and not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
        raise ValueError(f"Pipeline input_model must be a Pydantic BaseModel subclass, got {input_model}")
    
    if output_model is not None and not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
        raise ValueError(f"Pipeline output_model must be a Pydantic BaseModel subclass, got {output_model}")
        
    def decorator(method):
        @functools.wraps(method)
        async def wrapper(*args, **kwargs):
            """Wrapped pipeline method."""
            # Track execution stats
            start_time = datetime.datetime.now()
            
            try:
                # Execute the pipeline method and return the result directly
                result = await method(*args, **kwargs)
                return result
            finally:
                # Calculate execution time
                execution_time = datetime.datetime.now() - start_time
                
                # Log execution details
                if len(args) > 0 and hasattr(args[0], "__class__"):
                    flow_class = args[0].__class__.__name__
                    logger.debug(f"Pipeline execution: {flow_class}.{method.__name__} ({execution_time.total_seconds():.3f}s)")
        
        # Modern style - these are the preferred attributes we'll use
        wrapper.__pipeline__ = True
        wrapper.__input_model__ = input_model
        wrapper.__output_model__ = output_model
        
        return wrapper
        
    # Support both @pipeline and @pipeline() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def get_stage(self, stage_name: str) -> Stage:
    """
    Get a stage instance by name.
    
    Args:
        stage_name: The name of the stage to get.
        
    Returns:
        Stage: The stage instance.
        
    Raises:
        ValueError: If the stage is not found or cannot be instantiated.
    """
    # Validate input
    if not stage_name:
        raise ValueError("Stage name cannot be empty")
        
    # Initialize stage instances cache if needed
    if not hasattr(self, "__stage_instances__"):
        self.__stage_instances__ = {}
    
    # Return cached stage if available
    if stage_name in self.__stage_instances__:
        return self.__stage_instances__[stage_name]
    
    # Get the stage information from the registry
    try:
        stage_info = stage_registry.get_stage(stage_name, self.__class__.__name__)
        
        # Create the stage instance
        stage: Stage
        
        if stage_info.get("is_standalone", False):
            # Create standalone stage
            stage_func = stage_info.get("func")
            if not stage_func or not callable(stage_func):
                raise ValueError(f"Standalone stage '{stage_name}' has invalid or missing function")
                
            stage = Stage(
                name=stage_name,
                process_fn=stage_func,
                input_schema=stage_info.get("input_model"),
                output_schema=stage_info.get("output_model"),
                metadata=stage_info.get("metadata", {})
            )
        else:
            # Get the method from the instance or class
            method_name = stage_info.get("name", stage_name)
            
            try:
                # First try to get the method from the instance
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    if not callable(method):
                        raise ValueError(f"Stage method '{method_name}' is not callable")
                else:
                    # If not on instance, get from class and bind to instance
                    if hasattr(self.__class__, method_name):
                        class_method = getattr(self.__class__, method_name)
                        if not callable(class_method):
                            raise ValueError(f"Stage method '{method_name}' is not callable")
                        method = types.MethodType(class_method, self)
                    else:
                        raise ValueError(f"Stage method '{method_name}' not found on instance or class")
            except Exception as e:
                raise ValueError(f"Failed to retrieve method for stage '{stage_name}': {str(e)}")
            
            # Create stage instance
            stage = Stage(
                name=stage_name,
                process_fn=method,
                input_schema=stage_info.get("input_model"),
                output_schema=stage_info.get("output_model"),
                metadata=stage_info.get("metadata", {})
            )
        
        # Cache the stage instance
        self.__stage_instances__[stage_name] = stage
        return stage
        
    except KeyError:
        # Stage not found in registry
        available_stages = stage_registry.get_stages(self.__class__.__name__)
        suggestion = ""
        if available_stages:
            suggestion = f" Available stages: {', '.join(available_stages)}"
        
        raise ValueError(f"Stage '{stage_name}' not found in flow '{self.__class__.__name__}'.{suggestion}")


def get_stages(self) -> List[str]:
    """
    Get all available stages for this flow.
    
    Returns:
        List[str]: List of stage names registered for this flow.
    """
    # Get all registered stages for this flow
    return stage_registry.get_stages(self.__class__.__name__) 