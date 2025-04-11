from typing import Any, Type, TypeVar, cast

T = TypeVar('T')

def create_settings(settings_class: Type[T], **kwargs: Any) -> T:
    """Create settings instance with provided values.
    
    Args:
        settings_class: Settings class to instantiate
        **kwargs: Settings values
        
    Returns:
        Settings instance
    """
    return cast(T, settings_class(**kwargs))
