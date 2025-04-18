"""
Factory for creating state persisters.

This module provides a factory function for creating state persisters
based on the specified type.
"""

import logging
from typing import Optional, Union

from ..core.errors import StatePersistenceError
from .file import FileStatePersister
from .provider import ProviderStatePersister
from ..models.config import StatePersistenceConfig

logger = logging.getLogger(__name__)


def create_state_persister(
    persister_type: str = "file",
    **kwargs
) -> Optional[Union[FileStatePersister, ProviderStatePersister]]:
    """Create a state persister based on the specified type.
    
    Args:
        persister_type: Type of persister to create ("file" or "provider")
        **kwargs: Additional arguments for the persister. Common parameters include:
            - base_path/directory: Path for file storage (for 'file' persister)
            - provider_name/provider_id: Provider name (for 'provider' persister)
        
    Returns:
        Created state persister or None if type is invalid
        
    Raises:
        StatePersistenceError: If there is an error creating the persister
    """
    try:
        # Handle the case where a StatePersistenceConfig is provided
        if 'config' in kwargs and isinstance(kwargs['config'], StatePersistenceConfig):
            config = kwargs['config']
            persister_type = config.persistence_type
            
            if persister_type == "file":
                return FileStatePersister(directory=config.base_path)
            elif persister_type == "provider":
                return ProviderStatePersister(provider_name=config.provider_id)
        
        # Standard case with individual parameters
        if persister_type == "file":
            # Support both base_path (new) and directory (old) parameter names
            directory = kwargs.get("base_path", kwargs.get("directory", "./states"))
            return FileStatePersister(directory=directory)
            
        elif persister_type == "provider":
            # Support both provider_id (new) and provider_name (old) parameter names
            provider_name = kwargs.get("provider_id", kwargs.get("provider_name"))
            if not provider_name:
                raise StatePersistenceError(
                    message="Provider name is required for provider persister",
                    operation="create"
                )
            return ProviderStatePersister(provider_name=provider_name)
            
        else:
            logger.warning(f"Invalid persister type: {persister_type}")
            return None
            
    except Exception as e:
        error_msg = f"Error creating state persister: {str(e)}"
        logger.error(error_msg)
        raise StatePersistenceError(
            message=error_msg,
            operation="create",
            cause=e
        ) 