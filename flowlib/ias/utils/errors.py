"""Error handling utilities for the Integrated Agent System."""

from typing import Any, Dict, List, Optional

from ..models.base import ValidationResult

class IASError(Exception):
    """Base class for all IAS errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize IAS error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.details = details or {}

class StateError(IASError):
    """Error raised when state operations fail."""
    pass

class EventError(IASError):
    """Error raised when event operations fail."""
    pass

class ValidationError(IASError):
    """Error raised when validation fails."""
    
    def __init__(
        self,
        message: str,
        results: Optional[List[ValidationResult]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            results: List of validation results that caused the error
            details: Additional error details
        """
        super().__init__(message, details)
        self.results = results or []

class WorkflowError(IASError):
    """Error raised when workflow operations fail."""
    pass

class DomainError(IASError):
    """Error raised when domain-specific operations fail."""
    
    def __init__(
        self,
        message: str,
        domain: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize domain error.
        
        Args:
            message: Error message
            domain: Domain where the error occurred
            details: Additional error details
        """
        super().__init__(message, details)
        self.domain = domain 