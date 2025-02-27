"""Validation framework for the Integrated Agent System."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type
from uuid import UUID, uuid4

from ..models.base import DomainState, ValidationLevel, ValidationResult
from ..utils.errors import ValidationError

logger = logging.getLogger(__name__)

class Validator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    async def validate(self, state: DomainState) -> List[ValidationResult]:
        """Validate a domain state.
        
        Args:
            state: Domain state to validate
            
        Returns:
            List of validation results
            
        Raises:
            ValidationError: If validation fails
        """
        pass

class ValidationRegistry:
    """Registry for domain validators."""
    
    def __init__(self):
        """Initialize validation registry."""
        self._validators: Dict[str, Set[Validator]] = {}
        
    def register_validator(
        self,
        domain: str,
        validator: Validator
    ) -> None:
        """Register a validator for a domain.
        
        Args:
            domain: Domain to register validator for
            validator: Validator instance
        """
        if domain not in self._validators:
            self._validators[domain] = set()
        self._validators[domain].add(validator)
        
    def unregister_validator(
        self,
        domain: str,
        validator: Validator
    ) -> None:
        """Unregister a validator from a domain.
        
        Args:
            domain: Domain to unregister validator from
            validator: Validator instance to remove
        """
        if domain in self._validators:
            self._validators[domain].discard(validator)
            
    def get_validators(
        self,
        domain: str
    ) -> Set[Validator]:
        """Get all validators for a domain.
        
        Args:
            domain: Domain to get validators for
            
        Returns:
            Set of validators for the domain
        """
        return self._validators.get(domain, set()).copy()

class ValidationManager:
    """Manager for running validations across domains."""
    
    def __init__(self, registry: ValidationRegistry):
        """Initialize validation manager.
        
        Args:
            registry: Validation registry to use
        """
        self._registry = registry
        
    async def validate_state(
        self,
        state: DomainState,
        raise_on_error: bool = False
    ) -> List[ValidationResult]:
        """Validate a domain state using registered validators.
        
        Args:
            state: Domain state to validate
            raise_on_error: Whether to raise an exception on validation errors
            
        Returns:
            List of validation results
            
        Raises:
            ValidationError: If validation fails and raise_on_error is True
        """
        results = []
        validators = self._registry.get_validators(state.domain)
        
        for validator in validators:
            try:
                validator_results = await validator.validate(state)
                results.extend(validator_results)
            except Exception as e:
                logger.error(
                    f"Validator {validator.__class__.__name__} failed: {str(e)}"
                )
                results.append(
                    ValidationResult(
                        id=uuid4(),
                        valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Validator failed: {str(e)}",
                        domain=state.domain,
                        timestamp=datetime.utcnow(),
                        details={
                            "validator": validator.__class__.__name__,
                            "error": str(e)
                        }
                    )
                )
                
        if raise_on_error and any(
            r.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
            for r in results
        ):
            error_results = [
                r for r in results
                if r.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
            ]
            raise ValidationError(
                f"Validation failed with {len(error_results)} errors",
                results=error_results
            )
            
        return results

class CrossDomainValidator(Validator):
    """Base class for validators that check relationships between domains."""
    
    def __init__(
        self,
        source_domain: str,
        target_domain: str
    ):
        """Initialize cross-domain validator.
        
        Args:
            source_domain: Domain being validated
            target_domain: Domain to validate against
        """
        self.source_domain = source_domain
        self.target_domain = target_domain
        
    @abstractmethod
    async def validate_relationship(
        self,
        source_state: DomainState,
        target_state: DomainState
    ) -> List[ValidationResult]:
        """Validate relationship between domain states.
        
        Args:
            source_state: Source domain state
            target_state: Target domain state
            
        Returns:
            List of validation results
        """
        pass
        
    async def validate(
        self,
        state: DomainState
    ) -> List[ValidationResult]:
        """Validate source state against target state.
        
        This method should be implemented by subclasses to:
        1. Get the target state from the state manager
        2. Call validate_relationship with both states
        3. Return the validation results
        
        Args:
            state: Source domain state to validate
            
        Returns:
            List of validation results
        """
        raise NotImplementedError(
            "CrossDomainValidator.validate must be implemented by subclasses"
        ) 