from enum import Enum


class FlowStatus(str, Enum):
    """Enumeration of possible flow execution statuses."""
    
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    
    def is_terminal(self) -> bool:
        """Check if status is terminal."""
        return self in (
            self.SUCCESS,
            self.ERROR,
            self.CANCELED,
            self.TIMEOUT,
            self.SKIPPED
        )
    
    def is_error(self) -> bool:
        """Check if status indicates an error."""
        return self in (
            self.ERROR,
            self.TIMEOUT,
        )
    
    def __str__(self) -> str:
        """Return string representation."""
        return self.value