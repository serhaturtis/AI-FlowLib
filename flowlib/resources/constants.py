from enum import Enum

class ResourceType(str, Enum):
    """Enumeration of standard resource types.
    
    Using string enum ensures type checking while maintaining 
    string compatibility for storage and serialization.
    """
    MODEL = "model"
    PROMPT = "prompt"
    CONFIG = "config"
    EMBEDDING = "embedding"