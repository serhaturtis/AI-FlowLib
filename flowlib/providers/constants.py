"""Constants for the flowlib registry system.

This module defines standardized constants for resource types and provider types,
ensuring consistent typing throughout the system.
"""

from enum import Enum
    
class ProviderType(str, Enum):
    """Enumeration of standard provider types.
    
    Using string enum ensures type checking while maintaining
    string compatibility for storage and serialization.
    """
    LLM = "llm"
    VECTOR_DB = "vector_db"
    DATABASE = "database"
    CACHE = "cache"
    STORAGE = "storage"
    MESSAGE_QUEUE = "message_queue"
    GPU = "gpu"
    API = "api"
    CONVERSATION = "conversation"
    GRAPH_DB = "graph_db"  # Graph database provider 