"""Graph database provider interface.

This package provides providers for graph databases with entity and relationship operations,
supporting knowledge graph capabilities for the entity-centric memory system.
"""

from .base import GraphDBProvider, GraphDBProviderSettings
from .memory_graph import MemoryGraphProvider

# Import new providers
try:
    from .neo4j_provider import Neo4jProvider, Neo4jProviderSettings
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from .arango_provider import ArangoProvider, ArangoProviderSettings
    ARANGO_AVAILABLE = True
except ImportError:
    ARANGO_AVAILABLE = False

try:
    from .janus_provider import JanusProvider, JanusProviderSettings
    JANUS_AVAILABLE = True
except ImportError:
    JANUS_AVAILABLE = False

# Build exports dynamically
__all__ = [
    "GraphDBProvider",
    "GraphDBProviderSettings",
    "MemoryGraphProvider"
]

# Add new providers to exports if available
if NEO4J_AVAILABLE:
    __all__.extend(["Neo4jProvider", "Neo4jProviderSettings"])

if ARANGO_AVAILABLE:
    __all__.extend(["ArangoProvider", "ArangoProviderSettings"])

if JANUS_AVAILABLE:
    __all__.extend(["JanusProvider", "JanusProviderSettings"])

