# Hybrid Entity-Centric Memory System Implementation Plan

This document outlines the implementation plan for enhancing FlowLib's agent memory capabilities with a hybrid entity-centric memory system that leverages both vector and graph database capabilities.

## Target File Tree

```
flowlib/
└── agents/
    ├── memory/
    │   ├── __init__.py                 # Package exports
    │   ├── models.py                   # Entity and memory data models
    │   ├── manager.py                  # HybridMemoryManager implementation
    │   └── utils.py                    # Helper functions
    ├── flows/
    │   ├── __init__.py                 # Package exports
    │   └── memory_flows.py             # Memory-related flows
    ├── prompts/
    │   └── memory_prompts.py           # Memory-specific prompts
    └── providers/
        ├── graph/
        │   ├── __init__.py             # Package exports
        │   ├── base.py                 # GraphDBProvider base class
        │   └── memory_graph.py         # In-memory graph implementation
        └── __init__.py                 # Provider package exports
```

## Implementation Steps

### Phase 1: Core Memory Models

1. Create the basic directory structure:
   ```bash
   mkdir -p flowlib/agents/memory
   mkdir -p flowlib/agents/flows
   mkdir -p flowlib/agents/prompts
   mkdir -p flowlib/agents/providers/graph
   ```

2. Create `flowlib/agents/memory/models.py` with:
   - `EntityAttribute` model
   - `EntityRelationship` model
   - `Entity` model with conversion helpers
   - `ExtractedEntityInfo` model for LLM-generated content

3. Create `flowlib/agents/memory/__init__.py` to expose the models:
   ```python
   from .models import Entity, EntityAttribute, EntityRelationship, ExtractedEntityInfo
   ```

### Phase 2: Graph Provider Implementation

1. Update `flowlib/core/registry/constants.py` to add the graph database provider type:
   ```python
   class ProviderType:
       # Existing types...
       GRAPH_DB = "graph_db"
   ```

2. Create `flowlib/agents/providers/graph/base.py` with the `GraphDBProvider` abstract base class:
   - Define interface methods for entity and relationship management
   - Document the expected behavior of each method

3. Create `flowlib/agents/providers/graph/memory_graph.py` with the in-memory implementation:
   - Implement the `MemoryGraphProvider` class
   - Create data structures for entities and relationships
   - Implement graph traversal algorithm

4. Create `flowlib/agents/providers/graph/__init__.py` to expose the provider:
   ```python
   from .base import GraphDBProvider
   from .memory_graph import MemoryGraphProvider
   ```

### Phase 3: Memory Manager Implementation

1. Create `flowlib/agents/memory/manager.py` with the `HybridMemoryManager` class:
   - Initialize with providers for working memory, vector store, and graph store
   - Implement entity storage and retrieval methods
   - Implement semantic search and graph traversal methods
   - Add context management functionality

2. Add helper functions in `flowlib/agents/memory/utils.py`:
   - Conversion between different memory representations
   - Formatting utilities for human-readable output
   - Validation functions

### Phase 4: Memory Flows

1. Create `flowlib/agents/prompts/memory_prompts.py` with:
   - `EntityExtractionPrompt` for extracting entities from conversation
   - `MemoryRetrievalPrompt` for identifying what memory to retrieve

2. Create `flowlib/agents/flows/memory_flows.py` with:
   - `MemoryExtractionFlow` for extracting entities and storing them
   - `MemoryRetrievalFlow` for retrieving relevant memories
   - Input and output models for these flows

3. Update `flowlib/agents/flows/__init__.py` to expose the new flows:
   ```python
   from .memory_flows import MemoryExtractionFlow, MemoryRetrievalFlow
   ```

### Phase 5: Agent Integration

1. Update `flowlib/agents/base.py` with memory integration methods:
   - Add `retrieve_memories` method
   - Add `extract_and_store_memories` method
   - Update agent initialization to use the new memory manager

2. Modify `flowlib/agents/full.py` to leverage the new memory capabilities:
   - Update `handle_message` to retrieve memories before processing
   - Update post-processing to extract and store memories
   - Remove redundant memory extraction in reflection flow

## Implementation Details

### Entity Structure

Each entity is structured as:

```
Entity
├── id: Unique identifier (e.g., "john_smith", "ankara_city")
├── type: Entity type (e.g., "person", "location")
├── attributes: Dictionary of EntityAttribute objects
│   ├── name: Attribute name -> EntityAttribute
│   ├── age: Attribute name -> EntityAttribute
│   └── ...
├── relationships: List of EntityRelationship objects
│   ├── EntityRelationship(type="friend_of", target="amanda", ...)
│   ├── EntityRelationship(type="lives_in", target="ankara", ...)
│   └── ...
├── tags: List of categorization tags
├── importance: Overall importance score (0.0-1.0)
├── vector_id: ID in vector store if applicable
└── last_updated: Timestamp of last update
```

### Memory Flow Process

1. **Pre-execution memory retrieval**:
   - Analyze user message to identify relevant entities
   - Retrieve mentioned entities from memory
   - Retrieve related entities via graph traversal
   - Perform semantic search for relevant information
   - Format retrieved memories for context injection

2. **Post-execution memory storage**:
   - Extract entity information from conversation
   - Compare with existing entities for updates
   - Store entity nodes in graph database
   - Store entity attributes in vector database
   - Update working memory for quick access

### Hybrid Storage Strategy

1. **Vector Store**:
   - Stores entity attributes for semantic search
   - Each attribute is stored as a separate vector
   - Metadata includes entity ID, type, and attribute name
   - Enables semantic similarity search

2. **Graph Store**:
   - Stores entity nodes and relationships
   - Enables traversal-based queries
   - Maintains explicit relationship information
   - Provides structured knowledge representation

3. **Working Memory**:
   - Caches recently accessed entities
   - Maintains conversation context
   - Provides fast access to active entities
   - Uses TTL for automatic cleanup

## Testing Strategy

1. **Unit Tests**:
   - Test entity model conversions
   - Test graph provider operations
   - Test memory manager entity operations
   - Test memory flow components

2. **Integration Tests**:
   - Test end-to-end memory storage and retrieval
   - Test conversation extraction accuracy
   - Test semantic search effectiveness
   - Test graph traversal behavior

3. **Performance Benchmarks**:
   - Measure entity storage and retrieval times
   - Measure graph traversal performance
   - Measure vector search performance
   - Measure memory consumption

## Phased Delivery

1. **MVP (Minimum Viable Product)**:
   - Basic entity models
   - In-memory graph provider
   - Simplified memory manager
   - Basic memory flows

2. **Enhanced Features**:
   - Advanced relationship traversal
   - Multi-hop reasoning
   - Attribute confidence scoring
   - Temporal entity tracking

3. **Production Optimization**:
   - Connection to persistent graph database
   - Memory optimization strategies
   - Query performance enhancements
   - Scale testing with large entity counts 