# Graph Database Providers

This package provides graph database provider implementations for FlowLib, supporting entity-centric memory and knowledge graph capabilities.

## Available Providers

### MemoryGraphProvider

A simple in-memory graph database implementation for development and testing purposes. No additional installation required.

### Neo4jProvider

Integration with Neo4j, a popular open-source graph database.

Installation:
```bash
pip install neo4j
```

### ArangoProvider

Integration with ArangoDB, a multi-model database with strong graph capabilities.

Installation:
```bash
pip install python-arango
```

### JanusProvider

Integration with JanusGraph, a distributed graph database built on Apache TinkerPop.

Installation:
```bash
pip install gremlinpython
```

## Usage Example

```python
import asyncio
import flowlib as fl
from flowlib.core.registry.constants import ProviderType
from flowlib.agents.memory.models import Entity, EntityAttribute, EntityRelationship

async def main():
    # Initialize Neo4j provider with custom settings
    provider = await fl.provider_registry.get(
        ProviderType.GRAPH_DB, 
        "neo4j", 
        settings={
            "uri": "bolt://localhost:7687",
            "username": "neo4j", 
            "password": "password"
        }
    )
    
    # Create an entity
    entity = Entity(
        id="person-1",
        type="person",
        attributes={
            "name": EntityAttribute(
                name="name",
                value="John Doe",
                confidence=0.9,
                source="user"
            ),
            "age": EntityAttribute(
                name="age",
                value="30",
                confidence=0.8,
                source="user"
            )
        },
        relationships=[],
        source="user",
        importance=0.7
    )
    
    # Add entity to graph
    await provider.add_entity(entity)
    
    # Add a relationship
    await provider.add_relationship(
        "person-1", 
        "location-1", 
        "lives_in", 
        {"confidence": 0.9}
    )
    
    # Query relationships
    relationships = await provider.query_relationships("person-1")
    print(relationships)
    
    # Traverse graph
    entities = await provider.traverse("person-1", max_depth=2)
    for entity in entities:
        print(f"Entity: {entity.id}, Type: {entity.type}")
    
    # Clean up
    await provider.shutdown()

asyncio.run(main())
```

## JanusGraph Usage Example

```python
import asyncio
import flowlib as fl
from flowlib.core.registry.constants import ProviderType

async def main():
    # Initialize JanusGraph provider
    provider = await fl.provider_registry.get(
        ProviderType.GRAPH_DB, 
        "janusgraph", 
        settings={
            "url": "ws://localhost:8182/gremlin",
            "traversal_source": "g"
        }
    )
    
    # Use the provider
    # (See general example above for entity and relationship operations)
    
    # Example of a custom Gremlin query
    results = await provider.query(
        "gremlin", 
        {
            "gremlin_query": "g.V().hasLabel('Entity').count()",
            "params": {}
        }
    )
    print(f"Number of entities: {results[0]['value']}")
    
    # Clean up
    await provider.shutdown()

asyncio.run(main())
```

## Provider Selection

FlowLib will automatically select the appropriate provider based on availability:

1. If you specify a provider by name, it will use that provider
2. If you don't specify a provider, it will use a default:
   - Memory graph provider is always available as a fallback
   - Neo4j provider if neo4j is installed
   - ArangoDB provider if python-arango is installed
   - JanusGraph provider if gremlinpython is installed

## Tips for Working with Graph Providers

- For large-scale applications, use Neo4j, ArangoDB, or JanusGraph instead of the memory provider
- Set appropriate connection parameters (timeout, max connections) for production
- Use batch operations for better performance when adding many entities
- Consider using the optional query capabilities for provider-specific optimizations 

## Provider Features Comparison

| Feature | Memory | Neo4j | ArangoDB | JanusGraph |
|---------|--------|-------|----------|------------|
| Persistence | ❌ | ✅ | ✅ | ✅ |
| Distributed | ❌ | ✅ | ✅ | ✅ |
| Scalability | Low | High | High | Very High |
| Query Language | Custom | Cypher | AQL | Gremlin |
| Transaction Support | ❌ | ✅ | ✅ | ✅ |
| Schema Flexibility | High | Medium | High | Medium |
| Best For | Development | General purpose | Multi-model | Large scale | 