# Data Models for Learning/Teaching Agent System

This document describes the core data models that form the foundation of the Learning/Teaching Agent's knowledge representation, confidence scoring, and versioning systems.

## 1. Knowledge Representation Schema

The knowledge representation in the dual memory system uses complementary models for vector and graph storage to capture both semantic similarity and structured relationships.

### 1.1 Vector Memory Entry

```json
{
  "id": "vm-8e7af912-c3b0-4fe3-b2d2-89b654321098",
  "text": "The mitochondrion is the powerhouse of the cell, responsible for producing ATP through cellular respiration.",
  "embedding": [0.0234, -0.7652, 0.1287, ...],  // High-dimensional vector (typically 768-1536 dimensions)
  "metadata": {
    "source": {
      "type": "document",
      "id": "bio-textbook-ch7",
      "title": "Cell Biology Fundamentals, Chapter 7",
      "url": "https://example.org/textbooks/cell-biology/chapter7",
      "author": "Dr. Jane Smith"
    },
    "created_at": "2025-03-15T08:21:14Z",
    "updated_at": "2025-03-15T08:21:14Z",
    "domain": {
      "primary": "biology",
      "secondary": ["biochemistry", "cellular-biology"]
    },
    "confidence": {
      "score": 0.92,
      "factors": {
        "source_reliability": 0.95,
        "consistency": 0.97,
        "specificity": 0.93,
        "temporal_relevance": 0.98
      }
    },
    "version": {
      "number": 1,
      "history": []
    },
    "chunks": ["original-doc-chunk-54"],
    "tags": ["mitochondria", "cell-organelle", "atp-production"]
  }
}
```

### 1.2 Graph Memory Models

#### 1.2.1 Entity

```json
{
  "id": "ent-7db45c2a-9e31-4b7f-8a56-123456789abc",
  "type": "Organelle",
  "name": "Mitochondrion",
  "properties": {
    "description": "Membrane-bound organelle that generates most of the cell's ATP",
    "size_range": "0.5-10 μm",
    "membrane_structure": "Double membrane",
    "has_dna": true,
    "count_per_cell": "100-2000"
  },
  "metadata": {
    "source": {
      "type": "document",
      "id": "bio-textbook-ch7",
      "title": "Cell Biology Fundamentals, Chapter 7"
    },
    "created_at": "2025-03-15T08:24:37Z",
    "updated_at": "2025-03-15T08:24:37Z",
    "domain": {
      "primary": "biology",
      "secondary": ["cellular-biology"]
    },
    "confidence": {
      "score": 0.94,
      "factors": {
        "source_reliability": 0.95,
        "consistency": 0.97,
        "specificity": 0.95,
        "temporal_relevance": 0.98
      }
    },
    "version": {
      "number": 1,
      "history": []
    }
  }
}
```

#### 1.2.2 Relationship

```json
{
  "id": "rel-3f9a8d7e-6b5c-4312-a09f-fedcba987654",
  "type": "located_in",
  "source_entity": "ent-7db45c2a-9e31-4b7f-8a56-123456789abc",  // Mitochondrion
  "target_entity": "ent-2c3d4e5f-6a7b-8c9d-0e1f-23456789abcd",  // Eukaryotic Cell
  "properties": {
    "cardinality": "many",
    "essential": true,
    "distribution": "cytoplasm"
  },
  "metadata": {
    "source": {
      "type": "document",
      "id": "bio-textbook-ch7",
      "title": "Cell Biology Fundamentals, Chapter 7"
    },
    "created_at": "2025-03-15T08:27:19Z",
    "updated_at": "2025-03-15T08:27:19Z",
    "domain": {
      "primary": "biology",
      "secondary": ["cellular-biology"]
    },
    "confidence": {
      "score": 0.96,
      "factors": {
        "source_reliability": 0.95,
        "consistency": 0.98, 
        "specificity": 0.95,
        "temporal_relevance": 0.98
      }
    },
    "version": {
      "number": 1,
      "history": []
    }
  }
}
```

### 1.3 Knowledge Domain Taxonomy

The system organizes knowledge using a flexible domain taxonomy:

```json
{
  "domains": [
    {
      "name": "biology",
      "description": "The study of living organisms",
      "subdomains": [
        {
          "name": "cellular-biology",
          "description": "The study of cell structure and function",
          "related_domains": ["biochemistry", "molecular-biology"]
        },
        {
          "name": "genetics",
          "description": "The study of genes and heredity"
        },
        // Additional subdomains...
      ]
    },
    {
      "name": "physics",
      "description": "The study of matter, energy, and the interactions between them",
      "subdomains": [
        // Physics subdomains...
      ]
    }
    // Additional domains...
  ]
}
```

### 1.4 Source Reference Model

```json
{
  "id": "src-1a2b3c4d-5e6f-7a8b-9c0d-ef1234567890",
  "type": "document",
  "title": "Cell Biology Fundamentals",
  "authors": [
    {
      "name": "Dr. Jane Smith",
      "affiliation": "University of Science",
      "expertise": ["cellular-biology", "biochemistry"]
    }
  ],
  "publication_info": {
    "publisher": "Academic Press",
    "year": 2024,
    "edition": "3rd",
    "isbn": "978-0-12345-678-9"
  },
  "reliability_score": 0.95,
  "peer_reviewed": true,
  "citation_count": 87,
  "domains": ["biology", "cellular-biology", "biochemistry"],
  "content_summary": "Comprehensive textbook covering cellular structures, functions, and processes",
  "access_info": {
    "url": "https://example.org/textbooks/cell-biology",
    "last_accessed": "2025-03-14T11:32:45Z",
    "access_level": "full-text"
  }
}
```

## 2. Confidence Scoring Model

The confidence scoring system uses a multi-factor approach to assess the reliability of knowledge.

### 2.1 Confidence Score Component

```json
{
  "score": 0.87,
  "calculation_timestamp": "2025-03-15T14:23:17Z",
  "reasoning": "High confidence due to peer-reviewed source and multiple confirmations from independent sources",
  "factors": {
    "source_reliability": {
      "score": 0.92,
      "reasoning": "Peer-reviewed textbook by recognized expert"
    },
    "consistency": {
      "score": 0.89,
      "reasoning": "Consistent with 7 related knowledge items, minor variation in details"
    },
    "specificity": {
      "score": 0.85,
      "reasoning": "Contains specific details but some quantitative information is missing"
    },
    "temporal_relevance": {
      "score": 0.95,
      "reasoning": "Recent publication (2024) in a slowly changing field"
    }
  },
  "confirmations": [
    {
      "source_id": "src-abcdef123456",
      "confirmation_type": "exact",
      "timestamp": "2025-03-10T09:17:44Z"
    },
    {
      "source_id": "src-98765fedcba",
      "confirmation_type": "partial",
      "variation": "slight terminology differences",
      "timestamp": "2025-03-12T16:42:31Z"
    }
  ],
  "contradictions": [
    {
      "source_id": "src-1a2b3c4d5e6f",
      "contradiction_severity": "minor",
      "contradiction_details": "Claims different range for typical count per cell",
      "timestamp": "2025-03-13T11:05:27Z"
    }
  ],
  "history": [
    {
      "previous_score": 0.83,
      "timestamp": "2025-03-10T09:17:44Z",
      "change_reason": "Additional confirmation found"
    },
    {
      "previous_score": 0.85,
      "timestamp": "2025-03-12T16:42:31Z",
      "change_reason": "Additional partial confirmation"
    },
    {
      "previous_score": 0.85,
      "timestamp": "2025-03-13T11:05:27Z",
      "change_reason": "Minor contradiction found"
    }
  ]
}
```

### 2.2 Source Reliability Matrix

The system uses a standardized matrix to assign initial reliability scores to different source types:

| Source Type | Reliability Range | Factors Affecting Score |
|-------------|------------------|-------------------------|
| Peer-reviewed research | 0.85-0.97 | Journal impact factor, citation count, replicability |
| Textbooks | 0.80-0.95 | Author expertise, publisher reputation, edition recency |
| Government publications | 0.75-0.90 | Agency reputation, political independence, methodology |
| News sources | 0.40-0.85 | Editorial standards, fact-checking process, bias history |
| Industry reports | 0.50-0.80 | Research methodology, transparency, vested interests |
| Expert statements | 0.70-0.90 | Expertise relevance, recognition in field, context |
| Wikipedia | 0.60-0.85 | Citation quality, edit history, topic controversy |
| Personal blogs | 0.20-0.70 | Author credentials, citation practice, reasoning quality |
| Social media | 0.10-0.50 | Account verification, expertise, supporting evidence |
| Inferences by agent | 0.30-0.75 | Reasoning steps, evidence quality, assumption count |

### 2.3 Confidence Threshold Policies

The system employs configurable threshold policies for different operations:

```json
{
  "confidence_thresholds": {
    "knowledge_acceptance": 0.65,
    "teaching_default": 0.75,
    "critical_domain_teaching": 0.85,
    "contradiction_significance": 0.70,
    "automatic_update": 0.80,
    "flagging_for_review": 0.60
  },
  "domain_specific_overrides": [
    {
      "domain": "medicine",
      "knowledge_acceptance": 0.75,
      "teaching_default": 0.85
    },
    {
      "domain": "current_events",
      "temporal_decay_rate": "accelerated",
      "knowledge_acceptance": 0.70
    }
  ],
  "uncertainty_communication": {
    "below_threshold": "explicit disclaimer",
    "threshold_to_high_confidence": "qualified statement",
    "high_confidence": "direct statement"
  }
}
```

## 3. Versioning Model

The versioning system tracks how knowledge evolves over time.

### 3.1 Version Entry

```json
{
  "version_id": "ver-1a2b3c4d-5e6f-7a8b-9c0d-ef1234567890",
  "entity_id": "ent-7db45c2a-9e31-4b7f-8a56-123456789abc",
  "version_number": 3,
  "timestamp": "2025-03-17T14:23:17Z",
  "author": "learning-session-20250317",
  "previous_version_id": "ver-2b3c4d5e-6f7a-8b9c-0d1e-f23456789012",
  "change_type": "update",
  "change_summary": "Updated size range and added details about DNA content",
  "change_reason": "New information from recent research paper",
  "change_source": {
    "source_id": "src-5e6f7a8b9c0d",
    "source_type": "research-paper",
    "confidence": 0.89
  },
  "diff": {
    "properties": {
      "added": {
        "dna_gene_count": "37 genes"
      },
      "removed": {},
      "modified": {
        "size_range": {
          "from": "0.5-10 μm",
          "to": "0.75-8 μm"
        }
      }
    },
    "metadata": {
      "modified": {
        "confidence.score": {
          "from": 0.87,
          "to": 0.91
        }
      }
    }
  },
  "verification_status": "verified",
  "verified_by": "manual-review-user123",
  "verification_timestamp": "2025-03-17T16:45:22Z"
}
```

### 3.2 Version Chain

```json
{
  "entity_id": "ent-7db45c2a-9e31-4b7f-8a56-123456789abc",
  "entity_type": "Organelle",
  "entity_name": "Mitochondrion",
  "current_version": 3,
  "first_created": "2025-03-15T08:24:37Z",
  "last_updated": "2025-03-17T14:23:17Z",
  "version_history": [
    {
      "version_number": 1,
      "version_id": "ver-9c0d1e2f-3a4b-5c6d-7e8f-012345678901",
      "timestamp": "2025-03-15T08:24:37Z",
      "change_type": "create",
      "change_summary": "Initial creation"
    },
    {
      "version_number": 2,
      "version_id": "ver-2b3c4d5e-6f7a-8b9c-0d1e-f23456789012",
      "timestamp": "2025-03-16T10:42:19Z",
      "change_type": "update",
      "change_summary": "Added membrane structure information"
    },
    {
      "version_number": 3,
      "version_id": "ver-1a2b3c4d-5e6f-7a8b-9c0d-ef1234567890",
      "timestamp": "2025-03-17T14:23:17Z",
      "change_type": "update",
      "change_summary": "Updated size range and added DNA content details"
    }
  ],
  "branch_points": [],
  "merge_events": []
}
```

### 3.3 Knowledge Snapshot

```json
{
  "snapshot_id": "snap-1234567890abcdef",
  "name": "Biology Domain 2025-Q1",
  "description": "Complete snapshot of biology domain knowledge as of Q1 2025",
  "timestamp": "2025-03-31T23:59:59Z",
  "domains": ["biology", "biochemistry", "cellular-biology", "genetics"],
  "statistics": {
    "entity_count": 14289,
    "relationship_count": 27456,
    "vector_entry_count": 35672,
    "total_size_bytes": 178459023
  },
  "average_confidence": 0.84,
  "version_info": {
    "agent_version": "3.5.2",
    "memory_system_version": "2.1.0"
  },
  "created_by": "system-maintenance",
  "retention_policy": "permanent",
  "storage_location": "snapshots/biology/2025-Q1/",
  "verification_status": "complete",
  "restoration_instructions": {
    "command": "restore_snapshot --id snap-1234567890abcdef --target latest",
    "estimated_time": "10-15 minutes"
  }
}
```

## 4. Memory Query Models

These models define how the system represents queries to the memory system.

### 4.1 Vector Search Query

```json
{
  "query_id": "q-9876543210abcdef",
  "query_text": "What is the role of mitochondria in cellular energy production?",
  "embedding": [0.0453, -0.6523, 0.2341, ...],
  "search_parameters": {
    "similarity_metric": "cosine",
    "top_k": 10,
    "similarity_threshold": 0.75
  },
  "filters": {
    "domains": ["biology", "cellular-biology"],
    "confidence_min": 0.8,
    "created_after": "2024-01-01T00:00:00Z"
  },
  "context": {
    "conversation_id": "conv-12345abcde",
    "previous_queries": ["q-abcdef1234567890"],
    "user_expertise": "beginner"
  }
}
```

### 4.2 Graph Query

```json
{
  "query_id": "q-fedcba0987654321",
  "query_type": "relationship_traversal",
  "start_entity": {
    "type": "Organelle",
    "name": "Mitochondrion"
  },
  "relationship_path": [
    {
      "type": "produces",
      "direction": "outgoing"
    }
  ],
  "end_entity_filter": {
    "type": "Molecule"
  },
  "max_distance": 2,
  "traversal_strategy": "breadth_first",
  "property_filters": [
    {
      "entity_position": "end",
      "property_name": "molecular_weight",
      "operator": "lt",
      "value": 1000
    }
  ],
  "return_properties": ["name", "description", "chemical_formula"],
  "limit": 5,
  "confidence_threshold": 0.75
}
```

### 4.3 Hybrid Query

```json
{
  "query_id": "q-hybrid5678901234",
  "description": "Find information about how mitochondria produce ATP",
  "vector_component": {
    "query_text": "mitochondria ATP production process",
    "top_k": 5,
    "domains": ["biology", "biochemistry"]
  },
  "graph_component": {
    "start_entity_type": "Organelle",
    "start_entity_name": "Mitochondrion",
    "relationships": [
      {
        "type": "produces",
        "target_type": "Molecule",
        "target_properties": {
          "name": "ATP"
        }
      },
      {
        "type": "involves",
        "target_type": "Process"
      }
    ]
  },
  "integration_strategy": "complement",
  "confidence_threshold": 0.8,
  "result_limit": 10,
  "include_explanations": true
}
```

## 5. Memory Response Models

These models define how the system represents responses from memory queries.

### 5.1 Vector Search Results

```json
{
  "query_id": "q-9876543210abcdef",
  "timestamp": "2025-03-19T15:23:47Z",
  "execution_time_ms": 42,
  "results": [
    {
      "id": "vm-8e7af912-c3b0-4fe3-b2d2-89b654321098",
      "text": "The mitochondrion is the powerhouse of the cell, responsible for producing ATP through cellular respiration.",
      "similarity_score": 0.92,
      "confidence": 0.89,
      "metadata": {
        "source": {
          "type": "document",
          "id": "bio-textbook-ch7",
          "title": "Cell Biology Fundamentals, Chapter 7"
        },
        "domain": {
          "primary": "biology",
          "secondary": ["biochemistry", "cellular-biology"]
        }
      }
    },
    // Additional results...
  ],
  "metadata": {
    "total_candidates": 47,
    "filtered_out": 37,
    "filter_reasons": {
      "confidence_too_low": 12,
      "domain_mismatch": 25
    }
  }
}
```

### 5.2 Graph Query Results

```json
{
  "query_id": "q-fedcba0987654321",
  "timestamp": "2025-03-19T15:24:12Z",
  "execution_time_ms": 78,
  "results": [
    {
      "path": [
        {
          "id": "ent-7db45c2a-9e31-4b7f-8a56-123456789abc",
          "type": "Organelle",
          "name": "Mitochondrion",
          "properties": {
            "description": "Membrane-bound organelle that generates most of the cell's ATP"
          }
        },
        {
          "id": "rel-3f9a8d7e-6b5c-4312-a09f-abcdef123456",
          "type": "produces",
          "properties": {
            "mechanism": "oxidative phosphorylation",
            "efficiency": "high"
          }
        },
        {
          "id": "ent-123456789abc-def0-1234-5678-9abcdef01234",
          "type": "Molecule",
          "name": "ATP",
          "properties": {
            "description": "Adenosine triphosphate, the energy currency of the cell",
            "chemical_formula": "C10H16N5O13P3",
            "molecular_weight": 507.18
          }
        }
      ],
      "confidence": 0.94,
      "metadata": {
        "path_length": 2,
        "sources": [
          {
            "entity_id": "ent-7db45c2a-9e31-4b7f-8a56-123456789abc",
            "source_id": "bio-textbook-ch7"
          },
          {
            "entity_id": "ent-123456789abc-def0-1234-5678-9abcdef01234",
            "source_id": "bio-textbook-ch7"
          },
          {
            "relationship_id": "rel-3f9a8d7e-6b5c-4312-a09f-abcdef123456",
            "source_id": "bio-textbook-ch7"
          }
        ]
      }
    },
    // Additional results...
  ],
  "metadata": {
    "paths_explored": 23,
    "paths_returned": 3,
    "max_path_length": 2
  }
}
```

### 5.3 Knowledge Retrieval Result

This unified response model combines results from both vector and graph stores:

```json
{
  "query_id": "q-hybrid5678901234",
  "timestamp": "2025-03-19T15:25:03Z",
  "total_execution_time_ms": 156,
  "vector_results": {
    "execution_time_ms": 45,
    "result_count": 3,
    "results": [
      // Vector results (abbreviated)
    ]
  },
  "graph_results": {
    "execution_time_ms": 82,
    "result_count": 2,
    "results": [
      // Graph results (abbreviated)
    ]
  },
  "integrated_knowledge": {
    "main_concept": {
      "type": "Process",
      "name": "Oxidative Phosphorylation",
      "description": "The metabolic pathway in which cells use enzymes to oxidize nutrients, thereby releasing energy which is used to produce ATP.",
      "confidence": 0.95,
      "sources": ["bio-textbook-ch7", "biochem-journal-2024-03"]
    },
    "related_entities": [
      {
        "type": "Organelle",
        "name": "Mitochondrion",
        "relationship": "location_of",
        "confidence": 0.97
      },
      {
        "type": "Molecule", 
        "name": "ATP",
        "relationship": "product_of",
        "confidence": 0.96
      },
      {
        "type": "Process",
        "name": "Electron Transport Chain",
        "relationship": "part_of",
        "confidence": 0.93
      }
    ],
    "key_facts": [
      {
        "fact": "Oxidative phosphorylation produces 26-28 ATP molecules per glucose molecule",
        "confidence": 0.91,
        "source": "bio-textbook-ch7"
      },
      {
        "fact": "The process occurs in the inner mitochondrial membrane",
        "confidence": 0.94,
        "source": "biochem-journal-2024-03"
      }
    ]
  },
  "confidence_summary": {
    "overall_confidence": 0.93,
    "confidence_distribution": {
      "high": 7,
      "medium": 2,
      "low": 0
    },
    "contradictions_found": false
  }
}
```

## 6. Learning Process Models

These models represent how the system tracks its learning activities.

### 6.1 Learning Session

```json
{
  "session_id": "learn-abcdef123456",
  "timestamp_start": "2025-03-20T09:00:00Z",
  "timestamp_end": "2025-03-20T10:15:37Z",
  "domain": "biology",
  "subdomains": ["cellular-biology", "biochemistry"],
  "learning_objectives": [
    "Understand mitochondrial structure and function",
    "Map ATP production processes",
    "Identify relationships between organelles"
  ],
  "information_sources": [
    {
      "source_id": "src-bio-textbook-ch7",
      "source_type": "textbook",
      "exploration_depth": "comprehensive"
    },
    {
      "source_id": "src-biochem-journal-2024-03",
      "source_type": "journal-article",
      "exploration_depth": "selective"
    }
  ],
  "questions_generated": 37,
  "questions_answered": 35,
  "information_extracted": {
    "entities_created": 18,
    "entities_updated": 7,
    "relationships_created": 32,
    "relationships_updated": 5,
    "vector_entries_created": 43
  },
  "knowledge_integration": {
    "new_confidence_range": [0.72, 0.96],
    "average_confidence": 0.87,
    "conflicts_identified": 3,
    "conflicts_resolved": 2,
    "pending_review": 1
  },
  "learning_progress": {
    "objective_completion": 0.92,
    "knowledge_gaps_identified": [
      {
        "topic": "Mitochondrial DNA inheritance patterns",
        "importance": "medium",
        "recommended_sources": ["genetics-textbook-ch12"]
      }
    ],
    "knowledge_graph_metrics": {
      "connection_density": 0.76,
      "orphaned_entities": 0
    }
  },
  "summary": "Productive learning session focused on mitochondrial functions. Comprehensive understanding of ATP production achieved. Some knowledge gaps remain around inheritance patterns."
}
```

### 6.2 Question Generation

```json
{
  "question_id": "qgen-1234567890",
  "session_id": "learn-abcdef123456",
  "timestamp": "2025-03-20T09:12:47Z",
  "domain": "biology",
  "subdomain": "cellular-biology",
  "focus_entity": "Mitochondrion",
  "question_type": "relationship",
  "question_text": "What processes occur in the mitochondrial matrix?",
  "expected_knowledge_type": "process_list",
  "knowledge_gap_assessment": {
    "current_knowledge": {
      "completeness": 0.45,
      "confidence": 0.65
    },
    "importance": 0.85
  },
  "follow_up_questions": [
    "How does the citric acid cycle integrate with oxidative phosphorylation?",
    "What enzymes are involved in the citric acid cycle?"
  ],
  "sources_to_consult": ["bio-textbook-ch7", "biochem-journal-2024-03"]
}
```

### 6.3 Information Extraction Record

```json
{
  "extraction_id": "ext-a1b2c3d4e5f6",
  "session_id": "learn-abcdef123456",
  "timestamp": "2025-03-20T09:18:22Z",
  "source": {
    "source_id": "src-bio-textbook-ch7",
    "section": "7.4 - Mitochondrial Functions"
  },
  "question_id": "qgen-1234567890",
  "raw_text": "The mitochondrial matrix contains enzymes for the citric acid cycle, which is the second stage of cellular respiration after glycolysis. This cycle, also known as the Krebs cycle or TCA cycle, breaks down acetyl-CoA to produce NADH and FADH2, which carry electrons to the electron transport chain embedded in the inner mitochondrial membrane.",
  "extracted_entities": [
    {
      "type": "Process",
      "name": "Citric Acid Cycle",
      "properties": {
        "alternative_names": ["Krebs Cycle", "TCA Cycle"],
        "stage_number": 2,
        "location": "Mitochondrial Matrix",
        "precursor_process": "Glycolysis",
        "output_molecules": ["NADH", "FADH2"]
      },
      "confidence": 0.92
    },
    {
      "type": "Process",
      "name": "Electron Transport Chain",
      "properties": {
        "location": "Inner Mitochondrial Membrane"
      },
      "confidence": 0.88
    },
    {
      "type": "Molecule",
      "name": "Acetyl-CoA",
      "properties": {
        "role": "Substrate"
      },
      "confidence": 0.90
    }
  ],
  "extracted_relationships": [
    {
      "type": "occurs_in",
      "source_entity": "Citric Acid Cycle",
      "target_entity": "Mitochondrial Matrix",
      "properties": {
        "exclusivity": "exclusive"
      },
      "confidence": 0.94
    },
    {
      "type": "produces",
      "source_entity": "Citric Acid Cycle",
      "target_entity": "NADH",
      "confidence": 0.91
    },
    {
      "type": "produces",
      "source_entity": "Citric Acid Cycle",
      "target_entity": "FADH2",
      "confidence": 0.91
    },
    {
      "type": "inputs_to",
      "source_entity": "NADH",
      "target_entity": "Electron Transport Chain",
      "confidence": 0.87
    },
    {
      "type": "inputs_to",
      "source_entity": "FADH2",
      "target_entity": "Electron Transport Chain",
      "confidence": 0.87
    }
  ],
  "extraction_metrics": {
    "entity_count": 3,
    "relationship_count": 5,
    "confidence_range": [0.87, 0.94],
    "average_confidence": 0.90,
    "execution_time_ms": 287
  },
  "knowledge_integration": {
    "new_entities": 1,
    "updated_entities": 2,
    "new_relationships": 3,
    "updated_relationships": 2,
    "conflicts_detected": 0
  }
}
```

## 7. Teaching Process Models

These models represent how the system serves knowledge in teaching mode.

### 7.1 Query Understanding Record

```json
{
  "query_id": "qu-e1f2g3h4i5j6",
  "timestamp": "2025-03-21T14:32:17Z",
  "raw_query": "How do mitochondria produce energy for the cell?",
  "query_analysis": {
    "intent": {
      "primary": "explanation",
      "secondary": "process_description"
    },
    "entities": [
      {
        "name": "mitochondria",
        "type": "Organelle",
        "confidence": 0.98,
        "entity_id": "ent-7db45c2a-9e31-4b7f-8a56-123456789abc"
      },
      {
        "name": "energy",
        "type": "Concept",
        "confidence": 0.85,
        "resolved_to": "ATP"
      },
      {
        "name": "cell",
        "type": "Structure",
        "confidence": 0.97,
        "entity_id": "ent-2c3d4e5f-6a7b-8c9d-0e1f-23456789abcd"
      }
    ],
    "relationships": [
      {
        "type": "produces",
        "source": "mitochondria",
        "target": "energy",
        "confidence": 0.92
      }
    ],
    "domain": {
      "detected": "biology",
      "subdomain": "cellular-biology",
      "confidence": 0.96
    }
  },
  "knowledge_requirements": [
    {
      "entity_type": "Organelle",
      "entity_name": "Mitochondrion",
      "required_properties": ["function", "structure"],
      "priority": "high"
    },
    {
      "entity_type": "Process",
      "related_to": "energy production",
      "related_entity": "Mitochondrion",
      "required_properties": ["steps", "outputs"],
      "priority": "high"
    },
    {
      "entity_type": "Molecule",
      "entity_name": "ATP",
      "required_properties": ["function"],
      "priority": "medium"
    }
  ],
  "query_rewrite": {
    "vector_query": "mitochondria cellular respiration ATP production process",
    "graph_queries": [
      {
        "start_entity": "Mitochondrion",
        "relationship": "participates_in",
        "target_entity_type": "Process",
        "properties": {
          "related_to": "energy production"
        }
      },
      {
        "start_entity": "Mitochondrion",
        "relationship": "produces",
        "target_entity_type": "Molecule",
        "properties": {
          "name": "ATP"
        }
      }
    ]
  },
  "user_context": {
    "expertise_level": "beginner",
    "previous_queries": ["what are organelles"],
    "personalization_hints": ["prefer simple explanations", "use analogies"]
  }
}
```

### 7.2 Answer Generation Record

```json
{
  "answer_id": "ans-k7l8m9n0p1q2",
  "query_id": "qu-e1f2g3h4i5j6",
  "timestamp": "2025-03-21T14:32:22Z",
  "retrieved_knowledge": {
    "vector_results": [
      {
        "id": "vm-8e7af912-c3b0-4fe3-b2d2-89b654321098",
        "text": "The mitochondrion is the powerhouse of the cell, responsible for producing ATP through cellular respiration.",
        "confidence": 0.92,
        "similarity": 0.94
      },
      // Additional vector results...
    ],
    "graph_results": [
      {
        "central_entity": {
          "type": "Process",
          "name": "Oxidative Phosphorylation",
          "properties": {
            "description": "The metabolic pathway in which cells use enzymes to oxidize nutrients, thereby releasing energy which is used to produce ATP."
          },
          "confidence": 0.95
        },
        "related_entities": [
          // Related entities...
        ],
        "relationships": [
          // Relationships...
        ]
      }
    ],
    "knowledge_coverage": {
      "requirements_met": 0.92,
      "gaps": [
        {
          "topic": "Efficiency comparison with anaerobic processes",
          "importance": "low"
        }
      ]
    }
  },
  "answer_planning": {
    "structure": [
      {
        "section": "Introduction",
        "key_points": ["Mitochondria as energy producers", "ATP as energy currency"]
      },
      {
        "section": "Cellular Respiration Process",
        "key_points": ["Overview of stages", "Location within mitochondria"]
      },
      {
        "section": "ATP Production Mechanism",
        "key_points": ["Electron transport chain", "Proton gradient", "ATP synthase"]
      },
      {
        "section": "Importance and Efficiency",
        "key_points": ["Amount of ATP produced", "Comparison to other processes"]
      }
    ],
    "explanation_level": "beginner",
    "analogies": [
      {
        "concept": "Proton gradient",
        "analogy": "Like water behind a dam that can be used to generate electricity",
        "confidence": 0.85
      }
    ]
  },
  "generated_answer": {
    "text": "Mitochondria are often called the powerhouse of the cell because they produce most of the cell's energy in the form of ATP (adenosine triphosphate), which is like a rechargeable battery for cellular processes.\n\nThe energy production happens through a process called cellular respiration. This works similar to how we use oxygen to break down food for energy, but at a cellular level. This process has several stages:\n\n1. First, glucose (sugar) is partially broken down in the cell's cytoplasm through glycolysis.\n\n2. Then, inside the mitochondria, the breakdown products enter the citric acid cycle (also called the Krebs cycle), which takes place in the inner fluid portion called the matrix.\n\n3. The most important energy-producing step happens at the inner membrane of mitochondria. Here, an electron transport chain creates a concentration gradient of protons (hydrogen ions). Think of this like water building up behind a dam.\n\n4. Finally, as these protons flow back through a special enzyme called ATP synthase (like water flowing through a turbine in a dam), their energy is used to produce ATP molecules.\n\nThrough this entire process, a single glucose molecule can yield about 30-32 ATP molecules, which is much more efficient than the 2 ATP produced without mitochondria.\n\nThe mitochondria's unique structure with its folded inner membrane (cristae) increases the surface area where this energy production takes place, making it an efficient energy factory for the cell.",
    "confidence": 0.90,
    "sources": [
      {
        "id": "src-bio-textbook-ch7",
        "citation": "Smith, J. (2024). Cell Biology Fundamentals (3rd ed.). Academic Press.",
        "sections_used": ["7.4 - Mitochondrial Functions", "7.5 - Cellular Respiration"]
      }
    ],
    "uncertainty_indicators": [
      {
        "statement": "a single glucose molecule can yield about 30-32 ATP molecules",
        "type": "range",
        "reason": "Slightly different counts in different sources",
        "confidence": 0.87
      }
    ]
  },
  "metadata": {
    "execution_time_ms": {
      "knowledge_retrieval": 112,
      "answer_planning": 78,
      "answer_generation": 256,
      "total": 446
    },
    "token_usage": {
      "prompt": 1842,
      "completion": 387,
      "total": 2229
    }
  }
}
```

### 7.3 Explanation Building Record

```json
{
  "explanation_id": "exp-r3s4t5u6v7w8",
  "answer_id": "ans-k7l8m9n0p1q2",
  "timestamp": "2025-03-21T14:32:28Z",
  "explanation_request": {
    "focus_concept": "electron transport chain",
    "depth_level": "intermediate",
    "user_background": "high school biology student"
  },
  "knowledge_retrieval": {
    "focused_vector_query": "electron transport chain mitochondria mechanism",
    "focused_graph_query": {
      "start_entity": "Process",
      "name": "Electron Transport Chain",
      "traversal_depth": 2
    },
    "key_results": [
      // Retrieved knowledge details...
    ]
  },
  "conceptual_hierarchy": {
    "core_concept": "Electron Transport Chain",
    "prerequisite_concepts": [
      "Redox Reactions",
      "Membrane Proteins",
      "Proton Gradient"
    ],
    "dependent_concepts": [
      "Oxidative Phosphorylation",
      "ATP Synthesis"
    ],
    "learning_sequence": [
      "Basic electron carriers",
      "Protein complexes in the membrane",
      "Electron flow and energy release",
      "Proton pumping mechanism",
      "Coupling to ATP synthesis"
    ]
  },
  "analogy_development": {
    "primary_analogy": {
      "concept": "Electron Transport Chain",
      "analogy": "The electron transport chain works like a series of waterfalls. Just as water releases energy when flowing down multiple small waterfalls (rather than one big one), electrons release energy in small, manageable steps as they move through the chain of protein complexes.",
      "mapping": [
        {
          "concept_element": "Electrons",
          "analogy_element": "Water"
        },
        {
          "concept_element": "Protein complexes",
          "analogy_element": "Waterfalls"
        },
        {
          "concept_element": "Energy release",
          "analogy_element": "Falling water energy"
        },
        {
          "concept_element": "Proton pumping",
          "analogy_element": "Water wheels powered by falls"
        }
      ],
      "limitations": "Doesn't capture the cycling of electron carriers or the chemical nature of the process"
    },
    "secondary_analogies": [
      // Additional analogies...
    ]
  },
  "visual_aids": [
    {
      "type": "conceptual_diagram",
      "description": "Simple diagram showing the four main protein complexes in the inner mitochondrial membrane, with electrons flowing through and protons being pumped across the membrane",
      "key_labeled_elements": ["Complex I", "Complex II", "Complex III", "Complex IV", "Electron flow", "Proton pumping"]
    }
  ],
  "generated_explanation": {
    "text": "The electron transport chain (ETC) is like a series of small waterfalls that electrons flow down, releasing energy along the way.\n\nHere's how it works in more detail:\n\n1. Location: The ETC is embedded in the inner mitochondrial membrane, which separates the mitochondrial matrix from the intermembrane space.\n\n2. Components: The chain consists of four main protein complexes (I, II, III, and IV), along with two mobile electron carriers (ubiquinone and cytochrome c).\n\n3. The Process:\n   - The process begins when electron carriers NADH and FADH2 (made during earlier stages of cellular respiration) deliver high-energy electrons to the ETC.\n   - NADH delivers electrons to Complex I, while FADH2 delivers them to Complex II.\n   - These electrons then flow through the remaining complexes, losing energy at each step.\n   - The energy released at Complexes I, III, and IV is used to pump protons (H+) from the matrix into the intermembrane space.\n   - At Complex IV, the electrons are finally transferred to oxygen (which acts as the final electron acceptor), combining with protons to form water.\n\n4. The Result:\n   - The movement of protons creates a concentration gradient across the inner membrane (higher concentration in the intermembrane space).\n   - This proton gradient represents stored energy, similar to water behind a dam.\n   - Protons can only flow back into the matrix through ATP synthase, which uses their flow to generate ATP.\n\nThink of each protein complex as a waterfall of different height. As electrons \"fall\" through each complex, they release energy. Instead of this energy being lost as heat, the mitochondria capture it by pumping protons across the membrane, creating a kind of \"proton pressure\" that can later be used to make ATP.\n\nThe beauty of this system is that it breaks down the energy release into manageable steps, allowing the mitochondria to capture more of the energy in a usable form rather than losing it all at once as heat.",
    "confidence": 0.92,
    "complexity_metrics": {
      "readability_score": 65.3,
      "technical_terms_explained": 9,
      "sentence_complexity": "moderate"
    }
  },
  "metadata": {
    "execution_time_ms": 312,
    "knowledge_elements_used": 14,
    "sources_consulted": 3
  }
}
```

## 8. Agent State Models

These models represent the agent's internal state during operation.

### 8.1 Agent Configuration

```json
{
  "agent_id": "lta-3a4b5c6d7e8f",
  "name": "BioTeach",
  "version": "1.2.1",
  "description": "Learning/Teaching Agent specialized in biological sciences",
  "operation_modes": ["LEARNING", "TEACHING"],
  "default_mode": "TEACHING",
  "memory_configuration": {
    "vector_provider": {
      "provider_type": "chroma",
      "collection_name": "bio_knowledge",
      "embedding_dimensions": 1536,
      "similarity_metric": "cosine"
    },
    "graph_provider": {
      "provider_type": "neo4j",
      "database_name": "bio_graph",
      "relationship_types": ["standard", "domain_specific"],
      "index_configuration": {
        "node_labels": ["Entity", "Process", "Molecule", "Structure"],
        "relationship_types": ["produces", "participates_in", "located_in"]
      }
    },
    "confidence_thresholds": {
      "knowledge_acceptance": 0.65,
      "teaching_default": 0.75,
      "critical_domain_teaching": 0.85
    },
    "versioning_enabled": true,
    "versioning_policy": {
      "major_change_threshold": 0.5,
      "minor_change_threshold": 0.2,
      "retention_period_days": 365
    }
  },
  "llm_configuration": {
    "teaching_model": {
      "provider": "anthropic",
      "model_name": "claude-3-5-sonnet",
      "temperature": 0.3,
      "max_tokens": 4000,
      "system_prompt_template": "You are a biology tutor helping students understand {topic}. Explain concepts clearly at a {level} level."
    },
    "learning_model": {
      "provider": "anthropic",
      "model_name": "claude-3-5-opus",
      "temperature": 0.1,
      "max_tokens": 4000,
      "system_prompt_template": "You are a biology researcher extracting precise information about {topic}. Be comprehensive and specific."
    }
  },
  "learning_configuration": {
    "question_generation": {
      "breadth_vs_depth": 0.6,
      "max_questions_per_session": 50,
      "follow_up_depth": 3
    },
    "information_extraction": {
      "granularity": "fine",
      "relationship_focus": "high",
      "attribute_detail_level": "comprehensive"
    },
    "source_handling": {
      "source_prioritization": ["textbook", "journal", "reference", "web"],
      "source_recency_weight": 0.7
    }
  },
  "teaching_configuration": {
    "explanation_levels": ["beginner", "intermediate", "advanced", "expert"],
    "default_explanation_level": "intermediate",
    "response_style": {
      "formality": 0.6,
      "technical_precision": 0.7,
      "use_of_analogies": 0.8
    },
    "uncertainty_communication": {
      "mode": "explicit",
      "threshold_for_disclaimer": 0.7
    },
    "citation_policy": {
      "include_citations": true,
      "citation_format": "APA",
      "citation_threshold": 0.5
    }
  }
}
```

### 8.2 Agent Mode State

```json
{
  "agent_id": "lta-3a4b5c6d7e8f",
  "current_mode": "LEARNING",
  "mode_start_time": "2025-03-20T09:00:00Z",
  "active_session": "learn-abcdef123456",
  "session_configuration": {
    "domain": "biology",
    "subdomains": ["cellular-biology", "biochemistry"],
    "focus_entity": "Mitochondrion",
    "learning_depth": "comprehensive",
    "sources": ["bio-textbook-ch7", "biochem-journal-2024-03"]
  },
  "session_statistics": {
    "questions_generated": 28,
    "questions_answered": 25,
    "entities_added": 14,
    "relationships_added": 23,
    "confidence_range": [0.68, 0.95],
    "execution_time_seconds": 1284
  },
  "active_learning_flow": {
    "flow_name": "domain_exploration",
    "current_stage": "knowledge_gap_identification",
    "progress": 0.65,
    "next_actions": [
      "generate_questions_for_gap_1",
      "generate_questions_for_gap_2"
    ]
  },
  "pause_state": {
    "can_pause": true,
    "resume_point": {
      "description": "After question generation for Krebs cycle enzymes",
      "checkpoint_id": "cp-1234567890"
    }
  }
}
```

### 8.3 Memory System State

```json
{
  "memory_system_id": "mem-a1b2c3d4e5f6",
  "last_updated": "2025-03-20T14:37:22Z",
  "vector_store": {
    "status": "operational",
    "entry_count": 43856,
    "total_size_bytes": 267483921,
    "health": {
      "index_status": "optimized",
      "last_optimization": "2025-03-19T01:15:42Z"
    },
    "domains": [
      {
        "name": "biology",
        "entry_count": 32458,
        "last_updated": "2025-03-20T14:37:22Z"
      },
      {
        "name": "chemistry",
        "entry_count": 11398,
        "last_updated": "2025-03-18T22:14:09Z"
      }
    ],
    "performance_metrics": {
      "average_query_time_ms": 68,
      "p95_query_time_ms": 124,
      "cache_hit_rate": 0.87
    }
  },
  "graph_store": {
    "status": "operational",
    "entity_count": 27493,
    "relationship_count": 58726,
    "entity_types": [
      {
        "type": "Organelle",
        "count": 42
      },
      {
        "type": "Molecule",
        "count": 3871
      },
      {
        "type": "Process",
        "count": 982
      }
      // Additional types...
    ],
    "relationship_types": [
      {
        "type": "located_in",
        "count": 8753
      },
      {
        "type": "produces",
        "count": 4286
      }
      // Additional types...
    ],
    "performance_metrics": {
      "average_query_time_ms": 92,
      "p95_query_time_ms": 175,
      "index_utilization": 0.92
    }
  },
  "version_store": {
    "status": "operational",
    "version_count": 105372,
    "current_versions": 86219,
    "archived_versions": 19153,
    "storage_efficiency": {
      "compression_ratio": 4.3,
      "deduplication_ratio": 2.7
    }
  },
  "health_summary": {
    "overall_status": "healthy",
    "warnings": 0,
    "errors": 0,
    "maintenance_recommendations": [
      {
        "type": "scheduled_optimization",
        "target": "graph_indices",
        "priority": "low",
        "estimated_runtime_minutes": 45
      }
    ]
  },
  "memory_statistics": {
    "knowledge_domain_distribution": {
      "biology": 74,
      "chemistry": 26
    },
    "confidence_distribution": {
      "high": 67,
      "medium": 28,
      "low": 5
    },
    "knowledge_freshness": {
      "updated_last_week": 15,
      "updated_last_month": 35,
      "updated_last_year": 45,
      "older_than_one_year": 5
    }
  }
}
```

This document provides a comprehensive definition of the data models that power the Learning/Teaching Agent System. These models specify the structure, relationships, and metadata for knowledge representation, confidence scoring, and versioning, as well as models for handling learning and teaching processes.
