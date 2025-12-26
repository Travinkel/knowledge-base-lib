# Knowledge Base Foundation - Jules Session

You are implementing the Knowledge Base foundation layer for Project Astartes.

## Work Orders to Complete

### WO-KB-003: Knowledge Graph Storage and Retrieval (HIGH PRIORITY)
**Complexity:** Expert | **Effort:** 16-32 hours

Implement PostgreSQL + pgvector storage for the knowledge graph.

**Feature:** `features/graph_storage.feature` (16 scenarios)

**Acceptance Criteria:**
- [ ] Store and retrieve learning atoms
- [ ] Store concepts with hierarchy
- [ ] Store relationships between entities
- [ ] Query prerequisites (transitive closure)
- [ ] Query dependent concepts
- [ ] All Gherkin scenarios pass
- [ ] Unit test coverage >= 80%

**Deliverable:** `tests/step_defs/knowledge-graph-storage-and-retrieval_steps.py`

---

### WO-KB-004: Greenlight-KB Mastery Sync (LOW COMPLEXITY)
**Complexity:** Low | **Effort:** 2-4 hours

Sync IDE (Greenlight) performance to Knowledge Base.

**Feature:** `features/greenlight_kb_mastery_sync.feature` (3 scenarios)

**Acceptance Criteria:**
- [ ] Successful Code Submission Sync
- [ ] Failure with Misconception Diagnosis
- [ ] Adaptive Pathing Based on Greenlight Metrics

**Deliverable:** `tests/step_defs/greenlight-kb-mastery-sync_steps.py`

---

### WO-KB-006: Vector Search and Semantic Similarity (HIGH PRIORITY)
**Complexity:** Expert | **Effort:** 16-32 hours

Implement vector embeddings for semantic search.

**Feature:** `features/vector_search.feature` (18 scenarios)

**Acceptance Criteria:**
- [ ] Store embeddings for concept nodes
- [ ] Store embeddings for learning atoms
- [ ] Store code-specific embeddings
- [ ] Find k nearest neighbor concepts
- [ ] Expand search with semantic synonyms
- [ ] All Gherkin scenarios pass

**Deliverable:** `tests/step_defs/vector-search-and-semantic-similarity_steps.py`

---

## Existing Code Structure

```
libs/knowledge_base/
├── src/core/
│   ├── atom_store.py       # Learning atom storage
│   ├── vector_store.py     # Vector embedding storage
│   ├── concept_graph.py    # Concept hierarchy
│   ├── prerequisite.py     # Prerequisite chains
│   ├── repository.py       # Data access layer
│   ├── graph_analytics.py  # Graph queries
│   ├── evidence_connector.py
│   └── models.py           # Pydantic models
├── features/
│   ├── graph_storage.feature
│   ├── greenlight_kb_mastery_sync.feature
│   └── vector_search.feature
└── tests/
    ├── step_defs/
    │   └── graph_operations_steps.py
    └── test_*.py
```

## Quick Start

```bash
# Check existing implementation
cat src/core/vector_store.py
cat src/core/concept_graph.py

# Read the feature files
cat features/graph_storage.feature
cat features/vector_search.feature

# Run existing tests
pytest tests/ -v

# Implement step definitions
# Start with WO-KB-003 (graph storage) as foundation
```

## Implementation Order

1. **WO-KB-003 first** - Graph storage is the foundation
2. **WO-KB-006 second** - Vector search builds on storage
3. **WO-KB-004 last** - Greenlight sync uses both

## Tech Stack

- **PostgreSQL** with **pgvector** extension
- **SQLAlchemy** for ORM
- **Pydantic** for models
- **pytest-bdd** for Gherkin step definitions
- **sentence-transformers** or **OpenAI** for embeddings

## Commit Strategy

```bash
# One commit per scenario group
git add tests/step_defs/knowledge-graph-storage-and-retrieval_steps.py
git commit -m "feat(kb): Implement graph storage CRUD scenarios

WO-KB-003: Store/retrieve atoms, concepts, relationships

Generated with Claude Code

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

## Dependencies

- WO-MCP-001 (MCP Core) - COMPLETED
- WO-KB-001 (KB Core) - COMPLETED

## Success Metrics

- All 37 Gherkin scenarios pass (16 + 3 + 18)
- Test coverage >= 80%
- No N+1 query issues
- Vector search < 100ms for 10k embeddings
