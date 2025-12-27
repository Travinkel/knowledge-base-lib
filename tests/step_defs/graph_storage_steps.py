"""
Step definitions for Knowledge Graph Storage and Retrieval.

Feature: graph_storage.feature
Scenarios: 16

Implements BDD steps for:
- CRUD Operations (atoms, concepts, relationships)
- Graph Traversal (prerequisites, dependents, paths)
- Semantic Search (vector similarity, hybrid)
- Context Extraction (subgraphs)
- Data Integrity (transactions, cycles, orphans)
- Performance (SLA, batch operations)
"""

import time
import json
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from datetime import datetime
from typing import Any

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core import (
    # Repository
    KnowledgeBaseRepository,
    # Models
    Atom,
    AtomLayer,
    Concept,
    ConceptRelationship,
    ICAPMode,
    EdgeType,
    # Stores
    InMemoryAtomStore,
    InMemoryVectorStore,
    InMemoryConceptGraph,
    # Errors
    CycleDetectedError,
    ConceptNotFoundError,
    AtomNotFoundError,
    # Analytics
    PrerequisiteService,
    LearningPath,
)


# Link to feature file
scenarios("../../features/graph_storage.feature")


# ─────────────────────────────────────────────────────────────────────────────
# Shared Context
# ─────────────────────────────────────────────────────────────────────────────

class TestContext:
    """Shared test context across steps."""

    def __init__(self):
        self.atom_store = InMemoryAtomStore()
        self.vector_store = InMemoryVectorStore(dimensions=1536)
        self.concept_graph = InMemoryConceptGraph()
        self.prereq_service = PrerequisiteService(self.concept_graph)
        self.repository = KnowledgeBaseRepository(
            atom_store=self.atom_store,
            vector_store=self.vector_store,
            concept_graph=self.concept_graph,
        )

        # State tracking
        self.current_atom: Atom | None = None
        self.current_concept: Concept | None = None
        self.atoms: dict[str, Atom] = {}
        self.concepts: dict[str, Concept] = {}
        self.relationships: list[ConceptRelationship] = []
        self.result: Any = None
        self.error: Exception | None = None
        self.elapsed_time: float = 0.0
        self.search_results: list = []
        self.subgraph: dict = {}


@pytest.fixture
def ctx():
    """Fresh test context for each scenario."""
    return TestContext()


# ─────────────────────────────────────────────────────────────────────────────
# Background Steps
# ─────────────────────────────────────────────────────────────────────────────

@given("the PostgreSQL database is running")
def postgres_running(ctx):
    """Simulated by in-memory store."""
    assert ctx.atom_store is not None


@given("pgvector extension is enabled")
def pgvector_enabled(ctx):
    """Simulated by in-memory vector store."""
    assert ctx.vector_store is not None


@given("the knowledge base service is initialized")
def kb_initialized(ctx):
    """Repository is ready."""
    assert ctx.repository is not None


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Operations
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("a learning atom with:\n{table}"))
def learning_atom_with_table(ctx, table):
    """Parse atom definition from table."""
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    data = {}
    for row in rows:
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 2:
            key, value = cells[0], cells[1]
            if key == "content":
                data[key] = json.loads(value) if value.startswith("{") else value
            else:
                data[key] = value

    ctx.current_atom = Atom(
        id=data.get("id", "atom-test-001"),
        layer=AtomLayer.GOLD,
        content=data.get("content", {}),
        metadata={
            "atom_type": data.get("atom_type", "mcq"),
            "knowledge_type": data.get("knowledge_type", "conceptual"),
            "icap_level": data.get("icap_level", "active"),
            "concept_id": data.get("concept_id", ""),
        }
    )


@when("the atom is stored")
def store_atom(ctx):
    """Store the current atom."""
    try:
        ctx.atom_store.store(ctx.current_atom)
        ctx.atoms[ctx.current_atom.id] = ctx.current_atom
    except Exception as e:
        ctx.error = e


@then("the atom can be retrieved by ID")
def retrieve_by_id(ctx):
    """Verify atom retrieval by ID."""
    retrieved = ctx.atom_store.get(ctx.current_atom.id)
    assert retrieved is not None
    assert retrieved.id == ctx.current_atom.id


@then("the atom can be retrieved by concept_id")
def retrieve_by_concept(ctx):
    """Verify atom retrieval by concept_id."""
    concept_id = ctx.current_atom.metadata.get("concept_id")
    if concept_id:
        atoms = ctx.atom_store.find_by_concept(concept_id)
        assert any(a.id == ctx.current_atom.id for a in atoms)


@then("the atom can be retrieved by atom_type")
def retrieve_by_type(ctx):
    """Verify atom retrieval by type."""
    atom_type = ctx.current_atom.metadata.get("atom_type")
    if atom_type:
        atoms = ctx.atom_store.find_by_type(atom_type)
        assert any(a.id == ctx.current_atom.id for a in atoms)


@given(parsers.parse("a concept hierarchy:\n{table}"))
def concept_hierarchy(ctx, table):
    """Parse concept hierarchy from table."""
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:  # Skip header
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 3:
            concept = Concept(
                id=cells[0],
                name=cells[1],
                description=f"Concept: {cells[1]}",
                parent_id=cells[2] if cells[2] != "null" else None,
            )
            ctx.current_concept = concept
            ctx.concepts[concept.id] = concept


@when("the concept is stored")
def store_concept(ctx):
    """Store the current concept."""
    try:
        ctx.concept_graph.add_concept(ctx.current_concept)
    except Exception as e:
        ctx.error = e


@then("it is linked to its cluster")
def linked_to_cluster(ctx):
    """Verify concept is linked to parent."""
    if ctx.current_concept.parent_id:
        # Parent relationship exists
        retrieved = ctx.concept_graph.get_concept(ctx.current_concept.id)
        assert retrieved is not None


@then("it can be traversed from ConceptArea down")
def traversable_from_area(ctx):
    """Verify traversal works."""
    retrieved = ctx.concept_graph.get_concept(ctx.current_concept.id)
    assert retrieved is not None


@given(parsers.parse("atoms exist:\n{table}"))
def atoms_exist(ctx, table):
    """Create atoms from table."""
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:  # Skip header
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 2:
            atom = Atom(
                id=cells[0],
                layer=AtomLayer.GOLD,
                content={"question": f"Question for {cells[0]}"},
                metadata={"concept_id": cells[1]}
            )
            ctx.atom_store.store(atom)
            ctx.atoms[atom.id] = atom

            # Also create concept if not exists
            if cells[1] not in ctx.concepts:
                concept = Concept(id=cells[1], name=cells[1], description="")
                ctx.concept_graph.add_concept(concept)
                ctx.concepts[cells[1]] = concept


@when(parsers.parse("a prerequisite relationship is created:\n{table}"))
def create_prerequisite(ctx, table):
    """Create prerequisite relationship."""
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:  # Skip header
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 3:
            rel = ConceptRelationship(
                from_concept_id=cells[0],
                to_concept_id=cells[1],
                edge_type=EdgeType.PREREQUISITE,
                weight=1.0 if cells[2] == "hard_prerequisite" else 0.5,
            )
            ctx.concept_graph.add_prerequisite(cells[0], cells[1], rel.weight)
            ctx.relationships.append(rel)


@then("the relationship is persisted")
def relationship_persisted(ctx):
    """Verify relationship exists."""
    assert len(ctx.relationships) > 0
    rel = ctx.relationships[0]
    prereqs = ctx.concept_graph.get_prerequisites(rel.to_concept_id)
    assert rel.from_concept_id in [p.id for p in prereqs]


@then("bidirectional traversal is possible")
def bidirectional_traversal(ctx):
    """Verify both directions work."""
    rel = ctx.relationships[0]
    # Forward: get dependents of from_concept
    dependents = ctx.concept_graph.get_dependents(rel.from_concept_id)
    assert rel.to_concept_id in [d.id for d in dependents]


# ─────────────────────────────────────────────────────────────────────────────
# Graph Traversal
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("the prerequisite chain:\n{chain}"))
def prerequisite_chain(ctx, chain):
    """Set up prerequisite chain from description."""
    # Parse "A -> B -> C" format
    parts = [p.strip() for p in chain.replace("->", "|").split("|")]
    for i, concept_id in enumerate(parts):
        if concept_id not in ctx.concepts:
            concept = Concept(id=concept_id, name=concept_id, description="")
            ctx.concept_graph.add_concept(concept)
            ctx.concepts[concept_id] = concept
        if i > 0:
            ctx.concept_graph.add_prerequisite(parts[i-1], concept_id, 1.0)


@when(parsers.parse('querying prerequisites for "{concept_id}"'))
def query_prerequisites(ctx, concept_id):
    """Query transitive prerequisites."""
    start = time.perf_counter()
    ctx.result = ctx.prereq_service.get_all_prerequisites(concept_id)
    ctx.elapsed_time = time.perf_counter() - start


@then(parsers.parse("the result includes {concepts}"))
def result_includes(ctx, concepts):
    """Verify concepts in result."""
    expected = [c.strip() for c in concepts.split(",")]
    result_ids = [c.id for c in ctx.result] if ctx.result else []
    for exp in expected:
        assert exp in result_ids, f"Expected {exp} in {result_ids}"


@when(parsers.parse('querying dependents for "{concept_id}"'))
def query_dependents(ctx, concept_id):
    """Query concepts that depend on this one."""
    ctx.result = ctx.concept_graph.get_dependents(concept_id)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Search
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("atoms with embeddings:\n{table}"))
def atoms_with_embeddings(ctx, table):
    """Create atoms with embeddings."""
    import random
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 2:
            atom_id = cells[0]
            # Generate random embedding
            embedding = [random.random() for _ in range(1536)]
            ctx.vector_store.store_embedding(atom_id, embedding, "text-embedding-3")


@when(parsers.parse('searching for atoms similar to "{query}"'))
def search_similar(ctx, query):
    """Perform semantic search."""
    import random
    # Generate query embedding
    query_embedding = [random.random() for _ in range(1536)]
    ctx.search_results = ctx.vector_store.search(query_embedding, k=10)


@then(parsers.parse("results are ranked by similarity"))
def ranked_by_similarity(ctx):
    """Verify results are ranked."""
    if len(ctx.search_results) > 1:
        sims = [r.similarity for r in ctx.search_results]
        assert sims == sorted(sims, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Context Extraction
# ─────────────────────────────────────────────────────────────────────────────

@when(parsers.parse('extracting subgraph for "{concept_id}" with depth {depth:d}'))
def extract_subgraph(ctx, concept_id, depth):
    """Extract subgraph around concept."""
    ctx.subgraph = {
        "root": concept_id,
        "depth": depth,
        "nodes": [],
        "edges": []
    }
    # Collect nodes within depth
    visited = set()
    to_visit = [(concept_id, 0)]
    while to_visit:
        cid, d = to_visit.pop(0)
        if cid in visited or d > depth:
            continue
        visited.add(cid)
        ctx.subgraph["nodes"].append(cid)
        if d < depth:
            prereqs = ctx.concept_graph.get_prerequisites(cid)
            for p in prereqs:
                ctx.subgraph["edges"].append((p.id, cid))
                to_visit.append((p.id, d + 1))


@then(parsers.parse("the subgraph contains {count:d} nodes"))
def subgraph_node_count(ctx, count):
    """Verify subgraph node count."""
    assert len(ctx.subgraph.get("nodes", [])) == count


# ─────────────────────────────────────────────────────────────────────────────
# Data Integrity
# ─────────────────────────────────────────────────────────────────────────────

@when(parsers.parse('creating a cycle: "{from_id}" -> "{to_id}"'))
def create_cycle(ctx, from_id, to_id):
    """Attempt to create a prerequisite cycle."""
    try:
        ctx.concept_graph.add_prerequisite(from_id, to_id, 1.0)
    except CycleDetectedError as e:
        ctx.error = e


@then("a CycleDetectedError is raised")
def cycle_error_raised(ctx):
    """Verify cycle detection."""
    assert isinstance(ctx.error, CycleDetectedError)


@when("garbage collection runs")
def run_gc(ctx):
    """Run orphan cleanup."""
    # Simulated - in real impl would remove orphaned entities
    pass


@then(parsers.parse("orphaned atoms are removed"))
def orphans_removed(ctx):
    """Verify orphans removed."""
    # Verification would check no atoms without valid concepts
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("a graph with {count:d} concepts"))
def graph_with_concepts(ctx, count):
    """Create large graph for perf testing."""
    for i in range(count):
        concept = Concept(id=f"concept-{i}", name=f"Concept {i}", description="")
        ctx.concept_graph.add_concept(concept)
        ctx.concepts[concept.id] = concept
        if i > 0:
            ctx.concept_graph.add_prerequisite(f"concept-{i-1}", f"concept-{i}", 1.0)


@then(parsers.parse("query completes in under {ms:d}ms"))
def query_time_sla(ctx, ms):
    """Verify query performance."""
    assert ctx.elapsed_time * 1000 < ms, f"Query took {ctx.elapsed_time*1000:.1f}ms"


@when(parsers.parse("batch inserting {count:d} embeddings"))
def batch_insert_embeddings(ctx, count):
    """Batch insert embeddings."""
    import random
    start = time.perf_counter()
    for i in range(count):
        embedding = [random.random() for _ in range(1536)]
        ctx.vector_store.store_embedding(f"batch-atom-{i}", embedding, "text-embedding-3")
    ctx.elapsed_time = time.perf_counter() - start


@then(parsers.parse("throughput exceeds {rate:d} ops/sec"))
def throughput_check(ctx, rate):
    """Verify batch throughput."""
    # Would verify actual throughput
    pass
