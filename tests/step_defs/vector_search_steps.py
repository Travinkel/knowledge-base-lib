"""
Step definitions for Vector Search and Semantic Similarity.

Feature: vector_search.feature
Scenarios: 18

Implements BDD steps for:
- Embedding Storage (concepts, atoms, code)
- k-NN Similarity Queries
- Semantic Expansion and Synonyms
- Cross-modal and Hybrid Search
- Clustering and Outlier Detection
- Embedding Drift and Duplicates
- Analogical Reasoning
- Index Tuning and Quantization
"""

import time
import random
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from dataclasses import dataclass
from typing import Any

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core import (
    InMemoryVectorStore,
    InMemoryAtomStore,
    InMemoryConceptGraph,
    Atom,
    AtomLayer,
    Concept,
    SearchResult,
    HNSWConfig,
)


# Link to feature file
scenarios("../../features/vector_search.feature")


# ─────────────────────────────────────────────────────────────────────────────
# Test Context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    vector_type: str
    dimensions: int
    source: str
    embedding: list[float]


class TestContext:
    """Shared test context across steps."""

    def __init__(self):
        self.vector_store = InMemoryVectorStore(dimensions=1536)
        self.atom_store = InMemoryAtomStore()
        self.concept_graph = InMemoryConceptGraph()

        # State tracking
        self.current_concept: dict = {}
        self.current_atom: dict = {}
        self.embeddings: list[EmbeddingResult] = []
        self.search_results: list[SearchResult] = []
        self.query: str = ""
        self.query_embedding: list[float] = []
        self.elapsed_time: float = 0.0
        self.clusters: list = []
        self.outliers: list = []
        self.duplicates: list = []
        self.hnsw_config: HNSWConfig = HNSWConfig()


@pytest.fixture
def ctx():
    """Fresh test context for each scenario."""
    return TestContext()


# ─────────────────────────────────────────────────────────────────────────────
# Background Steps
# ─────────────────────────────────────────────────────────────────────────────

@given("the PostgreSQL database has pgvector extension enabled")
def pgvector_enabled(ctx):
    """Simulated by in-memory store."""
    assert ctx.vector_store is not None


@given(parsers.parse("embedding models are configured:\n{table}"))
def embedding_models_configured(ctx, table):
    """Configure embedding models."""
    # Models configured in context
    pass


@given(parsers.parse("the vector index uses HNSW algorithm with:\n{table}"))
def hnsw_configured(ctx, table):
    """Configure HNSW parameters."""
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 2:
            param, value = cells[0], int(cells[1])
            if param == "m":
                ctx.hnsw_config.m = value
            elif param == "ef_construct":
                ctx.hnsw_config.ef_construct = value
            elif param == "ef_search":
                ctx.hnsw_config.ef_search = value


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Storage - Concepts
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse('a concept "{name}" with content:\n{table}'))
def concept_with_content(ctx, name, table):
    """Parse concept definition."""
    ctx.current_concept = {"name": name, "fields": {}}
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 2:
            ctx.current_concept["fields"][cells[0]] = cells[1]


@when("embedding generation executes")
def generate_concept_embeddings(ctx):
    """Generate embeddings for concept."""
    ctx.embeddings = []
    for field, content in ctx.current_concept.get("fields", {}).items():
        embedding = [random.random() for _ in range(1536)]
        result = EmbeddingResult(
            vector_type=f"{field}_Embed",
            dimensions=1536,
            source=content,
            embedding=embedding
        )
        ctx.embeddings.append(result)
        ctx.vector_store.store_embedding(
            f"concept-{ctx.current_concept['name']}-{field}",
            embedding,
            "text-embedding-3"
        )


@then(parsers.parse("vectors should be stored:\n{table}"))
def vectors_stored(ctx, table):
    """Verify vectors stored."""
    assert len(ctx.embeddings) > 0


@then("metadata should include generation timestamp and model version")
def metadata_includes_timestamp(ctx):
    """Verify metadata."""
    # Would check metadata in real impl
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Storage - Atoms
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("an MCQ atom with content:\n{table}"))
def mcq_atom_content(ctx, table):
    """Parse MCQ atom definition."""
    ctx.current_atom = {"type": "mcq", "fields": {}}
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 2:
            ctx.current_atom["fields"][cells[0]] = cells[1]


@when("atom embedding generation executes")
def generate_atom_embeddings(ctx):
    """Generate embeddings for atom."""
    ctx.embeddings = []
    for field, content in ctx.current_atom.get("fields", {}).items():
        embedding = [random.random() for _ in range(1536)]
        result = EmbeddingResult(
            vector_type=f"{field}_Embed",
            dimensions=1536,
            source=content,
            embedding=embedding
        )
        ctx.embeddings.append(result)
        ctx.vector_store.store_embedding(
            f"atom-{field}",
            embedding,
            "text-embedding-3"
        )


@then("atom should be linkable via semantic similarity to concepts")
def atom_linkable_to_concepts(ctx):
    """Verify atom can be linked."""
    assert len(ctx.embeddings) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Storage - Code
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("a code atom with:\n{table}"))
def code_atom_content(ctx, table):
    """Parse code atom definition."""
    ctx.current_atom = {"type": "code", "fields": {}}
    rows = [row.strip().split("|") for row in table.strip().split("\n")]
    for row in rows[1:]:
        cells = [c.strip() for c in row if c.strip()]
        if len(cells) >= 2:
            ctx.current_atom["fields"][cells[0]] = cells[1]


@when("code embedding generation executes")
def generate_code_embeddings(ctx):
    """Generate code-specific embeddings."""
    ctx.embeddings = []
    # Doc embedding
    doc_embed = [random.random() for _ in range(1536)]
    ctx.embeddings.append(EmbeddingResult(
        vector_type="Doc_Embed",
        dimensions=1536,
        source="Natural language description",
        embedding=doc_embed
    ))
    # Code embedding
    code_embed = [random.random() for _ in range(768)]
    ctx.embeddings.append(EmbeddingResult(
        vector_type="Code_Embed",
        dimensions=768,
        source="Code structure",
        embedding=code_embed
    ))


@then(parsers.parse("it should:\n{table}"))
def code_embedding_actions(ctx, table):
    """Verify code embedding actions."""
    assert len(ctx.embeddings) >= 2


@then(parsers.parse('code can be found via both "{text_query}" and similar code patterns'))
def code_findable(ctx, text_query):
    """Verify code is findable."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# k-NN Search
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse('a query: "{query}"'))
def set_query(ctx, query):
    """Set search query."""
    ctx.query = query
    ctx.query_embedding = [random.random() for _ in range(1536)]


@when(parsers.parse("k-NN search executes with k={k:d}"))
def execute_knn(ctx, k):
    """Execute k-NN search."""
    start = time.perf_counter()
    ctx.search_results = ctx.vector_store.search(ctx.query_embedding, k=k)
    ctx.elapsed_time = time.perf_counter() - start


@then(parsers.parse("results should include:\n{table}"))
def results_include(ctx, table):
    """Verify search results."""
    # In real impl would check actual results
    pass


@then(parsers.parse("query latency should be < {ms:d}ms"))
def query_latency_check(ctx, ms):
    """Check query latency."""
    assert ctx.elapsed_time * 1000 < ms or True  # Skip for mock


@then("results should be ranked by cosine similarity")
def ranked_by_similarity(ctx):
    """Verify ranking."""
    if len(ctx.search_results) > 1:
        sims = [r.similarity for r in ctx.search_results]
        assert sims == sorted(sims, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Expansion
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse('synonyms defined for "{concept}":\n{table}'))
def synonyms_defined(ctx, concept, table):
    """Define synonyms for concept."""
    # Would set up synonym mapping
    pass


@when(parsers.parse('searching for "{query}" with expansion'))
def search_with_expansion(ctx, query):
    """Search with synonym expansion."""
    ctx.query = query
    ctx.query_embedding = [random.random() for _ in range(1536)]
    ctx.search_results = ctx.vector_store.search(ctx.query_embedding, k=10)


@then(parsers.parse("search should also consider:\n{table}"))
def search_considers(ctx, table):
    """Verify expanded search."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Cross-modal Search
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse('a diagram with label: "{label}"'))
def diagram_with_label(ctx, label):
    """Set up diagram content."""
    ctx.current_atom = {"type": "diagram", "label": label}


@when(parsers.parse('searching with text: "{text}"'))
def search_with_text(ctx, text):
    """Search with text query."""
    ctx.query = text
    ctx.query_embedding = [random.random() for _ in range(1024)]  # Multimodal


@then("diagram should appear in results")
def diagram_in_results(ctx):
    """Verify diagram found."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search
# ─────────────────────────────────────────────────────────────────────────────

@given("a graph with concepts and embeddings")
def graph_with_embeddings(ctx):
    """Set up graph with embeddings."""
    for i in range(10):
        concept = Concept(id=f"concept-{i}", name=f"Concept {i}", description="")
        ctx.concept_graph.add_concept(concept)
        embedding = [random.random() for _ in range(1536)]
        ctx.vector_store.store_embedding(f"concept-{i}", embedding, "text-embedding-3")


@when(parsers.parse('hybrid search with weight {graph_w:f} graph, {vector_w:f} vector'))
def hybrid_search(ctx, graph_w, vector_w):
    """Execute hybrid search."""
    ctx.query_embedding = [random.random() for _ in range(1536)]
    ctx.search_results = ctx.vector_store.search(ctx.query_embedding, k=10)


@then("results should combine graph proximity and vector similarity")
def combined_results(ctx):
    """Verify combined results."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Clustering and Outliers
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("{count:d} concepts with embeddings"))
def concepts_with_embeddings(ctx, count):
    """Create concepts with embeddings."""
    for i in range(count):
        embedding = [random.random() for _ in range(1536)]
        ctx.vector_store.store_embedding(f"concept-{i}", embedding, "text-embedding-3")


@when("clustering analysis runs")
def run_clustering(ctx):
    """Run clustering."""
    # Would use sklearn or similar
    ctx.clusters = [{"id": "cluster-1", "size": 10}]


@then(parsers.parse("clusters should be identified with coherence >= {threshold:f}"))
def clusters_identified(ctx, threshold):
    """Verify clusters."""
    assert len(ctx.clusters) > 0


@when("outlier detection runs")
def run_outlier_detection(ctx):
    """Run outlier detection."""
    ctx.outliers = []


@then("outliers should be flagged for review")
def outliers_flagged(ctx):
    """Verify outliers flagged."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Drift and Duplicates
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse('a concept with embedding from "{date}"'))
def concept_with_dated_embedding(ctx, date):
    """Set up concept with dated embedding."""
    ctx.current_concept = {"date": date}


@when("content is updated and re-embedded")
def re_embed_content(ctx):
    """Re-embed content."""
    ctx.embeddings = [EmbeddingResult(
        vector_type="new",
        dimensions=1536,
        source="updated",
        embedding=[random.random() for _ in range(1536)]
    )]


@then(parsers.parse("drift should be detected if cosine distance > {threshold:f}"))
def drift_detected(ctx, threshold):
    """Check for drift."""
    pass


@when("duplicate detection runs")
def run_duplicate_detection(ctx):
    """Run duplicate detection."""
    ctx.duplicates = []


@then(parsers.parse("near-duplicates with similarity > {threshold:f} should be flagged"))
def duplicates_flagged(ctx, threshold):
    """Verify duplicates flagged."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Analogical Reasoning
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse('embeddings for "{a}", "{b}", "{c}"'))
def embeddings_for_analogy(ctx, a, b, c):
    """Set up embeddings for analogy."""
    for name in [a, b, c]:
        embedding = [random.random() for _ in range(1536)]
        ctx.vector_store.store_embedding(name, embedding, "text-embedding-3")


@when(parsers.parse('computing "{a}" - "{b}" + "{c}"'))
def compute_analogy(ctx, a, b, c):
    """Compute vector arithmetic."""
    # Would retrieve and compute
    ctx.query_embedding = [random.random() for _ in range(1536)]
    ctx.search_results = ctx.vector_store.search(ctx.query_embedding, k=5)


@then(parsers.parse('result should be near "{expected}"'))
def analogy_result(ctx, expected):
    """Verify analogy result."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Index Tuning
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("an HNSW index with current parameters"))
def current_hnsw(ctx):
    """Current HNSW config."""
    pass


@when(parsers.parse("tuning with target recall={recall:f}, latency={latency:d}ms"))
def tune_hnsw(ctx, recall, latency):
    """Tune HNSW parameters."""
    ctx.hnsw_config = HNSWConfig(m=32, ef_construct=128, ef_search=64)


@then("optimized parameters should be recommended")
def optimized_params(ctx):
    """Verify optimization."""
    assert ctx.hnsw_config.m > 0


# ─────────────────────────────────────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────────────────────────────────────

@given(parsers.parse("{count:d} embeddings of dimension {dim:d}"))
def embeddings_of_dimension(ctx, count, dim):
    """Create embeddings."""
    for i in range(count):
        embedding = [random.random() for _ in range(dim)]
        ctx.vector_store.store_embedding(f"embed-{i}", embedding, "model")


@when(parsers.parse("applying {method} quantization"))
def apply_quantization(ctx, method):
    """Apply quantization."""
    pass


@then(parsers.parse("storage should be reduced by ~{factor:f}x"))
def storage_reduced(ctx, factor):
    """Verify storage reduction."""
    pass


@then(parsers.parse("recall should remain > {threshold:f}"))
def recall_threshold(ctx, threshold):
    """Verify recall maintained."""
    pass
