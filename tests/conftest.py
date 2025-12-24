"""
Pytest configuration for Knowledge Base tests.

Provides shared fixtures for BDD step definitions.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core import (
    InMemoryConceptGraph,
    InMemoryAtomStore,
    InMemoryVectorStore,
    KnowledgeBaseRepository,
    GraphAnalytics,
    Concept,
    Atom,
    AtomLayer,
    ICAPMode,
    Author,
    Domain,
    MasteryStatus,
)


@pytest.fixture
def concept_graph():
    """Fresh in-memory concept graph for each test."""
    return InMemoryConceptGraph()


@pytest.fixture
def atom_store():
    """Fresh in-memory atom store for each test."""
    return InMemoryAtomStore()


@pytest.fixture
def vector_store():
    """Fresh in-memory vector store for each test."""
    return InMemoryVectorStore()


@pytest.fixture
def kb_repository(concept_graph, atom_store, vector_store):
    """Knowledge Base repository with in-memory backends."""
    return KnowledgeBaseRepository(
        atom_store=atom_store,
        vector_store=vector_store,
        concept_graph=concept_graph,
        mode="memory"
    )


@pytest.fixture
def graph_analytics(concept_graph):
    """Graph analytics service for concept graph."""
    return GraphAnalytics(concept_graph)


@pytest.fixture
def sample_concepts(concept_graph):
    """Create sample concepts for testing."""
    concepts = {}

    # Foundation concepts
    concepts["algebra"] = Concept(
        id="algebra",
        title="Algebra Fundamentals",
        domain="Mathematics",
        difficulty=0.2
    )
    concept_graph.create_concept(concepts["algebra"])

    concepts["calculus_single"] = Concept(
        id="calculus_single",
        title="Single Variable Calculus",
        domain="Mathematics",
        difficulty=0.4,
        prerequisite_ids=["algebra"]
    )
    concept_graph.create_concept(concepts["calculus_single"])
    concept_graph.add_prerequisite("calculus_single", "algebra")

    concepts["calculus_multi"] = Concept(
        id="calculus_multi",
        title="Multivariable Calculus",
        domain="Mathematics",
        difficulty=0.6,
        prerequisite_ids=["calculus_single"]
    )
    concept_graph.create_concept(concepts["calculus_multi"])
    concept_graph.add_prerequisite("calculus_multi", "calculus_single")

    return concepts


@pytest.fixture
def sample_atoms(atom_store):
    """Create sample atoms for testing."""
    atoms = {}

    atoms["flashcard_1"] = Atom(
        id="atom_1",
        layer=AtomLayer.GOLD,
        content="What is the derivative of x^2?",
        icap_mode=ICAPMode.PASSIVE,
    )
    atom_store.create(atoms["flashcard_1"])

    atoms["mcq_1"] = Atom(
        id="atom_2",
        layer=AtomLayer.GOLD,
        content="Which is the correct chain rule formula?",
        icap_mode=ICAPMode.ACTIVE,
    )
    atom_store.create(atoms["mcq_1"])

    return atoms
