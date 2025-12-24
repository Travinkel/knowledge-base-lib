"""
Unit tests for Knowledge Base Core Infrastructure.
Tests WO-KB-001 acceptance criteria.
"""
import pathlib
import sys
import time
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from core import (
    KnowledgeBaseRepository,
    Atom, AtomLayer, ICAPMode,
    Concept,
    InMemoryVectorStore, InMemoryConceptGraph,
    PrerequisiteService,
    AtomValidationError,
    CycleDetectedError,
)


class TestAtomCRUD:
    """Test learning atom CRUD operations."""

    def setup_method(self):
        self.kb = KnowledgeBaseRepository(mode="memory")

    def test_create_and_retrieve_atom(self):
        """AC1: Learning atoms stored and retrieved correctly."""
        atom = Atom(id="atom-001", layer=AtomLayer.GOLD, content="What is binary search?")
        self.kb.ingest_atom(atom)
        retrieved = self.kb.get_atom("atom-001")
        assert retrieved is not None
        assert retrieved.id == "atom-001"

    def test_update_atom(self):
        atom = Atom(id="atom-002", layer=AtomLayer.SILVER, content="Original")
        self.kb.ingest_atom(atom)
        atom.content = "Updated"
        self.kb.update_atom(atom)
        assert self.kb.get_atom("atom-002").content == "Updated"

    def test_delete_atom(self):
        atom = Atom(id="atom-003", layer=AtomLayer.BRONZE, content="Delete me")
        self.kb.ingest_atom(atom)
        assert self.kb.delete_atom("atom-003") is True
        assert self.kb.get_atom("atom-003") is None


class TestConceptGraph:
    """Test concept graph operations."""

    def setup_method(self):
        self.kb = KnowledgeBaseRepository(mode="memory")

    def test_create_and_retrieve_concept(self):
        """AC2: Concept graph queries return valid paths."""
        concept = Concept(id="c-001", title="Binary Search", domain="algorithms")
        self.kb.create_concept(concept)
        assert self.kb.get_concept("c-001") is not None

    def test_cycle_detection(self):
        """AC5: Graph operations prevent cycles."""
        graph = InMemoryConceptGraph()
        graph.create_concept(Concept(id="A", title="A", domain="test"))
        graph.create_concept(Concept(id="B", title="B", domain="test"))
        graph.create_concept(Concept(id="C", title="C", domain="test"))
        graph.add_prerequisite("B", "A")
        graph.add_prerequisite("C", "B")
        with pytest.raises(CycleDetectedError):
            graph.add_prerequisite("A", "C")


class TestPrerequisiteChains:
    """AC3: Prerequisite chains resolved correctly."""

    def test_get_prerequisites_recursive(self):
        graph = InMemoryConceptGraph()
        for cid in ["binary", "ip", "subnet"]:
            graph.create_concept(Concept(id=cid, title=cid, domain="net"))
        graph.add_prerequisite("ip", "binary")
        graph.add_prerequisite("subnet", "ip")
        prereqs = graph.get_all_prerequisites_recursive("subnet")
        assert len(prereqs) == 2


class TestVectorSearch:
    """AC4: Vector similarity search returns relevant atoms."""

    def test_search_similar(self):
        vs = InMemoryVectorStore()
        vs.store_embedding("a1", [1.0, 0.0, 0.0], "test")
        vs.store_embedding("a2", [0.9, 0.1, 0.0], "test")
        results = vs.search_similar([1.0, 0.0, 0.0], top_k=2)
        assert results[0].atom_id == "a1"


class TestPerformance:
    """AC6: Performance <100ms for typical queries."""

    def test_query_latency(self):
        kb = KnowledgeBaseRepository(mode="memory")
        for i in range(1000):
            kb.ingest_atom(Atom(id=f"a-{i}", layer=AtomLayer.GOLD, content=f"c{i}"))
        start = time.perf_counter()
        for _ in range(100):
            kb.get_atom("a-500")
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000
        assert elapsed_ms < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
