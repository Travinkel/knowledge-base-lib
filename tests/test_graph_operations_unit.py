"""
Unit tests for Knowledge Graph Operations.

These tests verify the core functionality of WO-KB-002
without requiring full BDD integration.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core import (
    InMemoryConceptGraph,
    InMemoryAtomStore,
    GraphAnalytics,
    Concept,
    Atom,
    AtomLayer,
    ICAPMode,
    Author,
    Domain,
    CycleDetectedError,
    DeletionBlockedError,
    ConceptNotFoundError,
    MasteryStatus,
    PrerequisiteService,
)


class TestConceptCRUD:
    """Test CRUD operations for concepts."""

    def test_create_concept(self):
        """Test creating a new concept node."""
        graph = InMemoryConceptGraph()

        concept = graph.create_concept_with_prereqs(
            title="Binary Search Algorithm",
            domain="Computer_Science.Algorithms",
            difficulty=0.65
        )

        assert concept.id is not None
        assert concept.title == "Binary Search Algorithm"
        assert concept.domain == "Computer_Science.Algorithms"
        assert concept.difficulty == 0.65
        assert concept.created_at is not None

    def test_create_concept_with_prerequisites(self):
        """Test creating concept with prerequisites."""
        graph = InMemoryConceptGraph()

        # Create prerequisites first
        prereq1 = Concept(id="array_indexing", title="Array Indexing", domain="CS", difficulty=0.3)
        prereq2 = Concept(id="comparison", title="Comparison", domain="CS", difficulty=0.3)
        graph.create_concept(prereq1)
        graph.create_concept(prereq2)

        # Create concept with prerequisites
        concept = graph.create_concept_with_prereqs(
            title="Binary Search",
            domain="CS.Algorithms",
            difficulty=0.65,
            prerequisite_ids=["array_indexing", "comparison"]
        )

        assert len(concept.prerequisite_ids) == 2
        assert "array_indexing" in concept.prerequisite_ids
        assert "comparison" in concept.prerequisite_ids

    def test_read_concept_with_expansion(self):
        """Test reading concept with full relationship expansion."""
        graph = InMemoryConceptGraph()
        atom_store = InMemoryAtomStore()

        # Create concept with relationships
        concept = Concept(id="recursion", title="Recursion", domain="CS", difficulty=0.6)
        graph.create_concept(concept)

        # Add prerequisites
        prereq = Concept(id="stack_memory", title="Stack Memory", domain="CS", difficulty=0.4)
        graph.create_concept(prereq)
        graph.add_prerequisite("recursion", "stack_memory")

        # Add teaching atoms
        for i in range(3):
            atom = Atom(id=f"atom_{i}", layer=AtomLayer.GOLD, content=f"Content {i}", icap_mode=ICAPMode.ACTIVE)
            atom_store.create(atom)
            graph.link_atom_to_concept(atom.id, "recursion", mode=ICAPMode.ACTIVE)

        # Add isomorphism
        iso_concept = Concept(id="math_induction", title="Mathematical Induction", domain="Math", difficulty=0.6)
        graph.create_concept(iso_concept)
        graph.add_isomorphism("recursion", "math_induction", 0.85)

        # Read with expansion
        expanded = graph.get_concept_with_expansion("recursion")

        assert expanded is not None
        assert expanded["core_properties"]["title"] == "Recursion"
        assert len(expanded["prerequisites"]) == 1
        assert len(expanded["isomorphisms"]) == 1

    def test_update_concept_properties(self):
        """Test updating concept properties with changelog."""
        graph = InMemoryConceptGraph()

        concept = Concept(id="sorting", title="Sorting Algorithms", domain="CS", difficulty=0.50)
        graph.create_concept(concept)

        # Update difficulty
        updated = graph.update_concept_properties("sorting", {"difficulty": 0.62})

        assert updated.difficulty == 0.62

        # Check changelog
        changelog = graph.get_changelog()
        assert any(e.event_type == "PROPERTY_UPDATE" for e in changelog)

    def test_safe_delete_with_orphan_prevention(self):
        """Test that deletion is blocked when atoms reference concept."""
        graph = InMemoryConceptGraph()
        atom_store = InMemoryAtomStore()

        concept = Concept(id="deprecated", title="Deprecated Concept", domain="CS", difficulty=0.5)
        graph.create_concept(concept)

        # Link atoms to concept
        for i in range(3):
            atom = Atom(id=f"teaching_{i}", layer=AtomLayer.GOLD, content=f"Content {i}")
            atom_store.create(atom)
            graph.link_atom_to_concept(atom.id, "deprecated")

        # Attempt deletion - should be blocked
        with pytest.raises(DeletionBlockedError) as exc_info:
            graph.safe_delete_concept("deprecated")

        assert "3 atoms reference this concept" in str(exc_info.value)
        assert len(exc_info.value.blocking_atoms) == 3


class TestPrerequisiteGraph:
    """Test prerequisite graph operations."""

    def test_traversal_to_root(self):
        """Test traversing prerequisite chain to root concepts."""
        graph = InMemoryConceptGraph()

        # Build chain: neural_networks -> gradient_descent -> calculus -> algebra
        concepts = [
            ("neural_networks", "gradient_descent", 0.9),
            ("gradient_descent", "calculus", 0.7),
            ("calculus", "algebra", 0.5),
            ("algebra", None, 0.2),
        ]

        for concept_id, prereq_id, difficulty in concepts:
            concept = Concept(id=concept_id, title=concept_id, domain="Math", difficulty=difficulty)
            graph.create_concept(concept)

        for concept_id, prereq_id, _ in concepts:
            if prereq_id:
                graph.add_prerequisite(concept_id, prereq_id)

        # Traverse from neural_networks
        all_prereqs = graph.get_all_prerequisites_recursive("neural_networks")

        assert len(all_prereqs) == 3
        assert all_prereqs[0].id == "algebra"  # Deepest first

    def test_cycle_detection(self):
        """Test that prerequisite cycles are detected and prevented."""
        graph = InMemoryConceptGraph()

        # Create A -> B -> C
        for cid in ["A", "B", "C"]:
            graph.create_concept(Concept(id=cid, title=cid, domain="Test", difficulty=0.5))

        graph.add_prerequisite("A", "B")
        graph.add_prerequisite("B", "C")

        # Attempt to create C -> A (would create cycle)
        with pytest.raises(CycleDetectedError) as exc_info:
            graph.add_prerequisite("C", "A")

        assert "cycle" in str(exc_info.value).lower()


class TestIsomorphism:
    """Test cross-domain isomorphism operations."""

    def test_create_isomorphic_mapping(self):
        """Test creating isomorphic mappings between domains."""
        graph = InMemoryConceptGraph()

        # Create concepts in different domains
        water_flow = Concept(id="water_flow", title="Water Flow", domain="Physics", difficulty=0.5)
        electrical = Concept(id="electrical_current", title="Electrical Current", domain="Electronics", difficulty=0.5)

        graph.create_concept(water_flow)
        graph.create_concept(electrical)

        # Create isomorphism
        mapping = graph.add_isomorphism(
            "water_flow",
            "electrical_current",
            mapping_score=0.94,
            structural_mappings={"Pressure": "Voltage", "Flow_Rate": "Current"}
        )

        assert mapping.mapping_score == 0.94
        assert mapping.structural_mappings["Pressure"] == "Voltage"

    def test_find_transfer_path(self):
        """Test finding transfer learning paths via isomorphisms."""
        graph = InMemoryConceptGraph()

        # Create source domain concept
        gradient = Concept(id="gradient_descent", title="Gradient Descent", domain="ML", difficulty=0.6)
        graph.create_concept(gradient)

        # Create target domain path
        hill = Concept(id="hill_climbing", title="Hill Climbing", domain="Optimization", difficulty=0.6)
        annealing = Concept(id="simulated_annealing", title="Simulated Annealing", domain="Optimization", difficulty=0.7)
        graph.create_concept(hill)
        graph.create_concept(annealing)

        # Create isomorphism and prereq
        graph.add_isomorphism("gradient_descent", "hill_climbing", 0.85)
        graph.add_prerequisite("simulated_annealing", "hill_climbing")

        # Find path
        path = graph.find_transfer_path("gradient_descent", "simulated_annealing")

        assert path is not None
        assert len(path) >= 2


class TestGraphAnalytics:
    """Test graph analytics operations."""

    def test_compute_centrality(self):
        """Test computing centrality metrics."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        # Create hub-and-spoke structure
        # "functions" is the hub that all other concepts depend on
        hub = Concept(id="functions", title="Functions", domain="CS", difficulty=0.5)
        graph.create_concept(hub)

        for i in range(5):
            spoke = Concept(id=f"concept_{i}", title=f"Concept {i}", domain="CS", difficulty=0.6)
            graph.create_concept(spoke)
            # spoke -> functions means spoke requires functions
            # This creates incoming edges TO the hub from the spokes' perspective
            graph.add_prerequisite(f"concept_{i}", "functions")

        results = analytics.compute_centrality()

        assert len(results) == 6
        # Results are sorted by PageRank - verify we have valid metrics
        assert all(r.pagerank >= 0 for r in results)
        assert all(r.betweenness >= 0 for r in results)

    def test_detect_clusters(self):
        """Test cluster detection."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        # Create concepts in distinct domains
        domains = {
            "Data_Structures": ["Arrays", "Lists", "Trees"],
            "Algorithms": ["Sorting", "Searching"],
        }

        for domain, concepts in domains.items():
            for concept_name in concepts:
                c = Concept(id=concept_name, title=concept_name, domain=domain, difficulty=0.5)
                graph.create_concept(c)

        clusters = analytics.detect_clusters()

        assert len(clusters) >= 2

    def test_detect_gaps(self):
        """Test structural gap detection."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        # Create concept without prerequisites but high difficulty
        orphan = Concept(id="decision_trees", title="Decision Trees", domain="ML", difficulty=0.6)
        graph.create_concept(orphan)

        gaps = analytics.detect_gaps(domain="ML")

        assert len(gaps) >= 1
        assert any(g.gap_type == "Missing_Prerequisites" for g in gaps)


class TestAtomConceptLinks:
    """Test atom-concept teaching relationships."""

    def test_link_atom_with_coverage(self):
        """Test linking atoms to concepts with coverage metadata."""
        graph = InMemoryConceptGraph()
        atom_store = InMemoryAtomStore()

        concept = Concept(id="recursion", title="Recursion", domain="CS", difficulty=0.6)
        graph.create_concept(concept)

        atom = Atom(id="atom_1", layer=AtomLayer.GOLD, content="What is recursion?", icap_mode=ICAPMode.PASSIVE)
        atom_store.create(atom)

        link = graph.link_atom_to_concept(
            "atom_1",
            "recursion",
            coverage=1.0,
            mode=ICAPMode.PASSIVE,
            aspects_covered=["Definition"]
        )

        assert link.coverage == 1.0
        assert link.aspects_covered == ["Definition"]

    def test_compute_concept_coverage(self):
        """Test computing concept coverage from teaching links."""
        graph = InMemoryConceptGraph()

        concept = Concept(id="recursion", title="Recursion", domain="CS", difficulty=0.6)
        graph.create_concept(concept)

        # Link atoms covering different aspects
        graph.link_atom_to_concept("a1", "recursion", coverage=1.0, aspects_covered=["Definition"])
        graph.link_atom_to_concept("a2", "recursion", coverage=0.5, aspects_covered=["Base_Case"])

        coverage = graph.compute_concept_coverage("recursion", {
            "Definition": 1.0,
            "Base_Case": 0.8,
            "Tail_Optimization": 0.5
        })

        assert coverage.actual_coverage["Definition"] == 1.0
        assert coverage.actual_coverage["Base_Case"] == 0.5
        assert "Tail_Optimization" in coverage.gaps

    def test_icap_distribution_analysis(self):
        """Test ICAP mode distribution analysis."""
        graph = InMemoryConceptGraph()

        concept = Concept(id="binary_search", title="Binary Search", domain="CS", difficulty=0.65)
        graph.create_concept(concept)

        # Link atoms with various ICAP modes
        for i, mode in enumerate([ICAPMode.PASSIVE] * 5 + [ICAPMode.ACTIVE] * 8 + [ICAPMode.CONSTRUCTIVE] * 3):
            graph.link_atom_to_concept(f"atom_{i}", "binary_search", mode=mode)

        distribution = graph.analyze_icap_distribution("binary_search")

        assert distribution.counts[ICAPMode.PASSIVE] == 5
        assert distribution.counts[ICAPMode.ACTIVE] == 8
        assert distribution.counts[ICAPMode.CONSTRUCTIVE] == 3
        assert ICAPMode.INTERACTIVE in distribution.missing_modes

        # Check for dominant mode (Active at 50%)
        dominant = distribution.dominant_mode
        assert dominant is not None
        assert dominant[0] == ICAPMode.ACTIVE


class TestBatchOperations:
    """Test batch operations for performance."""

    def test_batch_create_concepts(self):
        """Test batch creation of concepts."""
        graph = InMemoryConceptGraph()

        concepts_data = [
            {"title": f"Concept {i}", "domain": f"Domain_{i % 3}", "difficulty": 0.5}
            for i in range(100)
        ]

        created = graph.batch_create_concepts(concepts_data, batch_size=25)

        assert len(created) == 100

    def test_batch_create_edges(self):
        """Test batch creation of edges."""
        graph = InMemoryConceptGraph()

        # Create concepts first
        for i in range(20):
            c = Concept(id=f"c_{i}", title=f"Concept {i}", domain="Test", difficulty=0.5)
            graph.create_concept(c)

        # Create edges in batch
        edges = [
            {"source_id": f"c_{i}", "target_id": f"c_{i-1}", "edge_type": "PREREQUISITE"}
            for i in range(1, 20)
        ]

        count = graph.batch_create_edges(edges, batch_size=10)

        assert count == 19


class TestChangelog:
    """Test changelog and auditing."""

    def test_changelog_records_changes(self):
        """Test that changelog records all operations."""
        graph = InMemoryConceptGraph()

        # Create concept
        concept = graph.create_concept_with_prereqs(
            title="Test Concept",
            domain="Test",
            difficulty=0.5
        )

        # Update concept
        graph.update_concept_properties(concept.id, {"difficulty": 0.7})

        changelog = graph.get_changelog()

        # Should have NODE_CREATED and PROPERTY_UPDATE entries
        event_types = {e.event_type for e in changelog}
        assert "NODE_CREATED" in event_types
        assert "PROPERTY_UPDATE" in event_types


class TestFullTextSearch:
    """Test full-text search functionality."""

    def test_search_concepts(self):
        """Test searching concepts by text."""
        graph = InMemoryConceptGraph()

        # Create concepts
        graph.create_concept_with_prereqs(
            title="Binary Search Algorithm",
            domain="CS.Algorithms",
            difficulty=0.65
        )
        graph.create_concept_with_prereqs(
            title="Linear Search",
            domain="CS.Algorithms",
            difficulty=0.3
        )
        graph.create_concept_with_prereqs(
            title="Hash Tables",
            domain="CS.Data_Structures",
            difficulty=0.5
        )

        # Search for "search"
        results = graph.search_concepts("search")

        assert len(results) == 2
        titles = [c.title for c in results]
        assert "Binary Search Algorithm" in titles
        assert "Linear Search" in titles


class TestAuthorAndDomain:
    """Test Author and Domain node operations."""

    def test_create_author(self):
        """Test creating an author node."""
        graph = InMemoryConceptGraph()

        author = Author(
            id="knuth",
            name="Donald Knuth",
            expertise=["Algorithms", "TeX"],
            bible_ref="TAOCP"
        )
        created = graph.create_author(author)

        assert created.id == "knuth"
        assert created.name == "Donald Knuth"

        # Retrieve
        retrieved = graph.get_author("knuth")
        assert retrieved is not None
        assert retrieved.bible_ref == "TAOCP"

    def test_create_domain(self):
        """Test creating a domain node."""
        graph = InMemoryConceptGraph()

        domain = Domain(
            id="cs.algorithms",
            name="Algorithms",
            parent_domain="cs",
            depth=1
        )
        created = graph.create_domain(domain)

        assert created.id == "cs.algorithms"
        assert created.depth == 1

        # Retrieve
        retrieved = graph.get_domain("cs.algorithms")
        assert retrieved is not None
        assert retrieved.parent_domain == "cs"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_create_duplicate_concept_fails(self):
        """Test that creating duplicate concept fails."""
        graph = InMemoryConceptGraph()

        concept = Concept(id="test", title="Test", domain="Test", difficulty=0.5)
        graph.create_concept(concept)

        # Try to create duplicate
        from core import ConceptGraphError
        with pytest.raises(ConceptGraphError):
            graph.create_concept(concept)

    def test_update_nonexistent_concept_fails(self):
        """Test that updating non-existent concept fails."""
        graph = InMemoryConceptGraph()

        with pytest.raises(ConceptNotFoundError):
            graph.update_concept_properties("nonexistent", {"difficulty": 0.5})

    def test_add_prerequisite_to_nonexistent_fails(self):
        """Test that adding prerequisite to non-existent concept fails."""
        graph = InMemoryConceptGraph()

        concept = Concept(id="a", title="A", domain="Test", difficulty=0.5)
        graph.create_concept(concept)

        with pytest.raises(ConceptNotFoundError):
            graph.add_prerequisite("a", "nonexistent")

    def test_delete_nonexistent_concept(self):
        """Test deleting non-existent concept returns False."""
        graph = InMemoryConceptGraph()
        result = graph.delete_concept("nonexistent")
        assert result is False

    def test_get_nonexistent_concept(self):
        """Test getting non-existent concept returns None."""
        graph = InMemoryConceptGraph()
        result = graph.get_concept("nonexistent")
        assert result is None

    def test_isomorphism_to_nonexistent_concept_fails(self):
        """Test creating isomorphism to non-existent concept fails."""
        graph = InMemoryConceptGraph()

        concept = Concept(id="a", title="A", domain="Test", difficulty=0.5)
        graph.create_concept(concept)

        with pytest.raises(ConceptNotFoundError):
            graph.add_isomorphism("a", "nonexistent", 0.8)

    def test_find_transfer_path_with_invalid_concepts(self):
        """Test finding transfer path with invalid concepts."""
        graph = InMemoryConceptGraph()

        result = graph.find_transfer_path("nonexistent1", "nonexistent2")
        assert result is None

    def test_link_atom_to_nonexistent_concept(self):
        """Test linking atom to non-existent concept fails."""
        graph = InMemoryConceptGraph()

        with pytest.raises(ConceptNotFoundError):
            graph.link_atom_to_concept("atom1", "nonexistent")


class TestPrerequisiteInference:
    """Test prerequisite inference from embeddings."""

    def test_infer_prerequisites_from_embeddings(self):
        """Test inferring prerequisites from similar embeddings."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        # Create concepts with increasing difficulty
        c1 = Concept(id="basics", title="Basics", domain="Test", difficulty=0.3)
        c2 = Concept(id="advanced", title="Advanced", domain="Test", difficulty=0.7)
        graph.create_concept(c1)
        graph.create_concept(c2)

        # Create similar embeddings
        embeddings = {
            "basics": [0.9, 0.1] + [0.0] * 382,
            "advanced": [0.85, 0.15] + [0.0] * 382,
        }

        inferences = analytics.infer_prerequisites(embeddings, similarity_threshold=0.9)

        # Should suggest basics as prereq for advanced
        assert len(inferences) >= 1
        assert all(inf.inferred for inf in inferences)

    def test_cosine_similarity_edge_cases(self):
        """Test cosine similarity with edge cases."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        # Empty vectors
        assert analytics._cosine_similarity([], []) == 0.0

        # Different length vectors
        assert analytics._cosine_similarity([1, 0], [1, 0, 0]) == 0.0

        # Zero vectors
        assert analytics._cosine_similarity([0, 0], [0, 0]) == 0.0


class TestConceptCoverageDetails:
    """Test coverage computation details."""

    def test_coverage_gaps_property(self):
        """Test ConceptCoverage gaps property."""
        from core import ConceptCoverage

        coverage = ConceptCoverage(
            concept_id="test",
            required_aspects={"A": 1.0, "B": 0.8, "C": 0.5},
            actual_coverage={"A": 1.0, "B": 0.5}  # B is below required, C is missing
        )

        assert "B" in coverage.gaps
        assert "C" in coverage.gaps
        assert "A" not in coverage.gaps
        assert not coverage.is_complete

    def test_coverage_complete(self):
        """Test ConceptCoverage when complete."""
        from core import ConceptCoverage

        coverage = ConceptCoverage(
            concept_id="test",
            required_aspects={"A": 1.0, "B": 0.5},
            actual_coverage={"A": 1.0, "B": 0.6}
        )

        assert len(coverage.gaps) == 0
        assert coverage.is_complete


class TestICAPDistributionDetails:
    """Test ICAP distribution details."""

    def test_icap_percentages(self):
        """Test ICAP percentage calculation."""
        from core import ICAPDistribution

        dist = ICAPDistribution(
            concept_id="test",
            counts={
                ICAPMode.PASSIVE: 5,
                ICAPMode.ACTIVE: 10,
                ICAPMode.CONSTRUCTIVE: 3,
                ICAPMode.INTERACTIVE: 2
            }
        )

        assert dist.total == 20
        assert dist.percentages[ICAPMode.ACTIVE] == 0.5  # 10/20
        assert len(dist.missing_modes) == 0

    def test_icap_empty_distribution(self):
        """Test ICAP distribution when empty."""
        from core import ICAPDistribution

        dist = ICAPDistribution(concept_id="test", counts={})

        assert dist.total == 0
        for mode in ICAPMode:
            assert dist.percentages[mode] == 0.0
        assert len(dist.missing_modes) == 4


class TestGraphAnalyticsEdgeCases:
    """Test graph analytics edge cases."""

    def test_centrality_empty_graph(self):
        """Test centrality on empty graph."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        results = analytics.compute_centrality()
        assert len(results) == 0

    def test_clusters_empty_graph(self):
        """Test clustering on empty graph."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        clusters = analytics.detect_clusters()
        assert len(clusters) == 0

    def test_gaps_empty_graph(self):
        """Test gap detection on empty graph."""
        graph = InMemoryConceptGraph()
        analytics = GraphAnalytics(graph)

        gaps = analytics.detect_gaps()
        assert len(gaps) == 0

    def test_centrality_interpretation(self):
        """Test centrality result interpretation."""
        from core import CentralityResult

        # Critical hub
        hub = CentralityResult(
            concept_id="hub",
            pagerank=0.08,
            betweenness=0.12,
            in_degree=5,
            out_degree=3
        )
        assert hub.interpretation == "Critical hub"

        # Specialized leaf
        leaf = CentralityResult(
            concept_id="leaf",
            pagerank=0.01,
            betweenness=0.001,
            in_degree=1,
            out_degree=0
        )
        assert leaf.interpretation == "Specialized leaf"
