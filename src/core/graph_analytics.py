"""
Graph analytics for the Knowledge Base.

Implements:
- Centrality metrics (PageRank, Betweenness)
- Community detection (clustering)
- Structural gap analysis
- Prerequisite inference from embeddings

Scientific Foundation:
- PageRank (Page et al., 1999)
- Betweenness Centrality (Freeman, 1977)
- Louvain Community Detection (Blondel et al., 2008)
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from .models import Concept, ConceptRelationship


logger = logging.getLogger(__name__)


@dataclass
class CentralityResult:
    """Centrality scores for a concept."""
    concept_id: str
    pagerank: float = 0.0
    betweenness: float = 0.0
    in_degree: int = 0
    out_degree: int = 0

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of centrality."""
        if self.pagerank > 0.07 and self.betweenness > 0.1:
            return "Critical hub"
        elif self.pagerank > 0.05 and self.in_degree == 0:
            return "Foundational gateway"
        elif self.betweenness > 0.08:
            return "Integration point"
        elif self.pagerank < 0.02 and self.out_degree == 0:
            return "Specialized leaf"
        return "Standard node"


@dataclass
class ConceptCluster:
    """A cluster of related concepts."""
    cluster_id: int
    concept_ids: list[str] = field(default_factory=list)
    suggested_module: str | None = None


@dataclass
class StructuralGap:
    """A detected gap in the knowledge graph structure."""
    gap_type: str  # Missing_Prerequisites, Orphan_Island, Weak_Connectivity
    concept_id: str | None
    issue: str
    recommendation: str | None = None


@dataclass
class PrerequisiteInference:
    """An inferred prerequisite relationship."""
    suggested_prereq_id: str
    for_concept_id: str
    confidence: float
    reasoning: str
    inferred: bool = True


class ConceptGraphProtocol(Protocol):
    """Protocol for concept graph backends (for type hints)."""

    def get_concept(self, concept_id: str) -> Concept | None: ...
    def get_all_concepts(self) -> list[Concept]: ...
    def get_prerequisites(self, concept_id: str) -> list[Concept]: ...


class GraphAnalytics:
    """
    Analytics service for the concept graph.

    Provides centrality metrics, clustering, and gap detection.
    """

    def __init__(self, concept_graph: ConceptGraphProtocol) -> None:
        self._graph = concept_graph

    # ─────────────────────────────────────────────────────────────────────────
    # Centrality Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def compute_centrality(
        self,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> list[CentralityResult]:
        """
        Compute centrality metrics for all concepts.

        Uses PageRank algorithm for importance and computes
        betweenness centrality for identifying bottlenecks.

        Args:
            damping: PageRank damping factor (default 0.85)
            max_iterations: Max iterations for PageRank convergence
            tolerance: Convergence tolerance

        Returns:
            List of CentralityResult sorted by PageRank (descending)
        """
        concepts = self._graph.get_all_concepts()
        if not concepts:
            return []

        n = len(concepts)
        concept_ids = [c.id for c in concepts]
        id_to_idx = {cid: i for i, cid in enumerate(concept_ids)}

        # Build adjacency matrix (incoming edges for PageRank)
        # For prereq graph: A requires B means edge B -> A (B flows into A)
        in_edges: dict[str, list[str]] = defaultdict(list)
        out_degree: dict[str, int] = defaultdict(int)

        for concept in concepts:
            for prereq_id in concept.prerequisite_ids:
                if prereq_id in id_to_idx:
                    in_edges[concept.id].append(prereq_id)
                    out_degree[prereq_id] += 1

        # PageRank iteration
        pagerank = {cid: 1.0 / n for cid in concept_ids}

        for iteration in range(max_iterations):
            new_pagerank = {}
            for cid in concept_ids:
                # Sum of incoming PageRank divided by out-degree
                incoming_rank = 0.0
                for prereq_id in in_edges[cid]:
                    out_deg = out_degree[prereq_id]
                    if out_deg > 0:
                        incoming_rank += pagerank[prereq_id] / out_deg

                new_pagerank[cid] = (1 - damping) / n + damping * incoming_rank

            # Check convergence
            diff = sum(abs(new_pagerank[cid] - pagerank[cid]) for cid in concept_ids)
            pagerank = new_pagerank
            if diff < tolerance:
                logger.debug(f"PageRank converged in {iteration + 1} iterations")
                break

        # Compute betweenness centrality (simplified Brandes algorithm)
        betweenness = self._compute_betweenness(concepts, id_to_idx)

        # Build results
        results = []
        for concept in concepts:
            in_deg = len(in_edges[concept.id])
            out_deg = out_degree[concept.id]
            results.append(CentralityResult(
                concept_id=concept.id,
                pagerank=pagerank[concept.id],
                betweenness=betweenness.get(concept.id, 0.0),
                in_degree=in_deg,
                out_degree=out_deg
            ))

        # Sort by PageRank descending
        results.sort(key=lambda r: r.pagerank, reverse=True)
        return results

    def _compute_betweenness(
        self,
        concepts: list[Concept],
        id_to_idx: dict[str, int]
    ) -> dict[str, float]:
        """
        Compute betweenness centrality using Brandes algorithm.

        Betweenness measures how often a node lies on shortest paths
        between other nodes - high betweenness = bottleneck.
        """
        betweenness = {c.id: 0.0 for c in concepts}

        for source in concepts:
            # BFS from source
            stack = []
            predecessors: dict[str, list[str]] = defaultdict(list)
            sigma = {c.id: 0.0 for c in concepts}
            sigma[source.id] = 1.0
            dist = {c.id: -1 for c in concepts}
            dist[source.id] = 0

            queue = [source.id]
            while queue:
                v = queue.pop(0)
                stack.append(v)
                concept = self._graph.get_concept(v)
                if not concept:
                    continue

                # Get neighbors (concepts that have this as prereq)
                for other in concepts:
                    if v in other.prerequisite_ids:
                        w = other.id
                        if dist[w] < 0:  # Not visited
                            dist[w] = dist[v] + 1
                            queue.append(w)
                        if dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            predecessors[w].append(v)

            # Back-propagation
            delta = {c.id: 0.0 for c in concepts}
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != source.id:
                    betweenness[w] += delta[w]

        # Normalize
        n = len(concepts)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            for cid in betweenness:
                betweenness[cid] *= norm

        return betweenness

    # ─────────────────────────────────────────────────────────────────────────
    # Clustering / Community Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_clusters(self) -> list[ConceptCluster]:
        """
        Detect concept clusters using a simplified Louvain-like algorithm.

        Groups concepts by domain similarity and prerequisite connectivity.

        Returns:
            List of ConceptCluster with suggested module names
        """
        concepts = self._graph.get_all_concepts()
        if not concepts:
            return []

        # Simple clustering: group by domain prefix
        domain_clusters: dict[str, list[str]] = defaultdict(list)

        for concept in concepts:
            # Extract top-level domain (e.g., "Computer_Science" from "Computer_Science.Algorithms")
            domain = concept.domain.split(".")[0] if "." in concept.domain else concept.domain
            domain_clusters[domain].append(concept.id)

        # Convert to ConceptCluster objects
        clusters = []
        for i, (domain, concept_ids) in enumerate(domain_clusters.items()):
            if len(concept_ids) >= 2:  # Only include clusters with 2+ concepts
                cluster = ConceptCluster(
                    cluster_id=i + 1,
                    concept_ids=concept_ids,
                    suggested_module=self._suggest_module_name(domain, concept_ids)
                )
                clusters.append(cluster)

        return clusters

    def _suggest_module_name(self, domain: str, concept_ids: list[str]) -> str:
        """Suggest a module name based on domain and concepts."""
        # Simple heuristic: use domain as module name
        return domain.replace("_", " ").title().replace(" ", "_")

    # ─────────────────────────────────────────────────────────────────────────
    # Gap Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_gaps(self, domain: str | None = None) -> list[StructuralGap]:
        """
        Detect structural gaps in the knowledge graph.

        Identifies:
        - Missing prerequisites (non-beginner concepts with no prereqs)
        - Orphan islands (disconnected concept groups)
        - Weak connectivity (concepts with very few connections)

        Args:
            domain: Optional domain filter

        Returns:
            List of detected StructuralGap
        """
        concepts = self._graph.get_all_concepts()
        if domain:
            concepts = [c for c in concepts if c.domain.startswith(domain)]

        gaps = []

        # Check for missing prerequisites
        for concept in concepts:
            if not concept.prerequisite_ids and concept.difficulty > 0.3:
                gaps.append(StructuralGap(
                    gap_type="Missing_Prerequisites",
                    concept_id=concept.id,
                    issue="No foundation defined",
                    recommendation=f"Add prerequisites for {concept.title}"
                ))

        # Check for orphan islands (concepts not connected to main graph)
        connected = self._find_connected_component(concepts)
        for concept in concepts:
            if concept.id not in connected:
                # Check if it's truly orphaned (no incoming or outgoing edges)
                has_outgoing = len(concept.prerequisite_ids) > 0
                has_incoming = any(
                    concept.id in c.prerequisite_ids
                    for c in concepts
                )
                if not has_outgoing and not has_incoming:
                    gaps.append(StructuralGap(
                        gap_type="Orphan_Island",
                        concept_id=concept.id,
                        issue="Completely disconnected from graph",
                        recommendation=f"Connect {concept.title} to prerequisite chain"
                    ))

        # Check for weak connectivity
        for concept in concepts:
            out_degree = len(concept.prerequisite_ids)
            in_degree = sum(
                1 for c in concepts
                if concept.id in c.prerequisite_ids
            )
            if out_degree + in_degree == 1 and concept.difficulty > 0.5:
                gaps.append(StructuralGap(
                    gap_type="Weak_Connectivity",
                    concept_id=concept.id,
                    issue=f"Only {out_degree + in_degree} connection(s) for intermediate concept",
                    recommendation=f"Add more connections for {concept.title}"
                ))

        return gaps

    def _find_connected_component(self, concepts: list[Concept]) -> set[str]:
        """Find the largest connected component using BFS."""
        if not concepts:
            return set()

        # Build adjacency (undirected for connectivity)
        adj: dict[str, set[str]] = defaultdict(set)
        for concept in concepts:
            for prereq_id in concept.prerequisite_ids:
                adj[concept.id].add(prereq_id)
                adj[prereq_id].add(concept.id)

        # Find largest component
        visited: set[str] = set()
        largest_component: set[str] = set()

        for concept in concepts:
            if concept.id in visited:
                continue

            # BFS
            component: set[str] = set()
            queue = [concept.id]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) > len(largest_component):
                largest_component = component

        return largest_component

    # ─────────────────────────────────────────────────────────────────────────
    # Prerequisite Inference
    # ─────────────────────────────────────────────────────────────────────────

    def infer_prerequisites(
        self,
        embeddings: dict[str, list[float]],
        similarity_threshold: float = 0.8
    ) -> list[PrerequisiteInference]:
        """
        Infer prerequisite relationships from content embeddings.

        Uses cosine similarity to identify concepts that should
        be prerequisites based on semantic similarity.

        Args:
            embeddings: Dict mapping concept_id to embedding vector
            similarity_threshold: Minimum similarity to suggest prerequisite

        Returns:
            List of PrerequisiteInference suggestions
        """
        inferences = []
        concept_ids = list(embeddings.keys())

        for i, cid_a in enumerate(concept_ids):
            for cid_b in concept_ids[i + 1:]:
                sim = self._cosine_similarity(embeddings[cid_a], embeddings[cid_b])
                if sim >= similarity_threshold:
                    # Determine direction: lower difficulty should be prereq
                    concept_a = self._graph.get_concept(cid_a)
                    concept_b = self._graph.get_concept(cid_b)

                    if not concept_a or not concept_b:
                        continue

                    if concept_a.difficulty < concept_b.difficulty:
                        prereq_id, dependent_id = cid_a, cid_b
                    else:
                        prereq_id, dependent_id = cid_b, cid_a

                    # Check if relationship already exists
                    dependent = self._graph.get_concept(dependent_id)
                    if dependent and prereq_id not in dependent.prerequisite_ids:
                        inferences.append(PrerequisiteInference(
                            suggested_prereq_id=prereq_id,
                            for_concept_id=dependent_id,
                            confidence=sim,
                            reasoning="High similarity" if sim > 0.9 else "Structural parent"
                        ))

        # Sort by confidence descending
        inferences.sort(key=lambda x: x.confidence, reverse=True)
        return inferences

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
