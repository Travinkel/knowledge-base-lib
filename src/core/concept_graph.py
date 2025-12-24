"""
Concept graph operations using Neo4j.

Implements:
- CRUD for concept nodes
- Prerequisite relationship management
- Graph analytics (centrality, clustering)
- Cross-domain isomorphism detection
- Atom-Concept teaching links with coverage
- Change auditing

Scientific Foundation:
- Knowledge Graph Embedding (Bordes et al., 2013)
- Semantic Networks (Sowa, 1987)
- Analogical Transfer of Meaning (Gentner, 1983)
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Protocol, Any

from .models import (
    Concept,
    ConceptRelationship,
    Author,
    Domain,
    IsomorphicMapping,
    TeachingLink,
    ConceptCoverage,
    ICAPDistribution,
    ICAPMode,
    ChangelogEntry,
    EdgeType,
)


logger = logging.getLogger(__name__)


class ConceptGraphError(Exception):
    """Base exception for concept graph operations."""
    pass


class CycleDetectedError(ConceptGraphError):
    """Raised when a prerequisite cycle is detected."""
    pass


class ConceptNotFoundError(ConceptGraphError):
    """Raised when a concept is not found."""
    pass


class DeletionBlockedError(ConceptGraphError):
    """Raised when deletion is blocked due to referencing entities."""

    def __init__(self, message: str, blocking_atoms: list[str] | None = None):
        super().__init__(message)
        self.blocking_atoms = blocking_atoms or []


class OrphanPreventionError(ConceptGraphError):
    """Raised when an operation would orphan atoms."""
    pass


class ConceptGraph(Protocol):
    """Protocol for concept graph backends."""

    def create_concept(self, concept: Concept) -> Concept: ...
    def get_concept(self, concept_id: str) -> Concept | None: ...
    def update_concept(self, concept: Concept) -> Concept: ...
    def delete_concept(self, concept_id: str) -> bool: ...
    def add_prerequisite(self, source_id: str, target_id: str, strength: float = 1.0) -> bool: ...
    def get_prerequisites(self, concept_id: str) -> list[Concept]: ...
    def detect_cycle(self, source_id: str, target_id: str) -> list[str] | None: ...


class InMemoryConceptGraph:
    """
    In-memory concept graph for testing.

    Uses adjacency list representation.
    Supports:
    - CRUD for concepts, authors, domains
    - Prerequisite edges with cycle detection
    - Isomorphic mappings (AToM)
    - Teaching links with coverage tracking
    - Changelog for auditing
    - Full-text search index (simulated)
    """

    def __init__(self) -> None:
        # Core graph storage
        self._concepts: dict[str, Concept] = {}
        self._edges: list[ConceptRelationship] = []

        # Extended node types
        self._authors: dict[str, Author] = {}
        self._domains: dict[str, Domain] = {}

        # Extended edge types
        self._isomorphisms: list[IsomorphicMapping] = []
        self._teaching_links: list[TeachingLink] = []

        # Auditing
        self._changelog: list[ChangelogEntry] = []

        # Search index (simulated full-text)
        self._search_index: dict[str, str] = {}

    def create_concept(self, concept: Concept) -> Concept:
        """Create a new concept node."""
        if concept.id in self._concepts:
            raise ConceptGraphError(f"Concept already exists: {concept.id}")
        self._concepts[concept.id] = concept
        logger.info(f"Created concept: {concept.id} - {concept.title}")
        return concept

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get a concept by ID."""
        return self._concepts.get(concept_id)

    def update_concept(self, concept: Concept) -> Concept:
        """Update an existing concept."""
        if concept.id not in self._concepts:
            raise ConceptNotFoundError(f"Concept not found: {concept.id}")
        self._concepts[concept.id] = concept
        logger.info(f"Updated concept: {concept.id}")
        return concept

    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept and its edges."""
        if concept_id not in self._concepts:
            return False
        # Remove concept
        del self._concepts[concept_id]
        # Remove edges involving this concept
        self._edges = [
            e for e in self._edges
            if e.source_id != concept_id and e.target_id != concept_id
        ]
        logger.info(f"Deleted concept: {concept_id}")
        return True

    def add_prerequisite(self, source_id: str, target_id: str, strength: float = 1.0) -> bool:
        """
        Add a prerequisite relationship: source requires target.

        Raises CycleDetectedError if this would create a cycle.
        """
        if source_id not in self._concepts:
            raise ConceptNotFoundError(f"Source concept not found: {source_id}")
        if target_id not in self._concepts:
            raise ConceptNotFoundError(f"Target concept not found: {target_id}")

        # Check for cycle
        cycle = self.detect_cycle(source_id, target_id)
        if cycle:
            cycle_str = " → ".join(cycle)
            raise CycleDetectedError(f"Prerequisite cycle detected: {cycle_str}")

        # Add edge
        edge = ConceptRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type="PREREQUISITE",
            strength=strength,
        )
        self._edges.append(edge)

        # Update concept's prerequisite list
        self._concepts[source_id].prerequisite_ids.append(target_id)

        logger.info(f"Added prerequisite: {source_id} requires {target_id}")
        return True

    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        """Get direct prerequisites for a concept."""
        concept = self.get_concept(concept_id)
        if not concept:
            return []
        return [
            self._concepts[pid]
            for pid in concept.prerequisite_ids
            if pid in self._concepts
        ]

    def detect_cycle(self, source_id: str, target_id: str) -> list[str] | None:
        """
        Detect if adding source → target would create a cycle.

        Uses DFS to check if target can reach source.
        Returns the cycle path if found, None otherwise.
        """
        # If adding source → target, check if target can reach source
        visited: set[str] = set()
        path: list[str] = []

        def dfs(current: str, goal: str) -> bool:
            if current == goal:
                path.append(current)
                return True
            if current in visited:
                return False
            visited.add(current)
            path.append(current)

            concept = self._concepts.get(current)
            if concept:
                for prereq_id in concept.prerequisite_ids:
                    if dfs(prereq_id, goal):
                        return True

            path.pop()
            return False

        if dfs(target_id, source_id):
            # Cycle found: target → ... → source → target
            path.append(target_id)
            return path

        return None

    def get_all_prerequisites_recursive(
        self,
        concept_id: str,
        visited: set[str] | None = None
    ) -> list[Concept]:
        """
        Get all prerequisites recursively (transitive closure).

        Returns prerequisites in topological order (deepest first).
        """
        if visited is None:
            visited = set()

        if concept_id in visited:
            return []

        visited.add(concept_id)
        concept = self.get_concept(concept_id)
        if not concept:
            return []

        result: list[Concept] = []
        for prereq_id in concept.prerequisite_ids:
            # Recurse first (deepest first)
            result.extend(self.get_all_prerequisites_recursive(prereq_id, visited))
            prereq = self.get_concept(prereq_id)
            if prereq and prereq not in result:
                result.append(prereq)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Extended Operations (WO-KB-002)
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_concepts(self) -> list[Concept]:
        """Get all concepts in the graph."""
        return list(self._concepts.values())

    def create_concept_with_prereqs(
        self,
        title: str,
        domain: str,
        difficulty: float = 0.5,
        prerequisite_ids: list[str] | None = None
    ) -> Concept:
        """
        Create a concept with prerequisites in a single transaction.

        Returns the created concept with auto-generated ID.
        """
        concept_id = str(uuid.uuid4())
        concept = Concept(
            id=concept_id,
            title=title,
            domain=domain,
            difficulty=difficulty,
            prerequisite_ids=[],
            created_at=datetime.utcnow()
        )

        self.create_concept(concept)

        # Add prerequisites
        if prerequisite_ids:
            for prereq_id in prerequisite_ids:
                try:
                    self.add_prerequisite(concept_id, prereq_id)
                except (ConceptNotFoundError, CycleDetectedError) as e:
                    # Rollback
                    self.delete_concept(concept_id)
                    raise

        # Record in changelog
        self._record_change(
            event_type="NODE_CREATED",
            entity_id=concept_id,
            entity_type="Concept",
            details={"title": title, "domain": domain}
        )

        # Index for full-text search
        self._index_concept(concept)

        return self._concepts[concept_id]

    def update_concept_properties(
        self,
        concept_id: str,
        updates: dict[str, Any]
    ) -> Concept:
        """
        Update concept properties and record changes.

        Args:
            concept_id: Concept to update
            updates: Dict of property name -> new value

        Returns:
            Updated concept
        """
        concept = self.get_concept(concept_id)
        if not concept:
            raise ConceptNotFoundError(f"Concept not found: {concept_id}")

        old_values = {}
        for field, new_value in updates.items():
            if hasattr(concept, field):
                old_values[field] = getattr(concept, field)
                setattr(concept, field, new_value)

        concept.updated_at = datetime.utcnow()
        self._concepts[concept_id] = concept

        # Record changes
        for field, old_value in old_values.items():
            self._record_change(
                event_type="PROPERTY_UPDATE",
                entity_id=concept_id,
                entity_type="Concept",
                details={
                    "field": field,
                    "old_value": old_value,
                    "new_value": updates[field]
                }
            )

        logger.info(f"Updated concept: {concept_id}, fields: {list(updates.keys())}")
        return concept

    def safe_delete_concept(self, concept_id: str) -> dict[str, Any]:
        """
        Attempt safe deletion with orphan prevention.

        Returns status dict with:
        - blocking_prereqs: Count of concepts that require this as prereq
        - orphaned_atoms: List of atoms that teach only this concept
        - can_delete: Whether deletion is safe

        Raises DeletionBlockedError if atoms would be orphaned.
        """
        concept = self.get_concept(concept_id)
        if not concept:
            raise ConceptNotFoundError(f"Concept not found: {concept_id}")

        # Check incoming prerequisites
        blocking_prereqs = 0
        for other in self._concepts.values():
            if concept_id in other.prerequisite_ids:
                blocking_prereqs += 1

        # Check teaching links (atoms that teach this concept)
        orphaned_atoms = self._get_teaching_atoms(concept_id)

        result = {
            "blocking_prereqs": blocking_prereqs,
            "orphaned_atoms": orphaned_atoms,
            "can_delete": blocking_prereqs == 0 and len(orphaned_atoms) == 0
        }

        if orphaned_atoms:
            raise DeletionBlockedError(
                f"{len(orphaned_atoms)} atoms reference this concept",
                blocking_atoms=orphaned_atoms
            )

        if blocking_prereqs > 0:
            raise DeletionBlockedError(
                f"{blocking_prereqs} concepts require this as a prerequisite"
            )

        # Safe to delete
        self.delete_concept(concept_id)
        self._record_change(
            event_type="NODE_DELETED",
            entity_id=concept_id,
            entity_type="Concept",
            details={"title": concept.title}
        )

        return result

    def get_concept_with_expansion(self, concept_id: str) -> dict[str, Any] | None:
        """
        Get concept with full relationship expansion.

        Returns dict with:
        - core_properties: id, title, domain, difficulty
        - prerequisites: list of prereq concepts with strength
        - teaching_atoms: atoms grouped by ICAP mode
        - isomorphisms: cross-domain mappings
        """
        concept = self.get_concept(concept_id)
        if not concept:
            return None

        prereqs = self.get_prerequisites(concept_id)
        prereq_data = [
            {
                "id": p.id,
                "title": p.title,
                "strength": self._get_prereq_strength(concept_id, p.id)
            }
            for p in prereqs
        ]

        # Get teaching atoms grouped by ICAP
        teaching_atoms = self._get_teaching_atoms_by_icap(concept_id)

        # Get isomorphisms
        isomorphisms = self._get_isomorphisms(concept_id)

        return {
            "core_properties": {
                "id": concept.id,
                "title": concept.title,
                "domain": concept.domain,
                "difficulty": concept.difficulty
            },
            "prerequisites": prereq_data,
            "teaching_atoms": teaching_atoms,
            "isomorphisms": isomorphisms
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Isomorphism Operations
    # ─────────────────────────────────────────────────────────────────────────

    def add_isomorphism(
        self,
        source_id: str,
        target_id: str,
        mapping_score: float,
        structural_mappings: dict[str, str] | None = None
    ) -> IsomorphicMapping:
        """
        Create an isomorphic mapping between concepts.

        Used for Analogical Transfer of Meaning (AToM).
        """
        if source_id not in self._concepts:
            raise ConceptNotFoundError(f"Source concept not found: {source_id}")
        if target_id not in self._concepts:
            raise ConceptNotFoundError(f"Target concept not found: {target_id}")

        mapping = IsomorphicMapping(
            source_concept_id=source_id,
            target_concept_id=target_id,
            mapping_score=mapping_score,
            structural_mappings=structural_mappings or {}
        )

        self._isomorphisms.append(mapping)

        # Update concept's isomorphic_to list
        self._concepts[source_id].isomorphic_to.append(target_id)

        self._record_change(
            event_type="EDGE_CREATED",
            entity_id=f"{source_id}->{target_id}",
            entity_type="ISOMORPHIC_TO",
            details={"mapping_score": mapping_score}
        )

        return mapping

    def find_transfer_path(
        self,
        source_concept_id: str,
        target_concept_id: str,
        max_depth: int = 5
    ) -> list[dict[str, Any]] | None:
        """
        Find a transfer learning path via isomorphisms.

        Returns path from mastered source to target via isomorphic links.
        """
        if source_concept_id not in self._concepts:
            return None
        if target_concept_id not in self._concepts:
            return None

        # BFS for shortest path through isomorphisms and prerequisites
        visited: set[str] = set()
        queue: list[tuple[str, list[dict[str, Any]]]] = [
            (source_concept_id, [{
                "concept": source_concept_id,
                "domain": self._concepts[source_concept_id].domain,
                "isomorphism_to": None
            }])
        ]

        while queue:
            current_id, path = queue.pop(0)
            if current_id == target_concept_id:
                return path

            if current_id in visited:
                continue
            visited.add(current_id)

            current = self._concepts.get(current_id)
            if not current:
                continue

            # Explore isomorphic neighbors
            for iso in self._isomorphisms:
                if iso.source_concept_id == current_id:
                    neighbor_id = iso.target_concept_id
                    if neighbor_id not in visited:
                        neighbor = self._concepts.get(neighbor_id)
                        if neighbor:
                            new_path = path + [{
                                "concept": neighbor_id,
                                "domain": neighbor.domain,
                                "isomorphism_to": current_id
                            }]
                            queue.append((neighbor_id, new_path))

            # Explore concepts that have this as prereq (reverse direction)
            for other_id, other in self._concepts.items():
                if current_id in other.prerequisite_ids and other_id not in visited:
                    new_path = path + [{
                        "concept": other_id,
                        "domain": other.domain,
                        "isomorphism_to": None
                    }]
                    queue.append((other_id, new_path))

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Teaching Links and Coverage
    # ─────────────────────────────────────────────────────────────────────────

    def link_atom_to_concept(
        self,
        atom_id: str,
        concept_id: str,
        coverage: float = 1.0,
        mode: ICAPMode | None = None,
        aspects_covered: list[str] | None = None
    ) -> TeachingLink:
        """
        Create a TEACHES edge from atom to concept.
        """
        if concept_id not in self._concepts:
            raise ConceptNotFoundError(f"Concept not found: {concept_id}")

        link = TeachingLink(
            atom_id=atom_id,
            concept_id=concept_id,
            coverage=coverage,
            mode=mode,
            aspects_covered=aspects_covered or []
        )

        self._teaching_links.append(link)

        # Update concept's teaching atom list
        if atom_id not in self._concepts[concept_id].teaching_atom_ids:
            self._concepts[concept_id].teaching_atom_ids.append(atom_id)

        self._record_change(
            event_type="EDGE_CREATED",
            entity_id=f"{atom_id}->{concept_id}",
            entity_type="TEACHES",
            details={"coverage": coverage, "mode": mode.value if mode else None}
        )

        return link

    def compute_concept_coverage(
        self,
        concept_id: str,
        required_aspects: dict[str, float]
    ) -> ConceptCoverage:
        """
        Compute coverage for a concept based on teaching links.
        """
        coverage = ConceptCoverage(
            concept_id=concept_id,
            required_aspects=required_aspects,
            actual_coverage={}
        )

        # Aggregate coverage from all teaching links
        for link in self._teaching_links:
            if link.concept_id == concept_id:
                for aspect in link.aspects_covered:
                    current = coverage.actual_coverage.get(aspect, 0.0)
                    coverage.actual_coverage[aspect] = current + link.coverage

        return coverage

    def analyze_icap_distribution(self, concept_id: str) -> ICAPDistribution:
        """
        Analyze ICAP mode distribution for a concept's atoms.
        """
        distribution = ICAPDistribution(
            concept_id=concept_id,
            counts={mode: 0 for mode in ICAPMode}
        )

        for link in self._teaching_links:
            if link.concept_id == concept_id and link.mode:
                distribution.counts[link.mode] = distribution.counts.get(link.mode, 0) + 1

        return distribution

    # ─────────────────────────────────────────────────────────────────────────
    # Author and Domain Operations
    # ─────────────────────────────────────────────────────────────────────────

    def create_author(self, author: Author) -> Author:
        """Create an author node."""
        if author.id in self._authors:
            raise ConceptGraphError(f"Author already exists: {author.id}")
        self._authors[author.id] = author
        return author

    def get_author(self, author_id: str) -> Author | None:
        """Get an author by ID."""
        return self._authors.get(author_id)

    def create_domain(self, domain: Domain) -> Domain:
        """Create a domain node."""
        if domain.id in self._domains:
            raise ConceptGraphError(f"Domain already exists: {domain.id}")
        self._domains[domain.id] = domain
        return domain

    def get_domain(self, domain_id: str) -> Domain | None:
        """Get a domain by ID."""
        return self._domains.get(domain_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Batch Operations
    # ─────────────────────────────────────────────────────────────────────────

    def batch_create_concepts(
        self,
        concepts: list[dict[str, Any]],
        batch_size: int = 50
    ) -> list[Concept]:
        """
        Efficiently create multiple concepts in batches.

        Each dict should have: title, domain, difficulty, prerequisites (optional)
        """
        created = []
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]
            for concept_data in batch:
                concept = self.create_concept_with_prereqs(
                    title=concept_data["title"],
                    domain=concept_data["domain"],
                    difficulty=concept_data.get("difficulty", 0.5),
                    prerequisite_ids=concept_data.get("prerequisites", [])
                )
                created.append(concept)

        logger.info(f"Batch created {len(created)} concepts")
        return created

    def batch_create_edges(
        self,
        edges: list[dict[str, Any]],
        batch_size: int = 50
    ) -> int:
        """
        Efficiently create multiple edges in batches.

        Each dict should have: source_id, target_id, edge_type, properties
        """
        created_count = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            for edge_data in batch:
                edge_type = edge_data.get("edge_type", EdgeType.PREREQUISITE)
                if edge_type == EdgeType.PREREQUISITE:
                    self.add_prerequisite(
                        edge_data["source_id"],
                        edge_data["target_id"],
                        edge_data.get("strength", 1.0)
                    )
                elif edge_type == EdgeType.ISOMORPHIC_TO:
                    self.add_isomorphism(
                        edge_data["source_id"],
                        edge_data["target_id"],
                        edge_data.get("mapping_score", 0.5)
                    )
                created_count += 1

        logger.info(f"Batch created {created_count} edges")
        return created_count

    # ─────────────────────────────────────────────────────────────────────────
    # Changelog and Caching
    # ─────────────────────────────────────────────────────────────────────────

    def get_changelog(
        self,
        entity_id: str | None = None,
        limit: int = 100
    ) -> list[ChangelogEntry]:
        """Get changelog entries, optionally filtered by entity."""
        entries = self._changelog
        if entity_id:
            entries = [e for e in entries if e.entity_id == entity_id]
        return entries[-limit:]

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _record_change(
        self,
        event_type: str,
        entity_id: str,
        entity_type: str,
        details: dict[str, Any]
    ) -> None:
        """Record a change in the changelog."""
        entry = ChangelogEntry(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            entity_id=entity_id,
            entity_type=entity_type,
            details=details
        )
        self._changelog.append(entry)

    def _index_concept(self, concept: Concept) -> None:
        """Index concept for full-text search (simulated)."""
        # In production, this would use Elasticsearch or similar
        search_text = f"{concept.title} {concept.domain}".lower()
        self._search_index[concept.id] = search_text

    def search_concepts(self, query: str, limit: int = 10) -> list[Concept]:
        """Full-text search across concepts."""
        query_lower = query.lower()
        matches = []
        for concept_id, indexed_text in self._search_index.items():
            if query_lower in indexed_text:
                concept = self._concepts.get(concept_id)
                if concept:
                    matches.append(concept)
                if len(matches) >= limit:
                    break
        return matches

    def _get_prereq_strength(self, source_id: str, target_id: str) -> float:
        """Get strength of prerequisite edge."""
        for edge in self._edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge.strength
        return 1.0

    def _get_teaching_atoms(self, concept_id: str) -> list[str]:
        """Get atom IDs that teach a concept."""
        return [
            link.atom_id for link in self._teaching_links
            if link.concept_id == concept_id
        ]

    def _get_teaching_atoms_by_icap(self, concept_id: str) -> dict[str, list[str]]:
        """Get teaching atoms grouped by ICAP mode."""
        result: dict[str, list[str]] = {}
        for mode in ICAPMode:
            result[mode.value] = []

        for link in self._teaching_links:
            if link.concept_id == concept_id and link.mode:
                result[link.mode.value].append(link.atom_id)

        return result

    def _get_isomorphisms(self, concept_id: str) -> list[dict[str, Any]]:
        """Get isomorphic mappings for a concept."""
        return [
            {
                "target_id": iso.target_concept_id,
                "mapping_score": iso.mapping_score,
                "structural_mappings": iso.structural_mappings
            }
            for iso in self._isomorphisms
            if iso.source_concept_id == concept_id
        ]


class Neo4jConceptGraph:
    """
    Neo4j-backed concept graph.

    Connection from: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None
    ) -> None:
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD")
        if not self._password:
            raise ConceptGraphError(
                "Neo4j password required. Set NEO4J_PASSWORD env var."
            )
        self._driver = None

    def _get_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password)
                )
                logger.info("Neo4j connection initialized")
            except ImportError:
                raise ConceptGraphError(
                    "neo4j driver not installed. Run: pip install neo4j"
                )
        return self._driver

    def create_concept(self, concept: Concept) -> Concept:
        """
        Create a concept node in Neo4j.

        Cypher:
            CREATE (c:Concept {id: $id, title: $title, domain: $domain})
            RETURN c
        """
        # TODO: Implement with Neo4j driver
        raise NotImplementedError("Neo4j graph not yet implemented")

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get a concept from Neo4j."""
        # TODO: Implement
        raise NotImplementedError("Neo4j graph not yet implemented")

    def update_concept(self, concept: Concept) -> Concept:
        """Update a concept in Neo4j."""
        # TODO: Implement
        raise NotImplementedError("Neo4j graph not yet implemented")

    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept and its relationships."""
        # TODO: Implement
        raise NotImplementedError("Neo4j graph not yet implemented")

    def add_prerequisite(self, source_id: str, target_id: str, strength: float = 1.0) -> bool:
        """
        Add prerequisite edge.

        Cypher:
            MATCH (s:Concept {id: $source}), (t:Concept {id: $target})
            CREATE (s)-[:PREREQUISITE {strength: $strength}]->(t)
        """
        # TODO: Implement
        raise NotImplementedError("Neo4j graph not yet implemented")

    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        """Get direct prerequisites."""
        # TODO: Implement
        raise NotImplementedError("Neo4j graph not yet implemented")

    def detect_cycle(self, source_id: str, target_id: str) -> list[str] | None:
        """
        Detect cycle using Cypher path query.

        Cypher:
            MATCH path = (t:Concept {id: $target})-[:PREREQUISITE*]->(s:Concept {id: $source})
            RETURN [n IN nodes(path) | n.id] as cycle
            LIMIT 1
        """
        # TODO: Implement
        raise NotImplementedError("Neo4j graph not yet implemented")
