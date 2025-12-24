"""
Unified repository interface for the Knowledge Base.

Provides a single facade for:
- Atom CRUD (PostgreSQL)
- Vector search (pgvector)
- Concept graph (Neo4j)
- Prerequisite traversal

This is the main entry point for Knowledge Base operations.
"""

import logging
import os
from typing import Any

from .models import Atom, AtomLayer, Concept
from .atom_store import AtomStore, InMemoryAtomStore, PostgresAtomStore
from .vector_store import VectorStore, InMemoryVectorStore, PgVectorStore, SearchResult
from .concept_graph import ConceptGraph, InMemoryConceptGraph, Neo4jConceptGraph
from .prerequisite import PrerequisiteService, MasteryStatus, LearningPath


logger = logging.getLogger(__name__)


class KnowledgeBaseRepository:
    """
    Unified Knowledge Base repository.

    Coordinates operations across:
    - PostgreSQL (atoms)
    - pgvector (embeddings)
    - Neo4j (concept graph)

    Configuration via environment variables:
    - ASTARTES_KB_CONNECTION: PostgreSQL connection string
    - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD: Neo4j connection
    - ASTARTES_KB_MODE: 'memory' for testing, 'production' for real DBs
    """

    def __init__(
        self,
        atom_store: AtomStore | None = None,
        vector_store: VectorStore | None = None,
        concept_graph: ConceptGraph | None = None,
        mode: str | None = None
    ) -> None:
        """
        Initialize the repository.

        Args:
            atom_store: Override atom storage backend
            vector_store: Override vector storage backend
            concept_graph: Override concept graph backend
            mode: 'memory' for testing, 'production' for real DBs
        """
        self._mode = mode or os.getenv("ASTARTES_KB_MODE", "memory")

        if self._mode == "production":
            self._atom_store = atom_store or PostgresAtomStore()
            self._vector_store = vector_store or PgVectorStore()
            self._concept_graph = concept_graph or Neo4jConceptGraph()
        else:
            # Memory mode for testing
            self._atom_store = atom_store or InMemoryAtomStore()
            self._vector_store = vector_store or InMemoryVectorStore()
            self._concept_graph = concept_graph or InMemoryConceptGraph()

        self._prerequisite_service = PrerequisiteService(
            atom_store=self._atom_store,
            concept_graph=self._concept_graph
        )

        logger.info(f"KnowledgeBaseRepository initialized in '{self._mode}' mode")

    # ─────────────────────────────────────────────────────────────────────────
    # Atom Operations
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_atom(self, atom: Atom) -> Atom:
        """
        Ingest an atom into the knowledge base.

        This:
        1. Stores the atom in PostgreSQL
        2. Indexes the embedding in pgvector (if present)
        3. Creates/updates concept links
        """
        # Store atom
        created = self._atom_store.create(atom)

        # Index embedding if present
        if atom.embedding:
            self._vector_store.store_embedding(
                atom_id=atom.id,
                embedding=atom.embedding,
                model=atom.embedding_model or "unknown"
            )

        logger.info(f"Ingested atom: {atom.id} ({atom.layer.value})")
        return created

    def get_atom(self, atom_id: str) -> Atom | None:
        """Get an atom by ID."""
        return self._atom_store.get(atom_id)

    def update_atom(self, atom: Atom) -> Atom:
        """Update an existing atom."""
        updated = self._atom_store.update(atom)

        # Re-index embedding if changed
        if atom.embedding:
            self._vector_store.store_embedding(
                atom_id=atom.id,
                embedding=atom.embedding,
                model=atom.embedding_model or "unknown"
            )

        return updated

    def delete_atom(self, atom_id: str) -> bool:
        """Delete an atom and its embedding."""
        self._vector_store.delete_embedding(atom_id)
        return self._atom_store.delete(atom_id)

    def enrich_to_platinum(self, atom_id: str, enrichment: dict[str, Any]) -> Atom | None:
        """Upgrade an atom to Platinum layer with enriched metadata."""
        atom = self.get_atom(atom_id)
        if not atom:
            return None

        atom.layer = AtomLayer.PLATINUM
        atom.metadata.update(enrichment)
        return self.update_atom(atom)

    # ─────────────────────────────────────────────────────────────────────────
    # Vector Search
    # ─────────────────────────────────────────────────────────────────────────

    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> list[Atom]:
        """
        Find semantically similar atoms.

        Returns atoms sorted by similarity (highest first).
        """
        results = self._vector_store.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            min_similarity=min_similarity
        )

        # Hydrate with full atom data
        atoms = []
        for result in results:
            atom = self._atom_store.get(result.atom_id)
            if atom:
                atoms.append(atom)

        return atoms

    # ─────────────────────────────────────────────────────────────────────────
    # Concept Graph
    # ─────────────────────────────────────────────────────────────────────────

    def create_concept(self, concept: Concept) -> Concept:
        """Create a new concept in the graph."""
        return self._concept_graph.create_concept(concept)

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get a concept by ID."""
        return self._concept_graph.get_concept(concept_id)

    def add_prerequisite(
        self,
        source_id: str,
        target_id: str,
        strength: float = 1.0
    ) -> bool:
        """Add a prerequisite relationship."""
        return self._concept_graph.add_prerequisite(source_id, target_id, strength)

    # ─────────────────────────────────────────────────────────────────────────
    # Prerequisite Traversal
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_prerequisites(self, atom_id: str) -> list[Atom]:
        """Get all prerequisites for an atom (recursive)."""
        return self._prerequisite_service.get_all_prerequisites(atom_id)

    def find_learning_path(
        self,
        target_concept_id: str,
        mastery_lookup: dict[str, MasteryStatus] | None = None
    ) -> LearningPath | None:
        """Find optimal learning path to a target concept."""
        return self._prerequisite_service.find_learning_path(
            target_concept_id,
            mastery_lookup
        )
