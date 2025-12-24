"""
Vector store for semantic search using pgvector.

Implements:
- Embedding storage with pgvector
- k-NN similarity search
- HNSW index configuration
- Hybrid search (graph + vector)

Scientific Foundation:
- Approximate Nearest Neighbors (Malkov & Yashunin, 2018)
- HNSW algorithm for sublinear search
"""

import logging
import os
from dataclasses import dataclass
from typing import Protocol

from .models import Atom


logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


@dataclass
class SearchResult:
    """Result from a vector similarity search."""
    atom_id: str
    similarity: float  # Cosine similarity, 0.0 to 1.0
    atom: Atom | None = None


@dataclass
class HNSWConfig:
    """HNSW index configuration per Gherkin spec."""
    m: int = 16           # Max connections per node
    ef_construct: int = 64  # Construction-time search width
    ef_search: int = 40     # Query-time search width


class VectorStore(Protocol):
    """Protocol for vector storage backends."""

    def store_embedding(self, atom_id: str, embedding: list[float], model: str) -> bool: ...
    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> list[SearchResult]: ...
    def delete_embedding(self, atom_id: str) -> bool: ...


class InMemoryVectorStore:
    """
    In-memory vector store for testing.

    Uses brute-force cosine similarity (O(n) per query).
    Not suitable for production - use PgVectorStore instead.
    """

    def __init__(self) -> None:
        self._embeddings: dict[str, tuple[list[float], str]] = {}  # atom_id -> (vector, model)

    def store_embedding(self, atom_id: str, embedding: list[float], model: str) -> bool:
        """Store an embedding for an atom."""
        if not embedding:
            return False
        self._embeddings[atom_id] = (embedding, model)
        logger.debug(f"Stored embedding for {atom_id} (dim={len(embedding)}, model={model})")
        return True

    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> list[SearchResult]:
        """Find k-nearest neighbors using cosine similarity."""
        results: list[SearchResult] = []

        for atom_id, (embedding, _) in self._embeddings.items():
            similarity = self._cosine_similarity(query_vector, embedding)
            if similarity >= min_similarity:
                results.append(SearchResult(atom_id=atom_id, similarity=similarity))

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def delete_embedding(self, atom_id: str) -> bool:
        """Delete an embedding."""
        if atom_id in self._embeddings:
            del self._embeddings[atom_id]
            return True
        return False

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class PgVectorStore:
    """
    PostgreSQL + pgvector storage.

    Features:
    - HNSW indexing for fast approximate search
    - Configurable vector dimensions (768, 1024, 1536)
    - Metadata filtering

    Connection from: ASTARTES_KB_CONNECTION env var
    """

    # Supported embedding dimensions
    SUPPORTED_DIMS = {768, 1024, 1536}

    def __init__(
        self,
        connection_string: str | None = None,
        hnsw_config: HNSWConfig | None = None
    ) -> None:
        self._connection_string = connection_string or os.getenv("ASTARTES_KB_CONNECTION")
        if not self._connection_string:
            raise VectorStoreError(
                "Database connection required. Set ASTARTES_KB_CONNECTION env var."
            )
        self._hnsw_config = hnsw_config or HNSWConfig()
        self._engine = None

    def _get_engine(self):
        """Lazy initialization of SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self._connection_string)
                logger.info("pgvector connection initialized")
            except ImportError:
                raise VectorStoreError(
                    "SQLAlchemy not installed. Run: pip install sqlalchemy pgvector"
                )
        return self._engine

    def store_embedding(self, atom_id: str, embedding: list[float], model: str) -> bool:
        """Store an embedding in pgvector."""
        # TODO: Implement with pgvector extension
        # SQL: INSERT INTO atom_embeddings (atom_id, embedding, model)
        #      VALUES ($1, $2::vector, $3)
        raise NotImplementedError("pgvector store not yet implemented")

    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> list[SearchResult]:
        """
        Find similar atoms using HNSW index.

        SQL:
            SELECT atom_id, 1 - (embedding <=> $1::vector) as similarity
            FROM atom_embeddings
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        # TODO: Implement with pgvector
        raise NotImplementedError("pgvector store not yet implemented")

    def delete_embedding(self, atom_id: str) -> bool:
        """Delete an embedding from pgvector."""
        # TODO: Implement
        raise NotImplementedError("pgvector store not yet implemented")

    def create_hnsw_index(self, vector_dimension: int) -> None:
        """
        Create HNSW index for fast approximate search.

        SQL:
            CREATE INDEX ON atom_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """
        if vector_dimension not in self.SUPPORTED_DIMS:
            raise VectorStoreError(
                f"Unsupported dimension: {vector_dimension}. "
                f"Supported: {self.SUPPORTED_DIMS}"
            )
        # TODO: Implement index creation
        raise NotImplementedError("Index creation not yet implemented")
