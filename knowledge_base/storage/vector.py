"""
Vector storage interface for Knowledge Base.

Provides operations for storing and querying vector embeddings
for semantic search and similarity matching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base import StorageBackend


@dataclass
class EmbeddingResult:
    """Result from embedding storage operation."""
    entity_id: UUID
    success: bool
    dimension: int
    model: str


@dataclass
class SimilarityResult:
    """Result from a similarity search."""
    entity_id: UUID
    similarity: float
    entity_type: str = "atom"
    metadata: Optional[Dict[str, Any]] = None


class VectorStorage(StorageBackend, ABC):
    """
    Abstract interface for vector storage operations.

    Implementations may use pgvector, Pinecone, Qdrant, etc.
    """

    # =========================================================
    # EMBEDDING STORAGE
    # =========================================================

    @abstractmethod
    async def store_embedding(
        self,
        entity_id: UUID,
        embedding: List[float],
        entity_type: str = "atom",
        model: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Store a vector embedding for an entity."""
        pass

    @abstractmethod
    async def get_embedding(
        self,
        entity_id: UUID,
    ) -> Optional[List[float]]:
        """Get the embedding for an entity."""
        pass

    @abstractmethod
    async def delete_embedding(
        self,
        entity_id: UUID,
    ) -> bool:
        """Delete an embedding."""
        pass

    @abstractmethod
    async def batch_store_embeddings(
        self,
        embeddings: List[tuple[UUID, List[float], str]],  # (id, vector, type)
        model: str = "unknown",
    ) -> List[EmbeddingResult]:
        """Batch store multiple embeddings."""
        pass

    # =========================================================
    # SIMILARITY SEARCH
    # =========================================================

    @abstractmethod
    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
        entity_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SimilarityResult]:
        """Find k-nearest neighbors by vector similarity."""
        pass

    @abstractmethod
    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        entity_type: Optional[str] = None,
    ) -> List[SimilarityResult]:
        """Search by text (uses configured embedding model)."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,  # Balance: 0 = pure vector, 1 = pure text
    ) -> List[SimilarityResult]:
        """Hybrid search combining vector and keyword matching."""
        pass

    # =========================================================
    # INDEX MANAGEMENT
    # =========================================================

    @abstractmethod
    async def create_index(
        self,
        dimension: int,
        index_type: str = "hnsw",
        **kwargs,
    ) -> bool:
        """Create a vector index."""
        pass

    @abstractmethod
    async def rebuild_index(self) -> bool:
        """Rebuild the vector index."""
        pass

    @abstractmethod
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        pass

    # =========================================================
    # EMBEDDING MODEL CONFIGURATION
    # =========================================================

    @abstractmethod
    async def configure_embedding_model(
        self,
        model_name: str,
        dimension: int,
        provider: str = "openai",
    ) -> bool:
        """Configure the embedding model for text-to-vector conversion."""
        pass

    @abstractmethod
    async def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get information about the configured embedding model."""
        pass
