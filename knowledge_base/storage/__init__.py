"""
Storage backends for Knowledge Base.

Provides abstract interfaces and concrete implementations
for graph storage and vector search.
"""

from .base import StorageBackend
from .graph import GraphStorage
from .vector import VectorStorage, EmbeddingResult, SimilarityResult

__all__ = [
    "StorageBackend",
    "GraphStorage",
    "VectorStorage",
    "EmbeddingResult",
    "SimilarityResult",
]
