"""
Knowledge Base Shared Library.

Provides storage and retrieval of Platinum-enriched learning atoms,
prerequisites, analogies, and evidence relationships.

This library abstracts storage backends (PostgreSQL, KuzuDB, etc.)
and provides a unified interface for knowledge graph operations.
"""

from .models import (
    KnowledgeItem,
    KnowledgeEdge,
    PrerequisiteLink,
    AnalogicalBridge,
    LearningAtom,
    PlatinumAtom,
)

from .storage import (
    StorageBackend,
    GraphStorage,
    VectorStorage,
)

from .services import (
    KnowledgeService,
    PrerequisiteService,
    AnalogyService,
    SearchService,
)

__all__ = [
    # Models
    "KnowledgeItem",
    "KnowledgeEdge",
    "PrerequisiteLink",
    "AnalogicalBridge",
    "LearningAtom",
    "PlatinumAtom",
    # Storage
    "StorageBackend",
    "GraphStorage",
    "VectorStorage",
    # Services
    "KnowledgeService",
    "PrerequisiteService",
    "AnalogyService",
    "SearchService",
]
