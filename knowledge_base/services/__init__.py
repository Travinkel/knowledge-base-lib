"""
Services layer for Knowledge Base.

Provides high-level operations that coordinate storage backends.
"""

from .knowledge_service import KnowledgeService
from .prerequisite_service import PrerequisiteService
from .analogy_service import AnalogyService
from .search_service import SearchService

__all__ = [
    "KnowledgeService",
    "PrerequisiteService",
    "AnalogyService",
    "SearchService",
]
