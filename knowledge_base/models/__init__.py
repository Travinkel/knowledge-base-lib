"""
Knowledge Base Domain Models.

These are storage-agnostic Pydantic models representing
the core entities in the knowledge graph.
"""

from .base import (
    KnowledgeItem,
    KnowledgeEdge,
    LearningAtom,
    KnowledgeType,
    StudyType,
    ICAPLevel,
    EdgePredicate,
)

from .platinum import (
    PrerequisiteLink,
    PrerequisiteStrength,
    AnalogicalBridge,
    BridgeType,
    PlatinumAtom,
)

__all__ = [
    # Base models
    "KnowledgeItem",
    "KnowledgeEdge",
    "LearningAtom",
    # Enums
    "KnowledgeType",
    "StudyType",
    "ICAPLevel",
    "EdgePredicate",
    # Platinum models
    "PrerequisiteLink",
    "PrerequisiteStrength",
    "AnalogicalBridge",
    "BridgeType",
    "PlatinumAtom",
]
