"""
Knowledge Base Core Module.

This module provides the core infrastructure for the Astartes Knowledge Base:
- Atom storage (PostgreSQL)
- Vector search (pgvector)
- Concept graph (Neo4j)
- Prerequisite traversal

Quick Start:
    from knowledge_base.core import KnowledgeBaseRepository, Atom, AtomLayer

    # Initialize (uses in-memory stores by default)
    kb = KnowledgeBaseRepository()

    # Create an atom
    atom = Atom(
        id="atom-001",
        layer=AtomLayer.GOLD,
        content="What is the time complexity of binary search?",
        metadata={"type": "mcq", "difficulty": 0.65}
    )
    kb.ingest_atom(atom)

    # Retrieve
    retrieved = kb.get_atom("atom-001")

For production, set environment variables:
    ASTARTES_KB_MODE=production
    ASTARTES_KB_CONNECTION=postgresql://...
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=...
"""

# Models
from .models import (
    Atom,
    AtomLayer,
    Concept,
    ConceptRelationship,
    ICAPMode,
    EdgeType,
    Author,
    Domain,
    IsomorphicMapping,
    TeachingLink,
    ConceptCoverage,
    ICAPDistribution,
    ChangelogEntry,
)

# Atom Store
from .atom_store import (
    AtomStore,
    AtomStoreError,
    AtomNotFoundError,
    AtomValidationError,
    InMemoryAtomStore,
    PostgresAtomStore,
)

# Vector Store
from .vector_store import (
    VectorStore,
    VectorStoreError,
    SearchResult,
    HNSWConfig,
    InMemoryVectorStore,
    PgVectorStore,
)

# Concept Graph
from .concept_graph import (
    ConceptGraph,
    ConceptGraphError,
    CycleDetectedError,
    ConceptNotFoundError,
    DeletionBlockedError,
    OrphanPreventionError,
    InMemoryConceptGraph,
    Neo4jConceptGraph,
)

# Graph Analytics
from .graph_analytics import (
    GraphAnalytics,
    CentralityResult,
    ConceptCluster,
    StructuralGap,
    PrerequisiteInference,
)

# Prerequisite Service
from .prerequisite import (
    PrerequisiteService,
    PrerequisiteNode,
    LearningPath,
    MasteryStatus,
)

# Repository (main entry point)
from .repository import KnowledgeBaseRepository

# Platinum Layer (research evidence integration)
from .platinum import (
    EvidenceQuality,
    ResearchDomain,
    PlatinumMetadata,
    EvidenceQuery,
    HyperparameterFeedback,
    EvidenceSynthesis,
)

# Feedback Store (Bayesian hyperparameter updates)
from .feedback_store import (
    FeedbackStore,
    FeedbackStoreError,
    FeedbackNotFoundError,
    PriorUpdateError,
    BayesianPrior,
    PriorUpdateResult,
    SQLiteFeedbackStore,
    InMemoryFeedbackStore,
)


__all__ = [
    # Models
    "Atom",
    "AtomLayer",
    "Concept",
    "ConceptRelationship",
    "ICAPMode",
    "EdgeType",
    "Author",
    "Domain",
    "IsomorphicMapping",
    "TeachingLink",
    "ConceptCoverage",
    "ICAPDistribution",
    "ChangelogEntry",
    # Atom Store
    "AtomStore",
    "AtomStoreError",
    "AtomNotFoundError",
    "AtomValidationError",
    "InMemoryAtomStore",
    "PostgresAtomStore",
    # Vector Store
    "VectorStore",
    "VectorStoreError",
    "SearchResult",
    "HNSWConfig",
    "InMemoryVectorStore",
    "PgVectorStore",
    # Concept Graph
    "ConceptGraph",
    "ConceptGraphError",
    "CycleDetectedError",
    "ConceptNotFoundError",
    "DeletionBlockedError",
    "OrphanPreventionError",
    "InMemoryConceptGraph",
    "Neo4jConceptGraph",
    # Graph Analytics
    "GraphAnalytics",
    "CentralityResult",
    "ConceptCluster",
    "StructuralGap",
    "PrerequisiteInference",
    # Prerequisite Service
    "PrerequisiteService",
    "PrerequisiteNode",
    "LearningPath",
    "MasteryStatus",
    # Repository
    "KnowledgeBaseRepository",
    # Platinum Layer
    "EvidenceQuality",
    "ResearchDomain",
    "PlatinumMetadata",
    "EvidenceQuery",
    "HyperparameterFeedback",
    "EvidenceSynthesis",
    # Feedback Store
    "FeedbackStore",
    "FeedbackStoreError",
    "FeedbackNotFoundError",
    "PriorUpdateError",
    "BayesianPrior",
    "PriorUpdateResult",
    "SQLiteFeedbackStore",
    "InMemoryFeedbackStore",
]
