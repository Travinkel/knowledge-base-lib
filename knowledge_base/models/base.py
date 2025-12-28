"""
Base domain models for Knowledge Base.

These models are storage-agnostic and can be persisted to
PostgreSQL, KuzuDB, or other backends.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field


class KnowledgeType(str, Enum):
    """Types of knowledge items."""
    DECLARATIVE = "declarative"
    PROCEDURAL = "procedural"
    CONDITIONAL = "conditional"
    METACOGNITIVE = "metacognitive"


class StudyType(str, Enum):
    """Types of research studies."""
    RCT = "rct"
    META_ANALYSIS = "meta_analysis"
    REVIEW = "review"
    CASE_STUDY = "case_study"
    OBSERVATIONAL = "observational"
    QUALITATIVE = "qualitative"


class ICAPLevel(str, Enum):
    """ICAP framework engagement levels."""
    PASSIVE = "passive"
    ACTIVE = "active"
    CONSTRUCTIVE = "constructive"
    INTERACTIVE = "interactive"


class KnowledgeItem(BaseModel):
    """
    A knowledge item representing a validated piece of evidence.

    This is typically derived from academic papers, textbooks,
    or other authoritative sources.
    """

    id: UUID = Field(default_factory=uuid4)
    reference_id: str = Field(..., description="Stable reference (DOI, OpenAlex ID, etc.)")

    # Content
    title: Optional[str] = None
    citation: Optional[str] = None
    abstract: Optional[str] = None
    key_findings: Optional[str] = Field(None, description="RAG-friendly summary")

    # Classification
    domain: Optional[str] = None
    study_type: Optional[StudyType] = None
    knowledge_type: Optional[KnowledgeType] = None
    applicable_concepts: List[str] = Field(default_factory=list)

    # Quality metrics
    adjusted_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    validation_passed: bool = True
    withdrawal_status: Optional[str] = None

    # Provenance
    authors: List[str] = Field(default_factory=list)
    published_year: Optional[int] = None
    cited_by_count: Optional[int] = None

    # Metadata
    source_metadata: Optional[Dict[str, Any]] = None
    embedding_metadata: Optional[Dict[str, Any]] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class EdgePredicate(str, Enum):
    """Standard predicates for knowledge graph edges."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"
    PREREQUISITE_OF = "prerequisite_of"
    ANALOGOUS_TO = "analogous_to"
    CITES = "cites"
    CITED_BY = "cited_by"
    APPLIES_TO = "applies_to"


class KnowledgeEdge(BaseModel):
    """
    An edge in the knowledge graph.

    Connects knowledge items, concepts, or atoms.
    """

    id: UUID = Field(default_factory=uuid4)

    # Nodes
    subject_id: str = Field(..., description="Source node ID")
    subject_kind: str = Field(default="knowledge_item")
    predicate: str = Field(..., description="Relationship type")
    object_id: str = Field(..., description="Target node ID")
    object_kind: str = Field(default="concept")

    # Metadata
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Optional[str] = None
    source_link: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    # Back-reference
    knowledge_item_id: Optional[UUID] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class LearningAtom(BaseModel):
    """
    A learning atom - the atomic unit of learning content.

    This is the Gold layer representation, ready for
    presentation to learners.
    """

    id: UUID = Field(default_factory=uuid4)

    # Content
    atom_type: str = Field(..., description="Type: mcq, parsons, socratic, etc.")
    content: Dict[str, Any] = Field(default_factory=dict)

    # ICAP classification
    icap_level: ICAPLevel = ICAPLevel.ACTIVE

    # Difficulty and hints
    difficulty: Optional[float] = Field(None, ge=0.0, le=1.0)
    hints: List[str] = Field(default_factory=list)

    # Grading
    grading_logic: Optional[Dict[str, Any]] = None

    # Provenance
    source_reference: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)

    # Platinum enrichment reference
    platinum_id: Optional[UUID] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}
