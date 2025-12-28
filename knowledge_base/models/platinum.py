"""
Platinum layer models for AI-enriched knowledge.

These models represent the enriched knowledge items that
are ready for agent consumption and RAG pipelines.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .base import ICAPLevel, LearningAtom


class PrerequisiteStrength(str):
    """Strength of prerequisite relationship."""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class PrerequisiteLink(BaseModel):
    """
    A prerequisite relationship between atoms/concepts.

    Represents what must be learned before understanding
    the target concept.
    """

    id: UUID = Field(default_factory=uuid4)

    # Relationship
    from_atom_id: UUID = Field(..., description="The atom that requires prerequisites")
    to_atom_id: UUID = Field(..., description="The prerequisite atom")
    strength: str = Field(default="recommended", description="required|recommended|optional")

    # Metadata
    rationale: str = Field(default="", description="Why this is a prerequisite")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Inference metadata
    inferred_by: str = Field(default="manual", description="Agent or source that inferred this")
    inference_metadata: Optional[Dict[str, Any]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class BridgeType(str):
    """Type of analogical bridge."""
    STRUCTURAL = "structural"
    SURFACE = "surface"
    MIXED = "mixed"


class AnalogicalBridge(BaseModel):
    """
    An analogical relationship between concepts from different domains.

    Based on Structure-Mapping Theory (Gentner), tracks both
    the structural correspondences and surface similarities.
    """

    id: UUID = Field(default_factory=uuid4)

    # Relationship
    source_atom_id: UUID = Field(..., description="Source concept")
    target_atom_id: UUID = Field(..., description="Target (analogous) concept")
    bridge_type: str = Field(default="structural")

    # Similarity
    similarity_score: float = Field(..., ge=0.0, le=1.0)

    # Structure mapping
    structural_mappings: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {source: X, target: Y} mappings"
    )

    # Metadata
    rationale: str = Field(default="")
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None

    # Inference metadata
    inferred_by: str = Field(default="manual")
    inference_metadata: Optional[Dict[str, Any]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class PlatinumAtom(BaseModel):
    """
    A fully enriched Platinum layer atom.

    Contains the Gold learning atom plus all AI-enrichments:
    - Prerequisite relationships
    - Analogical bridges
    - Vector embeddings
    - Evidence links
    """

    id: UUID = Field(default_factory=uuid4)

    # Reference to Gold atom
    gold_atom_id: UUID = Field(...)
    gold_atom: Optional[LearningAtom] = None

    # Enrichments
    prerequisites: List[PrerequisiteLink] = Field(default_factory=list)
    analogies: List[AnalogicalBridge] = Field(default_factory=list)

    # Vector embedding for semantic search
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None

    # Evidence and provenance
    evidence_item_ids: List[UUID] = Field(default_factory=list)
    provenance_chain: List[str] = Field(
        default_factory=list,
        description="Chain of IDs from Platinum back to Bronze"
    )

    # Quality metrics
    enrichment_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    prereq_count: int = Field(default=0)
    analogy_count: int = Field(default=0)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    enriched_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    def compute_quality_score(self) -> float:
        """Compute overall quality score for this Platinum atom."""
        scores = []

        # Prerequisite coverage
        if self.prerequisites:
            prereq_conf = sum(p.confidence for p in self.prerequisites) / len(self.prerequisites)
            scores.append(prereq_conf)

        # Analogy quality
        if self.analogies:
            analogy_score = sum(a.similarity_score for a in self.analogies) / len(self.analogies)
            scores.append(analogy_score)

        # Embedding presence
        if self.embedding:
            scores.append(0.8)

        # Base enrichment confidence
        scores.append(self.enrichment_confidence)

        return sum(scores) / len(scores) if scores else 0.0
