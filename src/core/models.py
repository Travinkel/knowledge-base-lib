"""
Knowledge Base domain models.

Scientific Foundation:
- Semantic Knowledge Networks (Collins & Quillian, 1969)
- Spreading Activation Theory (Anderson, 1983)
- Analogical Transfer of Meaning (AToM) (Gentner, 1983)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AtomLayer(str, Enum):
    """Layer in the Bronze-to-Platinum pipeline."""
    BRONZE = "Bronze"      # Raw extracted content
    SILVER = "Silver"      # Normalized, structured
    GOLD = "Gold"          # Learning atoms (144 types)
    PLATINUM = "Platinum"  # Verified, cross-linked


class ICAPMode(str, Enum):
    """ICAP framework engagement modes (Chi & Wylie, 2014)."""
    PASSIVE = "Passive"          # Receiving information
    ACTIVE = "Active"            # Manipulating/selecting
    CONSTRUCTIVE = "Constructive" # Generating new outputs
    INTERACTIVE = "Interactive"   # Collaborative dialogue


class EdgeType(str, Enum):
    """Edge types in the knowledge graph."""
    PREREQUISITE = "PREREQUISITE"      # Concept → Concept
    TEACHES = "TEACHES"                # Atom → Concept
    AUTHORED_BY = "AUTHORED_BY"        # Atom → Author
    BELONGS_TO = "BELONGS_TO"          # Concept → Domain
    ISOMORPHIC_TO = "ISOMORPHIC_TO"    # Concept → Concept (cross-domain)


@dataclass
class Atom:
    """
    A learning atom - the atomic unit of knowledge.

    Atoms are the smallest independently assessable pieces of knowledge.
    They map to specific IRT difficulty parameters and ICAP engagement modes.
    """
    id: str
    layer: AtomLayer
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Vector embedding for semantic search
    embedding: list[float] | None = None
    embedding_model: str | None = None

    # Prerequisite relationships
    prerequisites: list[str] = field(default_factory=list)

    # ICAP classification
    icap_mode: ICAPMode | None = None

    # IRT parameters (Item Response Theory)
    irt_difficulty: float | None = None      # b parameter
    irt_discrimination: float | None = None  # a parameter
    irt_guessing: float | None = None        # c parameter (for MCQ)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None


@dataclass
class Concept:
    """
    A concept node in the knowledge graph.

    Concepts are abstract ideas that atoms teach.
    They form a DAG with prerequisite relationships.
    """
    id: str
    title: str
    domain: str
    difficulty: float = 0.5  # 0.0 to 1.0

    # Graph relationships
    prerequisite_ids: list[str] = field(default_factory=list)

    # Atoms that teach this concept
    teaching_atom_ids: list[str] = field(default_factory=list)

    # Cross-domain isomorphisms (AToM bridges)
    isomorphic_to: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None


@dataclass
class ConceptRelationship:
    """An edge in the concept graph."""
    source_id: str
    target_id: str
    relationship_type: str  # PREREQUISITE, TEACHES, ISOMORPHIC_TO
    strength: float = 1.0   # 0.0 to 1.0
    inferred: bool = False  # True if ML-generated

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Author:
    """
    An author node in the knowledge graph.

    Represents content creators with expertise domains.
    Used for provenance tracking and authority weighting.
    """
    id: str
    name: str
    expertise: list[str] = field(default_factory=list)  # Domain IDs
    bible_ref: str | None = None  # Reference to authoritative source

    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Domain:
    """
    A domain node in the knowledge graph.

    Represents hierarchical knowledge domains (e.g., Computer_Science.Algorithms).
    Enables cross-domain isomorphism detection and transfer learning.
    """
    id: str
    name: str
    parent_domain: str | None = None  # Hierarchical parent
    depth: int = 0  # Depth in domain hierarchy (0 = root)

    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IsomorphicMapping:
    """
    An isomorphic mapping between concepts in different domains.

    Supports Analogical Transfer of Meaning (AToM) for transfer learning.
    Maps structural relationships between analogous concepts.
    """
    source_concept_id: str
    target_concept_id: str
    mapping_score: float  # 0.0 to 1.0 similarity
    structural_mappings: dict[str, str] = field(default_factory=dict)
    # e.g., {"Pressure": "Voltage", "Flow_Rate": "Current"}

    inferred: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TeachingLink:
    """
    A TEACHES edge linking an atom to a concept.

    Includes coverage metadata for tracking which aspects
    of a concept are taught by which atoms.
    """
    atom_id: str
    concept_id: str
    coverage: float = 1.0  # How much of the concept this atom covers
    mode: ICAPMode | None = None  # ICAP mode of the teaching
    aspects_covered: list[str] = field(default_factory=list)
    # e.g., ["Definition", "Base_Case"]


@dataclass
class ConceptCoverage:
    """
    Coverage analysis for a concept.

    Tracks which aspects are covered by atoms and identifies gaps.
    """
    concept_id: str
    required_aspects: dict[str, float] = field(default_factory=dict)
    # e.g., {"Definition": 1.0, "Base_Case": 1.0, "Tail_Optimization": 0.5}

    actual_coverage: dict[str, float] = field(default_factory=dict)
    # Computed from teaching links

    @property
    def gaps(self) -> list[str]:
        """Aspects that don't meet required coverage."""
        return [
            aspect for aspect, required in self.required_aspects.items()
            if self.actual_coverage.get(aspect, 0.0) < required
        ]

    @property
    def is_complete(self) -> bool:
        """Whether all required aspects are covered."""
        return len(self.gaps) == 0


@dataclass
class ICAPDistribution:
    """
    Distribution of ICAP modes for a concept's atoms.

    Used for ensuring balanced engagement modes.
    """
    concept_id: str
    counts: dict[ICAPMode, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def percentages(self) -> dict[ICAPMode, float]:
        if self.total == 0:
            return {mode: 0.0 for mode in ICAPMode}
        return {mode: count / self.total for mode, count in self.counts.items()}

    @property
    def missing_modes(self) -> list[ICAPMode]:
        """ICAP modes with zero atoms."""
        return [mode for mode in ICAPMode if self.counts.get(mode, 0) == 0]

    @property
    def dominant_mode(self) -> tuple[ICAPMode, float] | None:
        """Mode with >40% representation (potential imbalance)."""
        for mode, pct in self.percentages.items():
            if pct > 0.4:
                return (mode, pct)
        return None


@dataclass
class ChangelogEntry:
    """
    A changelog entry for tracking graph modifications.

    Supports audit trails and undo operations.
    """
    timestamp: datetime
    event_type: str  # PROPERTY_UPDATE, EDGE_CREATED, EDGE_DELETED, NODE_CREATED, NODE_DELETED
    entity_id: str
    entity_type: str  # Concept, Atom, Author, Domain
    details: dict[str, Any] = field(default_factory=dict)
    # e.g., {"field": "difficulty", "old_value": 0.50, "new_value": 0.62}
