"""
Platinum Layer: Research Evidence Integration Models.

This module provides data models for the platinum layer of the Bronze-to-Platinum
pipeline, enabling dynamic queries to the research-engine PostgreSQL database
rather than static file-based evidence loading.

Scientific Foundation:
- Evidence-Based Practice (Sackett et al., 1996)
- GRADE evidence quality framework (Guyatt et al., 2008)
- Bayesian updating for hyperparameter optimization

Architecture:
    Knowledge Base (this module) ──SQL Query──▶ Research Engine (PostgreSQL)
           │                                           │
           │ PlatinumMetadata                          │ Evidence Tables
           │ EvidenceQuery                             │ Phenomena, Citations
           │ HyperparameterFeedback                    │ Effect Sizes
           └───────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EvidenceQuality(str, Enum):
    """
    Quality tiers for research evidence based on GRADE framework.

    GRADE (Grading of Recommendations Assessment, Development and Evaluation)
    provides a systematic approach to rating the quality of evidence.
    """
    META_ANALYSIS = "meta_analysis"           # Highest: systematic review
    RCT = "randomized_controlled_trial"       # Gold standard single study
    QUASI_EXPERIMENTAL = "quasi_experimental" # Non-randomized controlled
    OBSERVATIONAL = "observational"           # Cohort/cross-sectional
    CASE_STUDY = "case_study"                 # Single case/series
    EXPERT_OPINION = "expert_opinion"         # Lowest tier


class ResearchDomain(str, Enum):
    """Domains where research evidence has been validated."""
    EDUCATION = "education"
    COGNITIVE_PSYCHOLOGY = "cognitive_psychology"
    NEUROSCIENCE = "neuroscience"
    MEDICINE = "medicine"
    PROGRAMMING = "programming"
    SPORTS = "sports"
    MUSIC = "music"
    LANGUAGE = "language"


@dataclass
class PlatinumMetadata:
    """
    Platinum-layer metadata linking atoms/concepts to research evidence.

    This is the primary data structure for connecting learning content
    to empirical research stored in the research-engine database.

    Usage:
        # Attach to an atom for evidence-backed difficulty calibration
        metadata = PlatinumMetadata(
            entity_id="atom-recursion-001",
            entity_type="atom",
            phenomenon_ids=["testing-effect", "spacing-effect"],
            evidence_quality=EvidenceQuality.META_ANALYSIS,
            confidence_level=0.95,
            study_count=36,
            effect_size_mean=0.70,
        )
    """
    entity_id: str  # Atom or Concept ID
    entity_type: str  # "atom" or "concept"

    # Research evidence links (query keys for research-engine DB)
    phenomenon_ids: list[str] = field(default_factory=list)
    # e.g., ["spacing-effect", "testing-effect", "interleaving-effect"]

    # Evidence quality metrics
    evidence_quality: EvidenceQuality = EvidenceQuality.OBSERVATIONAL
    confidence_level: float = 0.80  # 0.0 to 1.0
    study_count: int = 0
    total_sample_size: int | None = None

    # Effect size aggregation (Cohen's d)
    effect_size_mean: float | None = None
    effect_size_ci_lower: float | None = None  # 95% CI lower bound
    effect_size_ci_upper: float | None = None  # 95% CI upper bound

    # Citations (DOIs or OpenAlex IDs for database lookup)
    citation_ids: list[str] = field(default_factory=list)
    # e.g., ["10.1037/0033-2909.132.3.354", "W2963350374"]

    # Hyperparameter applicability
    applicable_hyperparameters: list[str] = field(default_factory=list)
    # e.g., ["spacing_ratio", "checkpoint_interval_hours"]

    # Cross-domain applicability
    domains_validated: list[str] = field(default_factory=list)
    # e.g., ["education", "medicine", "programming"]

    # Timestamps
    last_validated: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_high_quality(self) -> bool:
        """Whether evidence meets high-quality threshold (GRADE A/B)."""
        return (
            self.evidence_quality in (EvidenceQuality.META_ANALYSIS, EvidenceQuality.RCT)
            and self.confidence_level >= 0.85
            and self.study_count >= 3
        )

    @property
    def evidence_weight(self) -> float:
        """
        Combined weight for Bayesian hyperparameter updates.

        Factors:
        - Quality tier (meta-analysis = 1.0, expert opinion = 0.1)
        - Confidence level (0.0 to 1.0)
        - Effect size magnitude (normalized to 0.8 baseline)
        """
        quality_weights = {
            EvidenceQuality.META_ANALYSIS: 1.0,
            EvidenceQuality.RCT: 0.9,
            EvidenceQuality.QUASI_EXPERIMENTAL: 0.7,
            EvidenceQuality.OBSERVATIONAL: 0.5,
            EvidenceQuality.CASE_STUDY: 0.3,
            EvidenceQuality.EXPERT_OPINION: 0.1,
        }
        quality_factor = quality_weights.get(self.evidence_quality, 0.5)
        effect_factor = min(1.0, (self.effect_size_mean or 0.5) / 0.8)
        return quality_factor * self.confidence_level * effect_factor

    def to_db_query_params(self) -> dict[str, Any]:
        """Convert to parameters for research-engine DB query."""
        return {
            "phenomenon_ids": self.phenomenon_ids,
            "min_confidence": self.confidence_level * 0.9,  # Allow 10% variance
            "citation_ids": self.citation_ids,
        }


@dataclass
class EvidenceQuery:
    """
    Query specification for research-engine PostgreSQL database.

    Use this to dynamically fetch evidence instead of relying on static files.

    Usage:
        query = EvidenceQuery(
            phenomenon="spacing-effect",
            min_confidence=0.85,
            quality_threshold=EvidenceQuality.RCT,
        )
        results = research_engine.query_evidence(query)
    """
    phenomenon: str | None = None
    domain: str | None = None
    min_confidence: float = 0.0
    min_study_count: int = 0
    quality_threshold: EvidenceQuality | None = None
    hyperparameter: str | None = None  # Filter by applicable hyperparameter

    # Pagination
    limit: int = 50
    offset: int = 0

    # Ordering
    order_by: str = "confidence_level"  # or "effect_size", "study_count"
    descending: bool = True

    def to_sql_where(self) -> tuple[str, dict[str, Any]]:
        """Generate SQL WHERE clause and parameters."""
        conditions = []
        params = {}

        if self.phenomenon:
            conditions.append("phenomenon_id = :phenomenon")
            params["phenomenon"] = self.phenomenon

        if self.domain:
            conditions.append(":domain = ANY(domains_validated)")
            params["domain"] = self.domain

        if self.min_confidence > 0:
            conditions.append("confidence_level >= :min_confidence")
            params["min_confidence"] = self.min_confidence

        if self.min_study_count > 0:
            conditions.append("study_count >= :min_study_count")
            params["min_study_count"] = self.min_study_count

        if self.quality_threshold:
            quality_order = list(EvidenceQuality)
            threshold_idx = quality_order.index(self.quality_threshold)
            valid_qualities = [q.value for q in quality_order[:threshold_idx + 1]]
            conditions.append("evidence_quality = ANY(:quality_values)")
            params["quality_values"] = valid_qualities

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params


@dataclass
class HyperparameterFeedback:
    """
    Feedback record for Bayesian hyperparameter updates.

    Tracks actual outcomes when using research-derived hyperparameters,
    enabling the recursive optimization loop where the system learns
    to better apply learning science to its own development.

    Usage:
        feedback = HyperparameterFeedback(
            id="fb-001",
            hyperparameter="spacing_ratio",
            value_used=0.15,
            evidence_ids=["spacing-effect-001"],
            success=True,
            outcome_metric=0.92,  # e.g., retention rate
            context={"domain": "programming", "agent_id": "claude-1"},
        )
        feedback_store.record(feedback)
    """
    id: str
    hyperparameter: str  # e.g., "spacing_ratio", "interleave_frequency"
    value_used: float
    evidence_ids: list[str] = field(default_factory=list)

    # Outcome metrics
    success: bool = False
    outcome_metric: float | None = None  # e.g., retention rate, completion rate
    context: dict[str, Any] = field(default_factory=dict)
    # e.g., {"domain": "programming", "difficulty": 0.7, "agent_id": "claude-1"}

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_bayesian_observation(self) -> dict[str, Any]:
        """Format for Bayesian prior update."""
        return {
            "hyperparameter": self.hyperparameter,
            "value": self.value_used,
            "observation": self.outcome_metric or (1.0 if self.success else 0.0),
            "weight": 1.0,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvidenceSynthesis:
    """
    Aggregated evidence synthesis for a phenomenon or hyperparameter.

    Created by combining multiple PlatinumMetadata records and computing
    weighted averages of effect sizes and confidence levels.
    """
    phenomenon_id: str
    total_studies: int = 0
    total_sample_size: int = 0

    # Weighted effect size
    effect_size_weighted_mean: float = 0.0
    effect_size_heterogeneity: float = 0.0  # I² statistic

    # Confidence aggregation
    confidence_weighted_mean: float = 0.0

    # Domain coverage
    domains: list[str] = field(default_factory=list)
    domain_specific_effects: dict[str, float] = field(default_factory=dict)

    # Applicable hyperparameters with recommended values
    hyperparameter_recommendations: dict[str, float] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_metadata_list(cls, metadata_list: list[PlatinumMetadata]) -> "EvidenceSynthesis":
        """Synthesize evidence from multiple PlatinumMetadata records."""
        if not metadata_list:
            return cls(phenomenon_id="unknown")

        # Get phenomenon from first record
        phenomenon_id = metadata_list[0].phenomenon_ids[0] if metadata_list[0].phenomenon_ids else "unknown"

        # Aggregate metrics
        total_studies = sum(m.study_count for m in metadata_list)
        total_sample = sum(m.total_sample_size or 0 for m in metadata_list)

        # Weighted effect size
        weights = [m.evidence_weight for m in metadata_list]
        total_weight = sum(weights) or 1.0
        effect_sizes = [m.effect_size_mean or 0.5 for m in metadata_list]
        weighted_effect = sum(e * w for e, w in zip(effect_sizes, weights)) / total_weight

        # Weighted confidence
        confidences = [m.confidence_level for m in metadata_list]
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight

        # Collect domains
        all_domains = set()
        for m in metadata_list:
            all_domains.update(m.domains_validated)

        # Collect hyperparameters
        all_hyperparams = set()
        for m in metadata_list:
            all_hyperparams.update(m.applicable_hyperparameters)

        return cls(
            phenomenon_id=phenomenon_id,
            total_studies=total_studies,
            total_sample_size=total_sample,
            effect_size_weighted_mean=weighted_effect,
            confidence_weighted_mean=weighted_confidence,
            domains=list(all_domains),
            hyperparameter_recommendations={hp: weighted_effect for hp in all_hyperparams},
        )
