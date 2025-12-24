"""
Feedback Store: SQLite Persistence for Bayesian Hyperparameter Updates.

This module provides SQLite-backed storage for hyperparameter feedback records,
enabling the recursive optimization loop where learning outcomes update
research-derived hyperparameters.

Scientific Foundation:
- Bayesian Optimization (Mockus, 1989; Snoek et al., 2012)
- Thompson Sampling for contextual bandits (Chapelle & Li, 2011)
- Empirical Bayes methods (Casella, 1985)

Architecture:
    Navigation Engine                  Feedback Store (this module)
         │                                    │
         │ HyperparameterFeedback            │ SQLite Database
         │ (success/failure records)  ──────▶│ feedback_records
         │                                    │ prior_updates
         └────────────────────────────────────┘

The feedback loop enables recursive self-improvement:
1. Research evidence → initial hyperparameter values
2. AI agent uses hyperparameters for scheduling
3. Learning outcomes are recorded as feedback
4. Bayesian update adjusts hyperparameters
5. Improved parameters → better learning outcomes → repeat
"""

import sqlite3
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

from .platinum import HyperparameterFeedback, EvidenceQuality


# ============================================================================
# Exceptions
# ============================================================================

class FeedbackStoreError(Exception):
    """Base exception for feedback store operations."""
    pass


class FeedbackNotFoundError(FeedbackStoreError):
    """Raised when a feedback record is not found."""
    pass


class PriorUpdateError(FeedbackStoreError):
    """Raised when a prior update operation fails."""
    pass


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class BayesianPrior:
    """
    Bayesian prior for a hyperparameter.

    Uses a Beta distribution for bounded parameters (0-1)
    or Normal distribution for unbounded parameters.

    For Beta(alpha, beta):
    - Mean = alpha / (alpha + beta)
    - Variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))

    For Normal(mu, sigma):
    - Mean = mu
    - Variance = sigma^2
    """
    hyperparameter: str

    # Beta distribution parameters (for success rates, probabilities)
    alpha: float = 1.0  # Successes + prior
    beta: float = 1.0   # Failures + prior

    # Normal distribution parameters (for continuous values)
    mu: float = 0.5     # Mean estimate
    sigma: float = 0.2  # Uncertainty

    # Tracking
    observation_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Evidence weights
    total_evidence_weight: float = 0.0

    @property
    def beta_mean(self) -> float:
        """Expected value under Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def beta_variance(self) -> float:
        """Variance under Beta distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def confidence(self) -> float:
        """
        Confidence in the current estimate (0-1).

        Based on observation count and evidence weight.
        Asymptotes to 1.0 as evidence accumulates.
        """
        # Log-based confidence: 50% at 10 observations, 90% at 100
        import math
        if self.observation_count == 0:
            return 0.1  # Low prior confidence
        log_confidence = math.log10(self.observation_count + 1) / 3  # 0.33 at 10, 0.67 at 100
        return min(0.95, 0.1 + log_confidence * 0.85)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hyperparameter": self.hyperparameter,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "sigma": self.sigma,
            "observation_count": self.observation_count,
            "last_updated": self.last_updated.isoformat(),
            "total_evidence_weight": self.total_evidence_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BayesianPrior":
        """Deserialize from dictionary."""
        return cls(
            hyperparameter=data["hyperparameter"],
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            mu=data.get("mu", 0.5),
            sigma=data.get("sigma", 0.2),
            observation_count=data.get("observation_count", 0),
            last_updated=datetime.fromisoformat(data["last_updated"]) if "last_updated" in data else datetime.utcnow(),
            total_evidence_weight=data.get("total_evidence_weight", 0.0),
        )


@dataclass
class PriorUpdateResult:
    """Result of a Bayesian prior update operation."""
    hyperparameter: str
    old_mean: float
    new_mean: float
    old_confidence: float
    new_confidence: float
    observations_added: int

    @property
    def mean_shift(self) -> float:
        """Change in mean estimate."""
        return self.new_mean - self.old_mean

    @property
    def confidence_gain(self) -> float:
        """Increase in confidence."""
        return self.new_confidence - self.old_confidence


# ============================================================================
# Abstract Base Class
# ============================================================================

class FeedbackStore(ABC):
    """Abstract base class for feedback storage."""

    @abstractmethod
    def record_feedback(self, feedback: HyperparameterFeedback) -> None:
        """Record a hyperparameter feedback observation."""
        pass

    @abstractmethod
    def get_feedback(self, feedback_id: str) -> HyperparameterFeedback:
        """Retrieve a feedback record by ID."""
        pass

    @abstractmethod
    def query_feedback(
        self,
        hyperparameter: str | None = None,
        min_timestamp: datetime | None = None,
        max_timestamp: datetime | None = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> list[HyperparameterFeedback]:
        """Query feedback records with filters."""
        pass

    @abstractmethod
    def get_prior(self, hyperparameter: str) -> BayesianPrior:
        """Get the current Bayesian prior for a hyperparameter."""
        pass

    @abstractmethod
    def update_prior(
        self,
        hyperparameter: str,
        observation: float,
        weight: float = 1.0,
    ) -> PriorUpdateResult:
        """
        Update the Bayesian prior with a new observation.

        Args:
            hyperparameter: Name of the hyperparameter
            observation: Observed value (0-1 for Beta, any for Normal)
            weight: Weight of this observation (based on evidence quality)

        Returns:
            PriorUpdateResult with before/after comparison
        """
        pass

    @abstractmethod
    def get_recommended_value(self, hyperparameter: str) -> float:
        """
        Get the recommended value for a hyperparameter.

        Uses the posterior mean, weighted by confidence.
        """
        pass

    @abstractmethod
    def get_all_priors(self) -> list[BayesianPrior]:
        """Get all Bayesian priors."""
        pass


# ============================================================================
# SQLite Implementation
# ============================================================================

class SQLiteFeedbackStore(FeedbackStore):
    """
    SQLite-backed feedback store for Bayesian hyperparameter updates.

    Schema:
        feedback_records: Individual feedback observations
        bayesian_priors: Current prior estimates per hyperparameter
        prior_updates: History of prior update events

    Usage:
        store = SQLiteFeedbackStore("feedback.db")

        # Record feedback
        feedback = HyperparameterFeedback(
            id="fb-001",
            hyperparameter="spacing_ratio",
            value_used=0.15,
            success=True,
            outcome_metric=0.92,
        )
        store.record_feedback(feedback)

        # Get updated recommendation
        recommended = store.get_recommended_value("spacing_ratio")
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        """
        Initialize SQLite feedback store.

        Args:
            db_path: Path to SQLite database, or ":memory:" for in-memory
        """
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._shared_conn: sqlite3.Connection | None = None

        if self._is_memory:
            # For in-memory databases, keep a single shared connection
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row

        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with proper cleanup."""
        if self._is_memory and self._shared_conn:
            # For in-memory, use the shared connection
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            # For file-based, create a new connection each time
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Feedback records table
                CREATE TABLE IF NOT EXISTS feedback_records (
                    id TEXT PRIMARY KEY,
                    hyperparameter TEXT NOT NULL,
                    value_used REAL NOT NULL,
                    success INTEGER NOT NULL,
                    outcome_metric REAL,
                    evidence_ids TEXT,  -- JSON array
                    context TEXT,       -- JSON object
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_hyperparameter
                    ON feedback_records(hyperparameter);
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                    ON feedback_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_feedback_success
                    ON feedback_records(success);

                -- Bayesian priors table
                CREATE TABLE IF NOT EXISTS bayesian_priors (
                    hyperparameter TEXT PRIMARY KEY,
                    alpha REAL NOT NULL DEFAULT 1.0,
                    beta REAL NOT NULL DEFAULT 1.0,
                    mu REAL NOT NULL DEFAULT 0.5,
                    sigma REAL NOT NULL DEFAULT 0.2,
                    observation_count INTEGER NOT NULL DEFAULT 0,
                    total_evidence_weight REAL NOT NULL DEFAULT 0.0,
                    last_updated TEXT NOT NULL
                );

                -- Prior update history (audit trail)
                CREATE TABLE IF NOT EXISTS prior_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hyperparameter TEXT NOT NULL,
                    old_alpha REAL,
                    old_beta REAL,
                    new_alpha REAL,
                    new_beta REAL,
                    observation REAL,
                    weight REAL,
                    feedback_id TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (hyperparameter) REFERENCES bayesian_priors(hyperparameter)
                );

                CREATE INDEX IF NOT EXISTS idx_prior_updates_hyperparameter
                    ON prior_updates(hyperparameter);
            """)

    def record_feedback(self, feedback: HyperparameterFeedback) -> None:
        """Record a hyperparameter feedback observation."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feedback_records
                (id, hyperparameter, value_used, success, outcome_metric,
                 evidence_ids, context, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.id,
                feedback.hyperparameter,
                feedback.value_used,
                1 if feedback.success else 0,
                feedback.outcome_metric,
                json.dumps(feedback.evidence_ids),
                json.dumps(feedback.context),
                feedback.timestamp.isoformat(),
            ))

        # Auto-update prior with this observation
        observation = feedback.outcome_metric if feedback.outcome_metric is not None else (1.0 if feedback.success else 0.0)
        self.update_prior(feedback.hyperparameter, observation, weight=1.0)

    def get_feedback(self, feedback_id: str) -> HyperparameterFeedback:
        """Retrieve a feedback record by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM feedback_records WHERE id = ?",
                (feedback_id,)
            ).fetchone()

        if not row:
            raise FeedbackNotFoundError(f"Feedback not found: {feedback_id}")

        return self._row_to_feedback(row)

    def _row_to_feedback(self, row: sqlite3.Row) -> HyperparameterFeedback:
        """Convert a database row to HyperparameterFeedback."""
        return HyperparameterFeedback(
            id=row["id"],
            hyperparameter=row["hyperparameter"],
            value_used=row["value_used"],
            success=bool(row["success"]),
            outcome_metric=row["outcome_metric"],
            evidence_ids=json.loads(row["evidence_ids"]) if row["evidence_ids"] else [],
            context=json.loads(row["context"]) if row["context"] else {},
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def query_feedback(
        self,
        hyperparameter: str | None = None,
        min_timestamp: datetime | None = None,
        max_timestamp: datetime | None = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> list[HyperparameterFeedback]:
        """Query feedback records with filters."""
        conditions = []
        params: list[Any] = []

        if hyperparameter:
            conditions.append("hyperparameter = ?")
            params.append(hyperparameter)

        if min_timestamp:
            conditions.append("timestamp >= ?")
            params.append(min_timestamp.isoformat())

        if max_timestamp:
            conditions.append("timestamp <= ?")
            params.append(max_timestamp.isoformat())

        if success_only:
            conditions.append("success = 1")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM feedback_records WHERE {where_clause} "
                f"ORDER BY timestamp DESC LIMIT ?",
                params + [limit]
            ).fetchall()

        return [self._row_to_feedback(row) for row in rows]

    def get_prior(self, hyperparameter: str) -> BayesianPrior:
        """Get the current Bayesian prior for a hyperparameter."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM bayesian_priors WHERE hyperparameter = ?",
                (hyperparameter,)
            ).fetchone()

        if not row:
            # Return default prior (uninformative)
            return BayesianPrior(hyperparameter=hyperparameter)

        return BayesianPrior(
            hyperparameter=row["hyperparameter"],
            alpha=row["alpha"],
            beta=row["beta"],
            mu=row["mu"],
            sigma=row["sigma"],
            observation_count=row["observation_count"],
            last_updated=datetime.fromisoformat(row["last_updated"]),
            total_evidence_weight=row["total_evidence_weight"],
        )

    def update_prior(
        self,
        hyperparameter: str,
        observation: float,
        weight: float = 1.0,
    ) -> PriorUpdateResult:
        """
        Update the Bayesian prior with a new observation.

        Uses Beta-Bernoulli conjugate update for bounded (0-1) observations:
            alpha_new = alpha_old + weight * observation
            beta_new = beta_old + weight * (1 - observation)

        This is the core of the recursive optimization loop.
        """
        current = self.get_prior(hyperparameter)
        old_mean = current.beta_mean
        old_confidence = current.confidence

        # Clamp observation to [0, 1] for Beta update
        clamped_obs = max(0.0, min(1.0, observation))

        # Beta-Bernoulli conjugate update
        new_alpha = current.alpha + weight * clamped_obs
        new_beta = current.beta + weight * (1 - clamped_obs)

        # Update Normal parameters using online mean/variance update
        n = current.observation_count + 1
        delta = observation - current.mu
        new_mu = current.mu + delta / n
        # Welford's online algorithm for variance
        new_sigma = max(0.01, current.sigma * 0.99)  # Slowly decrease uncertainty

        now = datetime.utcnow()

        with self._get_connection() as conn:
            # Upsert prior
            conn.execute("""
                INSERT INTO bayesian_priors
                (hyperparameter, alpha, beta, mu, sigma, observation_count,
                 total_evidence_weight, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hyperparameter) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta,
                    mu = excluded.mu,
                    sigma = excluded.sigma,
                    observation_count = excluded.observation_count,
                    total_evidence_weight = excluded.total_evidence_weight,
                    last_updated = excluded.last_updated
            """, (
                hyperparameter,
                new_alpha,
                new_beta,
                new_mu,
                new_sigma,
                n,
                current.total_evidence_weight + weight,
                now.isoformat(),
            ))

            # Record update history
            conn.execute("""
                INSERT INTO prior_updates
                (hyperparameter, old_alpha, old_beta, new_alpha, new_beta,
                 observation, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                hyperparameter,
                current.alpha,
                current.beta,
                new_alpha,
                new_beta,
                observation,
                weight,
            ))

        # Calculate new metrics
        new_prior = BayesianPrior(
            hyperparameter=hyperparameter,
            alpha=new_alpha,
            beta=new_beta,
            mu=new_mu,
            sigma=new_sigma,
            observation_count=n,
        )

        return PriorUpdateResult(
            hyperparameter=hyperparameter,
            old_mean=old_mean,
            new_mean=new_prior.beta_mean,
            old_confidence=old_confidence,
            new_confidence=new_prior.confidence,
            observations_added=1,
        )

    def get_recommended_value(self, hyperparameter: str) -> float:
        """
        Get the recommended value for a hyperparameter.

        Returns the posterior mean of the Beta distribution.
        """
        prior = self.get_prior(hyperparameter)
        return prior.beta_mean

    def get_all_priors(self) -> list[BayesianPrior]:
        """Get all Bayesian priors."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM bayesian_priors ORDER BY hyperparameter"
            ).fetchall()

        return [
            BayesianPrior(
                hyperparameter=row["hyperparameter"],
                alpha=row["alpha"],
                beta=row["beta"],
                mu=row["mu"],
                sigma=row["sigma"],
                observation_count=row["observation_count"],
                last_updated=datetime.fromisoformat(row["last_updated"]),
                total_evidence_weight=row["total_evidence_weight"],
            )
            for row in rows
        ]

    def get_feedback_summary(self, hyperparameter: str) -> dict[str, Any]:
        """
        Get a summary of feedback for a hyperparameter.

        Returns aggregated statistics useful for debugging and monitoring.
        """
        with self._get_connection() as conn:
            # Aggregate stats
            stats = conn.execute("""
                SELECT
                    COUNT(*) as total_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    AVG(outcome_metric) as avg_outcome,
                    MIN(timestamp) as first_feedback,
                    MAX(timestamp) as last_feedback
                FROM feedback_records
                WHERE hyperparameter = ?
            """, (hyperparameter,)).fetchone()

        prior = self.get_prior(hyperparameter)

        return {
            "hyperparameter": hyperparameter,
            "total_feedback": stats["total_count"] or 0,
            "success_rate": (stats["success_count"] or 0) / max(1, stats["total_count"] or 1),
            "avg_outcome": stats["avg_outcome"],
            "first_feedback": stats["first_feedback"],
            "last_feedback": stats["last_feedback"],
            "current_prior": prior.to_dict(),
            "recommended_value": prior.beta_mean,
            "confidence": prior.confidence,
        }

    def initialize_from_evidence(
        self,
        hyperparameter: str,
        effect_size: float,
        confidence: float,
        study_count: int,
    ) -> BayesianPrior:
        """
        Initialize a prior from research evidence.

        Converts research evidence into informative priors:
        - effect_size → initial mean estimate
        - confidence + study_count → prior strength (pseudo-observations)

        Args:
            hyperparameter: Name of the hyperparameter
            effect_size: Cohen's d from research (converted to 0-1 scale)
            confidence: Confidence level from evidence (0-1)
            study_count: Number of supporting studies

        Returns:
            Initialized BayesianPrior
        """
        import math

        # Convert effect size to 0-1 scale (assumes d=0.8 is "strong")
        normalized_effect = max(0.1, min(0.9, effect_size / 1.0))

        # Prior strength based on evidence (log scale for study count)
        prior_strength = confidence * (1 + math.log10(max(1, study_count)))

        # Convert to Beta parameters
        # If normalized_effect = 0.7 and prior_strength = 2.0:
        #   alpha = 0.7 * 2.0 = 1.4
        #   beta = 0.3 * 2.0 = 0.6
        alpha = normalized_effect * prior_strength + 1  # +1 for uninformative component
        beta = (1 - normalized_effect) * prior_strength + 1

        now = datetime.utcnow()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO bayesian_priors
                (hyperparameter, alpha, beta, mu, sigma, observation_count,
                 total_evidence_weight, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hyperparameter) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta,
                    mu = excluded.mu,
                    sigma = excluded.sigma,
                    total_evidence_weight = excluded.total_evidence_weight,
                    last_updated = excluded.last_updated
            """, (
                hyperparameter,
                alpha,
                beta,
                normalized_effect,
                0.2 / (1 + math.log10(max(1, study_count))),  # Narrower with more studies
                0,  # No observations yet
                prior_strength,
                now.isoformat(),
            ))

        return self.get_prior(hyperparameter)


# ============================================================================
# In-Memory Implementation (for testing)
# ============================================================================

class InMemoryFeedbackStore(FeedbackStore):
    """In-memory feedback store for testing."""

    def __init__(self):
        self._feedback: dict[str, HyperparameterFeedback] = {}
        self._priors: dict[str, BayesianPrior] = {}

    def record_feedback(self, feedback: HyperparameterFeedback) -> None:
        self._feedback[feedback.id] = feedback
        observation = feedback.outcome_metric if feedback.outcome_metric is not None else (1.0 if feedback.success else 0.0)
        self.update_prior(feedback.hyperparameter, observation)

    def get_feedback(self, feedback_id: str) -> HyperparameterFeedback:
        if feedback_id not in self._feedback:
            raise FeedbackNotFoundError(f"Feedback not found: {feedback_id}")
        return self._feedback[feedback_id]

    def query_feedback(
        self,
        hyperparameter: str | None = None,
        min_timestamp: datetime | None = None,
        max_timestamp: datetime | None = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> list[HyperparameterFeedback]:
        results = list(self._feedback.values())

        if hyperparameter:
            results = [f for f in results if f.hyperparameter == hyperparameter]
        if min_timestamp:
            results = [f for f in results if f.timestamp >= min_timestamp]
        if max_timestamp:
            results = [f for f in results if f.timestamp <= max_timestamp]
        if success_only:
            results = [f for f in results if f.success]

        results.sort(key=lambda f: f.timestamp, reverse=True)
        return results[:limit]

    def get_prior(self, hyperparameter: str) -> BayesianPrior:
        return self._priors.get(hyperparameter, BayesianPrior(hyperparameter=hyperparameter))

    def update_prior(
        self,
        hyperparameter: str,
        observation: float,
        weight: float = 1.0,
    ) -> PriorUpdateResult:
        current = self.get_prior(hyperparameter)
        old_mean = current.beta_mean
        old_confidence = current.confidence

        clamped_obs = max(0.0, min(1.0, observation))
        new_alpha = current.alpha + weight * clamped_obs
        new_beta = current.beta + weight * (1 - clamped_obs)

        new_prior = BayesianPrior(
            hyperparameter=hyperparameter,
            alpha=new_alpha,
            beta=new_beta,
            mu=current.mu + (observation - current.mu) / (current.observation_count + 1),
            sigma=max(0.01, current.sigma * 0.99),
            observation_count=current.observation_count + 1,
            last_updated=datetime.utcnow(),
            total_evidence_weight=current.total_evidence_weight + weight,
        )
        self._priors[hyperparameter] = new_prior

        return PriorUpdateResult(
            hyperparameter=hyperparameter,
            old_mean=old_mean,
            new_mean=new_prior.beta_mean,
            old_confidence=old_confidence,
            new_confidence=new_prior.confidence,
            observations_added=1,
        )

    def get_recommended_value(self, hyperparameter: str) -> float:
        return self.get_prior(hyperparameter).beta_mean

    def get_all_priors(self) -> list[BayesianPrior]:
        return list(self._priors.values())
