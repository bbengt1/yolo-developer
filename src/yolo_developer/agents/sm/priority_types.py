"""Type definitions for priority scoring (Story 10.11).

This module provides the data types used by the priority scoring module:

- PriorityFactors: Input scoring factors for a story
- PriorityResult: Scored output with explanation for audit
- PriorityScoringConfig: Configuration for priority scoring behavior

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.priority_types import (
    ...     PriorityFactors,
    ...     PriorityResult,
    ...     PriorityScoringConfig,
    ... )
    >>>
    >>> factors = PriorityFactors(
    ...     story_id="1-1-user-auth",
    ...     value_score=0.9,
    ...     dependency_score=0.5,
    ...     velocity_impact=0.7,
    ...     tech_debt_score=0.2,
    ... )
    >>> factors.to_dict()
    {'story_id': '1-1-user-auth', 'value_score': 0.9, ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR65: SM Agent can calculate weighted priority scores for story selection
    - Story 10.3: Sprint Planning (defines PlanningConfig with weights)
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

MIN_SCORE: float = 0.0
"""Minimum valid score value."""

MAX_SCORE: float = 1.0
"""Maximum valid score value."""

DEFAULT_NORMALIZE_SCORES: bool = True
"""Default setting for score normalization across story set."""

DEFAULT_INCLUDE_EXPLANATION: bool = True
"""Default setting for including explanation in results."""

DEFAULT_MIN_SCORE_THRESHOLD: float = 0.0
"""Default minimum score threshold (stories below this are filtered out)."""

# Weight defaults imported from planning_types for consistency
DEFAULT_VALUE_WEIGHT: float = 0.4
"""Default weight for value score in priority calculation (per FR65)."""

DEFAULT_DEPENDENCY_WEIGHT: float = 0.3
"""Default weight for dependency score in priority calculation (per FR65)."""

DEFAULT_VELOCITY_WEIGHT: float = 0.2
"""Default weight for velocity impact in priority calculation (per FR65)."""

DEFAULT_TECH_DEBT_WEIGHT: float = 0.1
"""Default weight for tech debt score in priority calculation (per FR65)."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class PriorityFactors:
    """Input scoring factors for a story.

    Contains the raw scoring factors used to calculate a story's
    weighted priority score per FR65.

    Attributes:
        story_id: Unique story identifier (e.g., "1-2-user-auth")
        value_score: Business value component (0.0-1.0)
        dependency_score: Score based on how many stories depend on this (0.0-1.0)
        velocity_impact: Expected velocity impact (0.0-1.0)
        tech_debt_score: Tech debt reduction component (0.0-1.0)

    Example:
        >>> factors = PriorityFactors(
        ...     story_id="2-1-db-setup",
        ...     value_score=0.8,
        ...     dependency_score=0.6,
        ...     velocity_impact=0.5,
        ...     tech_debt_score=0.3,
        ... )
        >>> factors.value_score
        0.8
    """

    story_id: str
    value_score: float = 0.5
    dependency_score: float = 0.0
    velocity_impact: float = 0.5
    tech_debt_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate score ranges and log warnings for out-of-range values."""
        score_fields = [
            ("value_score", self.value_score),
            ("dependency_score", self.dependency_score),
            ("velocity_impact", self.velocity_impact),
            ("tech_debt_score", self.tech_debt_score),
        ]
        for field_name, value in score_fields:
            if value < MIN_SCORE or value > MAX_SCORE:
                _logger.warning(
                    "PriorityFactors %s=%s is outside valid range [%.1f, %.1f] "
                    "for story_id=%s",
                    field_name,
                    value,
                    MIN_SCORE,
                    MAX_SCORE,
                    self.story_id,
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the factors.
        """
        return {
            "story_id": self.story_id,
            "value_score": self.value_score,
            "dependency_score": self.dependency_score,
            "velocity_impact": self.velocity_impact,
            "tech_debt_score": self.tech_debt_score,
        }


@dataclass(frozen=True)
class PriorityResult:
    """Scored output with explanation for audit.

    Contains the calculated priority score along with the breakdown
    of how each factor contributed to the final score.

    Attributes:
        story_id: Unique story identifier (e.g., "1-2-user-auth")
        priority_score: Composite weighted score (0.0-1.0)
        normalized_score: Score after normalization across story set (0.0-1.0)
        explanation: Human-readable breakdown of score calculation
        factor_contributions: Dict mapping factor names to their weighted contribution

    Example:
        >>> result = PriorityResult(
        ...     story_id="2-1-db-setup",
        ...     priority_score=0.65,
        ...     normalized_score=0.8,
        ...     explanation="High value (0.32) + moderate dependency (0.18)...",
        ...     factor_contributions={"value": 0.32, "dependency": 0.18, ...},
        ... )
        >>> result.priority_score
        0.65
    """

    story_id: str
    priority_score: float
    normalized_score: float | None = None
    explanation: str = ""
    factor_contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "story_id": self.story_id,
            "priority_score": self.priority_score,
            "normalized_score": self.normalized_score,
            "explanation": self.explanation,
            "factor_contributions": dict(self.factor_contributions),
        }


@dataclass(frozen=True)
class PriorityScoringConfig:
    """Configuration for priority scoring behavior.

    Extends the weight configuration from PlanningConfig with additional
    options for normalization and explanation generation.

    Attributes:
        value_weight: Weight for business value in priority (default 0.4)
        dependency_weight: Weight for dependency score (default 0.3)
        velocity_weight: Weight for velocity impact (default 0.2)
        tech_debt_weight: Weight for tech debt reduction (default 0.1)
        normalize_scores: Whether to normalize scores across story set (default True)
        include_explanation: Whether to include explanation in results (default True)
        min_score_threshold: Minimum score threshold for filtering (default 0.0)

    Note:
        Weights should sum to 1.0 for normalized priority scores.
        This config is compatible with PlanningConfig weights.

    Example:
        >>> config = PriorityScoringConfig(
        ...     value_weight=0.5,
        ...     normalize_scores=False,
        ... )
        >>> config.value_weight
        0.5
    """

    value_weight: float = DEFAULT_VALUE_WEIGHT
    dependency_weight: float = DEFAULT_DEPENDENCY_WEIGHT
    velocity_weight: float = DEFAULT_VELOCITY_WEIGHT
    tech_debt_weight: float = DEFAULT_TECH_DEBT_WEIGHT
    normalize_scores: bool = DEFAULT_NORMALIZE_SCORES
    include_explanation: bool = DEFAULT_INCLUDE_EXPLANATION
    min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0 and log warning if not."""
        weight_sum = (
            self.value_weight
            + self.dependency_weight
            + self.velocity_weight
            + self.tech_debt_weight
        )
        # Allow small floating point tolerance
        if abs(weight_sum - 1.0) > 0.001:
            _logger.warning(
                "PriorityScoringConfig weights sum to %.3f (expected 1.0). "
                "Priority scores may not be normalized correctly.",
                weight_sum,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "value_weight": self.value_weight,
            "dependency_weight": self.dependency_weight,
            "velocity_weight": self.velocity_weight,
            "tech_debt_weight": self.tech_debt_weight,
            "normalize_scores": self.normalize_scores,
            "include_explanation": self.include_explanation,
            "min_score_threshold": self.min_score_threshold,
        }

    @classmethod
    def from_planning_config(
        cls,
        planning_config: Any,
        *,
        normalize_scores: bool = DEFAULT_NORMALIZE_SCORES,
        include_explanation: bool = DEFAULT_INCLUDE_EXPLANATION,
        min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
    ) -> PriorityScoringConfig:
        """Create PriorityScoringConfig from a PlanningConfig.

        This factory method ensures backward compatibility with existing
        PlanningConfig usage.

        Args:
            planning_config: A PlanningConfig instance with weight attributes.
            normalize_scores: Whether to normalize scores (default True).
            include_explanation: Whether to include explanation (default True).
            min_score_threshold: Minimum score threshold (default 0.0).

        Returns:
            PriorityScoringConfig with weights from planning_config.

        Example:
            >>> from yolo_developer.agents.sm.planning_types import PlanningConfig
            >>> planning = PlanningConfig(value_weight=0.5)
            >>> scoring = PriorityScoringConfig.from_planning_config(planning)
            >>> scoring.value_weight
            0.5
        """
        return cls(
            value_weight=planning_config.value_weight,
            dependency_weight=planning_config.dependency_weight,
            velocity_weight=planning_config.velocity_weight,
            tech_debt_weight=planning_config.tech_debt_weight,
            normalize_scores=normalize_scores,
            include_explanation=include_explanation,
            min_score_threshold=min_score_threshold,
        )
