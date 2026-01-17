"""Type definitions for velocity tracking (Story 10.12).

This module provides the data types used by the velocity tracking module:

- VelocityTrend: Literal type for velocity trend direction
- SprintVelocity: Velocity data for a single sprint
- VelocityMetrics: Aggregated velocity metrics across sprints
- VelocityConfig: Configuration for velocity calculations
- VelocityForecast: Forecasted velocity for planning

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.velocity_types import (
    ...     SprintVelocity,
    ...     VelocityMetrics,
    ...     VelocityConfig,
    ...     VelocityForecast,
    ... )
    >>>
    >>> velocity = SprintVelocity(
    ...     sprint_id="sprint-20260116",
    ...     stories_completed=5,
    ...     points_completed=5.0,
    ...     total_cycle_time_ms=18000000.0,
    ...     avg_cycle_time_ms=3600000.0,
    ... )
    >>> velocity.to_dict()
    {'sprint_id': 'sprint-20260116', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - Story 10.9: Sprint Progress Tracking (StoryProgress used as input)
    - Story 10.11: Priority Scoring (type patterns followed)
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types
# =============================================================================

VelocityTrend = Literal["improving", "stable", "declining"]
"""Trend direction for velocity metrics.

Values:
    improving: Recent velocity is significantly higher than historical average
    stable: Velocity is consistent with historical average
    declining: Recent velocity is significantly lower than historical average
"""


# =============================================================================
# Constants
# =============================================================================

DEFAULT_ROLLING_WINDOW: int = 5
"""Default number of sprints for rolling average calculations."""

DEFAULT_TREND_THRESHOLD: float = 0.1
"""Default threshold for trend detection (10% change = trend)."""

DEFAULT_MIN_SPRINTS_FOR_TREND: int = 3
"""Default minimum sprints required to calculate trend."""

DEFAULT_MIN_SPRINTS_FOR_FORECAST: int = 2
"""Default minimum sprints required for forecasting."""

MIN_CONFIDENCE: float = 0.0
"""Minimum confidence value."""

MAX_CONFIDENCE: float = 1.0
"""Maximum confidence value."""

CONFIDENCE_DECIMAL_PLACES: int = 3
"""Decimal places for rounding confidence values in forecasts."""

VALID_TRENDS: frozenset[str] = frozenset({"improving", "stable", "declining"})
"""Set of valid velocity trend values."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class SprintVelocity:
    """Velocity data for a single completed sprint.

    Captures the throughput and cycle time metrics for one sprint,
    used as input for aggregated velocity calculations.

    Attributes:
        sprint_id: Unique sprint identifier (e.g., "sprint-20260116")
        stories_completed: Number of stories completed in the sprint
        points_completed: Sum of story points completed (default 1.0 per story)
        total_cycle_time_ms: Total cycle time for all stories in milliseconds
        avg_cycle_time_ms: Average cycle time per story in milliseconds
        completed_at: ISO timestamp when sprint completed (auto-generated)

    Example:
        >>> velocity = SprintVelocity(
        ...     sprint_id="sprint-20260116",
        ...     stories_completed=5,
        ...     points_completed=5.0,
        ...     total_cycle_time_ms=18000000.0,
        ...     avg_cycle_time_ms=3600000.0,
        ... )
        >>> velocity.stories_completed
        5
    """

    sprint_id: str
    stories_completed: int
    points_completed: float
    total_cycle_time_ms: float
    avg_cycle_time_ms: float
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate sprint velocity data and log warnings for issues."""
        if self.stories_completed < 0:
            _logger.warning(
                "SprintVelocity stories_completed=%d is negative for sprint_id=%s",
                self.stories_completed,
                self.sprint_id,
            )
        if self.points_completed < 0:
            _logger.warning(
                "SprintVelocity points_completed=%.2f is negative for sprint_id=%s",
                self.points_completed,
                self.sprint_id,
            )
        if self.total_cycle_time_ms < 0:
            _logger.warning(
                "SprintVelocity total_cycle_time_ms=%.2f is negative for sprint_id=%s",
                self.total_cycle_time_ms,
                self.sprint_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the sprint velocity.
        """
        return {
            "sprint_id": self.sprint_id,
            "stories_completed": self.stories_completed,
            "points_completed": self.points_completed,
            "total_cycle_time_ms": self.total_cycle_time_ms,
            "avg_cycle_time_ms": self.avg_cycle_time_ms,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class VelocityMetrics:
    """Aggregated velocity metrics across multiple sprints.

    Provides rolling averages and trend analysis for velocity tracking,
    supporting planning and forecasting decisions.

    Attributes:
        average_stories_per_sprint: Rolling average of stories completed per sprint
        average_points_per_sprint: Rolling average of points completed per sprint
        average_cycle_time_ms: Rolling average cycle time in milliseconds
        cycle_time_p50_ms: 50th percentile (median) cycle time in milliseconds
        cycle_time_p90_ms: 90th percentile cycle time in milliseconds
        sprints_analyzed: Number of sprints included in the analysis
        trend: Current velocity trend (improving, stable, declining)
        variance_stories: Standard deviation of stories per sprint
        variance_points: Standard deviation of points per sprint
        calculated_at: ISO timestamp when metrics were calculated (auto-generated)

    Example:
        >>> metrics = VelocityMetrics(
        ...     average_stories_per_sprint=5.5,
        ...     average_points_per_sprint=5.5,
        ...     average_cycle_time_ms=3600000.0,
        ...     cycle_time_p50_ms=3500000.0,
        ...     cycle_time_p90_ms=5000000.0,
        ...     sprints_analyzed=5,
        ...     trend="stable",
        ... )
        >>> metrics.trend
        'stable'
    """

    average_stories_per_sprint: float
    average_points_per_sprint: float
    average_cycle_time_ms: float
    cycle_time_p50_ms: float
    cycle_time_p90_ms: float
    sprints_analyzed: int
    trend: VelocityTrend
    variance_stories: float = 0.0
    variance_points: float = 0.0
    calculated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate velocity metrics and log warnings for issues."""
        if self.average_stories_per_sprint < 0:
            _logger.warning(
                "VelocityMetrics average_stories_per_sprint=%.2f is negative",
                self.average_stories_per_sprint,
            )
        if self.average_points_per_sprint < 0:
            _logger.warning(
                "VelocityMetrics average_points_per_sprint=%.2f is negative",
                self.average_points_per_sprint,
            )
        if self.average_cycle_time_ms < 0:
            _logger.warning(
                "VelocityMetrics average_cycle_time_ms=%.2f is negative",
                self.average_cycle_time_ms,
            )
        if self.sprints_analyzed < 0:
            _logger.warning(
                "VelocityMetrics sprints_analyzed=%d is negative",
                self.sprints_analyzed,
            )
        if self.trend not in VALID_TRENDS:
            _logger.warning(
                "VelocityMetrics trend='%s' is not a valid trend value %s",
                self.trend,
                VALID_TRENDS,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the velocity metrics.
        """
        return {
            "average_stories_per_sprint": self.average_stories_per_sprint,
            "average_points_per_sprint": self.average_points_per_sprint,
            "average_cycle_time_ms": self.average_cycle_time_ms,
            "cycle_time_p50_ms": self.cycle_time_p50_ms,
            "cycle_time_p90_ms": self.cycle_time_p90_ms,
            "sprints_analyzed": self.sprints_analyzed,
            "trend": self.trend,
            "variance_stories": self.variance_stories,
            "variance_points": self.variance_points,
            "calculated_at": self.calculated_at,
        }


@dataclass(frozen=True)
class VelocityConfig:
    """Configuration for velocity calculations.

    Controls how velocity metrics are calculated including rolling window
    size and trend detection thresholds.

    Attributes:
        rolling_window: Number of sprints for rolling average (default 5)
        trend_threshold: Threshold for trend detection as ratio (default 0.1 = 10%)
        min_sprints_for_trend: Minimum sprints needed to calculate trend (default 3)
        min_sprints_for_forecast: Minimum sprints needed for forecasting (default 2)

    Example:
        >>> config = VelocityConfig(
        ...     rolling_window=3,
        ...     trend_threshold=0.15,
        ... )
        >>> config.rolling_window
        3
    """

    rolling_window: int = DEFAULT_ROLLING_WINDOW
    trend_threshold: float = DEFAULT_TREND_THRESHOLD
    min_sprints_for_trend: int = DEFAULT_MIN_SPRINTS_FOR_TREND
    min_sprints_for_forecast: int = DEFAULT_MIN_SPRINTS_FOR_FORECAST

    def __post_init__(self) -> None:
        """Validate configuration values and log warnings for issues."""
        if self.rolling_window < 1:
            _logger.warning(
                "VelocityConfig rolling_window=%d should be at least 1",
                self.rolling_window,
            )
        if self.trend_threshold < 0 or self.trend_threshold > 1:
            _logger.warning(
                "VelocityConfig trend_threshold=%.3f should be between 0.0 and 1.0",
                self.trend_threshold,
            )
        if self.min_sprints_for_trend < 1:
            _logger.warning(
                "VelocityConfig min_sprints_for_trend=%d should be at least 1",
                self.min_sprints_for_trend,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "rolling_window": self.rolling_window,
            "trend_threshold": self.trend_threshold,
            "min_sprints_for_trend": self.min_sprints_for_trend,
            "min_sprints_for_forecast": self.min_sprints_for_forecast,
        }


@dataclass(frozen=True)
class VelocityForecast:
    """Forecasted velocity for planning.

    Provides predictions for the next sprint based on historical
    velocity data with confidence levels.

    Attributes:
        expected_stories_next_sprint: Predicted number of stories for next sprint
        expected_points_next_sprint: Predicted story points for next sprint
        confidence: Confidence level of the forecast (0.0-1.0)
        forecast_factors: Tuple of factors explaining the forecast calculation
        generated_at: ISO timestamp when forecast was generated (auto-generated)

    Example:
        >>> forecast = VelocityForecast(
        ...     expected_stories_next_sprint=5,
        ...     expected_points_next_sprint=5.0,
        ...     confidence=0.85,
        ...     forecast_factors=("based_on_5_sprints", "stable_trend"),
        ... )
        >>> forecast.confidence
        0.85
    """

    expected_stories_next_sprint: int
    expected_points_next_sprint: float
    confidence: float
    forecast_factors: tuple[str, ...]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate forecast values and log warnings for issues."""
        if self.confidence < MIN_CONFIDENCE or self.confidence > MAX_CONFIDENCE:
            _logger.warning(
                "VelocityForecast confidence=%.3f is outside valid range [%.1f, %.1f]",
                self.confidence,
                MIN_CONFIDENCE,
                MAX_CONFIDENCE,
            )
        if self.expected_stories_next_sprint < 0:
            _logger.warning(
                "VelocityForecast expected_stories_next_sprint=%d is negative",
                self.expected_stories_next_sprint,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the forecast.
            Note: forecast_factors tuple is converted to list for JSON compatibility.
        """
        return {
            "expected_stories_next_sprint": self.expected_stories_next_sprint,
            "expected_points_next_sprint": self.expected_points_next_sprint,
            "confidence": self.confidence,
            "forecast_factors": list(self.forecast_factors),
            "generated_at": self.generated_at,
        }
