"""Type definitions for health telemetry dashboard data (Story 10.16).

This module provides the data types used by the telemetry module:

- MetricStatus: Literal type for metric health status
- MetricSummary: Summary of a single metric for dashboard display
- TelemetrySnapshot: Complete telemetry snapshot for dashboard
- TelemetryConfig: Configuration for telemetry collection
- DashboardMetrics: Display-optimized metrics for dashboard rendering

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.telemetry_types import (
    ...     MetricSummary,
    ...     TelemetrySnapshot,
    ...     TelemetryConfig,
    ...     DashboardMetrics,
    ... )
    >>>
    >>> # Create a metric summary
    >>> summary = MetricSummary(
    ...     name="velocity",
    ...     value=5.2,
    ...     unit="stories/sprint",
    ...     display_value="5.2 stories/sprint",
    ...     trend="stable",
    ...     status="healthy",
    ... )
    >>> summary.to_dict()
    {'name': 'velocity', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR72: SM Agent can maintain system health telemetry dashboard data
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - FR67: SM Agent can detect agent churn rate and idle time
    - FR11: SM Agent can monitor agent activity and health metrics
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
# Literal Types (Subtask 1.1)
# =============================================================================

MetricStatus = Literal["healthy", "warning", "critical"]
"""Health status for a metric.

Values:
    healthy: Metric is within normal thresholds
    warning: Metric is approaching thresholds
    critical: Metric has exceeded thresholds
"""

MetricTrend = Literal["improving", "stable", "declining"]
"""Trend direction for a metric.

Values:
    improving: Metric is trending in a positive direction
    stable: Metric is consistent with historical values
    declining: Metric is trending in a negative direction
"""

MetricCategory = Literal["velocity", "cycle_time", "churn_rate", "idle_time", "health"]
"""Category of metric for filtering and grouping.

Values:
    velocity: Sprint velocity metrics (stories/sprint)
    cycle_time: Task completion time metrics
    churn_rate: Agent exchange rate metrics
    idle_time: Agent idle time metrics
    health: Overall system health metrics
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

DEFAULT_TELEMETRY_INTERVAL_SECONDS: float = 60.0
"""Default interval between telemetry collections (1 minute)."""

DEFAULT_TELEMETRY_RETENTION_HOURS: int = 24
"""Default retention period for telemetry snapshots (24 hours)."""

MIN_TELEMETRY_INTERVAL_SECONDS: float = 5.0
"""Minimum allowed telemetry collection interval."""

MAX_TELEMETRY_INTERVAL_SECONDS: float = 3600.0
"""Maximum allowed telemetry collection interval (1 hour)."""

VALID_METRIC_STATUSES: frozenset[str] = frozenset({"healthy", "warning", "critical"})
"""Set of valid metric status values."""

VALID_METRIC_TRENDS: frozenset[str] = frozenset({"improving", "stable", "declining"})
"""Set of valid metric trend values."""

VALID_METRIC_CATEGORIES: frozenset[str] = frozenset(
    {"velocity", "cycle_time", "churn_rate", "idle_time", "health"}
)
"""Set of valid metric category values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class MetricSummary:
    """Summary of a single metric for dashboard display.

    Provides all information needed to render a metric on a dashboard,
    including the raw value, human-readable display, trend, and status.

    Attributes:
        name: Metric identifier (e.g., "velocity", "cycle_time")
        value: Raw numeric value of the metric
        unit: Unit of measurement (e.g., "stories/sprint", "ms")
        display_value: Human-readable formatted value (e.g., "5.2 stories/sprint")
        trend: Trend direction ("improving", "stable", "declining") or None
        status: Health status ("healthy", "warning", "critical")

    Example:
        >>> summary = MetricSummary(
        ...     name="velocity",
        ...     value=5.2,
        ...     unit="stories/sprint",
        ...     display_value="5.2 stories/sprint",
        ...     trend="stable",
        ...     status="healthy",
        ... )
        >>> summary.status
        'healthy'
    """

    name: str
    value: float
    unit: str
    display_value: str
    trend: MetricTrend | None
    status: MetricStatus

    def __post_init__(self) -> None:
        """Validate metric summary data and log warnings for issues."""
        if not self.name:
            _logger.warning("MetricSummary name is empty")
        if self.status not in VALID_METRIC_STATUSES:
            _logger.warning(
                "MetricSummary status='%s' is not a valid status for name=%s",
                self.status,
                self.name,
            )
        if self.trend is not None and self.trend not in VALID_METRIC_TRENDS:
            _logger.warning(
                "MetricSummary trend='%s' is not a valid trend for name=%s",
                self.trend,
                self.name,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the metric summary.
        """
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "display_value": self.display_value,
            "trend": self.trend,
            "status": self.status,
        }


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Complete telemetry snapshot for dashboard.

    Aggregates all metrics into a single point-in-time snapshot
    suitable for dashboard rendering.

    Attributes:
        burn_down_velocity: Velocity metric summary
        cycle_time: Cycle time metric summary
        churn_rate: Churn rate metric summary
        agent_idle_times: Mapping of agent name to idle time summary
        health_status: Overall system health status
        alert_count: Number of active health alerts
        collected_at: ISO timestamp when snapshot was taken (auto-generated)

    Example:
        >>> snapshot = TelemetrySnapshot(
        ...     burn_down_velocity=velocity_summary,
        ...     cycle_time=cycle_time_summary,
        ...     churn_rate=churn_summary,
        ...     agent_idle_times={"analyst": idle_summary},
        ...     health_status="healthy",
        ...     alert_count=0,
        ... )
        >>> snapshot.health_status
        'healthy'
    """

    burn_down_velocity: MetricSummary
    cycle_time: MetricSummary
    churn_rate: MetricSummary
    agent_idle_times: dict[str, MetricSummary]
    health_status: MetricStatus
    alert_count: int = 0
    collected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate telemetry snapshot data and log warnings for issues."""
        if self.health_status not in VALID_METRIC_STATUSES:
            _logger.warning(
                "TelemetrySnapshot health_status='%s' is not a valid status",
                self.health_status,
            )
        if self.alert_count < 0:
            _logger.warning(
                "TelemetrySnapshot alert_count=%d is negative",
                self.alert_count,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation with nested metric summaries.
        """
        return {
            "burn_down_velocity": self.burn_down_velocity.to_dict(),
            "cycle_time": self.cycle_time.to_dict(),
            "churn_rate": self.churn_rate.to_dict(),
            "agent_idle_times": {k: v.to_dict() for k, v in self.agent_idle_times.items()},
            "health_status": self.health_status,
            "alert_count": self.alert_count,
            "collected_at": self.collected_at,
        }


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for telemetry collection.

    Controls how frequently telemetry is collected and how long
    historical snapshots are retained.

    Attributes:
        collection_interval_seconds: Interval between collections (default 60s)
        retention_hours: How long to keep snapshots (default 24h)
        include_agent_details: Whether to include per-agent metrics (default True)
        enable_trend_analysis: Whether to calculate trends (default True)

    Example:
        >>> config = TelemetryConfig(collection_interval_seconds=30.0)
        >>> config.collection_interval_seconds
        30.0
    """

    collection_interval_seconds: float = DEFAULT_TELEMETRY_INTERVAL_SECONDS
    retention_hours: int = DEFAULT_TELEMETRY_RETENTION_HOURS
    include_agent_details: bool = True
    enable_trend_analysis: bool = True

    def __post_init__(self) -> None:
        """Validate config values and log warnings for issues."""
        if self.collection_interval_seconds < MIN_TELEMETRY_INTERVAL_SECONDS:
            _logger.warning(
                "TelemetryConfig collection_interval_seconds=%.1f is below minimum %.1f",
                self.collection_interval_seconds,
                MIN_TELEMETRY_INTERVAL_SECONDS,
            )
        if self.collection_interval_seconds > MAX_TELEMETRY_INTERVAL_SECONDS:
            _logger.warning(
                "TelemetryConfig collection_interval_seconds=%.1f exceeds maximum %.1f",
                self.collection_interval_seconds,
                MAX_TELEMETRY_INTERVAL_SECONDS,
            )
        if self.retention_hours <= 0:
            _logger.warning(
                "TelemetryConfig retention_hours=%d should be positive",
                self.retention_hours,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "collection_interval_seconds": self.collection_interval_seconds,
            "retention_hours": self.retention_hours,
            "include_agent_details": self.include_agent_details,
            "enable_trend_analysis": self.enable_trend_analysis,
        }


@dataclass(frozen=True)
class DashboardMetrics:
    """Display-optimized metrics for dashboard rendering.

    Provides pre-formatted strings and data structures optimized
    for rendering in a dashboard UI.

    Attributes:
        snapshot: The underlying telemetry snapshot
        velocity_display: Human-readable velocity (e.g., "5.2 stories/sprint (stable)")
        cycle_time_display: Human-readable cycle time (e.g., "45 min avg (p90: 1.2h)")
        health_summary: Overall health sentence (e.g., "System healthy, 0 alerts")
        agent_status_table: List of dicts with agent status info for table rendering

    Example:
        >>> metrics = DashboardMetrics(
        ...     snapshot=snapshot,
        ...     velocity_display="5.2 stories/sprint (stable)",
        ...     cycle_time_display="45 min avg (p90: 1.2h)",
        ...     health_summary="System healthy, 0 alerts",
        ...     agent_status_table=[{"agent": "analyst", "idle_time": "30s", "status": "healthy"}],
        ... )
        >>> metrics.health_summary
        'System healthy, 0 alerts'
    """

    snapshot: TelemetrySnapshot
    velocity_display: str
    cycle_time_display: str
    health_summary: str
    agent_status_table: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        """Validate dashboard metrics and log warnings for issues."""
        if not self.velocity_display:
            _logger.warning("DashboardMetrics velocity_display is empty")
        if not self.cycle_time_display:
            _logger.warning("DashboardMetrics cycle_time_display is empty")
        if not self.health_summary:
            _logger.warning("DashboardMetrics health_summary is empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation with nested snapshot.
        """
        return {
            "snapshot": self.snapshot.to_dict(),
            "velocity_display": self.velocity_display,
            "cycle_time_display": self.cycle_time_display,
            "health_summary": self.health_summary,
            "agent_status_table": list(self.agent_status_table),
        }
