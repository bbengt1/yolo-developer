"""Type definitions for health monitoring (Story 10.5).

This module provides the data types used by the health monitoring module:

- HealthSeverity: Literal type for overall health status levels
- AlertSeverity: Literal type for alert severity levels
- AgentHealthSnapshot: Point-in-time health snapshot for an agent
- HealthMetrics: Comprehensive health metrics for the system
- HealthAlert: Alert triggered by health monitoring
- HealthConfig: Configuration for health monitoring thresholds
- HealthStatus: Overall system health status

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.health_types import (
    ...     HealthMetrics,
    ...     HealthStatus,
    ...     HealthConfig,
    ...     HealthAlert,
    ... )
    >>>
    >>> # Create health config with custom thresholds
    >>> config = HealthConfig(max_idle_time_seconds=600.0)
    >>> config.max_idle_time_seconds
    600.0
    >>>
    >>> # Create health alert
    >>> alert = HealthAlert(
    ...     severity="warning",
    ...     alert_type="idle_time_warning",
    ...     message="Agent analyst idle for 250 seconds",
    ...     affected_agent="analyst",
    ...     metric_value=250.0,
    ...     threshold_value=210.0,
    ... )

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR11: SM Agent can monitor agent activity and health metrics
    - FR67: SM Agent can detect agent churn rate and idle time
    - FR17: SM Agent can trigger emergency protocols when system health degrades
    - FR72: SM Agent can maintain system health telemetry dashboard data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

HealthSeverity = Literal["healthy", "warning", "degraded", "critical"]
"""Overall system health status level.

Values:
    healthy: All metrics within normal thresholds
    warning: Some metrics approaching thresholds
    degraded: Multiple warnings or minor threshold violations
    critical: Critical threshold violations requiring immediate attention
"""

AlertSeverity = Literal["info", "warning", "critical"]
"""Severity level for health alerts.

Values:
    info: Informational alert, no action required
    warning: Potential issue, should be monitored
    critical: Serious issue requiring immediate attention
"""

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_IDLE_TIME_SECONDS: float = 300.0
"""Default maximum idle time before triggering alert (5 minutes)."""

DEFAULT_MAX_CYCLE_TIME_SECONDS: float = 600.0
"""Default maximum cycle time before triggering alert (10 minutes)."""

DEFAULT_MAX_CHURN_RATE: float = 10.0
"""Default maximum churn rate (exchanges per minute)."""

DEFAULT_WARNING_THRESHOLD_RATIO: float = 0.7
"""Default ratio of max threshold that triggers warning (70%)."""

VALID_HEALTH_SEVERITIES: frozenset[str] = frozenset({"healthy", "warning", "degraded", "critical"})
"""Set of all valid health severity values."""

VALID_ALERT_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})
"""Set of all valid alert severity values."""

VALID_AGENTS_FOR_HEALTH: frozenset[str] = frozenset(
    {"analyst", "pm", "architect", "dev", "tea", "sm"}
)
"""Set of agents tracked for health monitoring."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class HealthConfig:
    """Configuration for health monitoring thresholds.

    Controls what metric values trigger warnings and critical alerts.

    Attributes:
        max_idle_time_seconds: Max time an agent can be idle before critical alert
        max_cycle_time_seconds: Max cycle time before warning alert
        max_churn_rate: Max exchanges per minute before critical alert
        warning_threshold_ratio: Ratio of max that triggers warning (0.0-1.0)
        enable_alerts: Whether to generate alerts (default True)

    Example:
        >>> config = HealthConfig(
        ...     max_idle_time_seconds=600.0,
        ...     max_churn_rate=5.0,
        ... )
        >>> config.warning_threshold_ratio
        0.7
    """

    max_idle_time_seconds: float = DEFAULT_MAX_IDLE_TIME_SECONDS
    max_cycle_time_seconds: float = DEFAULT_MAX_CYCLE_TIME_SECONDS
    max_churn_rate: float = DEFAULT_MAX_CHURN_RATE
    warning_threshold_ratio: float = DEFAULT_WARNING_THRESHOLD_RATIO
    enable_alerts: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "max_idle_time_seconds": self.max_idle_time_seconds,
            "max_cycle_time_seconds": self.max_cycle_time_seconds,
            "max_churn_rate": self.max_churn_rate,
            "warning_threshold_ratio": self.warning_threshold_ratio,
            "enable_alerts": self.enable_alerts,
        }


@dataclass(frozen=True)
class AgentHealthSnapshot:
    """Point-in-time health snapshot for an agent.

    Captures the current health metrics for a single agent at a
    specific moment in time.

    Attributes:
        agent: Name of the agent (e.g., "analyst", "pm")
        idle_time_seconds: Time since agent last acted (seconds)
        last_activity: ISO timestamp of last activity
        cycle_time_seconds: Average time to complete tasks (None if no data)
        churn_rate: Exchanges per minute involving this agent
        is_healthy: Whether agent is within healthy thresholds
        captured_at: ISO timestamp when snapshot was taken (auto-generated)

    Example:
        >>> snapshot = AgentHealthSnapshot(
        ...     agent="analyst",
        ...     idle_time_seconds=120.5,
        ...     last_activity="2026-01-12T10:00:00+00:00",
        ...     cycle_time_seconds=45.0,
        ...     churn_rate=2.5,
        ...     is_healthy=True,
        ... )
        >>> snapshot.to_dict()
        {'agent': 'analyst', ...}
    """

    agent: str
    idle_time_seconds: float
    last_activity: str
    cycle_time_seconds: float | None
    churn_rate: float
    is_healthy: bool
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the snapshot.
        """
        return {
            "agent": self.agent,
            "idle_time_seconds": self.idle_time_seconds,
            "last_activity": self.last_activity,
            "cycle_time_seconds": self.cycle_time_seconds,
            "churn_rate": self.churn_rate,
            "is_healthy": self.is_healthy,
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class HealthMetrics:
    """Comprehensive health metrics for the system.

    Aggregates all health-related metrics from all agents and
    provides system-wide health indicators.

    Attributes:
        agent_idle_times: Mapping of agent name to idle time in seconds
        agent_cycle_times: Mapping of agent name to avg cycle time in seconds
        agent_churn_rates: Mapping of agent name to churn rate (exchanges/min)
        overall_cycle_time: System-wide average cycle time in seconds
        overall_churn_rate: Total system churn rate (exchanges/min)
        unproductive_churn_rate: Unproductive exchanges per minute (same topic back-and-forth)
        cycle_time_percentiles: Rolling percentiles (p50, p90, p95) for cycle times
        agent_snapshots: Tuple of health snapshots for each agent
        collected_at: ISO timestamp when metrics were collected (auto-generated)

    Example:
        >>> metrics = HealthMetrics(
        ...     agent_idle_times={"analyst": 100.0, "pm": 50.0},
        ...     agent_cycle_times={"analyst": 30.0},
        ...     agent_churn_rates={"analyst": 2.0, "pm": 1.5},
        ...     overall_cycle_time=35.0,
        ...     overall_churn_rate=3.5,
        ...     unproductive_churn_rate=0.5,
        ...     cycle_time_percentiles={"p50": 30.0, "p90": 50.0, "p95": 60.0},
        ...     agent_snapshots=(),
        ... )
        >>> metrics.to_dict()
        {'agent_idle_times': {...}, ...}
    """

    agent_idle_times: dict[str, float]
    agent_cycle_times: dict[str, float]
    agent_churn_rates: dict[str, float]
    overall_cycle_time: float
    overall_churn_rate: float
    unproductive_churn_rate: float = 0.0
    cycle_time_percentiles: dict[str, float] = field(
        default_factory=lambda: {"p50": 0.0, "p90": 0.0, "p95": 0.0}
    )
    agent_snapshots: tuple[AgentHealthSnapshot, ...] = ()
    collected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested snapshots.
        """
        return {
            "agent_idle_times": dict(self.agent_idle_times),
            "agent_cycle_times": dict(self.agent_cycle_times),
            "agent_churn_rates": dict(self.agent_churn_rates),
            "overall_cycle_time": self.overall_cycle_time,
            "overall_churn_rate": self.overall_churn_rate,
            "unproductive_churn_rate": self.unproductive_churn_rate,
            "cycle_time_percentiles": dict(self.cycle_time_percentiles),
            "agent_snapshots": [s.to_dict() for s in self.agent_snapshots],
            "collected_at": self.collected_at,
        }


@dataclass(frozen=True)
class HealthAlert:
    """Alert triggered by health monitoring.

    Represents a specific health issue that was detected and needs
    attention based on configured thresholds.

    Attributes:
        severity: How serious the alert is (info, warning, critical)
        alert_type: Category of alert (e.g., "idle_time_exceeded", "high_churn")
        message: Human-readable description of the issue
        affected_agent: Agent causing the alert (None for system-wide alerts)
        metric_value: The actual metric value that triggered the alert
        threshold_value: The threshold that was exceeded
        triggered_at: ISO timestamp when alert was triggered (auto-generated)

    Example:
        >>> alert = HealthAlert(
        ...     severity="warning",
        ...     alert_type="idle_time_warning",
        ...     message="Agent analyst idle for 250 seconds",
        ...     affected_agent="analyst",
        ...     metric_value=250.0,
        ...     threshold_value=210.0,
        ... )
        >>> alert.severity
        'warning'
    """

    severity: AlertSeverity
    alert_type: str
    message: str
    affected_agent: str | None
    metric_value: float
    threshold_value: float
    triggered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the alert.
        """
        return {
            "severity": self.severity,
            "alert_type": self.alert_type,
            "message": self.message,
            "affected_agent": self.affected_agent,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "triggered_at": self.triggered_at,
        }


@dataclass(frozen=True)
class HealthStatus:
    """Overall system health status.

    The complete result of a health monitoring check, including
    the overall status, all metrics, and any alerts generated.

    Attributes:
        status: Overall health severity (healthy, warning, degraded, critical)
        metrics: Complete health metrics snapshot
        alerts: Tuple of all triggered alerts
        summary: Human-readable summary of system health
        is_healthy: Whether system is considered healthy (healthy or warning)
        evaluated_at: ISO timestamp when evaluation occurred (auto-generated)

    Example:
        >>> status = HealthStatus(
        ...     status="healthy",
        ...     metrics=metrics,
        ...     alerts=(),
        ...     summary="All systems nominal",
        ...     is_healthy=True,
        ... )
        >>> status.is_healthy
        True
    """

    status: HealthSeverity
    metrics: HealthMetrics
    alerts: tuple[HealthAlert, ...]
    summary: str
    is_healthy: bool
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested metrics and alerts.
        """
        return {
            "status": self.status,
            "metrics": self.metrics.to_dict(),
            "alerts": [a.to_dict() for a in self.alerts],
            "summary": self.summary,
            "is_healthy": self.is_healthy,
            "evaluated_at": self.evaluated_at,
        }
