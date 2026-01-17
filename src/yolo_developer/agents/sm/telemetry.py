"""Health telemetry dashboard data module (Story 10.16).

This module provides health telemetry data for dashboard display:

- Collects and aggregates health metrics from HealthStatus
- Collects velocity metrics from VelocityMetrics
- Formats data for human-readable dashboard display
- Provides JSON-serializable output via to_dict() methods

Key Concepts:
- **Non-blocking**: Telemetry collection never blocks the main workflow
- **Graceful degradation**: Returns safe defaults when data is unavailable
- **Immutable outputs**: Returns frozen dataclasses

Example:
    >>> from yolo_developer.agents.sm.telemetry import get_dashboard_telemetry
    >>> from yolo_developer.agents.sm.telemetry_types import TelemetryConfig
    >>>
    >>> # Get dashboard telemetry with default config
    >>> metrics = await get_dashboard_telemetry(state)
    >>> metrics.health_summary
    'System healthy, 0 alerts'
    >>>
    >>> # Get telemetry with custom config
    >>> config = TelemetryConfig(include_agent_details=False)
    >>> metrics = await get_dashboard_telemetry(state, config)

Architecture Note:
    Per ADR-007, this module follows error handling patterns that ensure
    telemetry collection failures never block the main orchestration flow.

References:
    - FR72: SM Agent can maintain system health telemetry dashboard data
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - FR67: SM Agent can detect agent churn rate and idle time
    - FR11: SM Agent can monitor agent activity and health metrics
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from yolo_developer.agents.sm.telemetry_types import (
    DashboardMetrics,
    MetricStatus,
    MetricSummary,
    TelemetryConfig,
    TelemetrySnapshot,
)

if TYPE_CHECKING:
    from yolo_developer.agents.sm.health_types import HealthStatus
    from yolo_developer.agents.sm.velocity_types import VelocityMetrics
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)


# =============================================================================
# Telemetry Collection (Task 2)
# =============================================================================


def _aggregate_burn_down_velocity(velocity: VelocityMetrics | None) -> MetricSummary:
    """Aggregate burn-down velocity from VelocityMetrics.

    Creates a MetricSummary for velocity suitable for dashboard display,
    including trend information when available.

    Args:
        velocity: VelocityMetrics from velocity tracking, or None if unavailable.

    Returns:
        MetricSummary with velocity data or safe defaults.

    References:
        - FR66: SM Agent can track burn-down velocity and cycle time metrics
        - AC #1: burn-down velocity is available
    """
    if velocity is None:
        logger.debug("velocity_metrics_unavailable", message="Using default velocity summary")
        return MetricSummary(
            name="velocity",
            value=0.0,
            unit="stories/sprint",
            display_value="No data",
            trend=None,
            status="healthy",
        )

    value = velocity.average_stories_per_sprint
    trend = velocity.trend if velocity.sprints_analyzed >= 2 else None

    # Determine status based on trend
    status: MetricStatus = "healthy"
    if trend == "declining":
        status = "warning"

    # Format display value with trend
    display_parts = [f"{value:.1f} stories/sprint"]
    if trend:
        display_parts.append(f"({trend})")

    return MetricSummary(
        name="velocity",
        value=value,
        unit="stories/sprint",
        display_value=" ".join(display_parts),
        trend=trend,
        status=status,
    )


def _aggregate_cycle_time(
    health: HealthStatus | None,
    velocity: VelocityMetrics | None,
) -> MetricSummary:
    """Aggregate cycle time from health and velocity metrics.

    Combines cycle time data from both health monitoring (per-agent)
    and velocity tracking (overall sprint) for dashboard display.

    Args:
        health: HealthStatus from health monitoring, or None if unavailable.
        velocity: VelocityMetrics from velocity tracking, or None if unavailable.

    Returns:
        MetricSummary with cycle time data or safe defaults.

    References:
        - FR66: SM Agent can track burn-down velocity and cycle time metrics
        - AC #2: cycle time is available
    """
    # Prefer velocity metrics for cycle time (more accurate for sprint-level)
    if velocity is not None and velocity.average_cycle_time_ms > 0:
        value_ms = velocity.average_cycle_time_ms
        p90_ms = velocity.cycle_time_p90_ms

        # Convert to human-readable
        value_display = _format_duration_ms(value_ms)
        p90_display = _format_duration_ms(p90_ms)

        return MetricSummary(
            name="cycle_time",
            value=value_ms,
            unit="ms",
            display_value=f"{value_display} avg (p90: {p90_display})",
            trend=None,  # Cycle time trend not directly tracked
            status="healthy",
        )

    # Fall back to health metrics if velocity unavailable
    if health is not None and health.metrics.overall_cycle_time > 0:
        value_sec = health.metrics.overall_cycle_time
        value_ms = value_sec * 1000

        return MetricSummary(
            name="cycle_time",
            value=value_ms,
            unit="ms",
            display_value=_format_duration_ms(value_ms),
            trend=None,
            status="healthy",
        )

    logger.debug("cycle_time_unavailable", message="Using default cycle time summary")
    return MetricSummary(
        name="cycle_time",
        value=0.0,
        unit="ms",
        display_value="No data",
        trend=None,
        status="healthy",
    )


def _aggregate_churn_rate(health: HealthStatus | None) -> MetricSummary:
    """Aggregate churn rate from health metrics.

    Creates a MetricSummary for system-wide churn rate,
    with status based on configured thresholds.

    Args:
        health: HealthStatus from health monitoring, or None if unavailable.

    Returns:
        MetricSummary with churn rate data or safe defaults.

    References:
        - FR67: SM Agent can detect agent churn rate and idle time
        - AC #3: churn rate is available
    """
    if health is None:
        logger.debug("health_status_unavailable", message="Using default churn rate summary")
        return MetricSummary(
            name="churn_rate",
            value=0.0,
            unit="exchanges/min",
            display_value="No data",
            trend=None,
            status="healthy",
        )

    value = health.metrics.overall_churn_rate

    # Determine status based on churn rate thresholds
    # High churn (>10/min) indicates potential issues
    status: MetricStatus = "healthy"
    if value > 10.0:
        status = "critical"
    elif value > 7.0:
        status = "warning"

    return MetricSummary(
        name="churn_rate",
        value=value,
        unit="exchanges/min",
        display_value=f"{value:.1f} exchanges/min",
        trend=None,
        status=status,
    )


def _aggregate_idle_times(health: HealthStatus | None) -> dict[str, MetricSummary]:
    """Aggregate per-agent idle times from health metrics.

    Creates MetricSummary objects for each agent's idle time,
    with status based on configured thresholds.

    Args:
        health: HealthStatus from health monitoring, or None if unavailable.

    Returns:
        Dictionary mapping agent name to MetricSummary.

    References:
        - FR67: SM Agent can detect agent churn rate and idle time
        - AC #4: agent idle time is available
    """
    if health is None:
        logger.debug("health_status_unavailable", message="Using empty idle times")
        return {}

    result: dict[str, MetricSummary] = {}
    for agent, idle_seconds in health.metrics.agent_idle_times.items():
        # Determine status based on idle time thresholds
        # >300s (5 min) is warning, >600s (10 min) is critical
        status: MetricStatus = "healthy"
        if idle_seconds > 600:
            status = "critical"
        elif idle_seconds > 300:
            status = "warning"

        result[agent] = MetricSummary(
            name=f"idle_time_{agent}",
            value=idle_seconds,
            unit="seconds",
            display_value=_format_duration_seconds(idle_seconds),
            trend=None,
            status=status,
        )

    return result


def collect_telemetry(
    state: YoloState,
    health_status: HealthStatus | None,
    velocity_metrics: VelocityMetrics | None,
) -> TelemetrySnapshot:
    """Collect and aggregate all telemetry data into a snapshot.

    Combines health and velocity metrics into a single snapshot
    suitable for dashboard display.

    Args:
        state: Current orchestration state.
        health_status: HealthStatus from health monitoring, or None.
        velocity_metrics: VelocityMetrics from velocity tracking, or None.

    Returns:
        TelemetrySnapshot with all aggregated metrics.

    References:
        - FR72: SM Agent can maintain system health telemetry dashboard data
        - AC #1, #2, #3, #4: All metrics are available
    """
    logger.debug(
        "collecting_telemetry",
        has_health=health_status is not None,
        has_velocity=velocity_metrics is not None,
    )

    # Aggregate individual metrics
    burn_down_velocity = _aggregate_burn_down_velocity(velocity_metrics)
    cycle_time = _aggregate_cycle_time(health_status, velocity_metrics)
    churn_rate = _aggregate_churn_rate(health_status)
    agent_idle_times = _aggregate_idle_times(health_status)

    # Determine overall health status
    overall_status: MetricStatus = "healthy"
    alert_count = 0

    if health_status is not None:
        # Map health severity to metric status
        health_severity = health_status.status
        if health_severity in ("critical", "degraded"):
            overall_status = "critical"
        elif health_severity == "warning":
            overall_status = "warning"
        # else remains "healthy"
        alert_count = len(health_status.alerts)
    else:
        # Check individual metric statuses if no health status
        statuses = [burn_down_velocity.status, cycle_time.status, churn_rate.status]
        statuses.extend(m.status for m in agent_idle_times.values())
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"

    snapshot = TelemetrySnapshot(
        burn_down_velocity=burn_down_velocity,
        cycle_time=cycle_time,
        churn_rate=churn_rate,
        agent_idle_times=agent_idle_times,
        health_status=overall_status,
        alert_count=alert_count,
    )

    logger.info(
        "telemetry_collected",
        health_status=overall_status,
        alert_count=alert_count,
        velocity_value=burn_down_velocity.value,
        churn_rate_value=churn_rate.value,
        agent_count=len(agent_idle_times),
    )

    return snapshot


# =============================================================================
# Dashboard Formatting (Task 3)
# =============================================================================


def _format_duration_ms(ms: float) -> str:
    """Format milliseconds as human-readable duration.

    Args:
        ms: Duration in milliseconds.

    Returns:
        Human-readable string (e.g., "45 min", "1.2h", "30 sec").
    """
    if ms <= 0:
        return "0 sec"

    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.0f} sec"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min"

    hours = minutes / 60
    return f"{hours:.1f}h"


def _format_duration_seconds(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string (e.g., "45s", "5 min", "1.2h").
    """
    if seconds <= 0:
        return "0s"

    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min"

    hours = minutes / 60
    return f"{hours:.1f}h"


def _format_velocity_display(velocity: MetricSummary) -> str:
    """Format velocity metric for dashboard display.

    Args:
        velocity: MetricSummary containing velocity data.

    Returns:
        Human-readable velocity string.

    References:
        - AC #5: Data is formatted for display
    """
    if velocity.value <= 0 or velocity.display_value == "No data":
        return "No velocity data"

    return velocity.display_value


def _format_cycle_time_display(cycle_time: MetricSummary) -> str:
    """Format cycle time metric for dashboard display.

    Args:
        cycle_time: MetricSummary containing cycle time data.

    Returns:
        Human-readable cycle time string.

    References:
        - AC #5: Data is formatted for display
    """
    if cycle_time.value <= 0 or cycle_time.display_value == "No data":
        return "No cycle time data"

    return cycle_time.display_value


def _format_health_summary(snapshot: TelemetrySnapshot) -> str:
    """Format overall health summary for dashboard display.

    Args:
        snapshot: TelemetrySnapshot containing health data.

    Returns:
        Human-readable health summary sentence.

    References:
        - AC #5: Data is formatted for display
    """
    status_display = snapshot.health_status.capitalize()

    if snapshot.alert_count == 0:
        return f"System {status_display.lower()}, no alerts"
    elif snapshot.alert_count == 1:
        return f"System {status_display.lower()}, 1 alert"
    else:
        return f"System {status_display.lower()}, {snapshot.alert_count} alerts"


def _format_agent_status_table(
    idle_times: dict[str, MetricSummary],
) -> tuple[dict[str, str], ...]:
    """Format agent status data for table rendering.

    Args:
        idle_times: Dictionary mapping agent name to idle time summary.

    Returns:
        Tuple of dicts with agent status info for table rendering.

    References:
        - AC #5: Data is formatted for display
    """
    if not idle_times:
        return ()

    result: list[dict[str, str]] = []
    for agent, summary in sorted(idle_times.items()):
        result.append(
            {
                "agent": agent,
                "idle_time": summary.display_value,
                "status": summary.status,
            }
        )

    return tuple(result)


def format_for_dashboard(snapshot: TelemetrySnapshot) -> DashboardMetrics:
    """Format telemetry snapshot for dashboard rendering.

    Transforms raw metrics into human-readable strings and
    structures optimized for UI display.

    Args:
        snapshot: TelemetrySnapshot with collected metrics.

    Returns:
        DashboardMetrics with formatted display strings.

    References:
        - FR72: SM Agent can maintain system health telemetry dashboard data
        - AC #5: Data is formatted for display
    """
    logger.debug("formatting_for_dashboard", health_status=snapshot.health_status)

    return DashboardMetrics(
        snapshot=snapshot,
        velocity_display=_format_velocity_display(snapshot.burn_down_velocity),
        cycle_time_display=_format_cycle_time_display(snapshot.cycle_time),
        health_summary=_format_health_summary(snapshot),
        agent_status_table=_format_agent_status_table(snapshot.agent_idle_times),
    )


# =============================================================================
# Main Telemetry Function (Task 4)
# =============================================================================


async def get_dashboard_telemetry(
    state: YoloState,
    config: TelemetryConfig | None = None,
    health_status: HealthStatus | None = None,
    velocity_metrics: VelocityMetrics | None = None,
) -> DashboardMetrics:
    """Get formatted dashboard telemetry data.

    Main entry point for collecting and formatting telemetry.
    Handles errors gracefully and returns safe defaults on failure.

    Args:
        state: Current orchestration state.
        config: Optional telemetry configuration.
        health_status: Optional pre-collected HealthStatus.
        velocity_metrics: Optional pre-collected VelocityMetrics.

    Returns:
        DashboardMetrics with formatted telemetry data.

    References:
        - FR72: SM Agent can maintain system health telemetry dashboard data
        - AC #1, #2, #3, #4, #5: All metrics available and formatted
    """
    if config is None:
        config = TelemetryConfig()

    logger.info(
        "getting_dashboard_telemetry",
        include_agent_details=config.include_agent_details,
        enable_trend_analysis=config.enable_trend_analysis,
    )

    try:
        # Collect telemetry
        snapshot = collect_telemetry(state, health_status, velocity_metrics)

        # Filter agent details if disabled
        if not config.include_agent_details:
            # Create new snapshot without agent details
            snapshot = TelemetrySnapshot(
                burn_down_velocity=snapshot.burn_down_velocity,
                cycle_time=snapshot.cycle_time,
                churn_rate=snapshot.churn_rate,
                agent_idle_times={},
                health_status=snapshot.health_status,
                alert_count=snapshot.alert_count,
                collected_at=snapshot.collected_at,
            )

        # Format for dashboard
        metrics = format_for_dashboard(snapshot)

        logger.info(
            "dashboard_telemetry_complete",
            health_status=snapshot.health_status,
            velocity=metrics.velocity_display,
        )

        return metrics

    except Exception as e:
        # Per ADR-007: Return safe defaults, never block main workflow
        logger.exception(
            "telemetry_collection_failed",
            error=str(e),
            message="Returning safe default metrics",
        )
        return _create_default_dashboard_metrics()


def _create_default_dashboard_metrics() -> DashboardMetrics:
    """Create default DashboardMetrics for error cases.

    Returns:
        DashboardMetrics with safe default values.
    """
    default_summary = MetricSummary(
        name="unknown",
        value=0.0,
        unit="",
        display_value="Unavailable",
        trend=None,
        status="healthy",
    )

    snapshot = TelemetrySnapshot(
        burn_down_velocity=default_summary,
        cycle_time=default_summary,
        churn_rate=default_summary,
        agent_idle_times={},
        health_status="healthy",
        alert_count=0,
        collected_at=datetime.now(timezone.utc).isoformat(),
    )

    return DashboardMetrics(
        snapshot=snapshot,
        velocity_display="Data unavailable",
        cycle_time_display="Data unavailable",
        health_summary="Telemetry collection failed",
        agent_status_table=(),
    )
