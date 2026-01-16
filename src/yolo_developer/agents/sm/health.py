"""Health monitoring module for SM agent (Story 10.5).

This module provides health monitoring functionality for the orchestration system:

- Tracks agent idle time (AC #1, FR67)
- Measures cycle time (AC #2, FR11)
- Calculates churn rate (AC #3, FR67)
- Detects anomalies and triggers alerts (AC #4, FR17)

Key Concepts:
- **Non-blocking**: Health monitoring never blocks the main workflow
- **Immutable outputs**: Returns frozen dataclasses
- **Structured logging**: Uses structlog for audit trail

Example:
    >>> from yolo_developer.agents.sm.health import monitor_health
    >>> from yolo_developer.agents.sm.health_types import HealthConfig
    >>>
    >>> # Monitor health with default config
    >>> status = await monitor_health(state)
    >>> status.is_healthy
    True
    >>>
    >>> # Monitor with custom thresholds
    >>> config = HealthConfig(max_idle_time_seconds=600.0)
    >>> status = await monitor_health(state, config)

Architecture Note:
    Per ADR-007, this module follows error handling patterns that ensure
    health monitoring failures never block the main orchestration flow.

References:
    - FR11: SM Agent can monitor agent activity and health metrics
    - FR67: SM Agent can detect agent churn rate and idle time
    - FR17: SM Agent can trigger emergency protocols when system health degrades
    - FR72: SM Agent can maintain system health telemetry dashboard data
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import structlog

from yolo_developer.agents.sm.health_types import (
    VALID_AGENTS_FOR_HEALTH,
    AgentHealthSnapshot,
    HealthAlert,
    HealthConfig,
    HealthMetrics,
    HealthSeverity,
    HealthStatus,
)

if TYPE_CHECKING:
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)


# =============================================================================
# Idle Time Tracking (Task 2 - AC #1)
# =============================================================================


def _get_current_timestamp() -> str:
    """Get current UTC timestamp in ISO format.

    Returns:
        ISO formatted timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime.

    Args:
        timestamp_str: ISO formatted timestamp string.

    Returns:
        Parsed datetime object (UTC).
    """
    # Handle both Z suffix and +00:00 formats
    if timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1] + "+00:00"
    return datetime.fromisoformat(timestamp_str)


def _calculate_idle_time(last_activity: str, current_time: datetime) -> float:
    """Calculate idle time in seconds since last activity.

    Args:
        last_activity: ISO timestamp of last activity.
        current_time: Current datetime.

    Returns:
        Idle time in seconds.
    """
    last_dt = _parse_timestamp(last_activity)
    delta = current_time - last_dt
    return max(0.0, delta.total_seconds())


def _extract_agent_from_message(msg: Any) -> str | None:
    """Extract agent name from a message.

    Args:
        msg: Message object (may have additional_kwargs).

    Returns:
        Agent name or None if not found.
    """
    if hasattr(msg, "additional_kwargs"):
        agent = msg.additional_kwargs.get("agent")
        if isinstance(agent, str):
            return agent
    return None


def _extract_timestamp_from_message(msg: Any) -> str | None:
    """Extract timestamp from a message.

    Args:
        msg: Message object (may have additional_kwargs).

    Returns:
        ISO timestamp string or None if not found.
    """
    if hasattr(msg, "additional_kwargs"):
        timestamp = msg.additional_kwargs.get("timestamp")
        if isinstance(timestamp, str):
            return timestamp
    return None


def _calculate_agent_idle_times(state: YoloState) -> dict[str, float]:
    """Calculate idle time for each agent based on message history.

    Finds the most recent activity for each agent and calculates
    how long since they last acted.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary mapping agent name to idle time in seconds.
    """
    current_time = datetime.now(timezone.utc)
    messages = state.get("messages") or []

    # Handle invalid message types gracefully
    if not isinstance(messages, list):
        messages = []

    # Track last activity timestamp for each agent
    last_activity: dict[str, datetime] = {}

    for msg in messages:
        agent = _extract_agent_from_message(msg)
        timestamp_str = _extract_timestamp_from_message(msg)

        if agent and agent in VALID_AGENTS_FOR_HEALTH:
            if timestamp_str:
                try:
                    msg_time = _parse_timestamp(timestamp_str)
                    if agent not in last_activity or msg_time > last_activity[agent]:
                        last_activity[agent] = msg_time
                except ValueError:
                    # Skip invalid timestamps
                    pass
            else:
                # Use current time as fallback (message has no timestamp)
                if agent not in last_activity:
                    last_activity[agent] = current_time

    # Calculate idle times
    idle_times: dict[str, float] = {}
    for agent in VALID_AGENTS_FOR_HEALTH:
        if agent in last_activity:
            delta = current_time - last_activity[agent]
            idle_times[agent] = max(0.0, delta.total_seconds())
        else:
            # Agent never acted - consider idle since start
            idle_times[agent] = 0.0  # No activity means we can't measure idle

    return idle_times


def _track_agent_activity(state: YoloState, agent: str) -> str:
    """Get the timestamp of an agent's last activity.

    Args:
        state: Current orchestration state.
        agent: Agent name to look up.

    Returns:
        ISO timestamp of last activity (or current time if none).
    """
    messages = state.get("messages") or []

    # Handle invalid message types gracefully
    if not isinstance(messages, list):
        return _get_current_timestamp()

    for msg in reversed(messages):  # Most recent first
        msg_agent = _extract_agent_from_message(msg)
        if msg_agent == agent:
            timestamp = _extract_timestamp_from_message(msg)
            if timestamp:
                return timestamp

    return _get_current_timestamp()


# =============================================================================
# Cycle Time Measurement (Task 3 - AC #2)
# =============================================================================


def _calculate_agent_cycle_times(state: YoloState) -> dict[str, float]:
    """Calculate average cycle time for each agent.

    Cycle time is the average time between when an agent starts
    and finishes processing a task.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary mapping agent name to avg cycle time in seconds.
    """
    decisions = state.get("decisions") or []

    # Handle invalid decision types gracefully
    if not isinstance(decisions, list):
        decisions = []

    # Track decision timestamps per agent
    agent_times: dict[str, list[float]] = {agent: [] for agent in VALID_AGENTS_FOR_HEALTH}

    prev_timestamp: datetime | None = None
    prev_agent: str | None = None

    for decision in decisions:
        # Extract timestamp from decision
        timestamp = None
        if hasattr(decision, "timestamp"):
            timestamp = decision.timestamp
        elif isinstance(decision, dict) and "timestamp" in decision:
            ts_value = decision["timestamp"]
            if isinstance(ts_value, str):
                try:
                    timestamp = _parse_timestamp(ts_value)
                except ValueError:
                    pass
            elif isinstance(ts_value, datetime):
                timestamp = ts_value

        # Extract agent from decision
        agent = None
        if hasattr(decision, "agent"):
            agent = decision.agent
        elif isinstance(decision, dict) and "agent" in decision:
            agent = decision["agent"]

        if timestamp and agent and agent in VALID_AGENTS_FOR_HEALTH:
            # Calculate time since last decision by same agent
            if prev_timestamp and prev_agent == agent:
                delta = (timestamp - prev_timestamp).total_seconds()
                if delta > 0:
                    agent_times[agent].append(delta)

            prev_timestamp = timestamp
            prev_agent = agent

    # Calculate averages
    cycle_times: dict[str, float] = {}
    for agent, times in agent_times.items():
        if times:
            cycle_times[agent] = sum(times) / len(times)

    return cycle_times


def _calculate_overall_cycle_time(state: YoloState) -> float:
    """Calculate overall system cycle time.

    Args:
        state: Current orchestration state.

    Returns:
        Average cycle time across all agents (0.0 if no data).
    """
    agent_cycle_times = _calculate_agent_cycle_times(state)
    if not agent_cycle_times:
        return 0.0
    return sum(agent_cycle_times.values()) / len(agent_cycle_times)


# =============================================================================
# Percentile Calculations (Task 3.5)
# =============================================================================


def _calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate a percentile value from a list of numbers.

    Args:
        values: List of numeric values.
        percentile: Percentile to calculate (0-100).

    Returns:
        The percentile value, or 0.0 if no values.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Calculate position (using linear interpolation)
    position = (percentile / 100.0) * (n - 1)
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = position - lower_idx

    return sorted_values[lower_idx] + fraction * (sorted_values[upper_idx] - sorted_values[lower_idx])


def _calculate_cycle_time_percentiles(
    state: YoloState,
) -> dict[str, float]:
    """Calculate rolling percentiles (p50, p90, p95) for cycle times.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary with p50, p90, p95 percentile values.
    """
    decisions = state.get("decisions") or []

    if not isinstance(decisions, list):
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0}

    # Collect all cycle times from decisions
    cycle_times: list[float] = []
    prev_timestamp: datetime | None = None

    for decision in decisions:
        timestamp = None
        if hasattr(decision, "timestamp"):
            timestamp = decision.timestamp
        elif isinstance(decision, dict) and "timestamp" in decision:
            ts_value = decision["timestamp"]
            if isinstance(ts_value, str):
                try:
                    timestamp = _parse_timestamp(ts_value)
                except ValueError:
                    pass
            elif isinstance(ts_value, datetime):
                timestamp = ts_value

        if timestamp and prev_timestamp:
            delta = (timestamp - prev_timestamp).total_seconds()
            if delta > 0:
                cycle_times.append(delta)

        prev_timestamp = timestamp

    return {
        "p50": _calculate_percentile(cycle_times, 50),
        "p90": _calculate_percentile(cycle_times, 90),
        "p95": _calculate_percentile(cycle_times, 95),
    }


# =============================================================================
# Churn Rate Calculation (Task 4 - AC #3)
# =============================================================================


def _count_exchanges_in_window(state: YoloState, window_seconds: float = 60.0) -> int:
    """Count agent exchanges within a time window.

    Args:
        state: Current orchestration state.
        window_seconds: Time window in seconds (default 60s = 1 minute).

    Returns:
        Number of exchanges in the window.
    """
    messages = state.get("messages") or []

    # Handle invalid message types gracefully
    if not isinstance(messages, list):
        return 0
    current_time = datetime.now(timezone.utc)
    cutoff_time = current_time.timestamp() - window_seconds

    exchange_count = 0
    prev_agent: str | None = None

    for msg in messages:
        agent = _extract_agent_from_message(msg)
        timestamp_str = _extract_timestamp_from_message(msg)

        if not agent:
            continue

        # Check if within time window
        if timestamp_str:
            try:
                msg_time = _parse_timestamp(timestamp_str)
                if msg_time.timestamp() < cutoff_time:
                    continue
            except ValueError:
                pass

        # Count exchange if agent changed
        if prev_agent and agent != prev_agent:
            exchange_count += 1

        prev_agent = agent

    return exchange_count


def _calculate_churn_rate(state: YoloState) -> float:
    """Calculate overall churn rate (exchanges per minute).

    Args:
        state: Current orchestration state.

    Returns:
        Churn rate as exchanges per minute.
    """
    # Count exchanges in last minute
    exchange_count = _count_exchanges_in_window(state, window_seconds=60.0)
    return float(exchange_count)


def _calculate_agent_churn_rates(state: YoloState) -> dict[str, float]:
    """Calculate churn rate per agent.

    Churn rate for an agent is how many exchanges involve that agent
    per minute.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary mapping agent name to churn rate.
    """
    messages = state.get("messages") or []

    # Handle invalid message types gracefully
    if not isinstance(messages, list):
        return {}
    current_time = datetime.now(timezone.utc)
    cutoff_time = current_time.timestamp() - 60.0  # 1 minute window

    # Count exchanges involving each agent
    agent_exchange_counts: dict[str, int] = dict.fromkeys(VALID_AGENTS_FOR_HEALTH, 0)
    prev_agent: str | None = None

    for msg in messages:
        agent = _extract_agent_from_message(msg)
        timestamp_str = _extract_timestamp_from_message(msg)

        if not agent or agent not in VALID_AGENTS_FOR_HEALTH:
            continue

        # Check if within time window
        in_window = True
        if timestamp_str:
            try:
                msg_time = _parse_timestamp(timestamp_str)
                if msg_time.timestamp() < cutoff_time:
                    in_window = False
            except ValueError:
                pass

        # Count exchange if agent changed and in window
        if in_window and prev_agent and agent != prev_agent:
            agent_exchange_counts[agent] = agent_exchange_counts.get(agent, 0) + 1
            if prev_agent in VALID_AGENTS_FOR_HEALTH:
                agent_exchange_counts[prev_agent] = agent_exchange_counts.get(prev_agent, 0) + 1

        prev_agent = agent

    # Return only agents with activity
    return {agent: float(count) for agent, count in agent_exchange_counts.items() if count > 0}


def _extract_topic_from_message(msg: Any) -> str:
    """Extract topic/reason from a message for unproductive exchange detection.

    Args:
        msg: Message object (may have additional_kwargs).

    Returns:
        Topic string or empty string if not found.
    """
    if hasattr(msg, "additional_kwargs"):
        topic = msg.additional_kwargs.get("topic")
        if topic and isinstance(topic, str):
            return str(topic)
        # Fall back to extracting from reason if topic not available
        reason = msg.additional_kwargs.get("reason")
        if reason and isinstance(reason, str):
            return str(reason)
    return ""


def _count_unproductive_exchanges(state: YoloState, window_seconds: float = 60.0) -> int:
    """Count unproductive exchanges (same topic back-and-forth) within a time window.

    An unproductive exchange is when agents ping-pong on the same topic,
    indicating they may be stuck or in a loop.

    Args:
        state: Current orchestration state.
        window_seconds: Time window in seconds (default 60s).

    Returns:
        Number of unproductive exchanges detected.
    """
    messages = state.get("messages") or []

    if not isinstance(messages, list):
        return 0

    current_time = datetime.now(timezone.utc)
    cutoff_time = current_time.timestamp() - window_seconds

    # Track exchanges with their topics
    unproductive_count = 0
    prev_agent: str | None = None
    prev_topic: str | None = None
    prev_prev_agent: str | None = None
    prev_prev_topic: str | None = None

    for msg in messages:
        agent = _extract_agent_from_message(msg)
        topic = _extract_topic_from_message(msg)
        timestamp_str = _extract_timestamp_from_message(msg)

        if not agent:
            continue

        # Check if within time window
        if timestamp_str:
            try:
                msg_time = _parse_timestamp(timestamp_str)
                if msg_time.timestamp() < cutoff_time:
                    continue
            except ValueError:
                pass

        # Detect back-and-forth on same topic: A->B->A with same topic
        if (
            prev_prev_agent
            and prev_agent
            and agent == prev_prev_agent  # Same agent as 2 messages ago
            and prev_prev_topic
            and topic
            and (topic == prev_prev_topic or topic == prev_topic)  # Same or similar topic
        ):
            unproductive_count += 1

        # Shift history
        prev_prev_agent = prev_agent
        prev_prev_topic = prev_topic
        prev_agent = agent
        prev_topic = topic

    return unproductive_count


def _calculate_unproductive_churn_rate(state: YoloState) -> float:
    """Calculate unproductive churn rate (unproductive exchanges per minute).

    Args:
        state: Current orchestration state.

    Returns:
        Unproductive churn rate as exchanges per minute.
    """
    return float(_count_unproductive_exchanges(state, window_seconds=60.0))


# =============================================================================
# Anomaly Detection and Alerting (Task 5 - AC #4)
# =============================================================================


def _detect_anomalies(
    metrics: HealthMetrics,
    config: HealthConfig,
) -> list[dict[str, Any]]:
    """Detect anomalies in health metrics.

    Compares metrics against configured thresholds to identify
    potential issues.

    Args:
        metrics: Current health metrics.
        config: Health monitoring configuration.

    Returns:
        List of anomaly dictionaries with type, severity, agent, value, threshold.
    """
    anomalies: list[dict[str, Any]] = []

    # Check idle times against thresholds
    warning_idle = config.max_idle_time_seconds * config.warning_threshold_ratio
    for agent, idle_time in metrics.agent_idle_times.items():
        if idle_time > config.max_idle_time_seconds:
            anomalies.append(
                {
                    "type": "idle_time_exceeded",
                    "severity": "critical",
                    "agent": agent,
                    "value": idle_time,
                    "threshold": config.max_idle_time_seconds,
                }
            )
        elif idle_time > warning_idle:
            anomalies.append(
                {
                    "type": "idle_time_warning",
                    "severity": "warning",
                    "agent": agent,
                    "value": idle_time,
                    "threshold": warning_idle,
                }
            )

    # Check cycle times against thresholds
    for agent, cycle_time in metrics.agent_cycle_times.items():
        if cycle_time > config.max_cycle_time_seconds:
            anomalies.append(
                {
                    "type": "slow_cycle",
                    "severity": "warning",
                    "agent": agent,
                    "value": cycle_time,
                    "threshold": config.max_cycle_time_seconds,
                }
            )

    # Check overall churn rate
    if metrics.overall_churn_rate > config.max_churn_rate:
        anomalies.append(
            {
                "type": "high_churn",
                "severity": "critical",
                "agent": None,
                "value": metrics.overall_churn_rate,
                "threshold": config.max_churn_rate,
            }
        )
    elif metrics.overall_churn_rate > config.max_churn_rate * config.warning_threshold_ratio:
        anomalies.append(
            {
                "type": "churn_rate_warning",
                "severity": "warning",
                "agent": None,
                "value": metrics.overall_churn_rate,
                "threshold": config.max_churn_rate * config.warning_threshold_ratio,
            }
        )

    return anomalies


def _generate_alerts(
    anomalies: list[dict[str, Any]],
    config: HealthConfig,
) -> tuple[HealthAlert, ...]:
    """Generate HealthAlert objects from detected anomalies.

    Args:
        anomalies: List of detected anomalies.
        config: Health monitoring configuration.

    Returns:
        Tuple of HealthAlert objects.
    """
    if not config.enable_alerts:
        return ()

    alerts: list[HealthAlert] = []
    for anomaly in anomalies:
        severity = anomaly["severity"]
        agent = anomaly.get("agent")
        alert_type = anomaly["type"]
        value = anomaly["value"]
        threshold = anomaly["threshold"]

        # Generate human-readable message
        if agent:
            message = (
                f"Agent {agent}: {alert_type.replace('_', ' ')} ({value:.1f} > {threshold:.1f})"
            )
        else:
            message = f"System: {alert_type.replace('_', ' ')} ({value:.1f} > {threshold:.1f})"

        alert = HealthAlert(
            severity=severity,
            alert_type=alert_type,
            message=message,
            affected_agent=agent,
            metric_value=value,
            threshold_value=threshold,
        )
        alerts.append(alert)

        # Log alert with appropriate method
        if severity == "critical":
            logger.error(
                "health_alert_triggered",
                severity=severity,
                alert_type=alert_type,
                agent=agent,
                metric_value=value,
                threshold_value=threshold,
            )
        elif severity == "warning":
            logger.warning(
                "health_alert_triggered",
                severity=severity,
                alert_type=alert_type,
                agent=agent,
                metric_value=value,
                threshold_value=threshold,
            )
        else:
            logger.info(
                "health_alert_triggered",
                severity=severity,
                alert_type=alert_type,
                agent=agent,
                metric_value=value,
                threshold_value=threshold,
            )

    return tuple(alerts)


def _trigger_alerts(
    anomalies: list[dict[str, Any]],
    config: HealthConfig,
) -> tuple[HealthAlert, ...]:
    """Trigger alerts based on detected anomalies.

    This is the main entry point for alert generation, which
    calls _generate_alerts internally.

    Args:
        anomalies: List of detected anomalies.
        config: Health monitoring configuration.

    Returns:
        Tuple of triggered HealthAlert objects.
    """
    return _generate_alerts(anomalies, config)


# =============================================================================
# Status Determination
# =============================================================================


def _determine_status(
    metrics: HealthMetrics,
    alerts: tuple[HealthAlert, ...],
) -> HealthSeverity:
    """Determine overall system health status based on alerts.

    Args:
        metrics: Current health metrics.
        alerts: Generated alerts.

    Returns:
        Overall health severity level.
    """
    critical_count = sum(1 for a in alerts if a.severity == "critical")
    warning_count = sum(1 for a in alerts if a.severity == "warning")

    if critical_count > 0:
        return "critical"
    elif warning_count >= 2:
        return "degraded"
    elif warning_count > 0:
        return "warning"
    return "healthy"


def _generate_summary(
    status: HealthSeverity,
    metrics: HealthMetrics,
    alerts: tuple[HealthAlert, ...],
) -> str:
    """Generate human-readable health summary.

    Args:
        status: Overall health status.
        metrics: Current health metrics.
        alerts: Generated alerts.

    Returns:
        Summary string describing system health.
    """
    active_agents = len([t for t in metrics.agent_idle_times.values() if t < 300])
    alert_count = len(alerts)

    if status == "healthy":
        return f"All systems nominal. {active_agents} agents active."
    elif status == "warning":
        return f"Minor issues detected. {alert_count} alert(s). {active_agents} agents active."
    elif status == "degraded":
        return f"System degraded. {alert_count} alert(s) require attention."
    else:  # critical
        return f"CRITICAL: {alert_count} critical alert(s). Immediate attention required."


# =============================================================================
# Agent Snapshots
# =============================================================================


def _build_agent_snapshots(
    idle_times: dict[str, float],
    cycle_times: dict[str, float],
    churn_rates: dict[str, float],
    config: HealthConfig,
) -> list[AgentHealthSnapshot]:
    """Build health snapshots for all agents.

    Args:
        idle_times: Agent idle times.
        cycle_times: Agent cycle times.
        churn_rates: Agent churn rates.
        config: Health configuration for determining health status.

    Returns:
        List of AgentHealthSnapshot objects.
    """
    snapshots: list[AgentHealthSnapshot] = []
    current_time = _get_current_timestamp()

    for agent in VALID_AGENTS_FOR_HEALTH:
        idle_time = idle_times.get(agent, 0.0)
        cycle_time = cycle_times.get(agent)
        churn_rate = churn_rates.get(agent, 0.0)

        # Determine if agent is healthy
        is_healthy = idle_time <= config.max_idle_time_seconds

        # Calculate last activity timestamp
        if idle_time > 0:
            last_activity_dt = datetime.now(timezone.utc)
            # Subtract idle time to get last activity
            last_activity_dt = last_activity_dt - timedelta(seconds=idle_time)
            last_activity = last_activity_dt.isoformat()
        else:
            last_activity = current_time

        snapshot = AgentHealthSnapshot(
            agent=agent,
            idle_time_seconds=idle_time,
            last_activity=last_activity,
            cycle_time_seconds=cycle_time,
            churn_rate=churn_rate,
            is_healthy=is_healthy,
        )
        snapshots.append(snapshot)

    return snapshots


# =============================================================================
# Metrics Collection
# =============================================================================


def _collect_metrics(state: YoloState, config: HealthConfig) -> HealthMetrics:
    """Collect all health metrics from state.

    Args:
        state: Current orchestration state.
        config: Health configuration.

    Returns:
        HealthMetrics with all collected data.
    """
    # Collect individual metrics
    agent_idle_times = _calculate_agent_idle_times(state)
    agent_cycle_times = _calculate_agent_cycle_times(state)
    agent_churn_rates = _calculate_agent_churn_rates(state)
    overall_cycle_time = _calculate_overall_cycle_time(state)
    overall_churn_rate = _calculate_churn_rate(state)

    # Collect new metrics (Task 3.5 and Task 4.3)
    unproductive_churn_rate = _calculate_unproductive_churn_rate(state)
    cycle_time_percentiles = _calculate_cycle_time_percentiles(state)

    # Build agent snapshots
    agent_snapshots = _build_agent_snapshots(
        agent_idle_times, agent_cycle_times, agent_churn_rates, config
    )

    return HealthMetrics(
        agent_idle_times=agent_idle_times,
        agent_cycle_times=agent_cycle_times,
        agent_churn_rates=agent_churn_rates,
        overall_cycle_time=overall_cycle_time,
        overall_churn_rate=overall_churn_rate,
        unproductive_churn_rate=unproductive_churn_rate,
        cycle_time_percentiles=cycle_time_percentiles,
        agent_snapshots=tuple(agent_snapshots),
    )


# =============================================================================
# Main Health Monitoring Function (Task 6)
# =============================================================================


async def monitor_health(
    state: YoloState,
    config: HealthConfig | None = None,
) -> HealthStatus:
    """Monitor system health metrics (FR11, FR67).

    Main entry point for health monitoring. Collects metrics,
    detects anomalies, generates alerts, and returns overall status.

    This function is designed to be non-blocking - it should never
    fail the main orchestration workflow even if monitoring fails.

    Args:
        state: Current orchestration state.
        config: Health monitoring configuration (uses defaults if None).

    Returns:
        HealthStatus with metrics, alerts, and overall status.

    Example:
        >>> status = await monitor_health(state)
        >>> status.is_healthy
        True
        >>> status.status
        'healthy'
    """
    config = config or HealthConfig()

    logger.info(
        "health_monitoring_started",
        current_agent=state.get("current_agent"),
        enable_alerts=config.enable_alerts,
    )

    # Step 1: Collect metrics (AC #1, #2, #3)
    metrics = _collect_metrics(state, config)

    # Step 2: Detect anomalies
    anomalies = _detect_anomalies(metrics, config)

    # Step 3: Generate alerts (AC #4)
    alerts = _trigger_alerts(anomalies, config) if config.enable_alerts else ()

    # Step 4: Determine overall status
    status = _determine_status(metrics, alerts)

    # Step 5: Generate summary
    summary = _generate_summary(status, metrics, alerts)

    # Step 6: Create result
    result = HealthStatus(
        status=status,
        metrics=metrics,
        alerts=alerts,
        summary=summary,
        is_healthy=status in ("healthy", "warning"),
    )

    logger.info(
        "health_monitoring_complete",
        status=status,
        is_healthy=result.is_healthy,
        alert_count=len(alerts),
        overall_churn_rate=metrics.overall_churn_rate,
    )

    return result
