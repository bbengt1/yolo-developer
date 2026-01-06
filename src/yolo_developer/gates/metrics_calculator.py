"""Gate metrics calculator module (Story 3.9 - Task 5).

This module provides functions for calculating pass/fail rates and
summary statistics from gate metric records.

Example:
    >>> from datetime import datetime, timezone
    >>> from yolo_developer.gates.metrics_calculator import (
    ...     calculate_pass_rates,
    ...     calculate_gate_summary,
    ... )
    >>> from yolo_developer.gates.metrics_types import GateMetricRecord
    >>>
    >>> records = [
    ...     GateMetricRecord(
    ...         gate_name="testability",
    ...         passed=True,
    ...         score=0.85,
    ...         threshold=0.80,
    ...         timestamp=datetime.now(timezone.utc),
    ...     ),
    ... ]
    >>> rates = calculate_pass_rates(records)
    >>> rates["testability"]
    100.0
"""

from __future__ import annotations

from datetime import datetime, timezone

from yolo_developer.gates.metrics_types import (
    GateMetricRecord,
    GateMetricsSummary,
    GateTrend,
    TrendDirection,
)


def calculate_pass_rates(records: list[GateMetricRecord]) -> dict[str, float]:
    """Calculate pass rates by gate type.

    Groups records by gate name and calculates the pass rate
    (as a percentage) for each gate.

    Args:
        records: List of metric records to analyze.

    Returns:
        Dictionary mapping gate names to pass rates (0.0-100.0).

    Example:
        >>> from datetime import datetime, timezone
        >>> from yolo_developer.gates.metrics_types import GateMetricRecord
        >>> records = [
        ...     GateMetricRecord(
        ...         gate_name="testability",
        ...         passed=True,
        ...         score=0.85,
        ...         threshold=0.80,
        ...         timestamp=datetime.now(timezone.utc),
        ...     ),
        ...     GateMetricRecord(
        ...         gate_name="testability",
        ...         passed=False,
        ...         score=0.50,
        ...         threshold=0.80,
        ...         timestamp=datetime.now(timezone.utc),
        ...     ),
        ... ]
        >>> rates = calculate_pass_rates(records)
        >>> rates["testability"]
        50.0
    """
    if not records:
        return {}

    # Group records by gate name
    gate_records: dict[str, list[GateMetricRecord]] = {}
    for record in records:
        if record.gate_name not in gate_records:
            gate_records[record.gate_name] = []
        gate_records[record.gate_name].append(record)

    # Calculate pass rates
    rates: dict[str, float] = {}
    for gate_name, gate_data in gate_records.items():
        total = len(gate_data)
        passed = sum(1 for r in gate_data if r.passed)
        rates[gate_name] = (passed / total) * 100.0 if total > 0 else 0.0

    return rates


def calculate_gate_summary(
    gate_name: str,
    records: list[GateMetricRecord],
) -> GateMetricsSummary:
    """Calculate summary statistics for a specific gate.

    Filters records by gate name and calculates comprehensive
    statistics including pass/fail counts, rates, and time range.

    Args:
        gate_name: Name of the gate to calculate summary for.
        records: List of all metric records (will be filtered by gate).

    Returns:
        GateMetricsSummary with calculated statistics.

    Example:
        >>> from datetime import datetime, timezone
        >>> from yolo_developer.gates.metrics_types import GateMetricRecord
        >>> records = [
        ...     GateMetricRecord(
        ...         gate_name="testability",
        ...         passed=True,
        ...         score=0.85,
        ...         threshold=0.80,
        ...         timestamp=datetime.now(timezone.utc),
        ...     ),
        ... ]
        >>> summary = calculate_gate_summary("testability", records)
        >>> summary.pass_rate
        100.0
    """
    now = datetime.now(timezone.utc)

    # Filter to specified gate
    filtered = [r for r in records if r.gate_name == gate_name]

    if not filtered:
        return GateMetricsSummary(
            gate_name=gate_name,
            total_evaluations=0,
            pass_count=0,
            fail_count=0,
            pass_rate=0.0,
            avg_score=0.0,
            period_start=now,
            period_end=now,
        )

    total = len(filtered)
    pass_count = sum(1 for r in filtered if r.passed)
    fail_count = total - pass_count
    pass_rate = (pass_count / total) * 100.0 if total > 0 else 0.0
    avg_score = sum(r.score for r in filtered) / total if total > 0 else 0.0

    # Get time range
    timestamps = [r.timestamp for r in filtered]
    period_start = min(timestamps)
    period_end = max(timestamps)

    return GateMetricsSummary(
        gate_name=gate_name,
        total_evaluations=total,
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=pass_rate,
        avg_score=avg_score,
        period_start=period_start,
        period_end=period_end,
    )


def filter_records_by_time_range(
    records: list[GateMetricRecord],
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[GateMetricRecord]:
    """Filter records by time range.

    Returns records that fall within the specified time range.
    Both start and end times are inclusive.

    Args:
        records: List of records to filter.
        start_time: Minimum timestamp (inclusive). None for no lower bound.
        end_time: Maximum timestamp (inclusive). None for no upper bound.

    Returns:
        List of records within the time range.

    Example:
        >>> from datetime import datetime, timedelta, timezone
        >>> from yolo_developer.gates.metrics_types import GateMetricRecord
        >>> now = datetime.now(timezone.utc)
        >>> records = [
        ...     GateMetricRecord(
        ...         gate_name="testability",
        ...         passed=True,
        ...         score=0.85,
        ...         threshold=0.80,
        ...         timestamp=now - timedelta(hours=i),
        ...     )
        ...     for i in range(24)
        ... ]
        >>> # Get records from last 6 hours
        >>> filtered = filter_records_by_time_range(
        ...     records,
        ...     start_time=now - timedelta(hours=6),
        ... )
    """
    result = []

    for record in records:
        if start_time is not None and record.timestamp < start_time:
            continue
        if end_time is not None and record.timestamp > end_time:
            continue
        result.append(record)

    return result


def calculate_trends(
    records: list[GateMetricRecord],
    period: str,
    gate_name: str | None = None,
) -> list[GateTrend]:
    """Calculate trend analysis over time periods.

    Groups records by time period and calculates pass rates and
    trend direction for each period.

    Args:
        records: List of metric records to analyze.
        period: Aggregation period - "daily", "weekly", or "sprint".
        gate_name: Optional gate name filter.

    Returns:
        List of GateTrend objects sorted by time.

    Example:
        >>> from datetime import datetime, timezone
        >>> from yolo_developer.gates.metrics_types import GateMetricRecord
        >>> records = [
        ...     GateMetricRecord(
        ...         gate_name="testability",
        ...         passed=True,
        ...         score=0.85,
        ...         threshold=0.80,
        ...         timestamp=datetime.now(timezone.utc),
        ...     ),
        ... ]
        >>> trends = calculate_trends(records, "daily")
    """
    if not records:
        return []

    # Filter by gate name if specified
    if gate_name is not None:
        records = [r for r in records if r.gate_name == gate_name]

    if not records:
        return []

    # Group records by period
    period_groups = _group_by_period(records, period)

    # Sort periods chronologically
    sorted_periods = sorted(period_groups.keys())

    # Calculate trends for each period
    trends: list[GateTrend] = []
    prev_pass_rate: float | None = None

    for period_key in sorted_periods:
        period_records = period_groups[period_key]
        total = len(period_records)
        passed = sum(1 for r in period_records if r.passed)
        pass_rate = (passed / total) * 100.0 if total > 0 else 0.0
        avg_score = sum(r.score for r in period_records) / total if total > 0 else 0.0

        # Determine trend direction
        direction = _calculate_direction(pass_rate, prev_pass_rate)

        # Get period label
        period_label = _get_period_label(period_key, period)

        # Determine gate name for the trend
        trend_gate_name = gate_name or "all"
        if gate_name is None and period_records:
            # Use first record's gate name if all records are from same gate
            unique_gates = {r.gate_name for r in period_records}
            if len(unique_gates) == 1:
                trend_gate_name = period_records[0].gate_name

        trend = GateTrend(
            gate_name=trend_gate_name,
            period=period,
            pass_rate=pass_rate,
            avg_score=avg_score,
            evaluation_count=total,
            direction=direction,
            period_label=period_label,
        )
        trends.append(trend)
        prev_pass_rate = pass_rate

    return trends


def _group_by_period(
    records: list[GateMetricRecord],
    period: str,
) -> dict[str, list[GateMetricRecord]]:
    """Group records by time period.

    Args:
        records: Records to group.
        period: "daily", "weekly", or "sprint".

    Returns:
        Dictionary mapping period keys to records.
    """
    groups: dict[str, list[GateMetricRecord]] = {}

    for record in records:
        if period == "sprint":
            # Group by sprint_id
            key = record.sprint_id or "no-sprint"
        elif period == "weekly":
            # Group by ISO week number
            iso_calendar = record.timestamp.isocalendar()
            key = f"{iso_calendar.year}-W{iso_calendar.week:02d}"
        else:  # daily
            # Group by date
            key = record.timestamp.strftime("%Y-%m-%d")

        if key not in groups:
            groups[key] = []
        groups[key].append(record)

    return groups


def _calculate_direction(
    current_rate: float,
    previous_rate: float | None,
) -> TrendDirection:
    """Calculate trend direction based on pass rate change.

    Args:
        current_rate: Current period's pass rate.
        previous_rate: Previous period's pass rate (None for first period).

    Returns:
        TrendDirection enum value.
    """
    if previous_rate is None:
        return TrendDirection.STABLE

    # Use 5% threshold for significant change
    threshold = 5.0
    diff = current_rate - previous_rate

    if diff > threshold:
        return TrendDirection.IMPROVING
    elif diff < -threshold:
        return TrendDirection.DECLINING
    else:
        return TrendDirection.STABLE


def _get_period_label(period_key: str, period: str) -> str:
    """Get human-readable label for a period.

    Args:
        period_key: The period key (date string, week number, or sprint ID).
        period: The period type.

    Returns:
        Human-readable period label.
    """
    # For daily and weekly, the key is already a good label
    # For sprint, the sprint_id is used
    return period_key


def get_agent_breakdown(
    records: list[GateMetricRecord],
) -> dict[str, GateMetricsSummary]:
    """Get metrics breakdown by agent.

    Groups records by agent_name and calculates summary statistics
    for each agent. Records without agent_name are excluded.

    Args:
        records: List of metric records to analyze.

    Returns:
        Dictionary mapping agent names to their metrics summaries.

    Example:
        >>> from datetime import datetime, timezone
        >>> from yolo_developer.gates.metrics_types import GateMetricRecord
        >>> records = [
        ...     GateMetricRecord(
        ...         gate_name="testability",
        ...         passed=True,
        ...         score=0.85,
        ...         threshold=0.80,
        ...         timestamp=datetime.now(timezone.utc),
        ...         agent_name="analyst",
        ...     ),
        ... ]
        >>> breakdown = get_agent_breakdown(records)
        >>> breakdown["analyst"].pass_rate
        100.0
    """
    if not records:
        return {}

    # Group records by agent_name (excluding None)
    agent_records: dict[str, list[GateMetricRecord]] = {}
    for record in records:
        if record.agent_name is not None:
            if record.agent_name not in agent_records:
                agent_records[record.agent_name] = []
            agent_records[record.agent_name].append(record)

    if not agent_records:
        return {}

    # Calculate summary for each agent
    breakdown: dict[str, GateMetricsSummary] = {}
    for agent_name, agent_data in agent_records.items():
        total = len(agent_data)
        pass_count = sum(1 for r in agent_data if r.passed)
        fail_count = total - pass_count
        pass_rate = (pass_count / total) * 100.0 if total > 0 else 0.0
        avg_score = sum(r.score for r in agent_data) / total if total > 0 else 0.0

        # Get time range
        timestamps = [r.timestamp for r in agent_data]
        period_start = min(timestamps)
        period_end = max(timestamps)

        breakdown[agent_name] = GateMetricsSummary(
            gate_name=agent_name,  # Using agent_name as identifier
            total_evaluations=total,
            pass_count=pass_count,
            fail_count=fail_count,
            pass_rate=pass_rate,
            avg_score=avg_score,
            period_start=period_start,
            period_end=period_end,
        )

    return breakdown
