"""High-level metrics query API (Story 3.9 - Task 8).

This module provides a high-level API for querying gate metrics.
It wraps the lower-level calculator functions with a simpler interface.

Example:
    >>> from yolo_developer.gates.metrics_api import (
    ...     get_gate_metrics,
    ...     get_pass_rates,
    ...     get_trends,
    ...     get_agent_summary,
    ... )
    >>>
    >>> # Get all metrics for a specific gate
    >>> metrics = await get_gate_metrics(gate_name="testability")
    >>>
    >>> # Get pass rates for all gates
    >>> rates = await get_pass_rates()
    >>>
    >>> # Get daily trends
    >>> trends = await get_trends(period="daily")

Note:
    All functions require a metrics store to be configured via
    set_metrics_store() in the decorator module, or passed explicitly.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from yolo_developer.gates.decorator import get_metrics_store
from yolo_developer.gates.metrics_calculator import (
    calculate_pass_rates,
    calculate_trends,
    get_agent_breakdown,
)
from yolo_developer.gates.metrics_types import (
    GateMetricRecord,
    GateMetricsSummary,
    GateTrend,
)

if TYPE_CHECKING:
    from yolo_developer.gates.metrics_store import GateMetricsStore


async def get_gate_metrics(
    gate_name: str | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    store: GateMetricsStore | None = None,
) -> list[GateMetricRecord]:
    """Get gate metrics with optional filters.

    Retrieves metric records from the configured store, optionally
    filtered by gate name and/or time range.

    Args:
        gate_name: Filter by gate name (None for all gates).
        time_range: Tuple of (start_time, end_time) to filter by.
                   Both bounds are inclusive.
        store: Optional metrics store to use. If not provided,
               uses the globally configured store.

    Returns:
        List of matching GateMetricRecord objects.

    Raises:
        RuntimeError: If no metrics store is configured.

    Example:
        >>> from datetime import datetime, timedelta, timezone
        >>> now = datetime.now(timezone.utc)
        >>> start = now - timedelta(days=7)
        >>> metrics = await get_gate_metrics(
        ...     gate_name="testability",
        ...     time_range=(start, now),
        ... )
    """
    metrics_store = store or get_metrics_store()
    if metrics_store is None:
        raise RuntimeError(
            "No metrics store configured. Call set_metrics_store() first or pass store parameter."
        )

    start_time = time_range[0] if time_range else None
    end_time = time_range[1] if time_range else None

    return await metrics_store.get_metrics(
        gate_name=gate_name,
        start_time=start_time,
        end_time=end_time,
    )


async def get_pass_rates(
    gate_name: str | None = None,
    store: GateMetricsStore | None = None,
) -> dict[str, float]:
    """Get pass rates by gate type.

    Calculates pass rates as percentages for each gate type.
    If a gate_name is provided, returns only that gate's rate.

    Args:
        gate_name: Filter by gate name (None for all gates).
        store: Optional metrics store to use.

    Returns:
        Dictionary mapping gate names to pass rates (0.0-100.0).

    Raises:
        RuntimeError: If no metrics store is configured.

    Example:
        >>> rates = await get_pass_rates()
        >>> print(rates)
        {'testability': 85.0, 'architecture': 92.0}
    """
    metrics_store = store or get_metrics_store()
    if metrics_store is None:
        raise RuntimeError(
            "No metrics store configured. Call set_metrics_store() first or pass store parameter."
        )

    records = await metrics_store.get_metrics(gate_name=gate_name)
    return calculate_pass_rates(records)


async def get_trends(
    period: str = "daily",
    gate_name: str | None = None,
    store: GateMetricsStore | None = None,
) -> list[GateTrend]:
    """Get trends over time.

    Calculates quality trends aggregated by the specified period.

    Args:
        period: Aggregation period - "daily", "weekly", or "sprint".
        gate_name: Filter by gate name (None for all gates).
        store: Optional metrics store to use.

    Returns:
        List of GateTrend objects sorted by time.

    Raises:
        RuntimeError: If no metrics store is configured.

    Example:
        >>> trends = await get_trends(period="weekly", gate_name="testability")
        >>> for trend in trends:
        ...     print(f"{trend.period_label}: {trend.pass_rate}%")
    """
    metrics_store = store or get_metrics_store()
    if metrics_store is None:
        raise RuntimeError(
            "No metrics store configured. Call set_metrics_store() first or pass store parameter."
        )

    records = await metrics_store.get_metrics(gate_name=gate_name)
    return calculate_trends(records, period=period, gate_name=gate_name)


async def get_agent_summary(
    store: GateMetricsStore | None = None,
) -> dict[str, GateMetricsSummary]:
    """Get metrics summary by agent.

    Returns a breakdown of gate metrics for each agent,
    including pass rates and failure counts.

    Args:
        store: Optional metrics store to use.

    Returns:
        Dictionary mapping agent names to their GateMetricsSummary.

    Raises:
        RuntimeError: If no metrics store is configured.

    Example:
        >>> summary = await get_agent_summary()
        >>> for agent, stats in summary.items():
        ...     print(f"{agent}: {stats.pass_rate}% pass rate")
    """
    metrics_store = store or get_metrics_store()
    if metrics_store is None:
        raise RuntimeError(
            "No metrics store configured. Call set_metrics_store() first or pass store parameter."
        )

    records = await metrics_store.get_metrics()
    return get_agent_breakdown(records)
