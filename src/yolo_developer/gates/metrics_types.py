"""Metrics data types for gate evaluation tracking (Story 3.9).

This module provides data models for tracking quality gate metrics over time:
- GateMetricRecord: Individual evaluation record
- GateMetricsSummary: Aggregated metrics summary
- GateTrend: Trend analysis data
- TrendDirection: Enum for trend direction

Example:
    >>> from datetime import datetime, timezone
    >>> from yolo_developer.gates.metrics_types import (
    ...     GateMetricRecord,
    ...     GateMetricsSummary,
    ...     GateTrend,
    ...     TrendDirection,
    ... )
    >>>
    >>> # Record a gate evaluation
    >>> record = GateMetricRecord(
    ...     gate_name="testability",
    ...     passed=True,
    ...     score=0.85,
    ...     threshold=0.80,
    ...     timestamp=datetime.now(timezone.utc),
    ...     agent_name="analyst",
    ... )
    >>> print(record.to_dict())

Security Note:
    All dataclasses are frozen (immutable) to prevent accidental mutation.
    This ensures metrics data integrity throughout the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class TrendDirection(Enum):
    """Direction of quality trend over time.

    Used to indicate whether gate pass rates are improving,
    stable, or declining compared to previous periods.

    Values:
        IMPROVING: Pass rate is increasing
        STABLE: Pass rate is roughly constant
        DECLINING: Pass rate is decreasing
    """

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass(frozen=True)
class GateMetricRecord:
    """Single gate evaluation metric record.

    Represents one gate evaluation event for metrics tracking.
    Records pass/fail status, score, and contextual information
    about when and where the evaluation occurred.

    Attributes:
        gate_name: Name of the evaluated gate (e.g., "testability")
        passed: Whether the gate passed (True) or failed (False)
        score: Numeric score achieved (0.0-1.0)
        threshold: Required threshold for passing (0.0-1.0)
        timestamp: When the evaluation occurred (UTC)
        agent_name: Agent that triggered the gate (optional)
        sprint_id: Sprint identifier for grouping (optional)

    Example:
        >>> record = GateMetricRecord(
        ...     gate_name="testability",
        ...     passed=True,
        ...     score=0.85,
        ...     threshold=0.80,
        ...     timestamp=datetime.now(timezone.utc),
        ... )
    """

    gate_name: str
    passed: bool
    score: float
    threshold: float
    timestamp: datetime
    agent_name: str | None = None
    sprint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, timestamp as ISO 8601 string.
        """
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "sprint_id": self.sprint_id,
        }


@dataclass(frozen=True)
class GateMetricsSummary:
    """Aggregated metrics summary for a gate or agent.

    Provides a statistical summary of gate evaluations over
    a specified time period, including pass rates and averages.

    Attributes:
        gate_name: Gate name (or "all" for aggregate across gates)
        total_evaluations: Total number of evaluations in period
        pass_count: Number of evaluations that passed
        fail_count: Number of evaluations that failed
        pass_rate: Pass rate as percentage (0.0-100.0)
        avg_score: Average score across all evaluations (0.0-1.0)
        period_start: Start of measurement period (UTC)
        period_end: End of measurement period (UTC)

    Example:
        >>> summary = GateMetricsSummary(
        ...     gate_name="testability",
        ...     total_evaluations=100,
        ...     pass_count=85,
        ...     fail_count=15,
        ...     pass_rate=85.0,
        ...     avg_score=0.87,
        ...     period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ...     period_end=datetime(2026, 1, 5, tzinfo=timezone.utc),
        ... )
    """

    gate_name: str
    total_evaluations: int
    pass_count: int
    fail_count: int
    pass_rate: float
    avg_score: float
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, datetimes as ISO 8601 strings.
        """
        return {
            "gate_name": self.gate_name,
            "total_evaluations": self.total_evaluations,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


@dataclass(frozen=True)
class GateTrend:
    """Trend analysis for a gate over time.

    Represents quality metrics for a specific time period with
    trend direction indicating change from previous period.

    Attributes:
        gate_name: Name of the gate being analyzed
        period: Time period type ("daily", "weekly", "sprint")
        pass_rate: Pass rate for this period (0.0-100.0)
        avg_score: Average score for this period (0.0-1.0)
        evaluation_count: Number of evaluations in this period
        direction: Trend direction compared to previous period
        period_label: Human-readable period label (e.g., "2026-01-05", "2026-W01")

    Example:
        >>> trend = GateTrend(
        ...     gate_name="testability",
        ...     period="daily",
        ...     pass_rate=85.0,
        ...     avg_score=0.87,
        ...     evaluation_count=25,
        ...     direction=TrendDirection.IMPROVING,
        ...     period_label="2026-01-05",
        ... )
    """

    gate_name: str
    period: str
    pass_rate: float
    avg_score: float
    evaluation_count: int
    direction: TrendDirection
    period_label: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, direction as string value.
        """
        return {
            "gate_name": self.gate_name,
            "period": self.period,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "evaluation_count": self.evaluation_count,
            "direction": self.direction.value,
            "period_label": self.period_label,
        }
