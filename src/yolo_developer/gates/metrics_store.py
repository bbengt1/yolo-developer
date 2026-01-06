"""Gate metrics storage protocol and implementations (Story 3.9 - Tasks 2 & 3).

This module provides the storage infrastructure for gate metrics:
- GateMetricsStore: Protocol defining storage interface
- JsonGateMetricsStore: JSON file-based implementation

Example:
    >>> from pathlib import Path
    >>> from datetime import datetime, timezone
    >>> from yolo_developer.gates.metrics_store import JsonGateMetricsStore
    >>> from yolo_developer.gates.metrics_types import GateMetricRecord
    >>>
    >>> store = JsonGateMetricsStore(base_path=Path(".yolo"))
    >>> record = GateMetricRecord(
    ...     gate_name="testability",
    ...     passed=True,
    ...     score=0.85,
    ...     threshold=0.80,
    ...     timestamp=datetime.now(timezone.utc),
    ... )
    >>> await store.record_evaluation(record)

Security Note:
    File operations use atomic writes with file locking to prevent
    data corruption from concurrent access.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import structlog

from yolo_developer.gates.metrics_types import (
    GateMetricRecord,
    GateMetricsSummary,
    GateTrend,
)

logger = structlog.get_logger(__name__)


class GateMetricsStore(Protocol):
    """Protocol for gate metrics storage backends.

    Defines the interface that all metrics storage implementations
    must follow. Supports recording evaluations and querying metrics
    with various filters.

    All methods are async to support non-blocking I/O operations.
    """

    async def record_evaluation(self, record: GateMetricRecord) -> None:
        """Record a gate evaluation metric.

        Args:
            record: The metric record to store.
        """
        ...

    async def get_metrics(
        self,
        gate_name: str | None = None,
        agent_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        sprint_id: str | None = None,
    ) -> list[GateMetricRecord]:
        """Query metrics with optional filters.

        Args:
            gate_name: Filter by gate name.
            agent_name: Filter by agent name.
            start_time: Filter by start time (inclusive).
            end_time: Filter by end time (inclusive).
            sprint_id: Filter by sprint identifier.

        Returns:
            List of matching metric records.
        """
        ...

    async def get_summary(
        self,
        gate_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> GateMetricsSummary:
        """Get aggregated summary for metrics.

        Args:
            gate_name: Filter by gate name (None for all gates).
            start_time: Filter by start time.
            end_time: Filter by end time.

        Returns:
            Aggregated metrics summary.
        """
        ...

    async def get_agent_breakdown(self) -> dict[str, GateMetricsSummary]:
        """Get metrics breakdown by agent.

        Returns:
            Dictionary mapping agent names to their metrics summaries.
        """
        ...

    async def get_trends(
        self,
        period: str = "daily",
        gate_name: str | None = None,
    ) -> list[GateTrend]:
        """Get trends over time periods.

        Args:
            period: Aggregation period - "daily", "weekly", or "sprint".
            gate_name: Filter by gate name (None for all gates).

        Returns:
            List of trend objects sorted by time.
        """
        ...


class JsonGateMetricsStore:
    """JSON file-based metrics storage implementation.

    Stores gate metrics in a JSON file with support for:
    - Atomic writes with file locking
    - Append-only storage pattern
    - Filtering by gate, agent, time, and sprint
    - Persistence across application restarts
    - Configurable retention with cleanup/rotation

    Attributes:
        base_path: Base directory for metrics storage.
        metrics_file: Path to the metrics JSON file.
        retention_days: Number of days to retain metrics (default: 90).
    """

    def __init__(
        self,
        base_path: Path,
        retention_days: int = 90,
    ) -> None:
        """Initialize the JSON metrics store.

        Args:
            base_path: Base directory for storing metrics.
                      Metrics file will be at {base_path}/metrics/gate_metrics.json
            retention_days: Number of days to retain metrics. Older metrics
                           will be removed during cleanup. Default is 90 days.
        """
        self.base_path = base_path
        self.metrics_file = base_path / "metrics" / "gate_metrics.json"
        self._lock_file = base_path / "metrics" / ".gate_metrics.lock"
        self.retention_days = retention_days
        self._lock = asyncio.Lock()

    @contextmanager
    def _file_lock(self) -> Generator[None, None, None]:
        """Cross-process file lock for safe concurrent access.

        Uses fcntl.flock for file-based locking that works across
        multiple processes on Unix-like systems.

        Yields:
            None - the lock is held for the duration of the context.
        """
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(self._lock_file, "w")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()

    async def record_evaluation(self, record: GateMetricRecord) -> None:
        """Record a gate evaluation metric.

        Appends the record to the metrics file using atomic writes.

        Args:
            record: The metric record to store.
        """
        async with self._lock:
            # Ensure directory exists
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing metrics
            metrics = await self._load_metrics()

            # Append new record
            metrics.append(record.to_dict())

            # Write atomically
            await self._save_metrics(metrics)

            logger.info(
                "Recorded gate metric",
                gate_name=record.gate_name,
                passed=record.passed,
                score=record.score,
            )

    async def get_metrics(
        self,
        gate_name: str | None = None,
        agent_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        sprint_id: str | None = None,
    ) -> list[GateMetricRecord]:
        """Query metrics with optional filters.

        Args:
            gate_name: Filter by gate name.
            agent_name: Filter by agent name.
            start_time: Filter by start time (inclusive).
            end_time: Filter by end time (inclusive).
            sprint_id: Filter by sprint identifier.

        Returns:
            List of matching metric records.
        """
        metrics_data = await self._load_metrics()

        # Convert to records and filter
        records = []
        for data in metrics_data:
            record = self._dict_to_record(data)

            # Apply filters
            if gate_name is not None and record.gate_name != gate_name:
                continue
            if agent_name is not None and record.agent_name != agent_name:
                continue
            if sprint_id is not None and record.sprint_id != sprint_id:
                continue
            if start_time is not None and record.timestamp < start_time:
                continue
            if end_time is not None and record.timestamp > end_time:
                continue

            records.append(record)

        return records

    async def get_summary(
        self,
        gate_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> GateMetricsSummary:
        """Get aggregated summary for metrics.

        Args:
            gate_name: Filter by gate name (None for all gates).
            start_time: Filter by start time.
            end_time: Filter by end time.

        Returns:
            Aggregated metrics summary.
        """
        records = await self.get_metrics(
            gate_name=gate_name,
            start_time=start_time,
            end_time=end_time,
        )

        return self._calculate_summary(
            records=records,
            gate_name=gate_name or "all",
        )

    async def get_agent_breakdown(self) -> dict[str, GateMetricsSummary]:
        """Get metrics breakdown by agent.

        Returns:
            Dictionary mapping agent names to their metrics summaries.
            Records without agent_name are excluded.
        """
        records = await self.get_metrics()

        # Group by agent
        agent_records: dict[str, list[GateMetricRecord]] = {}
        for record in records:
            if record.agent_name is not None:
                if record.agent_name not in agent_records:
                    agent_records[record.agent_name] = []
                agent_records[record.agent_name].append(record)

        # Calculate summaries per agent
        breakdown = {}
        for agent_name, agent_data in agent_records.items():
            breakdown[agent_name] = self._calculate_summary(
                records=agent_data,
                gate_name=agent_name,
            )

        return breakdown

    async def get_trends(
        self,
        period: str = "daily",
        gate_name: str | None = None,
    ) -> list[GateTrend]:
        """Get trends over time periods.

        Calculates quality metrics trends aggregated by the specified period.

        Args:
            period: Aggregation period - "daily", "weekly", or "sprint".
            gate_name: Filter by gate name (None for all gates).

        Returns:
            List of GateTrend objects sorted by time.

        Example:
            >>> store = JsonGateMetricsStore(base_path=Path(".yolo"))
            >>> trends = await store.get_trends(period="weekly", gate_name="testability")
            >>> for trend in trends:
            ...     print(f"{trend.period_label}: {trend.pass_rate}%")
        """
        # Import here to avoid circular dependency
        from yolo_developer.gates.metrics_calculator import calculate_trends

        records = await self.get_metrics(gate_name=gate_name)
        return calculate_trends(records, period=period, gate_name=gate_name)

    async def cleanup_old_metrics(self) -> int:
        """Remove metrics older than retention_days.

        Cleans up old metrics to prevent unbounded storage growth.
        This should be called periodically (e.g., at application startup
        or on a schedule).

        Returns:
            Number of metrics records removed.

        Example:
            >>> store = JsonGateMetricsStore(base_path=Path(".yolo"), retention_days=30)
            >>> removed = await store.cleanup_old_metrics()
            >>> print(f"Removed {removed} old metrics")
        """
        async with self._lock:
            metrics_data = await self._load_metrics()
            if not metrics_data:
                return 0

            cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            original_count = len(metrics_data)

            # Filter to keep only recent metrics
            filtered_metrics = []
            for data in metrics_data:
                try:
                    timestamp = datetime.fromisoformat(data["timestamp"])
                    if timestamp >= cutoff_time:
                        filtered_metrics.append(data)
                except (KeyError, ValueError):
                    # Keep records with invalid timestamps (don't lose data)
                    filtered_metrics.append(data)

            removed_count = original_count - len(filtered_metrics)

            if removed_count > 0:
                await self._save_metrics(filtered_metrics)
                logger.info(
                    "Cleaned up old metrics",
                    removed_count=removed_count,
                    retention_days=self.retention_days,
                    remaining_count=len(filtered_metrics),
                )

            return removed_count

    async def _load_metrics(self) -> list[dict[str, Any]]:
        """Load metrics from JSON file.

        Uses cross-process file locking for safe concurrent access.

        Returns:
            List of metric dictionaries, or empty list if file doesn't exist.
        """
        if not self.metrics_file.exists():
            return []

        try:
            with self._file_lock():
                with open(self.metrics_file) as f:
                    data: list[dict[str, Any]] = json.load(f)
                    return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load metrics file", error=str(e))
            return []

    async def _save_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Save metrics to JSON file atomically.

        Uses cross-process file locking combined with atomic rename
        for safe concurrent access.

        Args:
            metrics: List of metric dictionaries to save.
        """
        # Write to temp file first, then rename (atomic on most filesystems)
        temp_file = self.metrics_file.with_suffix(".tmp")

        try:
            with self._file_lock():
                with open(temp_file, "w") as f:
                    json.dump(metrics, f, indent=2)

                # Atomic rename
                temp_file.rename(self.metrics_file)
        except OSError as e:
            logger.error("Failed to save metrics", error=str(e))
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _dict_to_record(self, data: dict[str, Any]) -> GateMetricRecord:
        """Convert dictionary to GateMetricRecord.

        Args:
            data: Dictionary with metric data.

        Returns:
            GateMetricRecord instance.
        """
        timestamp = datetime.fromisoformat(data["timestamp"])

        return GateMetricRecord(
            gate_name=data["gate_name"],
            passed=data["passed"],
            score=data["score"],
            threshold=data["threshold"],
            timestamp=timestamp,
            agent_name=data.get("agent_name"),
            sprint_id=data.get("sprint_id"),
        )

    def _calculate_summary(
        self,
        records: list[GateMetricRecord],
        gate_name: str,
    ) -> GateMetricsSummary:
        """Calculate summary statistics for a set of records.

        Args:
            records: List of metric records.
            gate_name: Name to use in summary.

        Returns:
            GateMetricsSummary with calculated statistics.
        """
        now = datetime.now(timezone.utc)

        if not records:
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

        pass_count = sum(1 for r in records if r.passed)
        fail_count = len(records) - pass_count
        total = len(records)
        pass_rate = (pass_count / total) * 100.0 if total > 0 else 0.0
        avg_score = sum(r.score for r in records) / total if total > 0 else 0.0

        # Get time range
        timestamps = [r.timestamp for r in records]
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
