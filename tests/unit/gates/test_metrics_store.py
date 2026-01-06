"""Unit tests for gate metrics storage (Story 3.9 - Task 2 & 3).

These tests verify the metrics storage protocol and JSON implementation:
- GateMetricsStore protocol definition
- JsonGateMetricsStore implementation
- Record/query operations
- Persistence and filtering
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from yolo_developer.gates.metrics_store import (
    GateMetricsStore,
    JsonGateMetricsStore,
)
from yolo_developer.gates.metrics_types import (
    GateMetricRecord,
)


class TestGateMetricsStoreProtocol:
    """Tests for GateMetricsStore protocol."""

    def test_protocol_is_importable(self) -> None:
        """GateMetricsStore protocol should be importable."""
        assert GateMetricsStore is not None

    def test_json_store_implements_protocol(self) -> None:
        """JsonGateMetricsStore should implement GateMetricsStore protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            # Protocol conformance - these methods should exist
            assert hasattr(store, "record_evaluation")
            assert hasattr(store, "get_metrics")
            assert hasattr(store, "get_summary")
            assert hasattr(store, "get_agent_breakdown")
            assert callable(store.record_evaluation)
            assert callable(store.get_metrics)
            assert callable(store.get_summary)
            assert callable(store.get_agent_breakdown)


class TestJsonGateMetricsStoreInit:
    """Tests for JsonGateMetricsStore initialization."""

    def test_create_with_base_path(self) -> None:
        """JsonGateMetricsStore should accept base_path parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            assert store.base_path == Path(tmpdir)

    def test_creates_metrics_directory(self) -> None:
        """JsonGateMetricsStore should create metrics directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / ".yolo"
            store = JsonGateMetricsStore(base_path=base)
            # Directory creation is lazy - happens on first write
            assert store.metrics_file.parent == base / "metrics"


class TestJsonGateMetricsStoreRecordEvaluation:
    """Tests for recording gate evaluations."""

    @pytest.mark.asyncio
    async def test_record_single_evaluation(self) -> None:
        """Should record a single gate evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            record = GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
            )

            await store.record_evaluation(record)

            # Verify record was stored
            metrics = await store.get_metrics()
            assert len(metrics) == 1
            assert metrics[0].gate_name == "testability"
            assert metrics[0].passed is True

    @pytest.mark.asyncio
    async def test_record_multiple_evaluations(self) -> None:
        """Should record multiple gate evaluations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            for i in range(5):
                record = GateMetricRecord(
                    gate_name=f"gate_{i}",
                    passed=i % 2 == 0,
                    score=0.5 + (i * 0.1),
                    threshold=0.70,
                    timestamp=timestamp + timedelta(minutes=i),
                )
                await store.record_evaluation(record)

            metrics = await store.get_metrics()
            assert len(metrics) == 5

    @pytest.mark.asyncio
    async def test_record_with_optional_fields(self) -> None:
        """Should record evaluations with optional agent_name and sprint_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            record = GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
                sprint_id="sprint-3",
            )

            await store.record_evaluation(record)

            metrics = await store.get_metrics()
            assert metrics[0].agent_name == "analyst"
            assert metrics[0].sprint_id == "sprint-3"


class TestJsonGateMetricsStoreGetMetrics:
    """Tests for querying metrics."""

    @pytest.mark.asyncio
    async def test_get_all_metrics(self) -> None:
        """Should return all metrics when no filters provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add some records
            for gate in ["testability", "architecture", "dod"]:
                record = GateMetricRecord(
                    gate_name=gate,
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                )
                await store.record_evaluation(record)

            metrics = await store.get_metrics()
            assert len(metrics) == 3

    @pytest.mark.asyncio
    async def test_filter_by_gate_name(self) -> None:
        """Should filter metrics by gate_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add records for different gates
            for gate in ["testability", "testability", "architecture"]:
                record = GateMetricRecord(
                    gate_name=gate,
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                )
                await store.record_evaluation(record)

            metrics = await store.get_metrics(gate_name="testability")
            assert len(metrics) == 2
            assert all(m.gate_name == "testability" for m in metrics)

    @pytest.mark.asyncio
    async def test_filter_by_agent_name(self) -> None:
        """Should filter metrics by agent_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add records for different agents
            for agent in ["analyst", "analyst", "dev", None]:
                record = GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                    agent_name=agent,
                )
                await store.record_evaluation(record)

            metrics = await store.get_metrics(agent_name="analyst")
            assert len(metrics) == 2
            assert all(m.agent_name == "analyst" for m in metrics)

    @pytest.mark.asyncio
    async def test_filter_by_time_range(self) -> None:
        """Should filter metrics by time range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            base_time = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)

            # Add records at different times
            for i in range(5):
                record = GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=base_time + timedelta(hours=i),
                )
                await store.record_evaluation(record)

            # Filter to middle 3 hours
            start = base_time + timedelta(hours=1)
            end = base_time + timedelta(hours=3)

            metrics = await store.get_metrics(start_time=start, end_time=end)
            assert len(metrics) == 3

    @pytest.mark.asyncio
    async def test_filter_by_sprint_id(self) -> None:
        """Should filter metrics by sprint_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add records for different sprints
            for sprint in ["sprint-1", "sprint-1", "sprint-2", None]:
                record = GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                    sprint_id=sprint,
                )
                await store.record_evaluation(record)

            metrics = await store.get_metrics(sprint_id="sprint-1")
            assert len(metrics) == 2
            assert all(m.sprint_id == "sprint-1" for m in metrics)

    @pytest.mark.asyncio
    async def test_combined_filters(self) -> None:
        """Should support combining multiple filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add varied records
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                    agent_name="analyst",
                )
            )
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=False,
                    score=0.60,
                    threshold=0.80,
                    timestamp=timestamp,
                    agent_name="dev",
                )
            )
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="architecture",
                    passed=True,
                    score=0.90,
                    threshold=0.80,
                    timestamp=timestamp,
                    agent_name="analyst",
                )
            )

            metrics = await store.get_metrics(gate_name="testability", agent_name="analyst")
            assert len(metrics) == 1
            assert metrics[0].gate_name == "testability"
            assert metrics[0].agent_name == "analyst"


class TestJsonGateMetricsStoreGetSummary:
    """Tests for getting metrics summary."""

    @pytest.mark.asyncio
    async def test_get_summary_all_gates(self) -> None:
        """Should return summary for all gates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add 10 records: 7 pass, 3 fail
            for i in range(10):
                record = GateMetricRecord(
                    gate_name="testability",
                    passed=i < 7,
                    score=0.85 if i < 7 else 0.60,
                    threshold=0.80,
                    timestamp=timestamp,
                )
                await store.record_evaluation(record)

            summary = await store.get_summary()

            assert summary.total_evaluations == 10
            assert summary.pass_count == 7
            assert summary.fail_count == 3
            assert summary.pass_rate == 70.0

    @pytest.mark.asyncio
    async def test_get_summary_specific_gate(self) -> None:
        """Should return summary for specific gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add records for multiple gates
            for gate in ["testability", "testability", "architecture"]:
                record = GateMetricRecord(
                    gate_name=gate,
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                )
                await store.record_evaluation(record)

            summary = await store.get_summary(gate_name="testability")

            assert summary.gate_name == "testability"
            assert summary.total_evaluations == 2

    @pytest.mark.asyncio
    async def test_get_summary_empty_store(self) -> None:
        """Should handle empty metrics store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))

            summary = await store.get_summary()

            assert summary.total_evaluations == 0
            assert summary.pass_count == 0
            assert summary.fail_count == 0
            assert summary.pass_rate == 0.0


class TestJsonGateMetricsStoreGetAgentBreakdown:
    """Tests for getting per-agent metrics breakdown."""

    @pytest.mark.asyncio
    async def test_get_agent_breakdown(self) -> None:
        """Should return metrics breakdown by agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add records for different agents
            for agent, passed in [
                ("analyst", True),
                ("analyst", True),
                ("dev", False),
                ("dev", True),
            ]:
                record = GateMetricRecord(
                    gate_name="testability",
                    passed=passed,
                    score=0.85 if passed else 0.60,
                    threshold=0.80,
                    timestamp=timestamp,
                    agent_name=agent,
                )
                await store.record_evaluation(record)

            breakdown = await store.get_agent_breakdown()

            assert "analyst" in breakdown
            assert "dev" in breakdown
            assert breakdown["analyst"].pass_count == 2
            assert breakdown["dev"].pass_count == 1
            assert breakdown["dev"].fail_count == 1

    @pytest.mark.asyncio
    async def test_agent_breakdown_excludes_none_agents(self) -> None:
        """Should handle records without agent_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Add records with and without agent
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                    agent_name="analyst",
                )
            )
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                    agent_name=None,  # No agent
                )
            )

            breakdown = await store.get_agent_breakdown()

            # Should only have "analyst", not None
            assert "analyst" in breakdown
            assert len(breakdown) == 1


class TestJsonGateMetricsStorePersistence:
    """Tests for metrics persistence."""

    @pytest.mark.asyncio
    async def test_metrics_persist_across_instances(self) -> None:
        """Metrics should persist when store is recreated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            timestamp = datetime.now(timezone.utc)

            # First store instance - write records
            store1 = JsonGateMetricsStore(base_path=base_path)
            await store1.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                )
            )

            # Second store instance - read records
            store2 = JsonGateMetricsStore(base_path=base_path)
            metrics = await store2.get_metrics()

            assert len(metrics) == 1
            assert metrics[0].gate_name == "testability"

    @pytest.mark.asyncio
    async def test_metrics_file_is_valid_json(self) -> None:
        """Metrics file should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            timestamp = datetime.now(timezone.utc)

            store = JsonGateMetricsStore(base_path=base_path)
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                )
            )

            # Read and parse the JSON file directly
            with open(store.metrics_file) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["gate_name"] == "testability"


class TestJsonGateMetricsStoreConcurrency:
    """Tests for concurrent access handling."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(self) -> None:
        """Should handle concurrent write operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            # Create multiple records concurrently
            async def write_record(i: int) -> None:
                record = GateMetricRecord(
                    gate_name=f"gate_{i}",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                )
                await store.record_evaluation(record)

            # Write 10 records concurrently
            await asyncio.gather(*[write_record(i) for i in range(10)])

            metrics = await store.get_metrics()
            assert len(metrics) == 10
