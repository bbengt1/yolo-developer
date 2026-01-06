"""Unit tests for gate metrics data types (Story 3.9 - Task 1).

These tests verify the metrics data models:
- GateMetricRecord: Individual metric records
- GateMetricsSummary: Aggregated summaries
- GateTrend: Trend analysis data
- TrendDirection: Enum for trend direction
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from yolo_developer.gates.metrics_types import (
    GateMetricRecord,
    GateMetricsSummary,
    GateTrend,
    TrendDirection,
)


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_trend_direction_values(self) -> None:
        """TrendDirection should have expected values."""
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DECLINING.value == "declining"

    def test_trend_direction_is_enum(self) -> None:
        """TrendDirection should be usable as enum."""
        assert len(TrendDirection) == 3
        assert TrendDirection("improving") == TrendDirection.IMPROVING


class TestGateMetricRecord:
    """Tests for GateMetricRecord dataclass."""

    def test_create_metric_record_with_required_fields(self) -> None:
        """GateMetricRecord should be created with required fields."""
        timestamp = datetime.now(timezone.utc)
        record = GateMetricRecord(
            gate_name="testability",
            passed=True,
            score=0.85,
            threshold=0.80,
            timestamp=timestamp,
        )

        assert record.gate_name == "testability"
        assert record.passed is True
        assert record.score == 0.85
        assert record.threshold == 0.80
        assert record.timestamp == timestamp
        assert record.agent_name is None
        assert record.sprint_id is None

    def test_create_metric_record_with_optional_fields(self) -> None:
        """GateMetricRecord should support optional fields."""
        timestamp = datetime.now(timezone.utc)
        record = GateMetricRecord(
            gate_name="architecture_validation",
            passed=False,
            score=0.65,
            threshold=0.70,
            timestamp=timestamp,
            agent_name="analyst",
            sprint_id="sprint-3",
        )

        assert record.agent_name == "analyst"
        assert record.sprint_id == "sprint-3"

    def test_metric_record_is_frozen(self) -> None:
        """GateMetricRecord should be immutable (frozen)."""
        timestamp = datetime.now(timezone.utc)
        record = GateMetricRecord(
            gate_name="testability",
            passed=True,
            score=0.85,
            threshold=0.80,
            timestamp=timestamp,
        )

        with pytest.raises(AttributeError):
            record.gate_name = "other"  # type: ignore[misc]

    def test_metric_record_to_dict(self) -> None:
        """GateMetricRecord.to_dict() should return serializable dict."""
        timestamp = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        record = GateMetricRecord(
            gate_name="testability",
            passed=True,
            score=0.85,
            threshold=0.80,
            timestamp=timestamp,
            agent_name="dev",
            sprint_id="sprint-1",
        )

        result = record.to_dict()

        assert result["gate_name"] == "testability"
        assert result["passed"] is True
        assert result["score"] == 0.85
        assert result["threshold"] == 0.80
        assert result["timestamp"] == "2026-01-05T12:00:00+00:00"
        assert result["agent_name"] == "dev"
        assert result["sprint_id"] == "sprint-1"

    def test_metric_record_to_dict_with_none_optionals(self) -> None:
        """GateMetricRecord.to_dict() should handle None optional fields."""
        timestamp = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        record = GateMetricRecord(
            gate_name="testability",
            passed=True,
            score=0.85,
            threshold=0.80,
            timestamp=timestamp,
        )

        result = record.to_dict()

        assert result["agent_name"] is None
        assert result["sprint_id"] is None


class TestGateMetricsSummary:
    """Tests for GateMetricsSummary dataclass."""

    def test_create_metrics_summary(self) -> None:
        """GateMetricsSummary should be created with all fields."""
        period_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        period_end = datetime(2026, 1, 5, 23, 59, 59, tzinfo=timezone.utc)

        summary = GateMetricsSummary(
            gate_name="testability",
            total_evaluations=100,
            pass_count=85,
            fail_count=15,
            pass_rate=85.0,
            avg_score=0.87,
            period_start=period_start,
            period_end=period_end,
        )

        assert summary.gate_name == "testability"
        assert summary.total_evaluations == 100
        assert summary.pass_count == 85
        assert summary.fail_count == 15
        assert summary.pass_rate == 85.0
        assert summary.avg_score == 0.87
        assert summary.period_start == period_start
        assert summary.period_end == period_end

    def test_metrics_summary_is_frozen(self) -> None:
        """GateMetricsSummary should be immutable (frozen)."""
        period_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        period_end = datetime(2026, 1, 5, 23, 59, 59, tzinfo=timezone.utc)

        summary = GateMetricsSummary(
            gate_name="testability",
            total_evaluations=100,
            pass_count=85,
            fail_count=15,
            pass_rate=85.0,
            avg_score=0.87,
            period_start=period_start,
            period_end=period_end,
        )

        with pytest.raises(AttributeError):
            summary.pass_rate = 90.0  # type: ignore[misc]

    def test_metrics_summary_to_dict(self) -> None:
        """GateMetricsSummary.to_dict() should return serializable dict."""
        period_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        period_end = datetime(2026, 1, 5, 23, 59, 59, tzinfo=timezone.utc)

        summary = GateMetricsSummary(
            gate_name="testability",
            total_evaluations=100,
            pass_count=85,
            fail_count=15,
            pass_rate=85.0,
            avg_score=0.87,
            period_start=period_start,
            period_end=period_end,
        )

        result = summary.to_dict()

        assert result["gate_name"] == "testability"
        assert result["total_evaluations"] == 100
        assert result["pass_count"] == 85
        assert result["fail_count"] == 15
        assert result["pass_rate"] == 85.0
        assert result["avg_score"] == 0.87
        assert result["period_start"] == "2026-01-01T00:00:00+00:00"
        assert result["period_end"] == "2026-01-05T23:59:59+00:00"


class TestGateTrend:
    """Tests for GateTrend dataclass."""

    def test_create_gate_trend(self) -> None:
        """GateTrend should be created with all fields."""
        trend = GateTrend(
            gate_name="testability",
            period="daily",
            pass_rate=85.0,
            avg_score=0.87,
            evaluation_count=25,
            direction=TrendDirection.IMPROVING,
            period_label="2026-01-05",
        )

        assert trend.gate_name == "testability"
        assert trend.period == "daily"
        assert trend.pass_rate == 85.0
        assert trend.avg_score == 0.87
        assert trend.evaluation_count == 25
        assert trend.direction == TrendDirection.IMPROVING
        assert trend.period_label == "2026-01-05"

    def test_gate_trend_is_frozen(self) -> None:
        """GateTrend should be immutable (frozen)."""
        trend = GateTrend(
            gate_name="testability",
            period="daily",
            pass_rate=85.0,
            avg_score=0.87,
            evaluation_count=25,
            direction=TrendDirection.IMPROVING,
            period_label="2026-01-05",
        )

        with pytest.raises(AttributeError):
            trend.direction = TrendDirection.DECLINING  # type: ignore[misc]

    def test_gate_trend_to_dict(self) -> None:
        """GateTrend.to_dict() should return serializable dict."""
        trend = GateTrend(
            gate_name="testability",
            period="weekly",
            pass_rate=92.5,
            avg_score=0.91,
            evaluation_count=150,
            direction=TrendDirection.STABLE,
            period_label="2026-W01",
        )

        result = trend.to_dict()

        assert result["gate_name"] == "testability"
        assert result["period"] == "weekly"
        assert result["pass_rate"] == 92.5
        assert result["avg_score"] == 0.91
        assert result["evaluation_count"] == 150
        assert result["direction"] == "stable"  # Enum converted to string
        assert result["period_label"] == "2026-W01"

    def test_gate_trend_all_directions(self) -> None:
        """GateTrend should work with all TrendDirection values."""
        for direction in TrendDirection:
            trend = GateTrend(
                gate_name="test",
                period="daily",
                pass_rate=50.0,
                avg_score=0.5,
                evaluation_count=10,
                direction=direction,
                period_label="test",
            )
            result = trend.to_dict()
            assert result["direction"] == direction.value


class TestMetricRecordFromDict:
    """Tests for creating GateMetricRecord from dict (deserialization)."""

    def test_create_from_dict_values(self) -> None:
        """GateMetricRecord should be creatable from dict values."""
        data: dict[str, Any] = {
            "gate_name": "testability",
            "passed": True,
            "score": 0.85,
            "threshold": 0.80,
            "timestamp": "2026-01-05T12:00:00+00:00",
            "agent_name": "dev",
            "sprint_id": "sprint-1",
        }

        # Parse timestamp and create record
        timestamp = datetime.fromisoformat(data["timestamp"])
        record = GateMetricRecord(
            gate_name=data["gate_name"],
            passed=data["passed"],
            score=data["score"],
            threshold=data["threshold"],
            timestamp=timestamp,
            agent_name=data["agent_name"],
            sprint_id=data["sprint_id"],
        )

        assert record.gate_name == "testability"
        assert record.passed is True
        assert record.timestamp.year == 2026


class TestJsonSerializability:
    """Tests for JSON serialization compatibility."""

    def test_metric_record_json_serializable(self) -> None:
        """GateMetricRecord.to_dict() should produce JSON-serializable data."""
        import json

        timestamp = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        record = GateMetricRecord(
            gate_name="testability",
            passed=True,
            score=0.85,
            threshold=0.80,
            timestamp=timestamp,
        )

        # Should not raise
        json_str = json.dumps(record.to_dict())
        assert json_str is not None

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["gate_name"] == "testability"

    def test_metrics_summary_json_serializable(self) -> None:
        """GateMetricsSummary.to_dict() should produce JSON-serializable data."""
        import json

        period_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        period_end = datetime(2026, 1, 5, 23, 59, 59, tzinfo=timezone.utc)

        summary = GateMetricsSummary(
            gate_name="testability",
            total_evaluations=100,
            pass_count=85,
            fail_count=15,
            pass_rate=85.0,
            avg_score=0.87,
            period_start=period_start,
            period_end=period_end,
        )

        json_str = json.dumps(summary.to_dict())
        parsed = json.loads(json_str)
        assert parsed["total_evaluations"] == 100

    def test_gate_trend_json_serializable(self) -> None:
        """GateTrend.to_dict() should produce JSON-serializable data."""
        import json

        trend = GateTrend(
            gate_name="testability",
            period="daily",
            pass_rate=85.0,
            avg_score=0.87,
            evaluation_count=25,
            direction=TrendDirection.IMPROVING,
            period_label="2026-01-05",
        )

        json_str = json.dumps(trend.to_dict())
        parsed = json.loads(json_str)
        assert parsed["direction"] == "improving"
