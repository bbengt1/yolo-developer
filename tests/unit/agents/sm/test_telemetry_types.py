"""Tests for telemetry type definitions (Story 10.16).

This module tests the type definitions in telemetry_types.py:
- MetricSummary frozen dataclass
- TelemetrySnapshot frozen dataclass
- TelemetryConfig frozen dataclass
- DashboardMetrics frozen dataclass
- Constants and validation

References:
    - FR72: SM Agent can maintain system health telemetry dashboard data
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from yolo_developer.agents.sm.telemetry_types import (
    DEFAULT_TELEMETRY_INTERVAL_SECONDS,
    DEFAULT_TELEMETRY_RETENTION_HOURS,
    MAX_TELEMETRY_INTERVAL_SECONDS,
    MIN_TELEMETRY_INTERVAL_SECONDS,
    VALID_METRIC_CATEGORIES,
    VALID_METRIC_STATUSES,
    VALID_METRIC_TRENDS,
    DashboardMetrics,
    MetricSummary,
    TelemetryConfig,
    TelemetrySnapshot,
)

# =============================================================================
# Constants Tests
# =============================================================================


class TestTelemetryConstants:
    """Tests for telemetry constants."""

    def test_default_telemetry_interval_seconds(self) -> None:
        """Test default telemetry interval is 60 seconds."""
        assert DEFAULT_TELEMETRY_INTERVAL_SECONDS == 60.0

    def test_default_telemetry_retention_hours(self) -> None:
        """Test default retention is 24 hours."""
        assert DEFAULT_TELEMETRY_RETENTION_HOURS == 24

    def test_min_telemetry_interval_seconds(self) -> None:
        """Test minimum telemetry interval is 5 seconds."""
        assert MIN_TELEMETRY_INTERVAL_SECONDS == 5.0

    def test_max_telemetry_interval_seconds(self) -> None:
        """Test maximum telemetry interval is 1 hour."""
        assert MAX_TELEMETRY_INTERVAL_SECONDS == 3600.0

    def test_valid_metric_statuses(self) -> None:
        """Test valid metric statuses contains expected values."""
        assert VALID_METRIC_STATUSES == frozenset({"healthy", "warning", "critical"})

    def test_valid_metric_trends(self) -> None:
        """Test valid metric trends contains expected values."""
        assert VALID_METRIC_TRENDS == frozenset({"improving", "stable", "declining"})

    def test_valid_metric_categories(self) -> None:
        """Test valid metric categories contains expected values."""
        assert VALID_METRIC_CATEGORIES == frozenset(
            {"velocity", "cycle_time", "churn_rate", "idle_time", "health"}
        )


# =============================================================================
# MetricSummary Tests
# =============================================================================


class TestMetricSummary:
    """Tests for MetricSummary dataclass."""

    def test_create_valid_metric_summary(self) -> None:
        """Test creating a valid MetricSummary."""
        summary = MetricSummary(
            name="velocity",
            value=5.2,
            unit="stories/sprint",
            display_value="5.2 stories/sprint",
            trend="stable",
            status="healthy",
        )
        assert summary.name == "velocity"
        assert summary.value == 5.2
        assert summary.unit == "stories/sprint"
        assert summary.display_value == "5.2 stories/sprint"
        assert summary.trend == "stable"
        assert summary.status == "healthy"

    def test_metric_summary_with_none_trend(self) -> None:
        """Test MetricSummary with None trend (no trend data)."""
        summary = MetricSummary(
            name="cycle_time",
            value=45000.0,
            unit="ms",
            display_value="45 sec",
            trend=None,
            status="healthy",
        )
        assert summary.trend is None

    def test_metric_summary_is_frozen(self) -> None:
        """Test that MetricSummary is immutable."""
        summary = MetricSummary(
            name="velocity",
            value=5.0,
            unit="stories/sprint",
            display_value="5.0 stories/sprint",
            trend="stable",
            status="healthy",
        )
        with pytest.raises(AttributeError):
            summary.value = 6.0  # type: ignore[misc]

    def test_metric_summary_to_dict(self) -> None:
        """Test MetricSummary serialization to dict."""
        summary = MetricSummary(
            name="churn_rate",
            value=3.5,
            unit="exchanges/min",
            display_value="3.5 exchanges/min",
            trend="declining",
            status="warning",
        )
        result = summary.to_dict()
        assert result["name"] == "churn_rate"
        assert result["value"] == 3.5
        assert result["unit"] == "exchanges/min"
        assert result["display_value"] == "3.5 exchanges/min"
        assert result["trend"] == "declining"
        assert result["status"] == "warning"

    def test_metric_summary_to_dict_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        summary = MetricSummary(
            name="velocity",
            value=5.2,
            unit="stories/sprint",
            display_value="5.2 stories/sprint",
            trend="stable",
            status="healthy",
        )
        # Should not raise
        json_str = json.dumps(summary.to_dict())
        assert "velocity" in json_str

    def test_metric_summary_warns_on_empty_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged for empty name."""
        with caplog.at_level(logging.WARNING):
            MetricSummary(
                name="",
                value=0.0,
                unit="",
                display_value="",
                trend=None,
                status="healthy",
            )
        assert "name is empty" in caplog.text

    def test_metric_summary_warns_on_invalid_status(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged for invalid status."""
        with caplog.at_level(logging.WARNING):
            MetricSummary(
                name="test",
                value=0.0,
                unit="",
                display_value="",
                trend=None,
                status="invalid",  # type: ignore[arg-type]
            )
        assert "not a valid status" in caplog.text

    def test_metric_summary_warns_on_invalid_trend(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged for invalid trend."""
        with caplog.at_level(logging.WARNING):
            MetricSummary(
                name="test",
                value=0.0,
                unit="",
                display_value="",
                trend="invalid",  # type: ignore[arg-type]
                status="healthy",
            )
        assert "not a valid trend" in caplog.text


# =============================================================================
# TelemetrySnapshot Tests
# =============================================================================


class TestTelemetrySnapshot:
    """Tests for TelemetrySnapshot dataclass."""

    @pytest.fixture
    def sample_metric_summary(self) -> MetricSummary:
        """Create a sample MetricSummary for testing."""
        return MetricSummary(
            name="velocity",
            value=5.0,
            unit="stories/sprint",
            display_value="5.0 stories/sprint",
            trend="stable",
            status="healthy",
        )

    def test_create_valid_telemetry_snapshot(self, sample_metric_summary: MetricSummary) -> None:
        """Test creating a valid TelemetrySnapshot."""
        snapshot = TelemetrySnapshot(
            burn_down_velocity=sample_metric_summary,
            cycle_time=sample_metric_summary,
            churn_rate=sample_metric_summary,
            agent_idle_times={"analyst": sample_metric_summary},
            health_status="healthy",
            alert_count=0,
        )
        assert snapshot.health_status == "healthy"
        assert snapshot.alert_count == 0
        assert "analyst" in snapshot.agent_idle_times

    def test_telemetry_snapshot_auto_timestamp(self, sample_metric_summary: MetricSummary) -> None:
        """Test that collected_at is auto-generated."""
        snapshot = TelemetrySnapshot(
            burn_down_velocity=sample_metric_summary,
            cycle_time=sample_metric_summary,
            churn_rate=sample_metric_summary,
            agent_idle_times={},
            health_status="healthy",
        )
        # Should be valid ISO timestamp
        parsed = datetime.fromisoformat(snapshot.collected_at)
        assert parsed.tzinfo == timezone.utc

    def test_telemetry_snapshot_is_frozen(self, sample_metric_summary: MetricSummary) -> None:
        """Test that TelemetrySnapshot is immutable."""
        snapshot = TelemetrySnapshot(
            burn_down_velocity=sample_metric_summary,
            cycle_time=sample_metric_summary,
            churn_rate=sample_metric_summary,
            agent_idle_times={},
            health_status="healthy",
        )
        with pytest.raises(AttributeError):
            snapshot.health_status = "critical"  # type: ignore[misc]

    def test_telemetry_snapshot_to_dict(self, sample_metric_summary: MetricSummary) -> None:
        """Test TelemetrySnapshot serialization to dict."""
        snapshot = TelemetrySnapshot(
            burn_down_velocity=sample_metric_summary,
            cycle_time=sample_metric_summary,
            churn_rate=sample_metric_summary,
            agent_idle_times={"analyst": sample_metric_summary, "pm": sample_metric_summary},
            health_status="warning",
            alert_count=2,
        )
        result = snapshot.to_dict()
        assert result["health_status"] == "warning"
        assert result["alert_count"] == 2
        assert "burn_down_velocity" in result
        assert "agent_idle_times" in result
        assert "analyst" in result["agent_idle_times"]
        assert "pm" in result["agent_idle_times"]

    def test_telemetry_snapshot_to_dict_json_serializable(
        self, sample_metric_summary: MetricSummary
    ) -> None:
        """Test that to_dict output is JSON serializable."""
        snapshot = TelemetrySnapshot(
            burn_down_velocity=sample_metric_summary,
            cycle_time=sample_metric_summary,
            churn_rate=sample_metric_summary,
            agent_idle_times={"analyst": sample_metric_summary},
            health_status="healthy",
        )
        # Should not raise
        json_str = json.dumps(snapshot.to_dict())
        assert "healthy" in json_str

    def test_telemetry_snapshot_warns_on_invalid_health_status(
        self, sample_metric_summary: MetricSummary, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for invalid health status."""
        with caplog.at_level(logging.WARNING):
            TelemetrySnapshot(
                burn_down_velocity=sample_metric_summary,
                cycle_time=sample_metric_summary,
                churn_rate=sample_metric_summary,
                agent_idle_times={},
                health_status="invalid",  # type: ignore[arg-type]
            )
        assert "not a valid status" in caplog.text

    def test_telemetry_snapshot_warns_on_negative_alert_count(
        self, sample_metric_summary: MetricSummary, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for negative alert count."""
        with caplog.at_level(logging.WARNING):
            TelemetrySnapshot(
                burn_down_velocity=sample_metric_summary,
                cycle_time=sample_metric_summary,
                churn_rate=sample_metric_summary,
                agent_idle_times={},
                health_status="healthy",
                alert_count=-1,
            )
        assert "is negative" in caplog.text


# =============================================================================
# TelemetryConfig Tests
# =============================================================================


class TestTelemetryConfig:
    """Tests for TelemetryConfig dataclass."""

    def test_create_default_config(self) -> None:
        """Test creating TelemetryConfig with defaults."""
        config = TelemetryConfig()
        assert config.collection_interval_seconds == DEFAULT_TELEMETRY_INTERVAL_SECONDS
        assert config.retention_hours == DEFAULT_TELEMETRY_RETENTION_HOURS
        assert config.include_agent_details is True
        assert config.enable_trend_analysis is True

    def test_create_custom_config(self) -> None:
        """Test creating TelemetryConfig with custom values."""
        config = TelemetryConfig(
            collection_interval_seconds=30.0,
            retention_hours=48,
            include_agent_details=False,
            enable_trend_analysis=False,
        )
        assert config.collection_interval_seconds == 30.0
        assert config.retention_hours == 48
        assert config.include_agent_details is False
        assert config.enable_trend_analysis is False

    def test_telemetry_config_is_frozen(self) -> None:
        """Test that TelemetryConfig is immutable."""
        config = TelemetryConfig()
        with pytest.raises(AttributeError):
            config.retention_hours = 12  # type: ignore[misc]

    def test_telemetry_config_to_dict(self) -> None:
        """Test TelemetryConfig serialization to dict."""
        config = TelemetryConfig(
            collection_interval_seconds=45.0,
            retention_hours=12,
        )
        result = config.to_dict()
        assert result["collection_interval_seconds"] == 45.0
        assert result["retention_hours"] == 12
        assert result["include_agent_details"] is True
        assert result["enable_trend_analysis"] is True

    def test_telemetry_config_to_dict_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        config = TelemetryConfig()
        # Should not raise
        json_str = json.dumps(config.to_dict())
        assert "collection_interval_seconds" in json_str

    def test_telemetry_config_warns_on_interval_below_minimum(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for interval below minimum."""
        with caplog.at_level(logging.WARNING):
            TelemetryConfig(collection_interval_seconds=1.0)
        assert "below minimum" in caplog.text

    def test_telemetry_config_warns_on_interval_above_maximum(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for interval above maximum."""
        with caplog.at_level(logging.WARNING):
            TelemetryConfig(collection_interval_seconds=7200.0)
        assert "exceeds maximum" in caplog.text

    def test_telemetry_config_warns_on_non_positive_retention(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for non-positive retention hours."""
        with caplog.at_level(logging.WARNING):
            TelemetryConfig(retention_hours=0)
        assert "should be positive" in caplog.text


# =============================================================================
# DashboardMetrics Tests
# =============================================================================


class TestDashboardMetrics:
    """Tests for DashboardMetrics dataclass."""

    @pytest.fixture
    def sample_snapshot(self) -> TelemetrySnapshot:
        """Create a sample TelemetrySnapshot for testing."""
        summary = MetricSummary(
            name="velocity",
            value=5.0,
            unit="stories/sprint",
            display_value="5.0 stories/sprint",
            trend="stable",
            status="healthy",
        )
        return TelemetrySnapshot(
            burn_down_velocity=summary,
            cycle_time=summary,
            churn_rate=summary,
            agent_idle_times={"analyst": summary},
            health_status="healthy",
            alert_count=0,
        )

    def test_create_valid_dashboard_metrics(self, sample_snapshot: TelemetrySnapshot) -> None:
        """Test creating valid DashboardMetrics."""
        metrics = DashboardMetrics(
            snapshot=sample_snapshot,
            velocity_display="5.0 stories/sprint (stable)",
            cycle_time_display="45 min avg (p90: 1.2h)",
            health_summary="System healthy, 0 alerts",
            agent_status_table=({"agent": "analyst", "idle_time": "30s", "status": "healthy"},),
        )
        assert metrics.velocity_display == "5.0 stories/sprint (stable)"
        assert metrics.cycle_time_display == "45 min avg (p90: 1.2h)"
        assert metrics.health_summary == "System healthy, 0 alerts"
        assert len(metrics.agent_status_table) == 1

    def test_dashboard_metrics_is_frozen(self, sample_snapshot: TelemetrySnapshot) -> None:
        """Test that DashboardMetrics is immutable."""
        metrics = DashboardMetrics(
            snapshot=sample_snapshot,
            velocity_display="5.0 stories/sprint",
            cycle_time_display="45 min",
            health_summary="Healthy",
            agent_status_table=(),
        )
        with pytest.raises(AttributeError):
            metrics.health_summary = "Not healthy"  # type: ignore[misc]

    def test_dashboard_metrics_to_dict(self, sample_snapshot: TelemetrySnapshot) -> None:
        """Test DashboardMetrics serialization to dict."""
        metrics = DashboardMetrics(
            snapshot=sample_snapshot,
            velocity_display="5.0 stories/sprint (stable)",
            cycle_time_display="45 min avg",
            health_summary="System healthy",
            agent_status_table=(
                {"agent": "analyst", "idle_time": "30s", "status": "healthy"},
                {"agent": "pm", "idle_time": "45s", "status": "healthy"},
            ),
        )
        result = metrics.to_dict()
        assert result["velocity_display"] == "5.0 stories/sprint (stable)"
        assert result["cycle_time_display"] == "45 min avg"
        assert result["health_summary"] == "System healthy"
        assert len(result["agent_status_table"]) == 2
        assert "snapshot" in result

    def test_dashboard_metrics_to_dict_json_serializable(
        self, sample_snapshot: TelemetrySnapshot
    ) -> None:
        """Test that to_dict output is JSON serializable."""
        metrics = DashboardMetrics(
            snapshot=sample_snapshot,
            velocity_display="5.0 stories/sprint",
            cycle_time_display="45 min",
            health_summary="Healthy",
            agent_status_table=(),
        )
        # Should not raise
        json_str = json.dumps(metrics.to_dict())
        assert "velocity_display" in json_str

    def test_dashboard_metrics_warns_on_empty_velocity_display(
        self, sample_snapshot: TelemetrySnapshot, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for empty velocity display."""
        with caplog.at_level(logging.WARNING):
            DashboardMetrics(
                snapshot=sample_snapshot,
                velocity_display="",
                cycle_time_display="45 min",
                health_summary="Healthy",
                agent_status_table=(),
            )
        assert "velocity_display is empty" in caplog.text

    def test_dashboard_metrics_warns_on_empty_cycle_time_display(
        self, sample_snapshot: TelemetrySnapshot, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for empty cycle time display."""
        with caplog.at_level(logging.WARNING):
            DashboardMetrics(
                snapshot=sample_snapshot,
                velocity_display="5.0 stories/sprint",
                cycle_time_display="",
                health_summary="Healthy",
                agent_status_table=(),
            )
        assert "cycle_time_display is empty" in caplog.text

    def test_dashboard_metrics_warns_on_empty_health_summary(
        self, sample_snapshot: TelemetrySnapshot, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for empty health summary."""
        with caplog.at_level(logging.WARNING):
            DashboardMetrics(
                snapshot=sample_snapshot,
                velocity_display="5.0 stories/sprint",
                cycle_time_display="45 min",
                health_summary="",
                agent_status_table=(),
            )
        assert "health_summary is empty" in caplog.text
