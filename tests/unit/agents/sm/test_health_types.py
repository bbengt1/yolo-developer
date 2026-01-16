"""Tests for health monitoring types (Story 10.5).

This module tests the health monitoring type definitions:
- HealthConfig: Configuration for thresholds
- AgentHealthSnapshot: Point-in-time agent health
- HealthMetrics: Comprehensive system metrics
- HealthAlert: Alert data structure
- HealthStatus: Overall health status
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.sm.health_types import (
    DEFAULT_MAX_CHURN_RATE,
    DEFAULT_MAX_CYCLE_TIME_SECONDS,
    DEFAULT_MAX_IDLE_TIME_SECONDS,
    DEFAULT_WARNING_THRESHOLD_RATIO,
    VALID_AGENTS_FOR_HEALTH,
    VALID_ALERT_SEVERITIES,
    VALID_HEALTH_SEVERITIES,
    AgentHealthSnapshot,
    HealthAlert,
    HealthConfig,
    HealthMetrics,
    HealthStatus,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestHealthConstants:
    """Tests for health monitoring constants."""

    def test_default_max_idle_time_seconds(self) -> None:
        """Default idle time should be 300 seconds (5 minutes)."""
        assert DEFAULT_MAX_IDLE_TIME_SECONDS == 300.0

    def test_default_max_cycle_time_seconds(self) -> None:
        """Default cycle time should be 600 seconds (10 minutes)."""
        assert DEFAULT_MAX_CYCLE_TIME_SECONDS == 600.0

    def test_default_max_churn_rate(self) -> None:
        """Default churn rate should be 10 exchanges per minute."""
        assert DEFAULT_MAX_CHURN_RATE == 10.0

    def test_default_warning_threshold_ratio(self) -> None:
        """Default warning threshold should be 0.7 (70%)."""
        assert DEFAULT_WARNING_THRESHOLD_RATIO == 0.7

    def test_valid_health_severities(self) -> None:
        """Valid health severities should include all expected values."""
        expected = {"healthy", "warning", "degraded", "critical"}
        assert VALID_HEALTH_SEVERITIES == expected

    def test_valid_alert_severities(self) -> None:
        """Valid alert severities should include all expected values."""
        expected = {"info", "warning", "critical"}
        assert VALID_ALERT_SEVERITIES == expected

    def test_valid_agents_for_health(self) -> None:
        """Valid agents should include all six agents."""
        expected = {"analyst", "pm", "architect", "dev", "tea", "sm"}
        assert VALID_AGENTS_FOR_HEALTH == expected


# =============================================================================
# HealthConfig Tests
# =============================================================================


class TestHealthConfig:
    """Tests for HealthConfig dataclass."""

    def test_default_values(self) -> None:
        """HealthConfig should have sensible defaults."""
        config = HealthConfig()
        assert config.max_idle_time_seconds == DEFAULT_MAX_IDLE_TIME_SECONDS
        assert config.max_cycle_time_seconds == DEFAULT_MAX_CYCLE_TIME_SECONDS
        assert config.max_churn_rate == DEFAULT_MAX_CHURN_RATE
        assert config.warning_threshold_ratio == DEFAULT_WARNING_THRESHOLD_RATIO
        assert config.enable_alerts is True

    def test_custom_values(self) -> None:
        """HealthConfig should accept custom values."""
        config = HealthConfig(
            max_idle_time_seconds=600.0,
            max_cycle_time_seconds=1200.0,
            max_churn_rate=20.0,
            warning_threshold_ratio=0.8,
            enable_alerts=False,
        )
        assert config.max_idle_time_seconds == 600.0
        assert config.max_cycle_time_seconds == 1200.0
        assert config.max_churn_rate == 20.0
        assert config.warning_threshold_ratio == 0.8
        assert config.enable_alerts is False

    def test_to_dict(self) -> None:
        """to_dict should return complete dictionary representation."""
        config = HealthConfig(max_idle_time_seconds=500.0)
        result = config.to_dict()

        assert result["max_idle_time_seconds"] == 500.0
        assert result["max_cycle_time_seconds"] == DEFAULT_MAX_CYCLE_TIME_SECONDS
        assert result["max_churn_rate"] == DEFAULT_MAX_CHURN_RATE
        assert result["warning_threshold_ratio"] == DEFAULT_WARNING_THRESHOLD_RATIO
        assert result["enable_alerts"] is True

    def test_frozen(self) -> None:
        """HealthConfig should be immutable (frozen)."""
        config = HealthConfig()
        with pytest.raises(AttributeError):
            config.max_idle_time_seconds = 999.0  # type: ignore[misc]


# =============================================================================
# AgentHealthSnapshot Tests
# =============================================================================


class TestAgentHealthSnapshot:
    """Tests for AgentHealthSnapshot dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """AgentHealthSnapshot should store all fields correctly."""
        snapshot = AgentHealthSnapshot(
            agent="analyst",
            idle_time_seconds=120.5,
            last_activity="2026-01-12T10:00:00+00:00",
            cycle_time_seconds=45.0,
            churn_rate=2.5,
            is_healthy=True,
        )
        assert snapshot.agent == "analyst"
        assert snapshot.idle_time_seconds == 120.5
        assert snapshot.last_activity == "2026-01-12T10:00:00+00:00"
        assert snapshot.cycle_time_seconds == 45.0
        assert snapshot.churn_rate == 2.5
        assert snapshot.is_healthy is True
        assert snapshot.captured_at is not None

    def test_cycle_time_can_be_none(self) -> None:
        """cycle_time_seconds can be None when no data available."""
        snapshot = AgentHealthSnapshot(
            agent="pm",
            idle_time_seconds=60.0,
            last_activity="2026-01-12T10:00:00+00:00",
            cycle_time_seconds=None,
            churn_rate=1.0,
            is_healthy=True,
        )
        assert snapshot.cycle_time_seconds is None

    def test_to_dict(self) -> None:
        """to_dict should return complete dictionary representation."""
        snapshot = AgentHealthSnapshot(
            agent="dev",
            idle_time_seconds=30.0,
            last_activity="2026-01-12T10:00:00+00:00",
            cycle_time_seconds=60.0,
            churn_rate=3.0,
            is_healthy=True,
        )
        result = snapshot.to_dict()

        assert result["agent"] == "dev"
        assert result["idle_time_seconds"] == 30.0
        assert result["last_activity"] == "2026-01-12T10:00:00+00:00"
        assert result["cycle_time_seconds"] == 60.0
        assert result["churn_rate"] == 3.0
        assert result["is_healthy"] is True
        assert "captured_at" in result

    def test_frozen(self) -> None:
        """AgentHealthSnapshot should be immutable (frozen)."""
        snapshot = AgentHealthSnapshot(
            agent="tea",
            idle_time_seconds=10.0,
            last_activity="2026-01-12T10:00:00+00:00",
            cycle_time_seconds=20.0,
            churn_rate=0.5,
            is_healthy=True,
        )
        with pytest.raises(AttributeError):
            snapshot.agent = "analyst"  # type: ignore[misc]


# =============================================================================
# HealthMetrics Tests
# =============================================================================


class TestHealthMetrics:
    """Tests for HealthMetrics dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """HealthMetrics should store all fields correctly."""
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 100.0, "pm": 50.0},
            agent_cycle_times={"analyst": 30.0, "pm": 40.0},
            agent_churn_rates={"analyst": 2.0, "pm": 1.5},
            overall_cycle_time=35.0,
            overall_churn_rate=3.5,
            agent_snapshots=(),
        )
        assert metrics.agent_idle_times == {"analyst": 100.0, "pm": 50.0}
        assert metrics.agent_cycle_times == {"analyst": 30.0, "pm": 40.0}
        assert metrics.agent_churn_rates == {"analyst": 2.0, "pm": 1.5}
        assert metrics.overall_cycle_time == 35.0
        assert metrics.overall_churn_rate == 3.5
        assert metrics.agent_snapshots == ()
        assert metrics.collected_at is not None

    def test_to_dict_with_snapshots(self) -> None:
        """to_dict should serialize nested snapshots correctly."""
        snapshot = AgentHealthSnapshot(
            agent="analyst",
            idle_time_seconds=100.0,
            last_activity="2026-01-12T10:00:00+00:00",
            cycle_time_seconds=30.0,
            churn_rate=2.0,
            is_healthy=True,
        )
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 100.0},
            agent_cycle_times={"analyst": 30.0},
            agent_churn_rates={"analyst": 2.0},
            overall_cycle_time=30.0,
            overall_churn_rate=2.0,
            agent_snapshots=(snapshot,),
        )
        result = metrics.to_dict()

        assert result["agent_idle_times"] == {"analyst": 100.0}
        assert result["agent_cycle_times"] == {"analyst": 30.0}
        assert result["agent_churn_rates"] == {"analyst": 2.0}
        assert result["overall_cycle_time"] == 30.0
        assert result["overall_churn_rate"] == 2.0
        assert len(result["agent_snapshots"]) == 1
        assert result["agent_snapshots"][0]["agent"] == "analyst"
        assert "collected_at" in result

    def test_empty_metrics(self) -> None:
        """HealthMetrics should handle empty data."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        result = metrics.to_dict()
        assert result["agent_idle_times"] == {}
        assert result["agent_snapshots"] == []

    def test_frozen(self) -> None:
        """HealthMetrics should be immutable (frozen)."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        with pytest.raises(AttributeError):
            metrics.overall_cycle_time = 999.0  # type: ignore[misc]


# =============================================================================
# HealthAlert Tests
# =============================================================================


class TestHealthAlert:
    """Tests for HealthAlert dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """HealthAlert should store all fields correctly."""
        alert = HealthAlert(
            severity="warning",
            alert_type="idle_time_warning",
            message="Agent analyst idle for 250 seconds",
            affected_agent="analyst",
            metric_value=250.0,
            threshold_value=210.0,
        )
        assert alert.severity == "warning"
        assert alert.alert_type == "idle_time_warning"
        assert alert.message == "Agent analyst idle for 250 seconds"
        assert alert.affected_agent == "analyst"
        assert alert.metric_value == 250.0
        assert alert.threshold_value == 210.0
        assert alert.triggered_at is not None

    def test_affected_agent_can_be_none(self) -> None:
        """affected_agent can be None for system-wide alerts."""
        alert = HealthAlert(
            severity="critical",
            alert_type="high_churn",
            message="System-wide churn rate exceeded",
            affected_agent=None,
            metric_value=15.0,
            threshold_value=10.0,
        )
        assert alert.affected_agent is None

    def test_to_dict(self) -> None:
        """to_dict should return complete dictionary representation."""
        alert = HealthAlert(
            severity="critical",
            alert_type="idle_time_exceeded",
            message="Agent pm exceeded idle threshold",
            affected_agent="pm",
            metric_value=400.0,
            threshold_value=300.0,
        )
        result = alert.to_dict()

        assert result["severity"] == "critical"
        assert result["alert_type"] == "idle_time_exceeded"
        assert result["message"] == "Agent pm exceeded idle threshold"
        assert result["affected_agent"] == "pm"
        assert result["metric_value"] == 400.0
        assert result["threshold_value"] == 300.0
        assert "triggered_at" in result

    def test_frozen(self) -> None:
        """HealthAlert should be immutable (frozen)."""
        alert = HealthAlert(
            severity="info",
            alert_type="test",
            message="Test",
            affected_agent=None,
            metric_value=1.0,
            threshold_value=2.0,
        )
        with pytest.raises(AttributeError):
            alert.severity = "critical"  # type: ignore[misc]


# =============================================================================
# HealthStatus Tests
# =============================================================================


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    @pytest.fixture
    def sample_metrics(self) -> HealthMetrics:
        """Create sample HealthMetrics for testing."""
        return HealthMetrics(
            agent_idle_times={"analyst": 100.0},
            agent_cycle_times={"analyst": 30.0},
            agent_churn_rates={"analyst": 2.0},
            overall_cycle_time=30.0,
            overall_churn_rate=2.0,
            agent_snapshots=(),
        )

    def test_creation_healthy(self, sample_metrics: HealthMetrics) -> None:
        """HealthStatus should represent healthy state correctly."""
        status = HealthStatus(
            status="healthy",
            metrics=sample_metrics,
            alerts=(),
            summary="All systems nominal",
            is_healthy=True,
        )
        assert status.status == "healthy"
        assert status.metrics == sample_metrics
        assert status.alerts == ()
        assert status.summary == "All systems nominal"
        assert status.is_healthy is True
        assert status.evaluated_at is not None

    def test_creation_with_alerts(self, sample_metrics: HealthMetrics) -> None:
        """HealthStatus should include alerts when present."""
        alert = HealthAlert(
            severity="warning",
            alert_type="test",
            message="Test alert",
            affected_agent="analyst",
            metric_value=5.0,
            threshold_value=3.0,
        )
        status = HealthStatus(
            status="warning",
            metrics=sample_metrics,
            alerts=(alert,),
            summary="Warning: 1 alert active",
            is_healthy=True,
        )
        assert status.status == "warning"
        assert len(status.alerts) == 1
        assert status.alerts[0].severity == "warning"

    def test_to_dict(self, sample_metrics: HealthMetrics) -> None:
        """to_dict should serialize nested structures correctly."""
        alert = HealthAlert(
            severity="critical",
            alert_type="test",
            message="Critical alert",
            affected_agent=None,
            metric_value=10.0,
            threshold_value=5.0,
        )
        status = HealthStatus(
            status="critical",
            metrics=sample_metrics,
            alerts=(alert,),
            summary="System critical",
            is_healthy=False,
        )
        result = status.to_dict()

        assert result["status"] == "critical"
        assert "metrics" in result
        assert result["metrics"]["overall_cycle_time"] == 30.0
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["severity"] == "critical"
        assert result["summary"] == "System critical"
        assert result["is_healthy"] is False
        assert "evaluated_at" in result

    def test_frozen(self, sample_metrics: HealthMetrics) -> None:
        """HealthStatus should be immutable (frozen)."""
        status = HealthStatus(
            status="healthy",
            metrics=sample_metrics,
            alerts=(),
            summary="OK",
            is_healthy=True,
        )
        with pytest.raises(AttributeError):
            status.status = "critical"  # type: ignore[misc]

    def test_degraded_state(self, sample_metrics: HealthMetrics) -> None:
        """HealthStatus can represent degraded state."""
        status = HealthStatus(
            status="degraded",
            metrics=sample_metrics,
            alerts=(),
            summary="System degraded",
            is_healthy=False,
        )
        assert status.status == "degraded"
        assert status.is_healthy is False
