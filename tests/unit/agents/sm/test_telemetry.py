"""Tests for telemetry module (Story 10.16).

This module tests the telemetry collection and formatting functions:
- collect_telemetry aggregation
- format_for_dashboard formatting
- get_dashboard_telemetry main function
- Graceful handling of None inputs

References:
    - FR72: SM Agent can maintain system health telemetry dashboard data
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - FR67: SM Agent can detect agent churn rate and idle time
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import structlog

from yolo_developer.agents.sm.health_types import (
    HealthAlert,
    HealthMetrics,
    HealthStatus,
)
from yolo_developer.agents.sm.telemetry import (
    _aggregate_burn_down_velocity,
    _aggregate_churn_rate,
    _aggregate_cycle_time,
    _aggregate_idle_times,
    _format_agent_status_table,
    _format_cycle_time_display,
    _format_duration_ms,
    _format_duration_seconds,
    _format_health_summary,
    _format_velocity_display,
    collect_telemetry,
    format_for_dashboard,
    get_dashboard_telemetry,
)
from yolo_developer.agents.sm.telemetry_types import (
    DashboardMetrics,
    MetricSummary,
    TelemetryConfig,
    TelemetrySnapshot,
)
from yolo_developer.agents.sm.velocity_types import VelocityMetrics

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_health_status() -> HealthStatus:
    """Create a sample HealthStatus for testing."""
    metrics = HealthMetrics(
        agent_idle_times={"analyst": 120.0, "pm": 60.0, "dev": 300.0},
        agent_cycle_times={"analyst": 45.0, "pm": 30.0},
        agent_churn_rates={"analyst": 2.0, "pm": 1.5},
        overall_cycle_time=40.0,
        overall_churn_rate=3.5,
        unproductive_churn_rate=0.5,
        cycle_time_percentiles={"p50": 35.0, "p90": 60.0, "p95": 75.0},
        agent_snapshots=(),
    )
    return HealthStatus(
        status="healthy",
        metrics=metrics,
        alerts=(),
        summary="All systems nominal",
        is_healthy=True,
    )


@pytest.fixture
def sample_velocity_metrics() -> VelocityMetrics:
    """Create a sample VelocityMetrics for testing."""
    return VelocityMetrics(
        average_stories_per_sprint=5.2,
        average_points_per_sprint=5.2,
        average_cycle_time_ms=3600000.0,  # 1 hour
        cycle_time_p50_ms=3000000.0,
        cycle_time_p90_ms=5400000.0,  # 1.5 hours
        sprints_analyzed=5,
        trend="stable",
    )


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create a sample YoloState for testing."""
    return {
        "messages": [],
        "current_agent": "sm",
    }


# =============================================================================
# Duration Formatting Tests
# =============================================================================


class TestDurationFormatting:
    """Tests for duration formatting helper functions."""

    def test_format_duration_ms_zero(self) -> None:
        """Test formatting zero milliseconds."""
        assert _format_duration_ms(0.0) == "0 sec"

    def test_format_duration_ms_seconds(self) -> None:
        """Test formatting milliseconds as seconds."""
        assert _format_duration_ms(30000.0) == "30 sec"

    def test_format_duration_ms_minutes(self) -> None:
        """Test formatting milliseconds as minutes."""
        assert _format_duration_ms(180000.0) == "3 min"

    def test_format_duration_ms_hours(self) -> None:
        """Test formatting milliseconds as hours."""
        assert _format_duration_ms(5400000.0) == "1.5h"

    def test_format_duration_seconds_zero(self) -> None:
        """Test formatting zero seconds."""
        assert _format_duration_seconds(0.0) == "0s"

    def test_format_duration_seconds_small(self) -> None:
        """Test formatting small seconds value."""
        assert _format_duration_seconds(45.0) == "45s"

    def test_format_duration_seconds_minutes(self) -> None:
        """Test formatting seconds as minutes."""
        assert _format_duration_seconds(300.0) == "5 min"

    def test_format_duration_seconds_hours(self) -> None:
        """Test formatting seconds as hours."""
        assert _format_duration_seconds(7200.0) == "2.0h"


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregateVelocity:
    """Tests for _aggregate_burn_down_velocity function."""

    def test_aggregate_velocity_with_data(
        self, sample_velocity_metrics: VelocityMetrics
    ) -> None:
        """Test velocity aggregation with valid data."""
        result = _aggregate_burn_down_velocity(sample_velocity_metrics)
        assert result.name == "velocity"
        assert result.value == 5.2
        assert result.unit == "stories/sprint"
        assert "5.2" in result.display_value
        assert result.trend == "stable"
        assert result.status == "healthy"

    def test_aggregate_velocity_none(self) -> None:
        """Test velocity aggregation with None input."""
        result = _aggregate_burn_down_velocity(None)
        assert result.name == "velocity"
        assert result.value == 0.0
        assert result.display_value == "No data"
        assert result.trend is None

    def test_aggregate_velocity_declining_trend(self) -> None:
        """Test velocity aggregation with declining trend shows warning."""
        velocity = VelocityMetrics(
            average_stories_per_sprint=3.0,
            average_points_per_sprint=3.0,
            average_cycle_time_ms=4000000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=6000000.0,
            sprints_analyzed=5,
            trend="declining",
        )
        result = _aggregate_burn_down_velocity(velocity)
        assert result.status == "warning"
        assert "declining" in result.display_value

    def test_aggregate_velocity_insufficient_data_no_trend(self) -> None:
        """Test velocity with insufficient sprints has no trend."""
        velocity = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3000000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=1,  # Not enough for trend
            trend="stable",
        )
        result = _aggregate_burn_down_velocity(velocity)
        assert result.trend is None


class TestAggregateCycleTime:
    """Tests for _aggregate_cycle_time function."""

    def test_aggregate_cycle_time_from_velocity(
        self, sample_health_status: HealthStatus, sample_velocity_metrics: VelocityMetrics
    ) -> None:
        """Test cycle time aggregation prefers velocity metrics."""
        result = _aggregate_cycle_time(sample_health_status, sample_velocity_metrics)
        assert result.name == "cycle_time"
        assert result.value == 3600000.0  # From velocity, not health
        assert "avg" in result.display_value
        assert "p90" in result.display_value

    def test_aggregate_cycle_time_from_health_only(
        self, sample_health_status: HealthStatus
    ) -> None:
        """Test cycle time falls back to health metrics when no velocity."""
        result = _aggregate_cycle_time(sample_health_status, None)
        assert result.name == "cycle_time"
        # Health metrics overall_cycle_time is in seconds, converted to ms
        assert result.value == 40000.0  # 40 seconds * 1000

    def test_aggregate_cycle_time_none_inputs(self) -> None:
        """Test cycle time aggregation with all None inputs."""
        result = _aggregate_cycle_time(None, None)
        assert result.name == "cycle_time"
        assert result.value == 0.0
        assert result.display_value == "No data"


class TestAggregateChurnRate:
    """Tests for _aggregate_churn_rate function."""

    def test_aggregate_churn_rate_healthy(
        self, sample_health_status: HealthStatus
    ) -> None:
        """Test churn rate aggregation with healthy value."""
        result = _aggregate_churn_rate(sample_health_status)
        assert result.name == "churn_rate"
        assert result.value == 3.5
        assert result.status == "healthy"
        assert "exchanges/min" in result.display_value

    def test_aggregate_churn_rate_none(self) -> None:
        """Test churn rate aggregation with None input."""
        result = _aggregate_churn_rate(None)
        assert result.name == "churn_rate"
        assert result.value == 0.0
        assert result.display_value == "No data"

    def test_aggregate_churn_rate_warning(self) -> None:
        """Test churn rate shows warning when high."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=8.0,  # High but not critical
            agent_snapshots=(),
        )
        health = HealthStatus(
            status="warning",
            metrics=metrics,
            alerts=(),
            summary="High churn",
            is_healthy=True,
        )
        result = _aggregate_churn_rate(health)
        assert result.status == "warning"

    def test_aggregate_churn_rate_critical(self) -> None:
        """Test churn rate shows critical when very high."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=12.0,  # Very high
            agent_snapshots=(),
        )
        health = HealthStatus(
            status="critical",
            metrics=metrics,
            alerts=(),
            summary="Very high churn",
            is_healthy=False,
        )
        result = _aggregate_churn_rate(health)
        assert result.status == "critical"


class TestAggregateIdleTimes:
    """Tests for _aggregate_idle_times function."""

    def test_aggregate_idle_times_with_data(
        self, sample_health_status: HealthStatus
    ) -> None:
        """Test idle times aggregation with valid data."""
        result = _aggregate_idle_times(sample_health_status)
        assert "analyst" in result
        assert "pm" in result
        assert "dev" in result
        assert result["analyst"].value == 120.0
        assert result["pm"].status == "healthy"

    def test_aggregate_idle_times_none(self) -> None:
        """Test idle times aggregation with None input."""
        result = _aggregate_idle_times(None)
        assert result == {}

    def test_aggregate_idle_times_warning_threshold(self) -> None:
        """Test idle times shows warning for medium idle time."""
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 400.0},  # >300s, <600s
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        health = HealthStatus(
            status="warning",
            metrics=metrics,
            alerts=(),
            summary="Agent idle",
            is_healthy=True,
        )
        result = _aggregate_idle_times(health)
        assert result["analyst"].status == "warning"

    def test_aggregate_idle_times_critical_threshold(self) -> None:
        """Test idle times shows critical for very high idle time."""
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 700.0},  # >600s
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        health = HealthStatus(
            status="critical",
            metrics=metrics,
            alerts=(),
            summary="Agent very idle",
            is_healthy=False,
        )
        result = _aggregate_idle_times(health)
        assert result["analyst"].status == "critical"


# =============================================================================
# Collect Telemetry Tests
# =============================================================================


class TestCollectTelemetry:
    """Tests for collect_telemetry function."""

    def test_collect_telemetry_full_data(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
        sample_velocity_metrics: VelocityMetrics,
    ) -> None:
        """Test telemetry collection with all data available."""
        result = collect_telemetry(sample_state, sample_health_status, sample_velocity_metrics)
        assert isinstance(result, TelemetrySnapshot)
        assert result.health_status == "healthy"
        assert result.alert_count == 0
        assert result.burn_down_velocity.value == 5.2
        assert len(result.agent_idle_times) == 3

    def test_collect_telemetry_no_health(
        self,
        sample_state: dict[str, Any],
        sample_velocity_metrics: VelocityMetrics,
    ) -> None:
        """Test telemetry collection without health data."""
        result = collect_telemetry(sample_state, None, sample_velocity_metrics)
        assert isinstance(result, TelemetrySnapshot)
        assert result.churn_rate.display_value == "No data"
        assert len(result.agent_idle_times) == 0

    def test_collect_telemetry_no_velocity(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
    ) -> None:
        """Test telemetry collection without velocity data."""
        result = collect_telemetry(sample_state, sample_health_status, None)
        assert isinstance(result, TelemetrySnapshot)
        assert result.burn_down_velocity.display_value == "No data"

    def test_collect_telemetry_all_none(
        self,
        sample_state: dict[str, Any],
    ) -> None:
        """Test telemetry collection with no data available."""
        result = collect_telemetry(sample_state, None, None)
        assert isinstance(result, TelemetrySnapshot)
        assert result.health_status == "healthy"  # Default
        assert result.alert_count == 0

    def test_collect_telemetry_with_alerts(
        self,
        sample_state: dict[str, Any],
        sample_velocity_metrics: VelocityMetrics,
    ) -> None:
        """Test telemetry collection with alerts."""
        alert = HealthAlert(
            severity="warning",
            alert_type="idle_time_warning",
            message="Agent idle",
            affected_agent="analyst",
            metric_value=400.0,
            threshold_value=300.0,
        )
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 400.0},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=5.0,
            agent_snapshots=(),
        )
        health = HealthStatus(
            status="warning",
            metrics=metrics,
            alerts=(alert,),
            summary="Agent idle warning",
            is_healthy=True,
        )
        result = collect_telemetry(sample_state, health, sample_velocity_metrics)
        assert result.alert_count == 1
        assert result.health_status == "warning"

    def test_collect_telemetry_to_dict_json_serializable(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
        sample_velocity_metrics: VelocityMetrics,
    ) -> None:
        """Test that collected telemetry is JSON serializable."""
        result = collect_telemetry(sample_state, sample_health_status, sample_velocity_metrics)
        # Should not raise
        json_str = json.dumps(result.to_dict())
        assert "burn_down_velocity" in json_str


# =============================================================================
# Format for Dashboard Tests
# =============================================================================


class TestFormatForDashboard:
    """Tests for format_for_dashboard function."""

    @pytest.fixture
    def sample_snapshot(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
        sample_velocity_metrics: VelocityMetrics,
    ) -> TelemetrySnapshot:
        """Create a sample TelemetrySnapshot for formatting tests."""
        return collect_telemetry(sample_state, sample_health_status, sample_velocity_metrics)

    def test_format_for_dashboard_produces_dashboard_metrics(
        self, sample_snapshot: TelemetrySnapshot
    ) -> None:
        """Test that format_for_dashboard returns DashboardMetrics."""
        result = format_for_dashboard(sample_snapshot)
        assert isinstance(result, DashboardMetrics)
        assert result.snapshot == sample_snapshot

    def test_format_velocity_display(self, sample_snapshot: TelemetrySnapshot) -> None:
        """Test velocity display formatting."""
        result = format_for_dashboard(sample_snapshot)
        assert "5.2" in result.velocity_display
        assert "stories/sprint" in result.velocity_display

    def test_format_cycle_time_display(self, sample_snapshot: TelemetrySnapshot) -> None:
        """Test cycle time display formatting."""
        result = format_for_dashboard(sample_snapshot)
        assert "avg" in result.cycle_time_display or "min" in result.cycle_time_display or "h" in result.cycle_time_display

    def test_format_health_summary(self, sample_snapshot: TelemetrySnapshot) -> None:
        """Test health summary formatting."""
        result = format_for_dashboard(sample_snapshot)
        assert "healthy" in result.health_summary.lower()
        assert "0 alerts" in result.health_summary or "no alerts" in result.health_summary

    def test_format_agent_status_table(self, sample_snapshot: TelemetrySnapshot) -> None:
        """Test agent status table formatting."""
        result = format_for_dashboard(sample_snapshot)
        assert len(result.agent_status_table) == 3  # analyst, pm, dev
        # Check sorted order
        agents = [row["agent"] for row in result.agent_status_table]
        assert agents == sorted(agents)

    def test_format_for_dashboard_to_dict_json_serializable(
        self, sample_snapshot: TelemetrySnapshot
    ) -> None:
        """Test that formatted dashboard metrics are JSON serializable."""
        result = format_for_dashboard(sample_snapshot)
        # Should not raise
        json_str = json.dumps(result.to_dict())
        assert "velocity_display" in json_str


class TestFormatHelpers:
    """Tests for format helper functions."""

    def test_format_velocity_display_no_data(self) -> None:
        """Test velocity display with no data."""
        summary = MetricSummary(
            name="velocity",
            value=0.0,
            unit="stories/sprint",
            display_value="No data",
            trend=None,
            status="healthy",
        )
        result = _format_velocity_display(summary)
        assert "No velocity data" in result

    def test_format_cycle_time_display_no_data(self) -> None:
        """Test cycle time display with no data."""
        summary = MetricSummary(
            name="cycle_time",
            value=0.0,
            unit="ms",
            display_value="No data",
            trend=None,
            status="healthy",
        )
        result = _format_cycle_time_display(summary)
        assert "No cycle time data" in result

    def test_format_health_summary_with_alerts(self) -> None:
        """Test health summary with multiple alerts."""
        summary = MetricSummary(
            name="test",
            value=0.0,
            unit="",
            display_value="",
            trend=None,
            status="healthy",
        )
        snapshot = TelemetrySnapshot(
            burn_down_velocity=summary,
            cycle_time=summary,
            churn_rate=summary,
            agent_idle_times={},
            health_status="warning",
            alert_count=3,
        )
        result = _format_health_summary(snapshot)
        assert "3 alerts" in result
        assert "warning" in result.lower()

    def test_format_agent_status_table_empty(self) -> None:
        """Test agent status table with empty data."""
        result = _format_agent_status_table({})
        assert result == ()


# =============================================================================
# Get Dashboard Telemetry Tests
# =============================================================================


class TestGetDashboardTelemetry:
    """Tests for get_dashboard_telemetry async function."""

    @pytest.mark.asyncio
    async def test_get_dashboard_telemetry_full_data(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
        sample_velocity_metrics: VelocityMetrics,
    ) -> None:
        """Test get_dashboard_telemetry with full data."""
        result = await get_dashboard_telemetry(
            sample_state,
            health_status=sample_health_status,
            velocity_metrics=sample_velocity_metrics,
        )
        assert isinstance(result, DashboardMetrics)
        assert "5.2" in result.velocity_display

    @pytest.mark.asyncio
    async def test_get_dashboard_telemetry_default_config(
        self,
        sample_state: dict[str, Any],
    ) -> None:
        """Test get_dashboard_telemetry with default config."""
        result = await get_dashboard_telemetry(sample_state)
        assert isinstance(result, DashboardMetrics)

    @pytest.mark.asyncio
    async def test_get_dashboard_telemetry_custom_config(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
        sample_velocity_metrics: VelocityMetrics,
    ) -> None:
        """Test get_dashboard_telemetry with custom config."""
        config = TelemetryConfig(include_agent_details=False)
        result = await get_dashboard_telemetry(
            sample_state,
            config=config,
            health_status=sample_health_status,
            velocity_metrics=sample_velocity_metrics,
        )
        assert isinstance(result, DashboardMetrics)
        assert len(result.agent_status_table) == 0  # Agent details disabled

    @pytest.mark.asyncio
    async def test_get_dashboard_telemetry_no_data(
        self,
        sample_state: dict[str, Any],
    ) -> None:
        """Test get_dashboard_telemetry with no data."""
        result = await get_dashboard_telemetry(sample_state)
        assert isinstance(result, DashboardMetrics)
        assert result.velocity_display == "No velocity data"

    @pytest.mark.asyncio
    async def test_get_dashboard_telemetry_error_handling(
        self,
        sample_state: dict[str, Any],
    ) -> None:
        """Test get_dashboard_telemetry handles errors gracefully."""
        # Create a mock health status that will raise an exception when accessed
        bad_health = MagicMock()
        # Make the metrics attribute raise when accessed
        type(bad_health).metrics = property(lambda self: (_ for _ in ()).throw(Exception("Test error")))

        # Should not raise, should return default metrics
        result = await get_dashboard_telemetry(sample_state, health_status=bad_health)
        assert isinstance(result, DashboardMetrics)
        # Error case returns "Telemetry collection failed" and "Data unavailable"
        assert "failed" in result.health_summary.lower() or "unavailable" in result.velocity_display.lower()


# =============================================================================
# Logging Tests
# =============================================================================


class TestTelemetryLogging:
    """Tests for telemetry logging output.

    Note: structlog integrates with Python's stdlib logging, which pytest's caplog
    can capture. We configure structlog with stdlib processors to enable log capture.
    """

    @pytest.mark.asyncio
    async def test_get_dashboard_telemetry_logs_completion(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
        sample_velocity_metrics: VelocityMetrics,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that telemetry collection logs completion events."""
        import logging

        # Configure structlog to output through stdlib logging for test capture
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        # Capture INFO level logs from the telemetry module
        with caplog.at_level(logging.INFO, logger="yolo_developer.agents.sm.telemetry"):
            await get_dashboard_telemetry(
                sample_state,
                health_status=sample_health_status,
                velocity_metrics=sample_velocity_metrics,
            )

        # Verify that telemetry completion was logged
        log_messages = [record.message for record in caplog.records]

        # Check for key log events from get_dashboard_telemetry
        assert any(
            "dashboard_telemetry" in msg.lower() or "telemetry" in msg.lower()
            for msg in log_messages
        ), f"Expected telemetry log events, got: {log_messages}"

    @pytest.mark.asyncio
    async def test_collect_telemetry_logs_aggregation(
        self,
        sample_state: dict[str, Any],
        sample_health_status: HealthStatus,
        sample_velocity_metrics: VelocityMetrics,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that collect_telemetry logs aggregation info."""
        import logging

        # Import collect_telemetry directly
        from yolo_developer.agents.sm.telemetry import collect_telemetry

        # Configure structlog for test capture
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        with caplog.at_level(logging.DEBUG, logger="yolo_developer.agents.sm.telemetry"):
            collect_telemetry(
                sample_state,
                sample_health_status,
                sample_velocity_metrics,
            )

        # Verify telemetry collection logged
        log_text = " ".join(record.message for record in caplog.records)
        assert "telemetry" in log_text.lower(), f"Expected telemetry logs, got: {log_text}"
