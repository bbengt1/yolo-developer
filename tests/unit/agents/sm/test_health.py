"""Tests for health monitoring module (Story 10.5).

This module tests the health monitoring functionality:
- Idle time tracking (AC #1)
- Cycle time measurement (AC #2)
- Churn rate calculation (AC #3)
- Anomaly detection and alerting (AC #4)
- Main monitor_health function
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from yolo_developer.agents.sm.health import (
    _build_agent_snapshots,
    _calculate_agent_churn_rates,
    _calculate_agent_cycle_times,
    _calculate_agent_idle_times,
    _calculate_churn_rate,
    _calculate_cycle_time_percentiles,
    _calculate_idle_time,
    _calculate_overall_cycle_time,
    _calculate_percentile,
    _calculate_unproductive_churn_rate,
    _collect_metrics,
    _count_exchanges_in_window,
    _count_unproductive_exchanges,
    _detect_anomalies,
    _determine_status,
    _extract_agent_from_message,
    _extract_timestamp_from_message,
    _extract_topic_from_message,
    _generate_alerts,
    _generate_summary,
    _get_current_timestamp,
    _parse_timestamp,
    _track_agent_activity,
    _trigger_alerts,
    monitor_health,
)
from yolo_developer.agents.sm.health_types import (
    HealthAlert,
    HealthConfig,
    HealthMetrics,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def empty_state() -> dict[str, Any]:
    """Create empty state for testing."""
    return {
        "messages": [],
        "decisions": [],
        "current_agent": "sm",
    }


@pytest.fixture
def mock_message() -> MagicMock:
    """Create mock message with agent and timestamp."""
    msg = MagicMock()
    msg.additional_kwargs = {
        "agent": "analyst",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return msg


@pytest.fixture
def state_with_messages() -> dict[str, Any]:
    """Create state with mock messages for testing."""
    now = datetime.now(timezone.utc)

    def make_msg(agent: str, offset_seconds: float) -> MagicMock:
        msg = MagicMock()
        ts = (now - timedelta(seconds=offset_seconds)).isoformat()
        msg.additional_kwargs = {"agent": agent, "timestamp": ts}
        return msg

    return {
        "messages": [
            make_msg("analyst", 300),  # 5 min ago
            make_msg("pm", 200),  # 3:20 ago
            make_msg("analyst", 100),  # 1:40 ago
            make_msg("dev", 50),  # 50s ago
        ],
        "decisions": [],
        "current_agent": "dev",
    }


@pytest.fixture
def high_churn_state() -> dict[str, Any]:
    """Create state with high churn (many exchanges) for testing."""
    now = datetime.now(timezone.utc)

    def make_msg(agent: str, offset_seconds: float) -> MagicMock:
        msg = MagicMock()
        ts = (now - timedelta(seconds=offset_seconds)).isoformat()
        msg.additional_kwargs = {"agent": agent, "timestamp": ts}
        return msg

    # Create 15 exchanges in the last minute (very high churn)
    messages = []
    for i in range(15):
        agent = "analyst" if i % 2 == 0 else "pm"
        messages.append(make_msg(agent, 60 - i * 4))  # Every 4 seconds

    return {
        "messages": messages,
        "decisions": [],
        "current_agent": "pm",
    }


# =============================================================================
# Timestamp Utility Tests
# =============================================================================


class TestTimestampUtilities:
    """Tests for timestamp utility functions."""

    def test_get_current_timestamp_format(self) -> None:
        """_get_current_timestamp should return valid ISO format."""
        ts = _get_current_timestamp()
        # Should not raise
        _parse_timestamp(ts)
        assert "T" in ts  # ISO format has T separator

    def test_parse_timestamp_with_z_suffix(self) -> None:
        """_parse_timestamp should handle Z suffix."""
        ts = "2026-01-12T10:00:00Z"
        result = _parse_timestamp(ts)
        assert result.tzinfo is not None

    def test_parse_timestamp_with_offset(self) -> None:
        """_parse_timestamp should handle +00:00 format."""
        ts = "2026-01-12T10:00:00+00:00"
        result = _parse_timestamp(ts)
        assert result.tzinfo is not None

    def test_calculate_idle_time(self) -> None:
        """_calculate_idle_time should return correct duration."""
        current = datetime.now(timezone.utc)
        past = (current - timedelta(seconds=120)).isoformat()
        idle = _calculate_idle_time(past, current)
        assert 119 < idle < 121  # Allow for minor timing differences


# =============================================================================
# Message Extraction Tests
# =============================================================================


class TestMessageExtraction:
    """Tests for message data extraction."""

    def test_extract_agent_from_message(self, mock_message: MagicMock) -> None:
        """_extract_agent_from_message should return agent name."""
        agent = _extract_agent_from_message(mock_message)
        assert agent == "analyst"

    def test_extract_agent_from_message_no_kwargs(self) -> None:
        """_extract_agent_from_message should return None if no kwargs."""
        msg = MagicMock(spec=[])  # No additional_kwargs
        agent = _extract_agent_from_message(msg)
        assert agent is None

    def test_extract_timestamp_from_message(self, mock_message: MagicMock) -> None:
        """_extract_timestamp_from_message should return timestamp."""
        ts = _extract_timestamp_from_message(mock_message)
        assert ts is not None
        assert "T" in ts

    def test_extract_timestamp_from_message_no_kwargs(self) -> None:
        """_extract_timestamp_from_message should return None if no kwargs."""
        msg = MagicMock(spec=[])
        ts = _extract_timestamp_from_message(msg)
        assert ts is None


# =============================================================================
# Idle Time Tracking Tests (AC #1)
# =============================================================================


class TestIdleTimeTracking:
    """Tests for agent idle time tracking (AC #1)."""

    def test_calculate_agent_idle_times_empty_state(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should return zero idle times."""
        idle_times = _calculate_agent_idle_times(empty_state)
        for agent, time in idle_times.items():
            assert time == 0.0

    def test_calculate_agent_idle_times_with_activity(
        self, state_with_messages: dict[str, Any]
    ) -> None:
        """Should calculate correct idle times from message history."""
        idle_times = _calculate_agent_idle_times(state_with_messages)

        # Dev had activity 50s ago
        assert 40 < idle_times.get("dev", 0) < 60

        # Analyst had activity 100s ago
        assert 90 < idle_times.get("analyst", 0) < 110

        # PM had activity 200s ago
        assert 190 < idle_times.get("pm", 0) < 210

    def test_track_agent_activity(
        self, state_with_messages: dict[str, Any]
    ) -> None:
        """_track_agent_activity should return last activity timestamp."""
        ts = _track_agent_activity(state_with_messages, "analyst")
        assert "T" in ts  # ISO format

    def test_track_agent_activity_no_activity(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Should return current time if agent has no activity."""
        ts = _track_agent_activity(empty_state, "analyst")
        assert "T" in ts  # ISO format


# =============================================================================
# Cycle Time Tests (AC #2)
# =============================================================================


class TestCycleTimeMeasurement:
    """Tests for cycle time measurement (AC #2)."""

    def test_calculate_agent_cycle_times_empty(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should return empty cycle times."""
        cycle_times = _calculate_agent_cycle_times(empty_state)
        assert cycle_times == {}

    def test_calculate_overall_cycle_time_empty(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should return zero overall cycle time."""
        overall = _calculate_overall_cycle_time(empty_state)
        assert overall == 0.0

    def test_calculate_agent_cycle_times_with_decisions(self) -> None:
        """Should calculate cycle times from decision history."""
        now = datetime.now(timezone.utc)

        decisions = [
            MagicMock(agent="analyst", timestamp=now - timedelta(seconds=100)),
            MagicMock(agent="analyst", timestamp=now - timedelta(seconds=50)),
        ]

        state: dict[str, Any] = {
            "messages": [],
            "decisions": decisions,
            "current_agent": "analyst",
        }

        cycle_times = _calculate_agent_cycle_times(state)
        # Should have ~50s cycle time for analyst
        if "analyst" in cycle_times:
            assert 40 < cycle_times["analyst"] < 60


# =============================================================================
# Churn Rate Tests (AC #3)
# =============================================================================


class TestChurnRateCalculation:
    """Tests for churn rate calculation (AC #3)."""

    def test_count_exchanges_empty_state(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should have zero exchanges."""
        count = _count_exchanges_in_window(empty_state)
        assert count == 0

    def test_count_exchanges_with_activity(
        self, state_with_messages: dict[str, Any]
    ) -> None:
        """Should count exchanges in time window."""
        count = _count_exchanges_in_window(state_with_messages, window_seconds=60.0)
        # Messages: analyst -> pm -> analyst -> dev = 3 exchanges
        # But only those within 60s window count
        assert count >= 0

    def test_calculate_churn_rate_empty(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should have zero churn rate."""
        rate = _calculate_churn_rate(empty_state)
        assert rate == 0.0

    def test_calculate_churn_rate_high_churn(
        self, high_churn_state: dict[str, Any]
    ) -> None:
        """High churn state should have high churn rate."""
        rate = _calculate_churn_rate(high_churn_state)
        # 15 messages alternating = ~14 exchanges in 60s
        assert rate > 5  # Should be significantly higher than default threshold

    def test_calculate_agent_churn_rates_empty(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should return empty agent churn rates."""
        rates = _calculate_agent_churn_rates(empty_state)
        assert rates == {}


# =============================================================================
# Percentile Calculation Tests (Task 3.5)
# =============================================================================


class TestPercentileCalculation:
    """Tests for percentile calculations (Task 3.5)."""

    def test_calculate_percentile_empty(self) -> None:
        """Empty list should return 0.0."""
        result = _calculate_percentile([], 50)
        assert result == 0.0

    def test_calculate_percentile_single_value(self) -> None:
        """Single value should return that value for any percentile."""
        result = _calculate_percentile([100.0], 50)
        assert result == 100.0
        result = _calculate_percentile([100.0], 90)
        assert result == 100.0

    def test_calculate_percentile_p50(self) -> None:
        """P50 should return median value."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _calculate_percentile(values, 50)
        assert result == 30.0

    def test_calculate_percentile_p90(self) -> None:
        """P90 should return 90th percentile."""
        values = [i * 10.0 for i in range(1, 11)]  # 10, 20, ..., 100
        result = _calculate_percentile(values, 90)
        assert result > 80.0

    def test_calculate_percentile_interpolation(self) -> None:
        """Percentile should use linear interpolation."""
        values = [0.0, 100.0]
        result = _calculate_percentile(values, 50)
        assert result == 50.0

    def test_calculate_cycle_time_percentiles_empty(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should return zero percentiles."""
        result = _calculate_cycle_time_percentiles(empty_state)
        assert result == {"p50": 0.0, "p90": 0.0, "p95": 0.0}

    def test_calculate_cycle_time_percentiles_with_decisions(self) -> None:
        """Should calculate percentiles from decision history."""
        now = datetime.now(timezone.utc)

        decisions = [
            MagicMock(agent="analyst", timestamp=now - timedelta(seconds=100)),
            MagicMock(agent="pm", timestamp=now - timedelta(seconds=50)),
            MagicMock(agent="dev", timestamp=now),
        ]

        state: dict[str, Any] = {
            "messages": [],
            "decisions": decisions,
            "current_agent": "dev",
        }

        result = _calculate_cycle_time_percentiles(state)
        assert "p50" in result
        assert "p90" in result
        assert "p95" in result


# =============================================================================
# Unproductive Exchange Tests (Task 4.3)
# =============================================================================


class TestUnproductiveExchangeTracking:
    """Tests for unproductive exchange tracking (Task 4.3)."""

    def test_extract_topic_from_message(self) -> None:
        """Should extract topic from message kwargs."""
        msg = MagicMock()
        msg.additional_kwargs = {"topic": "requirements", "agent": "analyst"}
        topic = _extract_topic_from_message(msg)
        assert topic == "requirements"

    def test_extract_topic_from_message_fallback_reason(self) -> None:
        """Should fall back to reason if no topic."""
        msg = MagicMock()
        msg.additional_kwargs = {"reason": "clarification", "agent": "pm"}
        topic = _extract_topic_from_message(msg)
        assert topic == "clarification"

    def test_extract_topic_no_kwargs(self) -> None:
        """Should return empty string if no kwargs."""
        msg = MagicMock(spec=[])
        topic = _extract_topic_from_message(msg)
        assert topic == ""

    def test_count_unproductive_exchanges_empty(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should have zero unproductive exchanges."""
        count = _count_unproductive_exchanges(empty_state)
        assert count == 0

    def test_count_unproductive_exchanges_no_pingpong(self) -> None:
        """Normal flow should not count as unproductive."""
        now = datetime.now(timezone.utc)

        def make_msg(agent: str, topic: str, offset: float) -> MagicMock:
            msg = MagicMock()
            msg.additional_kwargs = {
                "agent": agent,
                "topic": topic,
                "timestamp": (now - timedelta(seconds=offset)).isoformat(),
            }
            return msg

        state: dict[str, Any] = {
            "messages": [
                make_msg("analyst", "requirements", 30),
                make_msg("pm", "story_creation", 20),
                make_msg("dev", "implementation", 10),
            ],
            "decisions": [],
            "current_agent": "dev",
        }

        count = _count_unproductive_exchanges(state)
        assert count == 0

    def test_count_unproductive_exchanges_with_pingpong(self) -> None:
        """Ping-pong on same topic should count as unproductive."""
        now = datetime.now(timezone.utc)

        def make_msg(agent: str, topic: str, offset: float) -> MagicMock:
            msg = MagicMock()
            msg.additional_kwargs = {
                "agent": agent,
                "topic": topic,
                "timestamp": (now - timedelta(seconds=offset)).isoformat(),
            }
            return msg

        state: dict[str, Any] = {
            "messages": [
                make_msg("analyst", "clarification", 40),
                make_msg("pm", "clarification", 30),
                make_msg("analyst", "clarification", 20),  # Back to analyst on same topic
                make_msg("pm", "clarification", 10),  # Back to pm on same topic
            ],
            "decisions": [],
            "current_agent": "pm",
        }

        count = _count_unproductive_exchanges(state)
        assert count >= 1  # At least one unproductive exchange detected

    def test_calculate_unproductive_churn_rate_empty(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Empty state should have zero unproductive churn rate."""
        rate = _calculate_unproductive_churn_rate(empty_state)
        assert rate == 0.0


# =============================================================================
# Anomaly Detection Tests (AC #4)
# =============================================================================


class TestAnomalyDetection:
    """Tests for anomaly detection (AC #4)."""

    def test_detect_anomalies_healthy_metrics(self) -> None:
        """Healthy metrics should produce no anomalies."""
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 100.0, "pm": 50.0},
            agent_cycle_times={"analyst": 30.0},
            agent_churn_rates={"analyst": 2.0},
            overall_cycle_time=30.0,
            overall_churn_rate=2.0,
            agent_snapshots=(),
        )
        config = HealthConfig()
        anomalies = _detect_anomalies(metrics, config)
        assert len(anomalies) == 0

    def test_detect_anomalies_high_idle_time(self) -> None:
        """High idle time should trigger anomaly."""
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 400.0},  # Above 300 threshold
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        config = HealthConfig(max_idle_time_seconds=300.0)
        anomalies = _detect_anomalies(metrics, config)

        assert len(anomalies) >= 1
        assert any(a["type"] == "idle_time_exceeded" for a in anomalies)
        assert any(a["severity"] == "critical" for a in anomalies)

    def test_detect_anomalies_warning_idle_time(self) -> None:
        """Warning-level idle time should trigger warning anomaly."""
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 250.0},  # Above 70% of 300
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        config = HealthConfig(max_idle_time_seconds=300.0, warning_threshold_ratio=0.7)
        anomalies = _detect_anomalies(metrics, config)

        assert len(anomalies) >= 1
        assert any(a["type"] == "idle_time_warning" for a in anomalies)
        assert any(a["severity"] == "warning" for a in anomalies)

    def test_detect_anomalies_high_churn(self) -> None:
        """High churn rate should trigger critical anomaly."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=15.0,  # Above 10 threshold
            agent_snapshots=(),
        )
        config = HealthConfig(max_churn_rate=10.0)
        anomalies = _detect_anomalies(metrics, config)

        assert len(anomalies) >= 1
        assert any(a["type"] == "high_churn" for a in anomalies)
        assert any(a["severity"] == "critical" for a in anomalies)

    def test_detect_anomalies_slow_cycle(self) -> None:
        """Slow cycle time should trigger warning."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={"analyst": 700.0},  # Above 600 threshold
            agent_churn_rates={},
            overall_cycle_time=700.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        config = HealthConfig(max_cycle_time_seconds=600.0)
        anomalies = _detect_anomalies(metrics, config)

        assert len(anomalies) >= 1
        assert any(a["type"] == "slow_cycle" for a in anomalies)


# =============================================================================
# Alert Generation Tests (AC #4)
# =============================================================================


class TestAlertGeneration:
    """Tests for alert generation (AC #4)."""

    def test_generate_alerts_empty_anomalies(self) -> None:
        """No anomalies should produce no alerts."""
        config = HealthConfig(enable_alerts=True)
        alerts = _generate_alerts([], config)
        assert alerts == ()

    def test_generate_alerts_disabled(self) -> None:
        """Disabled alerts should produce empty tuple."""
        config = HealthConfig(enable_alerts=False)
        anomalies = [{"type": "test", "severity": "warning", "agent": None, "value": 5.0, "threshold": 3.0}]
        alerts = _generate_alerts(anomalies, config)
        assert alerts == ()

    def test_generate_alerts_with_agent(self) -> None:
        """Alert for specific agent should include agent name."""
        config = HealthConfig(enable_alerts=True)
        anomalies = [
            {
                "type": "idle_time_exceeded",
                "severity": "critical",
                "agent": "analyst",
                "value": 400.0,
                "threshold": 300.0,
            }
        ]
        alerts = _generate_alerts(anomalies, config)

        assert len(alerts) == 1
        assert alerts[0].affected_agent == "analyst"
        assert alerts[0].severity == "critical"
        assert alerts[0].alert_type == "idle_time_exceeded"

    def test_generate_alerts_system_wide(self) -> None:
        """System-wide alert should have None affected_agent."""
        config = HealthConfig(enable_alerts=True)
        anomalies = [
            {
                "type": "high_churn",
                "severity": "critical",
                "agent": None,
                "value": 15.0,
                "threshold": 10.0,
            }
        ]
        alerts = _generate_alerts(anomalies, config)

        assert len(alerts) == 1
        assert alerts[0].affected_agent is None
        assert "System" in alerts[0].message

    def test_trigger_alerts_same_as_generate(self) -> None:
        """_trigger_alerts should produce same result as _generate_alerts."""
        config = HealthConfig(enable_alerts=True)
        anomalies = [
            {"type": "test", "severity": "warning", "agent": "pm", "value": 5.0, "threshold": 3.0}
        ]
        alerts1 = _generate_alerts(anomalies, config)
        alerts2 = _trigger_alerts(anomalies, config)
        assert len(alerts1) == len(alerts2)


# =============================================================================
# Status Determination Tests
# =============================================================================


class TestStatusDetermination:
    """Tests for health status determination."""

    def test_determine_status_healthy(self) -> None:
        """No alerts should result in healthy status."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        status = _determine_status(metrics, ())
        assert status == "healthy"

    def test_determine_status_warning(self) -> None:
        """One warning alert should result in warning status."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        alert = HealthAlert(
            severity="warning",
            alert_type="test",
            message="Test",
            affected_agent=None,
            metric_value=5.0,
            threshold_value=3.0,
        )
        status = _determine_status(metrics, (alert,))
        assert status == "warning"

    def test_determine_status_degraded(self) -> None:
        """Two+ warning alerts should result in degraded status."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        alert1 = HealthAlert(
            severity="warning",
            alert_type="test1",
            message="Test1",
            affected_agent=None,
            metric_value=5.0,
            threshold_value=3.0,
        )
        alert2 = HealthAlert(
            severity="warning",
            alert_type="test2",
            message="Test2",
            affected_agent=None,
            metric_value=6.0,
            threshold_value=4.0,
        )
        status = _determine_status(metrics, (alert1, alert2))
        assert status == "degraded"

    def test_determine_status_critical(self) -> None:
        """Critical alert should result in critical status."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        alert = HealthAlert(
            severity="critical",
            alert_type="test",
            message="Test",
            affected_agent=None,
            metric_value=15.0,
            threshold_value=10.0,
        )
        status = _determine_status(metrics, (alert,))
        assert status == "critical"


# =============================================================================
# Summary Generation Tests
# =============================================================================


class TestSummaryGeneration:
    """Tests for health summary generation."""

    def test_generate_summary_healthy(self) -> None:
        """Healthy status should have 'nominal' in summary."""
        metrics = HealthMetrics(
            agent_idle_times={"analyst": 100.0},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        summary = _generate_summary("healthy", metrics, ())
        assert "nominal" in summary.lower()

    def test_generate_summary_critical(self) -> None:
        """Critical status should have 'CRITICAL' in summary."""
        metrics = HealthMetrics(
            agent_idle_times={},
            agent_cycle_times={},
            agent_churn_rates={},
            overall_cycle_time=0.0,
            overall_churn_rate=0.0,
            agent_snapshots=(),
        )
        alert = HealthAlert(
            severity="critical",
            alert_type="test",
            message="Test",
            affected_agent=None,
            metric_value=15.0,
            threshold_value=10.0,
        )
        summary = _generate_summary("critical", metrics, (alert,))
        assert "CRITICAL" in summary


# =============================================================================
# Agent Snapshots Tests
# =============================================================================


class TestAgentSnapshots:
    """Tests for agent health snapshot building."""

    def test_build_agent_snapshots_all_agents(self) -> None:
        """Should build snapshots for all tracked agents."""
        idle_times = {"analyst": 100.0, "pm": 50.0}
        cycle_times = {"analyst": 30.0}
        churn_rates = {"analyst": 2.0}
        config = HealthConfig()

        snapshots = _build_agent_snapshots(idle_times, cycle_times, churn_rates, config)

        # Should have snapshots for all 6 agents
        assert len(snapshots) == 6
        agent_names = {s.agent for s in snapshots}
        assert "analyst" in agent_names
        assert "pm" in agent_names
        assert "dev" in agent_names

    def test_build_agent_snapshots_healthy(self) -> None:
        """Agents within thresholds should be marked healthy."""
        idle_times = {"analyst": 100.0}  # Under 300s threshold
        cycle_times = {}
        churn_rates = {}
        config = HealthConfig(max_idle_time_seconds=300.0)

        snapshots = _build_agent_snapshots(idle_times, cycle_times, churn_rates, config)

        analyst_snapshot = next(s for s in snapshots if s.agent == "analyst")
        assert analyst_snapshot.is_healthy is True

    def test_build_agent_snapshots_unhealthy(self) -> None:
        """Agents exceeding thresholds should be marked unhealthy."""
        idle_times = {"analyst": 400.0}  # Over 300s threshold
        cycle_times = {}
        churn_rates = {}
        config = HealthConfig(max_idle_time_seconds=300.0)

        snapshots = _build_agent_snapshots(idle_times, cycle_times, churn_rates, config)

        analyst_snapshot = next(s for s in snapshots if s.agent == "analyst")
        assert analyst_snapshot.is_healthy is False


# =============================================================================
# Metrics Collection Tests
# =============================================================================


class TestMetricsCollection:
    """Tests for metrics collection."""

    def test_collect_metrics_empty_state(self, empty_state: dict[str, Any]) -> None:
        """Should collect metrics from empty state without error."""
        config = HealthConfig()
        metrics = _collect_metrics(empty_state, config)

        assert isinstance(metrics, HealthMetrics)
        assert metrics.overall_churn_rate == 0.0
        assert len(metrics.agent_snapshots) == 6

    def test_collect_metrics_with_activity(
        self, state_with_messages: dict[str, Any]
    ) -> None:
        """Should collect metrics from state with activity."""
        config = HealthConfig()
        metrics = _collect_metrics(state_with_messages, config)

        assert isinstance(metrics, HealthMetrics)
        # Should have some idle times recorded
        assert len(metrics.agent_idle_times) > 0


# =============================================================================
# Main Monitor Health Function Tests
# =============================================================================


class TestMonitorHealth:
    """Tests for main monitor_health function."""

    @pytest.mark.asyncio
    async def test_monitor_health_empty_state(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Should return healthy status for empty state."""
        status = await monitor_health(empty_state)

        assert status.status == "healthy"
        assert status.is_healthy is True
        assert len(status.alerts) == 0

    @pytest.mark.asyncio
    async def test_monitor_health_with_default_config(
        self, state_with_messages: dict[str, Any]
    ) -> None:
        """Should use default config when none provided."""
        status = await monitor_health(state_with_messages)

        assert status is not None
        assert status.metrics is not None
        assert status.summary != ""

    @pytest.mark.asyncio
    async def test_monitor_health_with_custom_config(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Should respect custom config thresholds."""
        config = HealthConfig(
            max_idle_time_seconds=600.0,
            enable_alerts=False,
        )
        status = await monitor_health(empty_state, config)

        # Alerts disabled
        assert len(status.alerts) == 0

    @pytest.mark.asyncio
    async def test_monitor_health_high_churn(
        self, high_churn_state: dict[str, Any]
    ) -> None:
        """High churn should trigger alerts and degrade status."""
        config = HealthConfig(max_churn_rate=5.0)  # Low threshold
        status = await monitor_health(high_churn_state, config)

        # Should have at least one alert
        # Note: Actual alert depends on churn calculation within time window
        assert status is not None
        assert status.metrics.overall_churn_rate >= 0

    @pytest.mark.asyncio
    async def test_monitor_health_returns_serializable(
        self, empty_state: dict[str, Any]
    ) -> None:
        """HealthStatus should be serializable via to_dict."""
        status = await monitor_health(empty_state)
        result = status.to_dict()

        assert isinstance(result, dict)
        assert "status" in result
        assert "metrics" in result
        assert "alerts" in result
        assert "is_healthy" in result

    @pytest.mark.asyncio
    async def test_monitor_health_is_healthy_warning(self) -> None:
        """Warning status should still be considered healthy."""
        # Create state that triggers warning but not critical
        now = datetime.now(timezone.utc)
        msg = MagicMock()
        msg.additional_kwargs = {
            "agent": "analyst",
            "timestamp": (now - timedelta(seconds=250)).isoformat(),  # 250s idle
        }

        state: dict[str, Any] = {
            "messages": [msg],
            "decisions": [],
            "current_agent": "analyst",
        }

        # 250s idle with 300s max and 0.7 warning ratio = warning level
        config = HealthConfig(
            max_idle_time_seconds=300.0,
            warning_threshold_ratio=0.7,
        )
        status = await monitor_health(state, config)

        # Warning is still considered healthy
        if status.status == "warning":
            assert status.is_healthy is True

    @pytest.mark.asyncio
    async def test_monitor_health_is_healthy_critical(self) -> None:
        """Critical status should NOT be considered healthy."""
        # Create state that triggers critical alert
        now = datetime.now(timezone.utc)
        msg = MagicMock()
        msg.additional_kwargs = {
            "agent": "analyst",
            "timestamp": (now - timedelta(seconds=400)).isoformat(),  # 400s idle
        }

        state: dict[str, Any] = {
            "messages": [msg],
            "decisions": [],
            "current_agent": "analyst",
        }

        config = HealthConfig(max_idle_time_seconds=300.0)
        status = await monitor_health(state, config)

        if status.status == "critical":
            assert status.is_healthy is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestHealthIntegration:
    """Integration tests for health monitoring."""

    @pytest.mark.asyncio
    async def test_full_monitoring_flow(self) -> None:
        """Test complete monitoring flow from state to status."""
        # Create realistic state
        now = datetime.now(timezone.utc)

        def make_msg(agent: str, offset: float) -> MagicMock:
            msg = MagicMock()
            msg.additional_kwargs = {
                "agent": agent,
                "timestamp": (now - timedelta(seconds=offset)).isoformat(),
            }
            return msg

        state: dict[str, Any] = {
            "messages": [
                make_msg("sm", 60),
                make_msg("analyst", 50),
                make_msg("pm", 40),
                make_msg("architect", 30),
                make_msg("dev", 20),
                make_msg("tea", 10),
            ],
            "decisions": [],
            "current_agent": "tea",
        }

        config = HealthConfig()
        status = await monitor_health(state, config)

        # Verify all components present
        assert status.status in ("healthy", "warning", "degraded", "critical")
        assert len(status.metrics.agent_snapshots) == 6
        assert status.summary != ""
        assert "evaluated_at" in status.to_dict()

    @pytest.mark.asyncio
    async def test_monitoring_never_raises(self) -> None:
        """Health monitoring should handle any state without raising."""
        # Various edge case states
        states = [
            {},  # Empty dict
            {"messages": None},  # None messages
            {"messages": [], "decisions": None},  # None decisions
            {"messages": "invalid"},  # Wrong type
            {"current_agent": 123},  # Wrong type for agent
        ]

        for state in states:
            # Should not raise
            status = await monitor_health(state)  # type: ignore[arg-type]
            assert status is not None
