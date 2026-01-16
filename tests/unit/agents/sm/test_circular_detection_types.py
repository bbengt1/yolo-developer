"""Tests for circular detection types (Story 10.6).

Tests the type definitions used by the circular logic detection system:
- CircularPattern: Detected circular pattern in agent exchanges
- CycleLog: Audit log entry for a detected cycle
- CircularLogicConfig: Configuration for circular logic detection
- CycleAnalysis: Complete analysis result from circular logic detection

All types are frozen dataclasses per ADR-001.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


class TestCycleSeverityLiteral:
    """Tests for CycleSeverity literal type."""

    def test_valid_severity_values(self) -> None:
        """Test that valid severity values are accepted."""
        from yolo_developer.agents.sm.circular_detection_types import CycleSeverity

        # These are the valid severity values
        valid_severities: list[CycleSeverity] = ["low", "medium", "high", "critical"]
        assert len(valid_severities) == 4

    def test_severity_type_annotation(self) -> None:
        """Test CycleSeverity is a Literal type."""
        from typing import get_args

        from yolo_developer.agents.sm.circular_detection_types import CycleSeverity

        args = get_args(CycleSeverity)
        assert "low" in args
        assert "medium" in args
        assert "high" in args
        assert "critical" in args


class TestInterventionStrategyLiteral:
    """Tests for InterventionStrategy literal type."""

    def test_valid_strategy_values(self) -> None:
        """Test that valid intervention strategy values are accepted."""
        from yolo_developer.agents.sm.circular_detection_types import (
            InterventionStrategy,
        )

        valid_strategies: list[InterventionStrategy] = [
            "break_cycle",
            "inject_context",
            "escalate_human",
            "none",
        ]
        assert len(valid_strategies) == 4

    def test_strategy_type_annotation(self) -> None:
        """Test InterventionStrategy is a Literal type."""
        from typing import get_args

        from yolo_developer.agents.sm.circular_detection_types import (
            InterventionStrategy,
        )

        args = get_args(InterventionStrategy)
        assert "break_cycle" in args
        assert "inject_context" in args
        assert "escalate_human" in args
        assert "none" in args


class TestCircularPattern:
    """Tests for CircularPattern dataclass."""

    def test_create_agent_pair_pattern(self) -> None:
        """Test creating an agent pair circular pattern."""
        from yolo_developer.agents.sm.circular_detection_types import CircularPattern

        pattern = CircularPattern(
            pattern_type="agent_pair",
            agents_involved=("analyst", "pm"),
            topic="requirements_clarification",
            exchange_count=5,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:30:00Z",
            duration_seconds=1800.0,
            severity="medium",
        )

        assert pattern.pattern_type == "agent_pair"
        assert pattern.agents_involved == ("analyst", "pm")
        assert pattern.topic == "requirements_clarification"
        assert pattern.exchange_count == 5
        assert pattern.severity == "medium"

    def test_create_multi_agent_pattern(self) -> None:
        """Test creating a multi-agent circular pattern."""
        from yolo_developer.agents.sm.circular_detection_types import CircularPattern

        pattern = CircularPattern(
            pattern_type="multi_agent",
            agents_involved=("analyst", "pm", "architect"),
            topic="design_decision",
            exchange_count=4,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:20:00Z",
            duration_seconds=1200.0,
            severity="low",
        )

        assert pattern.pattern_type == "multi_agent"
        assert len(pattern.agents_involved) == 3

    def test_create_topic_cycle_pattern(self) -> None:
        """Test creating a topic-based cycle pattern."""
        from yolo_developer.agents.sm.circular_detection_types import CircularPattern

        pattern = CircularPattern(
            pattern_type="topic_cycle",
            agents_involved=("dev", "tea"),
            topic="test_failure_resolution",
            exchange_count=8,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T11:00:00Z",
            duration_seconds=3600.0,
            severity="high",
        )

        assert pattern.pattern_type == "topic_cycle"
        assert pattern.severity == "high"

    def test_pattern_is_frozen(self) -> None:
        """Test that CircularPattern is immutable."""
        from yolo_developer.agents.sm.circular_detection_types import CircularPattern

        pattern = CircularPattern(
            pattern_type="agent_pair",
            agents_involved=("analyst", "pm"),
            topic="test",
            exchange_count=4,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:30:00Z",
            duration_seconds=1800.0,
            severity="low",
        )

        with pytest.raises(AttributeError):
            pattern.severity = "critical"  # type: ignore[misc]

    def test_pattern_to_dict(self) -> None:
        """Test CircularPattern.to_dict() serialization."""
        from yolo_developer.agents.sm.circular_detection_types import CircularPattern

        pattern = CircularPattern(
            pattern_type="agent_pair",
            agents_involved=("analyst", "pm"),
            topic="requirements",
            exchange_count=5,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:30:00Z",
            duration_seconds=1800.0,
            severity="medium",
        )

        result = pattern.to_dict()

        assert result["pattern_type"] == "agent_pair"
        assert result["agents_involved"] == ["analyst", "pm"]
        assert result["topic"] == "requirements"
        assert result["exchange_count"] == 5
        assert result["severity"] == "medium"


class TestCycleLog:
    """Tests for CycleLog dataclass."""

    def test_create_cycle_log(self) -> None:
        """Test creating a cycle log entry."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularPattern,
            CycleLog,
        )

        pattern = CircularPattern(
            pattern_type="agent_pair",
            agents_involved=("analyst", "pm"),
            topic="test",
            exchange_count=4,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:30:00Z",
            duration_seconds=1800.0,
            severity="medium",
        )

        log = CycleLog(
            cycle_id="cycle-123",
            detected_at="2026-01-16T10:30:00Z",
            patterns=(pattern,),
            intervention_taken="break_cycle",
            escalation_triggered=False,
            resolution="Routed to architect to break cycle",
        )

        assert log.cycle_id == "cycle-123"
        assert len(log.patterns) == 1
        assert log.intervention_taken == "break_cycle"
        assert log.escalation_triggered is False

    def test_cycle_log_is_frozen(self) -> None:
        """Test that CycleLog is immutable."""
        from yolo_developer.agents.sm.circular_detection_types import CycleLog

        log = CycleLog(
            cycle_id="cycle-123",
            detected_at="2026-01-16T10:30:00Z",
            patterns=(),
            intervention_taken="none",
            escalation_triggered=False,
            resolution="",
        )

        with pytest.raises(AttributeError):
            log.escalation_triggered = True  # type: ignore[misc]

    def test_cycle_log_to_dict(self) -> None:
        """Test CycleLog.to_dict() serialization."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularPattern,
            CycleLog,
        )

        pattern = CircularPattern(
            pattern_type="agent_pair",
            agents_involved=("analyst", "pm"),
            topic="test",
            exchange_count=4,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:30:00Z",
            duration_seconds=1800.0,
            severity="low",
        )

        log = CycleLog(
            cycle_id="cycle-456",
            detected_at="2026-01-16T10:30:00Z",
            patterns=(pattern,),
            intervention_taken="escalate_human",
            escalation_triggered=True,
            resolution="Escalated to human",
        )

        result = log.to_dict()

        assert result["cycle_id"] == "cycle-456"
        assert len(result["patterns"]) == 1
        assert result["intervention_taken"] == "escalate_human"
        assert result["escalation_triggered"] is True


class TestCircularLogicConfig:
    """Tests for CircularLogicConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Test default configuration values."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularLogicConfig,
        )

        config = CircularLogicConfig()

        assert config.exchange_threshold == 3  # Per FR12
        assert config.time_window_seconds == 600.0  # 10 minutes
        assert config.auto_escalate_severity == "critical"
        assert config.enable_topic_detection is True
        assert config.enable_multi_agent_detection is True

    def test_custom_config_values(self) -> None:
        """Test custom configuration values."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularLogicConfig,
        )

        config = CircularLogicConfig(
            exchange_threshold=5,
            time_window_seconds=300.0,
            auto_escalate_severity="high",
            enable_topic_detection=False,
            enable_multi_agent_detection=False,
        )

        assert config.exchange_threshold == 5
        assert config.time_window_seconds == 300.0
        assert config.auto_escalate_severity == "high"
        assert config.enable_topic_detection is False
        assert config.enable_multi_agent_detection is False

    def test_config_severity_thresholds(self) -> None:
        """Test severity threshold defaults."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularLogicConfig,
        )

        config = CircularLogicConfig()

        assert "low" in config.severity_thresholds
        assert "medium" in config.severity_thresholds
        assert "high" in config.severity_thresholds
        assert "critical" in config.severity_thresholds
        assert config.severity_thresholds["low"] == 3
        assert config.severity_thresholds["critical"] == 12

    def test_config_is_frozen(self) -> None:
        """Test that CircularLogicConfig is immutable."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularLogicConfig,
        )

        config = CircularLogicConfig()

        with pytest.raises(AttributeError):
            config.exchange_threshold = 10  # type: ignore[misc]


class TestCycleAnalysis:
    """Tests for CycleAnalysis dataclass."""

    def test_create_no_cycle_analysis(self) -> None:
        """Test creating analysis result with no circular logic detected."""
        from yolo_developer.agents.sm.circular_detection_types import CycleAnalysis

        analysis = CycleAnalysis(
            circular_detected=False,
            patterns_found=(),
            intervention_strategy="none",
            intervention_message="",
            escalation_triggered=False,
            escalation_reason=None,
            topic_exchanges={},
            total_exchange_count=2,
            cycle_log=None,
        )

        assert analysis.circular_detected is False
        assert len(analysis.patterns_found) == 0
        assert analysis.intervention_strategy == "none"
        assert analysis.escalation_triggered is False

    def test_create_cycle_detected_analysis(self) -> None:
        """Test creating analysis result with circular logic detected."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularPattern,
            CycleAnalysis,
            CycleLog,
        )

        pattern = CircularPattern(
            pattern_type="agent_pair",
            agents_involved=("analyst", "pm"),
            topic="requirements",
            exchange_count=5,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:30:00Z",
            duration_seconds=1800.0,
            severity="high",
        )

        log = CycleLog(
            cycle_id="cycle-789",
            detected_at="2026-01-16T10:30:00Z",
            patterns=(pattern,),
            intervention_taken="inject_context",
            escalation_triggered=False,
            resolution="Injecting context to break cycle",
        )

        analysis = CycleAnalysis(
            circular_detected=True,
            patterns_found=(pattern,),
            intervention_strategy="inject_context",
            intervention_message="High severity cycle - injecting context",
            escalation_triggered=False,
            escalation_reason=None,
            topic_exchanges={"requirements": ["ex-1", "ex-2", "ex-3"]},
            total_exchange_count=5,
            cycle_log=log,
        )

        assert analysis.circular_detected is True
        assert len(analysis.patterns_found) == 1
        assert analysis.intervention_strategy == "inject_context"
        assert analysis.cycle_log is not None

    def test_analysis_with_escalation(self) -> None:
        """Test creating analysis result with escalation triggered."""
        from yolo_developer.agents.sm.circular_detection_types import CycleAnalysis

        analysis = CycleAnalysis(
            circular_detected=True,
            patterns_found=(),
            intervention_strategy="escalate_human",
            intervention_message="Critical circular logic - human intervention required",
            escalation_triggered=True,
            escalation_reason="Circular logic severity reached critical",
            topic_exchanges={},
            total_exchange_count=12,
            cycle_log=None,
        )

        assert analysis.escalation_triggered is True
        assert analysis.escalation_reason is not None

    def test_analysis_is_frozen(self) -> None:
        """Test that CycleAnalysis is immutable."""
        from yolo_developer.agents.sm.circular_detection_types import CycleAnalysis

        analysis = CycleAnalysis(
            circular_detected=False,
            patterns_found=(),
            intervention_strategy="none",
            intervention_message="",
            escalation_triggered=False,
            escalation_reason=None,
            topic_exchanges={},
            total_exchange_count=0,
            cycle_log=None,
        )

        with pytest.raises(AttributeError):
            analysis.circular_detected = True  # type: ignore[misc]

    def test_analysis_to_dict(self) -> None:
        """Test CycleAnalysis.to_dict() serialization."""
        from yolo_developer.agents.sm.circular_detection_types import (
            CircularPattern,
            CycleAnalysis,
        )

        pattern = CircularPattern(
            pattern_type="agent_pair",
            agents_involved=("dev", "tea"),
            topic="test_failures",
            exchange_count=6,
            first_exchange_at="2026-01-16T10:00:00Z",
            last_exchange_at="2026-01-16T10:45:00Z",
            duration_seconds=2700.0,
            severity="medium",
        )

        analysis = CycleAnalysis(
            circular_detected=True,
            patterns_found=(pattern,),
            intervention_strategy="break_cycle",
            intervention_message="Breaking cycle",
            escalation_triggered=False,
            escalation_reason=None,
            topic_exchanges={"test_failures": ["ex-1", "ex-2"]},
            total_exchange_count=6,
            cycle_log=None,
        )

        result = analysis.to_dict()

        assert result["circular_detected"] is True
        assert len(result["patterns_found"]) == 1
        assert result["intervention_strategy"] == "break_cycle"
        assert result["escalation_triggered"] is False
        assert "analyzed_at" in result

    def test_analysis_has_default_timestamp(self) -> None:
        """Test that analyzed_at has a default timestamp."""
        from yolo_developer.agents.sm.circular_detection_types import CycleAnalysis

        analysis = CycleAnalysis(
            circular_detected=False,
            patterns_found=(),
            intervention_strategy="none",
            intervention_message="",
            escalation_triggered=False,
            escalation_reason=None,
            topic_exchanges={},
            total_exchange_count=0,
            cycle_log=None,
        )

        assert analysis.analyzed_at is not None
        # Should be a valid ISO timestamp
        datetime.fromisoformat(analysis.analyzed_at.replace("Z", "+00:00"))


class TestDefaultConstants:
    """Tests for default constants in circular detection types."""

    def test_default_exchange_threshold(self) -> None:
        """Test DEFAULT_EXCHANGE_THRESHOLD constant matches FR12."""
        from yolo_developer.agents.sm.circular_detection_types import (
            DEFAULT_EXCHANGE_THRESHOLD,
        )

        assert DEFAULT_EXCHANGE_THRESHOLD == 3

    def test_default_time_window_seconds(self) -> None:
        """Test DEFAULT_TIME_WINDOW_SECONDS constant."""
        from yolo_developer.agents.sm.circular_detection_types import (
            DEFAULT_TIME_WINDOW_SECONDS,
        )

        assert DEFAULT_TIME_WINDOW_SECONDS == 600.0  # 10 minutes

    def test_valid_cycle_severities(self) -> None:
        """Test VALID_CYCLE_SEVERITIES constant."""
        from yolo_developer.agents.sm.circular_detection_types import (
            VALID_CYCLE_SEVERITIES,
        )

        assert "low" in VALID_CYCLE_SEVERITIES
        assert "medium" in VALID_CYCLE_SEVERITIES
        assert "high" in VALID_CYCLE_SEVERITIES
        assert "critical" in VALID_CYCLE_SEVERITIES

    def test_valid_intervention_strategies(self) -> None:
        """Test VALID_INTERVENTION_STRATEGIES constant."""
        from yolo_developer.agents.sm.circular_detection_types import (
            VALID_INTERVENTION_STRATEGIES,
        )

        assert "break_cycle" in VALID_INTERVENTION_STRATEGIES
        assert "inject_context" in VALID_INTERVENTION_STRATEGIES
        assert "escalate_human" in VALID_INTERVENTION_STRATEGIES
        assert "none" in VALID_INTERVENTION_STRATEGIES

    def test_valid_pattern_types(self) -> None:
        """Test VALID_PATTERN_TYPES constant."""
        from yolo_developer.agents.sm.circular_detection_types import (
            VALID_PATTERN_TYPES,
        )

        assert "agent_pair" in VALID_PATTERN_TYPES
        assert "multi_agent" in VALID_PATTERN_TYPES
        assert "topic_cycle" in VALID_PATTERN_TYPES
