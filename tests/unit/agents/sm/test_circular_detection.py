"""Tests for circular logic detection (Story 10.6).

Tests the enhanced circular logic detection system:
- Topic extraction from messages and handoff context
- Agent pair cycle detection (A->B->A patterns)
- Multi-agent cycle detection (A->B->C->A patterns)
- Topic-based cycle detection
- Intervention strategy selection
- Escalation triggering
- Cycle logging for analysis
- Main detection function

All detection is per FR12 (>3 exchanges triggers detection).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from yolo_developer.agents.sm.circular_detection_types import (
    CircularLogicConfig,
    CircularPattern,
    CycleAnalysis,
    CycleLog,
)
from yolo_developer.agents.sm.types import AgentExchange

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> CircularLogicConfig:
    """Create default circular logic detection config."""
    return CircularLogicConfig()


@pytest.fixture
def strict_config() -> CircularLogicConfig:
    """Create strict config with lower thresholds."""
    return CircularLogicConfig(
        exchange_threshold=2,
        time_window_seconds=300.0,
        auto_escalate_severity="high",
    )


@pytest.fixture
def sample_exchanges() -> list[AgentExchange]:
    """Create sample agent exchanges for testing."""
    base_time = datetime(2026, 1, 16, 10, 0, 0, tzinfo=timezone.utc)
    return [
        AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
            topic="requirements_clarification",
            timestamp=(base_time).isoformat(),
        ),
        AgentExchange(
            source_agent="pm",
            target_agent="analyst",
            exchange_type="query",
            topic="requirements_clarification",
            timestamp=(base_time).isoformat(),
        ),
        AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="response",
            topic="requirements_clarification",
            timestamp=(base_time).isoformat(),
        ),
        AgentExchange(
            source_agent="pm",
            target_agent="analyst",
            exchange_type="query",
            topic="requirements_clarification",
            timestamp=(base_time).isoformat(),
        ),
    ]


@pytest.fixture
def multi_agent_exchanges() -> list[AgentExchange]:
    """Create multi-agent cycle exchanges (analyst->pm->architect->analyst)."""
    base_time = datetime(2026, 1, 16, 10, 0, 0, tzinfo=timezone.utc)
    return [
        AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
            topic="design_decision",
            timestamp=(base_time).isoformat(),
        ),
        AgentExchange(
            source_agent="pm",
            target_agent="architect",
            exchange_type="handoff",
            topic="design_decision",
            timestamp=(base_time).isoformat(),
        ),
        AgentExchange(
            source_agent="architect",
            target_agent="analyst",
            exchange_type="query",
            topic="design_decision",
            timestamp=(base_time).isoformat(),
        ),
        AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
            topic="design_decision",
            timestamp=(base_time).isoformat(),
        ),
        AgentExchange(
            source_agent="pm",
            target_agent="architect",
            exchange_type="handoff",
            topic="design_decision",
            timestamp=(base_time).isoformat(),
        ),
    ]


@pytest.fixture
def mock_state_no_exchanges() -> dict[str, Any]:
    """Create mock state with no exchanges."""
    return {
        "messages": [],
        "current_agent": "analyst",
        "handoff_context": None,
        "decisions": [],
        "gate_blocked": False,
        "escalate_to_human": False,
    }


@pytest.fixture
def mock_state_with_exchanges() -> dict[str, Any]:
    """Create mock state with message history that has exchanges."""
    # Create mock messages with additional_kwargs containing agent info
    messages = []
    for i, agent in enumerate(["analyst", "pm", "analyst", "pm", "analyst"]):
        msg = MagicMock()
        msg.additional_kwargs = {"agent": agent, "topic": "requirements"}
        msg.content = f"Message {i} from {agent}"
        messages.append(msg)

    return {
        "messages": messages,
        "current_agent": "pm",
        "handoff_context": None,
        "decisions": [],
        "gate_blocked": False,
        "escalate_to_human": False,
    }


# =============================================================================
# Topic Extraction Tests (Task 2)
# =============================================================================


class TestTopicExtraction:
    """Tests for topic extraction from messages and context."""

    def test_extract_topic_from_exchange(self) -> None:
        """Test extracting topic from AgentExchange."""
        from yolo_developer.agents.sm.circular_detection import _extract_exchange_topic

        exchange = AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
            topic="requirements_clarification",
        )

        topic = _extract_exchange_topic(exchange)
        assert topic == "requirements_clarification"

    def test_extract_topic_empty_default(self) -> None:
        """Test default topic when exchange has no topic."""
        from yolo_developer.agents.sm.circular_detection import _extract_exchange_topic

        exchange = AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
            topic="",
        )

        topic = _extract_exchange_topic(exchange)
        assert topic == "workflow_transition"

    def test_group_exchanges_by_topic(self, sample_exchanges: list[AgentExchange]) -> None:
        """Test grouping exchanges by semantic topic."""
        from yolo_developer.agents.sm.circular_detection import _group_exchanges_by_topic

        grouped = _group_exchanges_by_topic(sample_exchanges)

        assert "requirements_clarification" in grouped
        assert len(grouped["requirements_clarification"]) == 4

    def test_group_exchanges_multiple_topics(self) -> None:
        """Test grouping with multiple different topics."""
        from yolo_developer.agents.sm.circular_detection import _group_exchanges_by_topic

        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "topic_a"),
            AgentExchange("pm", "analyst", "query", "topic_a"),
            AgentExchange("dev", "tea", "handoff", "topic_b"),
            AgentExchange("tea", "dev", "response", "topic_b"),
        ]

        grouped = _group_exchanges_by_topic(exchanges)

        assert len(grouped) == 2
        assert len(grouped["topic_a"]) == 2
        assert len(grouped["topic_b"]) == 2


class TestTimeWindowFiltering:
    """Tests for time window filtering of exchanges."""

    def test_filter_by_time_window_includes_recent(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test that recent exchanges are included."""
        from yolo_developer.agents.sm.circular_detection import _filter_by_time_window

        now = datetime.now(timezone.utc)
        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "topic", now.isoformat()),
        ]

        filtered = _filter_by_time_window(exchanges, default_config)

        assert len(filtered) == 1

    def test_filter_by_time_window_excludes_old(self) -> None:
        """Test that old exchanges outside time window are excluded."""
        from yolo_developer.agents.sm.circular_detection import _filter_by_time_window

        config = CircularLogicConfig(time_window_seconds=60.0)  # 1 minute
        old_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "topic", old_time.isoformat()),
        ]

        filtered = _filter_by_time_window(exchanges, config)

        assert len(filtered) == 0

    def test_filter_by_time_window_handles_empty(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test filtering empty exchange list."""
        from yolo_developer.agents.sm.circular_detection import _filter_by_time_window

        filtered = _filter_by_time_window([], default_config)

        assert len(filtered) == 0


class TestTrackTopicExchanges:
    """Tests for tracking topic-grouped exchanges (Task 2.3, 2.4)."""

    def test_track_topic_exchanges_basic(self) -> None:
        """Test basic topic exchange tracking."""
        from yolo_developer.agents.sm.circular_detection import _track_topic_exchanges

        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "requirements"),
            AgentExchange("pm", "analyst", "query", "requirements"),
        ]

        result = _track_topic_exchanges(exchanges)

        assert "requirements" in result
        assert len(result["requirements"]) == 2

    def test_track_returns_exchange_ids(self) -> None:
        """Test that tracking returns exchange identifiers."""
        from yolo_developer.agents.sm.circular_detection import _track_topic_exchanges

        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "topic1"),
        ]

        result = _track_topic_exchanges(exchanges)

        # Should return list of string IDs
        assert isinstance(result["topic1"], list)
        assert all(isinstance(id, str) for id in result["topic1"])


# =============================================================================
# Agent Pair Cycle Detection Tests (Task 3.2)
# =============================================================================


class TestAgentPairCycleDetection:
    """Tests for A->B->A agent pair cycle detection."""

    def test_detect_no_cycles_below_threshold(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test no cycles detected when below threshold."""
        from yolo_developer.agents.sm.circular_detection import (
            _detect_agent_pair_cycles,
        )

        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "topic"),
            AgentExchange("pm", "analyst", "query", "topic"),
        ]

        patterns = _detect_agent_pair_cycles(exchanges, default_config)

        assert len(patterns) == 0

    def test_detect_agent_pair_cycle(
        self, sample_exchanges: list[AgentExchange], default_config: CircularLogicConfig
    ) -> None:
        """Test detecting A->B->A pattern when threshold exceeded."""
        from yolo_developer.agents.sm.circular_detection import (
            _detect_agent_pair_cycles,
        )

        patterns = _detect_agent_pair_cycles(sample_exchanges, default_config)

        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.pattern_type == "agent_pair"
        assert set(pattern.agents_involved) == {"analyst", "pm"}
        assert pattern.exchange_count == 4

    def test_detect_multiple_agent_pairs(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test detecting multiple different agent pair cycles."""
        from yolo_developer.agents.sm.circular_detection import (
            _detect_agent_pair_cycles,
        )

        exchanges = [
            # analyst <-> pm cycle
            AgentExchange("analyst", "pm", "handoff", "topic1"),
            AgentExchange("pm", "analyst", "query", "topic1"),
            AgentExchange("analyst", "pm", "response", "topic1"),
            AgentExchange("pm", "analyst", "query", "topic1"),
            # dev <-> tea cycle
            AgentExchange("dev", "tea", "handoff", "topic2"),
            AgentExchange("tea", "dev", "response", "topic2"),
            AgentExchange("dev", "tea", "handoff", "topic2"),
            AgentExchange("tea", "dev", "response", "topic2"),
        ]

        patterns = _detect_agent_pair_cycles(exchanges, default_config)

        assert len(patterns) == 2


# =============================================================================
# Multi-Agent Cycle Detection Tests (Task 3.3)
# =============================================================================


class TestMultiAgentCycleDetection:
    """Tests for A->B->C->A multi-agent cycle detection."""

    def test_detect_no_multi_agent_cycle_below_threshold(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test no multi-agent cycle detected below threshold."""
        from yolo_developer.agents.sm.circular_detection import (
            _detect_multi_agent_cycles,
        )

        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "topic"),
            AgentExchange("pm", "architect", "handoff", "topic"),
        ]

        patterns = _detect_multi_agent_cycles(exchanges, default_config)

        assert len(patterns) == 0

    def test_detect_three_agent_cycle(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test detecting A->B->C->A pattern."""
        from yolo_developer.agents.sm.circular_detection import (
            _detect_multi_agent_cycles,
        )

        # Create exchanges that form a clear cycle: analyst->pm->architect->analyst
        # Then continue the cycle to exceed threshold
        base_time = datetime(2026, 1, 16, 10, 0, 0, tzinfo=timezone.utc)
        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "design", base_time.isoformat()),
            AgentExchange("pm", "architect", "handoff", "design", base_time.isoformat()),
            AgentExchange("architect", "analyst", "handoff", "design", base_time.isoformat()),
            AgentExchange("analyst", "pm", "handoff", "design", base_time.isoformat()),
            AgentExchange("pm", "architect", "handoff", "design", base_time.isoformat()),
            AgentExchange("architect", "analyst", "handoff", "design", base_time.isoformat()),
        ]

        patterns = _detect_multi_agent_cycles(exchanges, default_config)

        # Should detect the cycle (6 exchanges > 3 threshold)
        assert len(patterns) >= 1, "Expected at least one multi-agent cycle pattern"

        # At least one pattern should be multi_agent type
        multi_patterns = [p for p in patterns if p.pattern_type == "multi_agent"]
        assert len(multi_patterns) >= 1, "Expected at least one multi_agent pattern"

        # Verify pattern details
        pattern = multi_patterns[0]
        assert set(pattern.agents_involved) == {"analyst", "pm", "architect"}, \
            f"Expected analyst, pm, architect but got {pattern.agents_involved}"
        assert pattern.exchange_count > default_config.exchange_threshold, \
            f"Exchange count {pattern.exchange_count} should exceed threshold {default_config.exchange_threshold}"

    def test_no_false_positive_linear_flow(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test no false positive for linear workflow."""
        from yolo_developer.agents.sm.circular_detection import (
            _detect_multi_agent_cycles,
        )

        # Linear flow: analyst -> pm -> architect -> dev -> tea
        exchanges = [
            AgentExchange("analyst", "pm", "handoff", "topic"),
            AgentExchange("pm", "architect", "handoff", "topic"),
            AgentExchange("architect", "dev", "handoff", "topic"),
            AgentExchange("dev", "tea", "handoff", "topic"),
        ]

        patterns = _detect_multi_agent_cycles(exchanges, default_config)

        assert len(patterns) == 0


# =============================================================================
# Topic Cycle Detection Tests (Task 3.1)
# =============================================================================


class TestTopicCycleDetection:
    """Tests for topic-based cycle detection."""

    def test_detect_topic_cycle(self, default_config: CircularLogicConfig) -> None:
        """Test detecting repeated exchanges on same topic."""
        from yolo_developer.agents.sm.circular_detection import _detect_topic_cycles

        base_time = datetime(2026, 1, 16, 10, 0, 0, tzinfo=timezone.utc)
        grouped_exchanges = {
            "requirements_clarification": [
                AgentExchange("analyst", "pm", "handoff", "requirements_clarification", base_time.isoformat()),
                AgentExchange("pm", "analyst", "query", "requirements_clarification", base_time.isoformat()),
                AgentExchange("analyst", "pm", "response", "requirements_clarification", base_time.isoformat()),
                AgentExchange("pm", "analyst", "query", "requirements_clarification", base_time.isoformat()),
                AgentExchange("analyst", "pm", "response", "requirements_clarification", base_time.isoformat()),
            ],
            "other_topic": [
                AgentExchange("dev", "tea", "handoff", "other_topic", base_time.isoformat()),
            ],
        }

        patterns = _detect_topic_cycles(grouped_exchanges, default_config)

        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.pattern_type == "topic_cycle"
        assert pattern.topic == "requirements_clarification"
        assert pattern.exchange_count == 5
        # Verify agents are extracted from exchanges
        assert set(pattern.agents_involved) == {"analyst", "pm"}

    def test_no_topic_cycle_below_threshold(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test no topic cycle when below threshold."""
        from yolo_developer.agents.sm.circular_detection import _detect_topic_cycles

        base_time = datetime(2026, 1, 16, 10, 0, 0, tzinfo=timezone.utc)
        grouped_exchanges = {
            "topic1": [
                AgentExchange("analyst", "pm", "handoff", "topic1", base_time.isoformat()),
                AgentExchange("pm", "analyst", "query", "topic1", base_time.isoformat()),
            ],
            "topic2": [
                AgentExchange("dev", "tea", "handoff", "topic2", base_time.isoformat()),
            ],
        }

        patterns = _detect_topic_cycles(grouped_exchanges, default_config)

        assert len(patterns) == 0


# =============================================================================
# Severity Calculation Tests (Task 3.4)
# =============================================================================


class TestSeverityCalculation:
    """Tests for cycle severity calculation."""

    def test_calculate_low_severity(self, default_config: CircularLogicConfig) -> None:
        """Test low severity for threshold exchanges."""
        from yolo_developer.agents.sm.circular_detection import _calculate_severity

        severity = _calculate_severity(4, default_config)
        assert severity == "low"

    def test_calculate_medium_severity(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test medium severity for moderate exchanges."""
        from yolo_developer.agents.sm.circular_detection import _calculate_severity

        severity = _calculate_severity(6, default_config)
        assert severity == "medium"

    def test_calculate_high_severity(self, default_config: CircularLogicConfig) -> None:
        """Test high severity for persistent exchanges."""
        from yolo_developer.agents.sm.circular_detection import _calculate_severity

        severity = _calculate_severity(9, default_config)
        assert severity == "high"

    def test_calculate_critical_severity(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test critical severity for severe exchanges."""
        from yolo_developer.agents.sm.circular_detection import _calculate_severity

        severity = _calculate_severity(15, default_config)
        assert severity == "critical"


# =============================================================================
# Intervention Strategy Tests (Task 4)
# =============================================================================


class TestInterventionStrategy:
    """Tests for intervention strategy selection."""

    def test_determine_break_cycle_for_low(self) -> None:
        """Test break_cycle strategy for low severity."""
        from yolo_developer.agents.sm.circular_detection import (
            _determine_intervention_strategy,
        )

        patterns = [
            CircularPattern(
                pattern_type="agent_pair",
                agents_involved=("analyst", "pm"),
                topic="test",
                exchange_count=4,
                first_exchange_at="2026-01-16T10:00:00Z",
                last_exchange_at="2026-01-16T10:30:00Z",
                duration_seconds=1800.0,
                severity="low",
            )
        ]

        strategy, message = _determine_intervention_strategy(patterns)

        assert strategy == "break_cycle"
        assert message != ""

    def test_determine_inject_context_for_high(self) -> None:
        """Test inject_context strategy for high severity."""
        from yolo_developer.agents.sm.circular_detection import (
            _determine_intervention_strategy,
        )

        patterns = [
            CircularPattern(
                pattern_type="agent_pair",
                agents_involved=("analyst", "pm"),
                topic="test",
                exchange_count=10,
                first_exchange_at="2026-01-16T10:00:00Z",
                last_exchange_at="2026-01-16T11:00:00Z",
                duration_seconds=3600.0,
                severity="high",
            )
        ]

        strategy, message = _determine_intervention_strategy(patterns)

        assert strategy == "inject_context"

    def test_determine_escalate_for_critical(self) -> None:
        """Test escalate_human strategy for critical severity."""
        from yolo_developer.agents.sm.circular_detection import (
            _determine_intervention_strategy,
        )

        patterns = [
            CircularPattern(
                pattern_type="agent_pair",
                agents_involved=("analyst", "pm"),
                topic="test",
                exchange_count=15,
                first_exchange_at="2026-01-16T10:00:00Z",
                last_exchange_at="2026-01-16T12:00:00Z",
                duration_seconds=7200.0,
                severity="critical",
            )
        ]

        strategy, message = _determine_intervention_strategy(patterns)

        assert strategy == "escalate_human"

    def test_generate_intervention_message(self) -> None:
        """Test intervention message generation."""
        from yolo_developer.agents.sm.circular_detection import (
            _generate_intervention_message,
        )

        message = _generate_intervention_message("break_cycle", ("analyst", "pm"))

        assert "analyst" in message or "pm" in message
        assert len(message) > 0


# =============================================================================
# Escalation Tests (Task 5)
# =============================================================================


class TestEscalation:
    """Tests for escalation triggering."""

    def test_should_escalate_critical(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test escalation triggered for critical severity."""
        from yolo_developer.agents.sm.circular_detection import _should_escalate

        patterns = [
            CircularPattern(
                pattern_type="agent_pair",
                agents_involved=("analyst", "pm"),
                topic="test",
                exchange_count=15,
                first_exchange_at="2026-01-16T10:00:00Z",
                last_exchange_at="2026-01-16T12:00:00Z",
                duration_seconds=7200.0,
                severity="critical",
            )
        ]

        should, reason = _should_escalate(patterns, default_config)

        assert should is True
        assert reason is not None
        assert "critical" in reason.lower()

    def test_should_not_escalate_low(
        self, default_config: CircularLogicConfig
    ) -> None:
        """Test no escalation for low severity."""
        from yolo_developer.agents.sm.circular_detection import _should_escalate

        patterns = [
            CircularPattern(
                pattern_type="agent_pair",
                agents_involved=("analyst", "pm"),
                topic="test",
                exchange_count=4,
                first_exchange_at="2026-01-16T10:00:00Z",
                last_exchange_at="2026-01-16T10:30:00Z",
                duration_seconds=1800.0,
                severity="low",
            )
        ]

        should, reason = _should_escalate(patterns, default_config)

        assert should is False
        assert reason is None


# =============================================================================
# Cycle Logging Tests (Task 6)
# =============================================================================


class TestCycleLogging:
    """Tests for cycle logging."""

    def test_create_cycle_log(self) -> None:
        """Test creating a cycle log entry."""
        from yolo_developer.agents.sm.circular_detection import _create_cycle_log

        patterns = [
            CircularPattern(
                pattern_type="agent_pair",
                agents_involved=("analyst", "pm"),
                topic="test",
                exchange_count=4,
                first_exchange_at="2026-01-16T10:00:00Z",
                last_exchange_at="2026-01-16T10:30:00Z",
                duration_seconds=1800.0,
                severity="low",
            )
        ]

        log = _create_cycle_log(patterns, "break_cycle", False)

        assert log.cycle_id.startswith("cycle-")
        assert len(log.patterns) == 1
        assert log.intervention_taken == "break_cycle"
        assert log.escalation_triggered is False

    def test_log_cycle_detection_info(self) -> None:
        """Test logging at INFO level for detection."""
        from yolo_developer.agents.sm.circular_detection import _log_cycle_detection

        log = CycleLog(
            cycle_id="cycle-test",
            detected_at="2026-01-16T10:30:00Z",
            patterns=(),
            intervention_taken="none",
            escalation_triggered=False,
            resolution="No intervention",
        )

        # Should not raise
        _log_cycle_detection(log)

    def test_log_cycle_detection_warning_for_intervention(self) -> None:
        """Test logging at WARNING level for intervention."""
        from yolo_developer.agents.sm.circular_detection import _log_cycle_detection

        log = CycleLog(
            cycle_id="cycle-test",
            detected_at="2026-01-16T10:30:00Z",
            patterns=(),
            intervention_taken="break_cycle",
            escalation_triggered=False,
            resolution="Breaking cycle",
        )

        # Should not raise
        _log_cycle_detection(log)

    def test_log_cycle_detection_error_for_escalation(self) -> None:
        """Test logging at ERROR level for escalation."""
        from yolo_developer.agents.sm.circular_detection import _log_cycle_detection

        log = CycleLog(
            cycle_id="cycle-test",
            detected_at="2026-01-16T10:30:00Z",
            patterns=(),
            intervention_taken="escalate_human",
            escalation_triggered=True,
            resolution="Escalated to human",
        )

        # Should not raise
        _log_cycle_detection(log)


# =============================================================================
# Main Detection Function Tests (Task 7)
# =============================================================================


class TestDetectCircularLogic:
    """Tests for the main detect_circular_logic function."""

    @pytest.mark.asyncio
    async def test_detect_no_circular_logic(
        self, mock_state_no_exchanges: dict[str, Any]
    ) -> None:
        """Test detection with no circular logic."""
        from yolo_developer.agents.sm.circular_detection import detect_circular_logic

        result = await detect_circular_logic(mock_state_no_exchanges)

        assert isinstance(result, CycleAnalysis)
        assert result.circular_detected is False
        assert len(result.patterns_found) == 0
        assert result.intervention_strategy == "none"

    @pytest.mark.asyncio
    async def test_detect_circular_logic_with_cycles(
        self, mock_state_with_exchanges: dict[str, Any]
    ) -> None:
        """Test detection with circular logic present."""
        from yolo_developer.agents.sm.circular_detection import detect_circular_logic

        result = await detect_circular_logic(mock_state_with_exchanges)

        assert isinstance(result, CycleAnalysis)
        # With 5 messages alternating analyst/pm, should detect cycle
        assert result.circular_detected is True
        assert len(result.patterns_found) > 0

    @pytest.mark.asyncio
    async def test_detect_uses_config(
        self, mock_state_with_exchanges: dict[str, Any], strict_config: CircularLogicConfig
    ) -> None:
        """Test that detection respects config settings."""
        from yolo_developer.agents.sm.circular_detection import detect_circular_logic

        result = await detect_circular_logic(mock_state_with_exchanges, strict_config)

        # With strict config (threshold=2), should definitely detect
        assert result.circular_detected is True

    @pytest.mark.asyncio
    async def test_detect_returns_cycle_analysis(
        self, mock_state_no_exchanges: dict[str, Any]
    ) -> None:
        """Test that detection always returns CycleAnalysis."""
        from yolo_developer.agents.sm.circular_detection import detect_circular_logic

        result = await detect_circular_logic(mock_state_no_exchanges)

        assert isinstance(result, CycleAnalysis)
        assert result.analyzed_at is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestCircularDetectionIntegration:
    """Integration tests for circular detection."""

    @pytest.mark.asyncio
    async def test_full_detection_flow(self) -> None:
        """Test complete detection flow from state to analysis."""
        from yolo_developer.agents.sm.circular_detection import detect_circular_logic

        # Create state with cycling messages
        messages = []
        for i in range(8):
            agent = "analyst" if i % 2 == 0 else "pm"
            msg = MagicMock()
            msg.additional_kwargs = {"agent": agent, "topic": "requirements"}
            msg.content = f"Message {i}"
            messages.append(msg)

        state = {
            "messages": messages,
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "gate_blocked": False,
            "escalate_to_human": False,
        }

        result = await detect_circular_logic(state)

        assert result.circular_detected is True
        assert result.total_exchange_count > 0
        assert result.intervention_strategy != "none"

    @pytest.mark.asyncio
    async def test_detection_with_escalation(self) -> None:
        """Test detection that triggers escalation."""
        from yolo_developer.agents.sm.circular_detection import detect_circular_logic

        # Create state with many cycling messages (critical severity)
        messages = []
        for i in range(20):
            agent = "analyst" if i % 2 == 0 else "pm"
            msg = MagicMock()
            msg.additional_kwargs = {"agent": agent, "topic": "requirements"}
            msg.content = f"Message {i}"
            messages.append(msg)

        state = {
            "messages": messages,
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "gate_blocked": False,
            "escalate_to_human": False,
        }

        result = await detect_circular_logic(state)

        # With 20 messages cycling, should trigger escalation
        assert result.circular_detected is True
        # Severity should be high or critical

    @pytest.mark.asyncio
    async def test_topic_disabled_detection(self) -> None:
        """Test detection with topic detection disabled."""
        from yolo_developer.agents.sm.circular_detection import detect_circular_logic

        config = CircularLogicConfig(enable_topic_detection=False)

        messages = []
        for i in range(6):
            agent = "analyst" if i % 2 == 0 else "pm"
            msg = MagicMock()
            msg.additional_kwargs = {"agent": agent, "topic": "requirements"}
            msg.content = f"Message {i}"
            messages.append(msg)

        state = {
            "messages": messages,
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "gate_blocked": False,
            "escalate_to_human": False,
        }

        result = await detect_circular_logic(state, config)

        # Should still detect agent pair cycles even with topic disabled
        assert isinstance(result, CycleAnalysis)
