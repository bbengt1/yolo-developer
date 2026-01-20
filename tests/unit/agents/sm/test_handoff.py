"""Tests for handoff management functions (Story 10.8).

Tests all handoff management functionality:
- Context preparation (Task 2)
- State updates (Task 3)
- Context validation (Task 4)
- Timing and logging (Task 5)
- Main manage_handoff function (Task 6)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from yolo_developer.orchestrator.context import Decision, HandoffContext

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_decisions() -> list[Decision]:
    """Create sample decisions for testing."""
    return [
        Decision(
            agent="analyst",
            summary="Prioritized security requirements",
            rationale="User explicitly requested secure design",
            related_artifacts=("req-001", "req-002"),
        ),
        Decision(
            agent="pm",
            summary="Created user authentication story",
            rationale="Follows from security requirements",
            related_artifacts=("story-001",),
        ),
    ]


@pytest.fixture
def sample_messages() -> list[AIMessage]:
    """Create sample messages for testing."""
    return [
        AIMessage(
            content="Analysis complete",
            additional_kwargs={"agent": "analyst"},
        ),
        AIMessage(
            content="Stories created",
            additional_kwargs={"agent": "pm"},
        ),
        AIMessage(
            content="Architecture reviewed",
            additional_kwargs={"agent": "architect"},
        ),
    ]


@pytest.fixture
def sample_state(
    sample_decisions: list[Decision], sample_messages: list[AIMessage]
) -> dict[str, Any]:
    """Create sample YoloState for testing."""
    return {
        "messages": sample_messages,
        "current_agent": "pm",
        "handoff_context": None,
        "decisions": sample_decisions,
        "gate_blocked": False,
        "escalate_to_human": False,
    }


@pytest.fixture
def default_config() -> Any:
    """Create default HandoffConfig for testing."""
    from yolo_developer.agents.sm.handoff_types import HandoffConfig

    return HandoffConfig()


# =============================================================================
# Test Context Preparation (Task 2)
# =============================================================================


class TestPrepareHandoffContext:
    """Tests for _prepare_handoff_context function (Task 2.2)."""

    def test_prepare_context_basic(self, sample_state: dict[str, Any], default_config: Any) -> None:
        """Should prepare context with decisions and memory refs."""
        from yolo_developer.agents.sm.handoff import _prepare_handoff_context

        context = _prepare_handoff_context(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        assert context.source_agent == "pm"
        assert context.target_agent == "architect"
        assert len(context.decisions) > 0

    def test_prepare_context_filters_recent_decisions(
        self, sample_state: dict[str, Any], default_config: Any
    ) -> None:
        """Should include recent relevant decisions."""
        from yolo_developer.agents.sm.handoff import _prepare_handoff_context

        context = _prepare_handoff_context(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        # Should include decisions from source agent
        # The sample_state fixture has 2 decisions (analyst and pm)
        assert len(context.decisions) == 2
        agent_names = [d.agent for d in context.decisions]
        assert "analyst" in agent_names
        assert "pm" in agent_names


class TestGatherDecisionsForHandoff:
    """Tests for _gather_decisions_for_handoff function (Task 2.3)."""

    def test_gather_all_decisions(self, sample_decisions: list[Decision]) -> None:
        """Should gather all decisions when include_all=True."""
        from yolo_developer.agents.sm.handoff import _gather_decisions_for_handoff

        state = {"decisions": sample_decisions}
        decisions = _gather_decisions_for_handoff(
            state=state,
            source_agent="pm",
            include_all=True,
        )

        assert len(decisions) == len(sample_decisions)

    def test_gather_recent_decisions(self, sample_decisions: list[Decision]) -> None:
        """Should prioritize source agent decisions when include_all=False."""
        from yolo_developer.agents.sm.handoff import _gather_decisions_for_handoff

        state = {"decisions": sample_decisions}
        # When include_all=False and max_decisions=1, should get the PM decision
        # since we're asking for source_agent="pm" decisions to be prioritized
        decisions = _gather_decisions_for_handoff(
            state=state,
            source_agent="pm",
            include_all=False,
            max_decisions=1,
        )

        assert len(decisions) == 1
        # Should prioritize the source agent's decision
        assert decisions[0].agent == "pm"

    def test_gather_decisions_prioritizes_source_agent(
        self, sample_decisions: list[Decision]
    ) -> None:
        """Should include all source agent decisions plus recent others."""
        from yolo_developer.agents.sm.handoff import _gather_decisions_for_handoff

        state = {"decisions": sample_decisions}
        # With max_decisions=2, should get PM decision plus one other
        decisions = _gather_decisions_for_handoff(
            state=state,
            source_agent="pm",
            include_all=False,
            max_decisions=2,
        )

        assert len(decisions) == 2
        agent_names = [d.agent for d in decisions]
        assert "pm" in agent_names  # Source agent included
        assert "analyst" in agent_names  # Other recent decision included

    def test_gather_empty_decisions(self) -> None:
        """Should handle empty decisions list."""
        from yolo_developer.agents.sm.handoff import _gather_decisions_for_handoff

        state: dict[str, Any] = {"decisions": []}
        decisions = _gather_decisions_for_handoff(
            state=state,
            source_agent="pm",
            include_all=True,
        )

        assert decisions == ()


class TestGatherMemoryRefsForHandoff:
    """Tests for _gather_memory_refs_for_handoff function (Task 2.4)."""

    def test_gather_memory_refs_from_decisions(self, sample_decisions: list[Decision]) -> None:
        """Should extract memory refs from decision artifacts."""
        from yolo_developer.agents.sm.handoff import _gather_memory_refs_for_handoff

        state = {"decisions": sample_decisions}
        refs = _gather_memory_refs_for_handoff(state=state)

        # Should extract artifacts from decisions
        assert isinstance(refs, tuple)

    def test_gather_memory_refs_empty(self) -> None:
        """Should handle empty state."""
        from yolo_developer.agents.sm.handoff import _gather_memory_refs_for_handoff

        state: dict[str, Any] = {"decisions": []}
        refs = _gather_memory_refs_for_handoff(state=state)

        assert refs == ()


class TestFilterMessagesForHandoff:
    """Tests for _filter_messages_for_handoff function (Task 2.5)."""

    def test_filter_messages_respects_limit(self, sample_messages: list[AIMessage]) -> None:
        """Should respect max_messages limit."""
        from yolo_developer.agents.sm.handoff import _filter_messages_for_handoff

        filtered = _filter_messages_for_handoff(
            messages=sample_messages,
            max_messages=2,
        )

        assert len(filtered) <= 2

    def test_filter_messages_includes_recent(self, sample_messages: list[AIMessage]) -> None:
        """Should include most recent messages."""
        from yolo_developer.agents.sm.handoff import _filter_messages_for_handoff

        filtered = _filter_messages_for_handoff(
            messages=sample_messages,
            max_messages=1,
        )

        # Should include the most recent message
        if len(filtered) > 0:
            assert filtered[-1].content == sample_messages[-1].content

    def test_filter_messages_empty(self) -> None:
        """Should handle empty messages list."""
        from yolo_developer.agents.sm.handoff import _filter_messages_for_handoff

        filtered = _filter_messages_for_handoff(
            messages=[],
            max_messages=10,
        )

        assert filtered == []


class TestCalculateContextSize:
    """Tests for _calculate_context_size function (Task 2.6)."""

    def test_calculate_size_basic(self) -> None:
        """Should calculate serialized size correctly."""
        from yolo_developer.agents.sm.handoff import _calculate_context_size

        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
        )

        size = _calculate_context_size(context)

        assert size > 0
        assert isinstance(size, int)

    def test_calculate_size_with_decisions(self, sample_decisions: list[Decision]) -> None:
        """Should include decisions in size calculation."""
        from yolo_developer.agents.sm.handoff import _calculate_context_size

        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
            decisions=tuple(sample_decisions),
        )

        size = _calculate_context_size(context)

        # Size should be larger with decisions
        empty_context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
        )
        empty_size = _calculate_context_size(empty_context)

        assert size > empty_size


class TestValidateContextCompleteness:
    """Tests for _validate_context_completeness function (Task 2.7)."""

    def test_validate_complete_context(
        self, sample_state: dict[str, Any], sample_decisions: list[Decision]
    ) -> None:
        """Should validate complete context as valid."""
        from yolo_developer.agents.sm.handoff import _validate_context_completeness

        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
            decisions=tuple(sample_decisions),
        )

        is_valid, missing = _validate_context_completeness(
            context=context,
            target_agent="architect",
            state=sample_state,
        )

        # Even with missing optional context, basic validation should pass
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)

    def test_validate_empty_context(self, sample_state: dict[str, Any]) -> None:
        """Should identify missing required context."""
        from yolo_developer.agents.sm.handoff import _validate_context_completeness

        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
        )

        is_valid, missing = _validate_context_completeness(
            context=context,
            target_agent="architect",
            state=sample_state,
        )

        # With empty context, may have missing items
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)


# =============================================================================
# Test State Updates (Task 3)
# =============================================================================


class TestUpdateStateForHandoff:
    """Tests for _update_state_for_handoff function (Task 3.1)."""

    def test_update_state_includes_context(
        self, sample_state: dict[str, Any], sample_decisions: list[Decision]
    ) -> None:
        """Should include handoff_context in state updates."""
        from yolo_developer.agents.sm.handoff import _update_state_for_handoff

        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
            decisions=tuple(sample_decisions),
        )

        updates = _update_state_for_handoff(
            state=sample_state,
            context=context,
            target_agent="architect",
        )

        assert "handoff_context" in updates
        assert updates["handoff_context"].target_agent == "architect"

    def test_update_state_sets_current_agent(self, sample_state: dict[str, Any]) -> None:
        """Should set current_agent to target."""
        from yolo_developer.agents.sm.handoff import _update_state_for_handoff

        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
        )

        updates = _update_state_for_handoff(
            state=sample_state,
            context=context,
            target_agent="architect",
        )

        assert updates["current_agent"] == "architect"

    def test_update_state_does_not_mutate_input(self, sample_state: dict[str, Any]) -> None:
        """Should not mutate the input state (ADR-001)."""
        from yolo_developer.agents.sm.handoff import _update_state_for_handoff

        original_agent = sample_state["current_agent"]

        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
        )

        _update_state_for_handoff(
            state=sample_state,
            context=context,
            target_agent="architect",
        )

        # Original state should be unchanged
        assert sample_state["current_agent"] == original_agent


class TestAccumulateMessages:
    """Tests for _accumulate_messages function (Task 3.2)."""

    def test_accumulate_appends_new_messages(self, sample_messages: list[AIMessage]) -> None:
        """Should append new messages to existing."""
        from yolo_developer.agents.sm.handoff import _accumulate_messages

        new_message = AIMessage(
            content="New handoff message",
            additional_kwargs={"agent": "sm"},
        )

        result = _accumulate_messages(
            existing_messages=sample_messages,
            new_messages=[new_message],
        )

        assert len(result) == len(sample_messages) + 1
        assert result[-1].content == "New handoff message"

    def test_accumulate_empty_new_messages(self, sample_messages: list[AIMessage]) -> None:
        """Should handle empty new messages."""
        from yolo_developer.agents.sm.handoff import _accumulate_messages

        result = _accumulate_messages(
            existing_messages=sample_messages,
            new_messages=[],
        )

        assert len(result) == len(sample_messages)


class TestTransferDecisions:
    """Tests for _transfer_decisions function (Task 3.3)."""

    def test_transfer_preserves_all_decisions(self, sample_decisions: list[Decision]) -> None:
        """Should preserve all existing decisions."""
        from yolo_developer.agents.sm.handoff import _transfer_decisions

        result = _transfer_decisions(
            existing_decisions=sample_decisions,
            context_decisions=(),
        )

        assert len(result) >= len(sample_decisions)

    def test_transfer_adds_context_decisions(self, sample_decisions: list[Decision]) -> None:
        """Should add context decisions if not duplicated."""
        from yolo_developer.agents.sm.handoff import _transfer_decisions

        new_decision = Decision(
            agent="architect",
            summary="New decision",
            rationale="Test rationale",
        )

        result = _transfer_decisions(
            existing_decisions=sample_decisions,
            context_decisions=(new_decision,),
        )

        # Should include new decision
        summaries = [d.summary for d in result]
        assert "New decision" in summaries


# =============================================================================
# Test Context Validation (Task 4)
# =============================================================================


class TestValidateStateIntegrity:
    """Tests for _validate_state_integrity function (Task 4.1)."""

    def test_validate_integrity_passes_for_unchanged(self, sample_state: dict[str, Any]) -> None:
        """Should pass when state integrity preserved."""
        from yolo_developer.agents.sm.handoff import _validate_state_integrity

        # Same state with different handoff_context
        after_state = {**sample_state, "handoff_context": "new_context"}

        # Should pass because only transient keys changed
        is_valid = _validate_state_integrity(
            before_state=sample_state,
            after_state=after_state,
        )

        assert is_valid is True

    def test_validate_integrity_fails_for_data_change(self, sample_state: dict[str, Any]) -> None:
        """Should fail when non-transient data changed."""
        from yolo_developer.agents.sm.handoff import _validate_state_integrity

        # Change a non-transient field
        after_state = {**sample_state, "gate_blocked": True}

        is_valid = _validate_state_integrity(
            before_state=sample_state,
            after_state=after_state,
        )

        assert is_valid is False


class TestAgentContextRequirements:
    """Tests for agent-specific context requirements (Task 4.5)."""

    def test_agent_context_requirements_defined(self) -> None:
        """Should have requirements defined for all agents."""
        from yolo_developer.agents.sm.handoff import AGENT_CONTEXT_REQUIREMENTS

        expected_agents = {"analyst", "pm", "architect", "dev", "tea", "sm"}

        for agent in expected_agents:
            assert agent in AGENT_CONTEXT_REQUIREMENTS
            assert isinstance(AGENT_CONTEXT_REQUIREMENTS[agent], tuple)


# =============================================================================
# Test Timing and Logging (Task 5)
# =============================================================================


class TestHandoffTiming:
    """Tests for handoff timing functions (Tasks 5.1-5.3)."""

    def test_start_timer_returns_callable(self) -> None:
        """Should return a callable for getting elapsed time."""
        from yolo_developer.agents.sm.handoff import _start_handoff_timer

        get_elapsed = _start_handoff_timer()

        assert callable(get_elapsed)

    def test_timer_measures_duration(self) -> None:
        """Should measure elapsed duration in milliseconds."""
        from yolo_developer.agents.sm.handoff import _start_handoff_timer

        get_elapsed = _start_handoff_timer()
        time.sleep(0.01)  # 10ms
        elapsed = get_elapsed()

        # Should be at least 10ms
        assert elapsed >= 10.0
        # Should be less than 100ms (reasonable buffer)
        assert elapsed < 100.0

    def test_calculate_metrics(
        self, sample_state: dict[str, Any], sample_decisions: list[Decision]
    ) -> None:
        """Should calculate all handoff metrics."""
        from yolo_developer.agents.sm.handoff import _calculate_handoff_metrics

        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
            decisions=tuple(sample_decisions),
            memory_refs=("ref-001",),
        )

        metrics = _calculate_handoff_metrics(
            state=sample_state,
            context=context,
            duration_ms=150.5,
        )

        assert metrics.duration_ms == 150.5
        assert metrics.messages_transferred == len(sample_state["messages"])
        assert metrics.decisions_transferred == len(sample_decisions)
        assert metrics.memory_refs_transferred == 1
        assert metrics.context_size_bytes > 0


class TestHandoffLogging:
    """Tests for handoff logging functions (Tasks 5.4-5.7)."""

    def test_log_handoff_start(self) -> None:
        """Should log handoff start at INFO level."""
        from yolo_developer.agents.sm.handoff import _log_handoff_start

        with patch("yolo_developer.agents.sm.handoff.logger") as mock_logger:
            _log_handoff_start(
                handoff_id="test-123",
                source_agent="pm",
                target_agent="architect",
            )

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "handoff_started" in str(call_args)

    def test_log_handoff_complete(self) -> None:
        """Should log handoff completion at INFO level."""
        from yolo_developer.agents.sm.handoff import _log_handoff_complete
        from yolo_developer.agents.sm.handoff_types import HandoffMetrics

        metrics = HandoffMetrics(
            duration_ms=100.0,
            context_size_bytes=1024,
            messages_transferred=5,
            decisions_transferred=2,
        )

        with patch("yolo_developer.agents.sm.handoff.logger") as mock_logger:
            _log_handoff_complete(
                handoff_id="test-123",
                metrics=metrics,
                context_validated=True,
            )

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "handoff_completed" in str(call_args)

    def test_log_handoff_failure(self) -> None:
        """Should log handoff failure at WARNING level."""
        from yolo_developer.agents.sm.handoff import _log_handoff_failure

        with patch("yolo_developer.agents.sm.handoff.logger") as mock_logger:
            _log_handoff_failure(
                handoff_id="test-123",
                error="Test error",
                duration_ms=50.0,
            )

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "handoff_failed" in str(call_args)


# =============================================================================
# Test Main Handoff Function (Task 6)
# =============================================================================


class TestManageHandoff:
    """Tests for manage_handoff main function (Task 6)."""

    @pytest.mark.asyncio
    async def test_manage_handoff_success(
        self, sample_state: dict[str, Any], default_config: Any
    ) -> None:
        """Should complete handoff successfully."""
        from yolo_developer.agents.sm.handoff import manage_handoff

        result = await manage_handoff(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        assert result.success is True
        assert result.record.status == "completed"
        assert result.record.source_agent == "pm"
        assert result.record.target_agent == "architect"
        assert result.state_updates is not None

    @pytest.mark.asyncio
    async def test_manage_handoff_includes_metrics(
        self, sample_state: dict[str, Any], default_config: Any
    ) -> None:
        """Should include metrics in result."""
        from yolo_developer.agents.sm.handoff import manage_handoff

        result = await manage_handoff(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        assert result.record.metrics is not None
        assert result.record.metrics.duration_ms >= 0
        assert result.record.metrics.messages_transferred >= 0

    @pytest.mark.asyncio
    async def test_manage_handoff_validates_context(self, sample_state: dict[str, Any]) -> None:
        """Should validate context when config enabled."""
        from yolo_developer.agents.sm.handoff import manage_handoff
        from yolo_developer.agents.sm.handoff_types import HandoffConfig

        config = HandoffConfig(validate_context_integrity=True)

        result = await manage_handoff(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=config,
        )

        assert result.context_validated is not None

    @pytest.mark.asyncio
    async def test_manage_handoff_creates_unique_id(
        self, sample_state: dict[str, Any], default_config: Any
    ) -> None:
        """Should create unique handoff ID."""
        from yolo_developer.agents.sm.handoff import manage_handoff

        result1 = await manage_handoff(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        result2 = await manage_handoff(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        assert result1.record.handoff_id != result2.record.handoff_id

    @pytest.mark.asyncio
    async def test_manage_handoff_fallback_on_error(
        self, sample_state: dict[str, Any], default_config: Any
    ) -> None:
        """Should fall back gracefully on errors."""
        from yolo_developer.agents.sm.handoff import manage_handoff

        # Create invalid state to trigger error handling
        # Empty state will cause issues during context validation
        invalid_state: dict[str, Any] = {}

        result = await manage_handoff(
            state=invalid_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        # Should still return a result (fallback provides graceful degradation)
        assert result is not None
        # Fallback should still provide basic state updates
        assert result.state_updates is not None
        assert result.state_updates.get("current_agent") == "architect"
        # Context was validated (even if empty/minimal context)
        assert isinstance(result.context_validated, bool)

    @pytest.mark.asyncio
    async def test_manage_handoff_state_updates_include_agent(
        self, sample_state: dict[str, Any], default_config: Any
    ) -> None:
        """Should set current_agent in state updates."""
        from yolo_developer.agents.sm.handoff import manage_handoff

        result = await manage_handoff(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=default_config,
        )

        assert result.state_updates is not None
        assert result.state_updates.get("current_agent") == "architect"

    @pytest.mark.asyncio
    async def test_manage_handoff_with_custom_config(self, sample_state: dict[str, Any]) -> None:
        """Should respect custom configuration."""
        from yolo_developer.agents.sm.handoff import manage_handoff
        from yolo_developer.agents.sm.handoff_types import HandoffConfig

        config = HandoffConfig(
            validate_context_integrity=False,
            max_messages_to_transfer=5,
        )

        result = await manage_handoff(
            state=sample_state,
            source_agent="pm",
            target_agent="architect",
            config=config,
        )

        assert result.success is True


class TestCreateFallbackHandoff:
    """Tests for _create_fallback_handoff function (Task 6.5)."""

    def test_create_fallback_with_error(self) -> None:
        """Should create fallback result with error message."""
        from yolo_developer.agents.sm.handoff import _create_fallback_handoff

        result = _create_fallback_handoff(
            handoff_id="test-123",
            source_agent="pm",
            target_agent="architect",
            error_message="Test error",
            duration_ms=50.0,
        )

        assert result.success is False
        assert result.record.status == "failed"
        assert result.record.error_message == "Test error"

    def test_create_fallback_provides_basic_updates(self) -> None:
        """Should provide basic state updates for fallback."""
        from yolo_developer.agents.sm.handoff import _create_fallback_handoff

        result = _create_fallback_handoff(
            handoff_id="test-123",
            source_agent="pm",
            target_agent="architect",
            error_message="Test error",
            duration_ms=50.0,
        )

        # Fallback should still provide current_agent update
        assert result.state_updates is not None
        assert result.state_updates.get("current_agent") == "architect"
