"""Tests for delegation module (Story 10.4).

Tests the task delegation functionality:
- Task analysis and agent matching
- Context preparation
- Delegation logging
- Acknowledgment verification
- Main delegate_task() function

All async tests use pytest-asyncio.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from yolo_developer.agents.sm.delegation import (
    _analyze_task,
    _get_agent_expertise,
    _get_relevant_state_keys,
    _handle_unacknowledged_delegation,
    _match_agent,
    _prepare_delegation_context,
    _validate_agent_availability,
    _verify_acknowledgment,
    delegate_task,
    routing_to_task_type,
)
from yolo_developer.agents.sm.delegation_types import (
    DelegationConfig,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def basic_state() -> dict[str, Any]:
    """Create a basic YoloState-like dict for testing."""
    return {
        "current_agent": "sm",
        "messages": [{"content": "test message"}],
        "decisions": [{"summary": "test decision"}],
        "gate_blocked": False,
        "escalate_to_human": False,
    }


@pytest.fixture
def state_with_context() -> dict[str, Any]:
    """Create a state with rich context for testing."""
    return {
        "current_agent": "sm",
        "messages": [
            {"content": "msg1"},
            {"content": "msg2"},
        ],
        "decisions": [{"summary": "dec1"}],
        "gate_blocked": False,
        "escalate_to_human": False,
        "seed_input": "Initial project requirements",
        "requirements": ["req1", "req2"],
        "stories": ["story1", "story2"],
        "current_story": "story-10-4",
        "design": {"pattern": "singleton"},
        "patterns": ["pattern1"],
        "implementation": "def foo(): pass",
        "test_results": {"passed": 10, "failed": 0},
        "coverage": 0.85,
        "sprint_plan": {"sprint_id": "sprint-1"},
        "health_metrics": {"status": "healthy"},
        "tech_stack": ["python", "fastapi"],
        "handoff_context": {"previous": "context"},
    }


# =============================================================================
# Task Analysis Tests (Task 2)
# =============================================================================


class TestAnalyzeTask:
    """Tests for _analyze_task function."""

    def test_extracts_keywords(self) -> None:
        """Should extract keywords from task description."""
        result = _analyze_task("Implement user authentication feature")
        assert "implement" in result["keywords"]
        assert "user" in result["keywords"]
        assert "authentication" in result["keywords"]

    def test_detects_high_complexity(self) -> None:
        """Should detect high complexity indicators."""
        result = _analyze_task("Complex architecture refactor needed")
        assert result["estimated_complexity"] == "high"

    def test_detects_low_complexity(self) -> None:
        """Should detect low complexity indicators."""
        result = _analyze_task("Simple fix for minor bug")
        assert result["estimated_complexity"] == "low"

    def test_default_medium_complexity(self) -> None:
        """Should default to medium complexity."""
        result = _analyze_task("Process the data")
        assert result["estimated_complexity"] == "medium"

    def test_identifies_test_context(self) -> None:
        """Should identify test-related context needs."""
        result = _analyze_task("Run validation tests")
        assert "test_results" in result["requires_context"]
        assert "coverage" in result["requires_context"]

    def test_identifies_implementation_context(self) -> None:
        """Should identify implementation context needs."""
        result = _analyze_task("Implement the code for feature")
        assert "current_story" in result["requires_context"]
        assert "design" in result["requires_context"]

    def test_identifies_requirement_context(self) -> None:
        """Should identify requirement analysis context needs."""
        result = _analyze_task("Analyze the requirement")
        assert "seed_input" in result["requires_context"]


class TestMatchAgent:
    """Tests for _match_agent function."""

    def test_matches_implementation_to_dev(self) -> None:
        """Implementation tasks should go to dev."""
        assert _match_agent("implementation") == "dev"

    def test_matches_requirement_analysis_to_analyst(self) -> None:
        """Requirement analysis should go to analyst."""
        assert _match_agent("requirement_analysis") == "analyst"

    def test_matches_story_creation_to_pm(self) -> None:
        """Story creation should go to pm."""
        assert _match_agent("story_creation") == "pm"

    def test_matches_architecture_design_to_architect(self) -> None:
        """Architecture design should go to architect."""
        assert _match_agent("architecture_design") == "architect"

    def test_matches_validation_to_tea(self) -> None:
        """Validation should go to tea."""
        assert _match_agent("validation") == "tea"

    def test_matches_orchestration_to_sm(self) -> None:
        """Orchestration should go to sm."""
        assert _match_agent("orchestration") == "sm"

    def test_raises_on_unknown_task_type(self) -> None:
        """Should raise ValueError for unknown task type."""
        with pytest.raises(ValueError, match="Unknown task type"):
            _match_agent("unknown_type")  # type: ignore[arg-type]


class TestValidateAgentAvailability:
    """Tests for _validate_agent_availability function."""

    @pytest.mark.asyncio
    async def test_valid_agent_available(self, basic_state: dict[str, Any]) -> None:
        """Valid agent should be available."""
        assert await _validate_agent_availability("dev", basic_state) is True

    @pytest.mark.asyncio
    async def test_invalid_agent_not_available(self, basic_state: dict[str, Any]) -> None:
        """Invalid agent name should not be available."""
        assert await _validate_agent_availability("unknown", basic_state) is False

    @pytest.mark.asyncio
    async def test_agent_blocked_by_gate(self, basic_state: dict[str, Any]) -> None:
        """Agent should not be available if gate blocked and same as current."""
        basic_state["gate_blocked"] = True
        basic_state["current_agent"] = "dev"
        assert await _validate_agent_availability("dev", basic_state) is False

    @pytest.mark.asyncio
    async def test_different_agent_available_when_gate_blocked(
        self, basic_state: dict[str, Any]
    ) -> None:
        """Different agent should be available even if gate blocked."""
        basic_state["gate_blocked"] = True
        basic_state["current_agent"] = "dev"
        assert await _validate_agent_availability("pm", basic_state) is True

    @pytest.mark.asyncio
    async def test_available_during_escalation(self, basic_state: dict[str, Any]) -> None:
        """Agent should be available during escalation for recovery."""
        basic_state["escalate_to_human"] = True
        assert await _validate_agent_availability("dev", basic_state) is True


class TestGetAgentExpertise:
    """Tests for _get_agent_expertise function."""

    def test_dev_expertise(self) -> None:
        """Dev should handle implementation."""
        assert _get_agent_expertise("dev") == ("implementation",)

    def test_analyst_expertise(self) -> None:
        """Analyst should handle requirement analysis."""
        assert _get_agent_expertise("analyst") == ("requirement_analysis",)

    def test_unknown_agent_empty(self) -> None:
        """Unknown agent should return empty tuple."""
        assert _get_agent_expertise("unknown") == ()


# =============================================================================
# Context Preparation Tests (Task 3)
# =============================================================================


class TestPrepareContext:
    """Tests for _prepare_delegation_context function."""

    def test_includes_core_context(self, basic_state: dict[str, Any]) -> None:
        """Should always include core context."""
        context = _prepare_delegation_context(basic_state, "implementation", "dev")
        assert context["message_count"] == 1
        assert context["decision_count"] == 1
        assert context["source_agent"] == "sm"

    def test_implementation_context(self, state_with_context: dict[str, Any]) -> None:
        """Should include implementation-specific context."""
        context = _prepare_delegation_context(state_with_context, "implementation", "dev")
        assert context["current_story"] == "story-10-4"
        assert context["design"] == {"pattern": "singleton"}
        assert context["patterns"] == ["pattern1"]

    def test_requirement_analysis_context(self, state_with_context: dict[str, Any]) -> None:
        """Should include requirement analysis context."""
        context = _prepare_delegation_context(state_with_context, "requirement_analysis", "analyst")
        assert context["seed_input"] == "Initial project requirements"

    def test_story_creation_context(self, state_with_context: dict[str, Any]) -> None:
        """Should include story creation context."""
        context = _prepare_delegation_context(state_with_context, "story_creation", "pm")
        assert context["requirements"] == ["req1", "req2"]

    def test_architecture_design_context(self, state_with_context: dict[str, Any]) -> None:
        """Should include architecture design context."""
        context = _prepare_delegation_context(
            state_with_context, "architecture_design", "architect"
        )
        assert context["stories"] == ["story1", "story2"]
        assert context["requirements"] == ["req1", "req2"]
        assert context["tech_stack"] == ["python", "fastapi"]

    def test_validation_context(self, state_with_context: dict[str, Any]) -> None:
        """Should include validation context."""
        context = _prepare_delegation_context(state_with_context, "validation", "tea")
        assert context["implementation"] == "def foo(): pass"
        assert context["test_results"] == {"passed": 10, "failed": 0}
        assert context["coverage"] == 0.85

    def test_orchestration_context(self, state_with_context: dict[str, Any]) -> None:
        """Should include orchestration context."""
        context = _prepare_delegation_context(state_with_context, "orchestration", "sm")
        assert context["sprint_plan"] == {"sprint_id": "sprint-1"}
        assert context["health_metrics"] == {"status": "healthy"}

    def test_includes_previous_handoff(self, state_with_context: dict[str, Any]) -> None:
        """Should include previous handoff context if present."""
        context = _prepare_delegation_context(state_with_context, "implementation", "dev")
        assert context["previous_handoff"] == {"previous": "context"}


class TestGetRelevantStateKeys:
    """Tests for _get_relevant_state_keys function."""

    def test_base_keys_always_included(self) -> None:
        """Base keys should always be included."""
        keys = _get_relevant_state_keys("implementation")
        assert "messages" in keys
        assert "decisions" in keys
        assert "current_agent" in keys

    def test_implementation_keys(self) -> None:
        """Implementation should include story and design keys."""
        keys = _get_relevant_state_keys("implementation")
        assert "current_story" in keys
        assert "design" in keys
        assert "patterns" in keys

    def test_validation_keys(self) -> None:
        """Validation should include test-related keys."""
        keys = _get_relevant_state_keys("validation")
        assert "implementation" in keys
        assert "test_results" in keys
        assert "coverage" in keys


# =============================================================================
# Acknowledgment Tests (Task 5)
# =============================================================================


class TestVerifyAcknowledgment:
    """Tests for _verify_acknowledgment function."""

    @pytest.mark.asyncio
    async def test_returns_acknowledged(self) -> None:
        """Should return acknowledged=True in current design."""
        config = DelegationConfig()
        acknowledged, timestamp = await _verify_acknowledgment("dev", config)
        assert acknowledged is True
        assert timestamp is not None

    @pytest.mark.asyncio
    async def test_returns_timestamp(self) -> None:
        """Should return ISO timestamp."""
        config = DelegationConfig()
        _, timestamp = await _verify_acknowledgment("dev", config)
        assert "T" in timestamp  # ISO format contains T


class TestHandleUnacknowledgedDelegation:
    """Tests for _handle_unacknowledged_delegation function."""

    def test_retry_when_attempts_available(self) -> None:
        """Should return retry if attempts available."""
        config = DelegationConfig(max_retry_attempts=3)
        action, rationale = _handle_unacknowledged_delegation("dev", "implementation", config)
        assert action == "retry"
        assert "retry" in rationale.lower()

    def test_escalate_when_no_retries(self) -> None:
        """Should escalate when no retries left."""
        config = DelegationConfig(max_retry_attempts=0)
        action, rationale = _handle_unacknowledged_delegation("dev", "implementation", config)
        assert action == "escalate"
        assert "unresponsive" in rationale.lower()


# =============================================================================
# Main Delegation Function Tests (Task 6)
# =============================================================================


class TestDelegateTask:
    """Tests for delegate_task main function."""

    @pytest.mark.asyncio
    async def test_successful_delegation(self, basic_state: dict[str, Any]) -> None:
        """Should successfully delegate to appropriate agent."""
        result = await delegate_task(
            state=basic_state,
            task_type="implementation",
            task_description="Implement feature X",
        )
        assert result.success is True
        assert result.acknowledged is True
        assert result.request.target_agent == "dev"
        assert result.request.task_type == "implementation"

    @pytest.mark.asyncio
    async def test_delegation_to_analyst(self, basic_state: dict[str, Any]) -> None:
        """Should delegate requirement analysis to analyst."""
        result = await delegate_task(
            state=basic_state,
            task_type="requirement_analysis",
            task_description="Analyze requirements",
        )
        assert result.success is True
        assert result.request.target_agent == "analyst"

    @pytest.mark.asyncio
    async def test_delegation_with_priority(self, basic_state: dict[str, Any]) -> None:
        """Should pass priority to delegation request."""
        result = await delegate_task(
            state=basic_state,
            task_type="implementation",
            task_description="Critical fix",
            priority="critical",
        )
        assert result.request.priority == "critical"

    @pytest.mark.asyncio
    async def test_delegation_includes_handoff_context(self, basic_state: dict[str, Any]) -> None:
        """Should include handoff context in result."""
        result = await delegate_task(
            state=basic_state,
            task_type="implementation",
            task_description="Implement feature",
        )
        assert result.handoff_context is not None
        assert result.handoff_context["source_agent"] == "sm"
        assert result.handoff_context["target_agent"] == "dev"

    @pytest.mark.asyncio
    async def test_delegation_to_unavailable_agent_fails(self, basic_state: dict[str, Any]) -> None:
        """Should fail when agent is unavailable."""
        basic_state["gate_blocked"] = True
        basic_state["current_agent"] = "dev"

        result = await delegate_task(
            state=basic_state,
            task_type="implementation",
            task_description="Implement feature",
        )
        assert result.success is False
        assert result.acknowledged is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_delegation_uses_custom_config(self, basic_state: dict[str, Any]) -> None:
        """Should use custom config when provided."""
        config = DelegationConfig(acknowledgment_timeout_seconds=60.0)
        result = await delegate_task(
            state=basic_state,
            task_type="implementation",
            task_description="Implement feature",
            config=config,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_delegation_includes_context(self, state_with_context: dict[str, Any]) -> None:
        """Should include prepared context in request."""
        result = await delegate_task(
            state=state_with_context,
            task_type="implementation",
            task_description="Implement feature",
        )
        assert "current_story" in result.request.context
        assert result.request.context["current_story"] == "story-10-4"

    @pytest.mark.asyncio
    async def test_delegation_logs_events(self, basic_state: dict[str, Any]) -> None:
        """Should log delegation events."""
        with patch("yolo_developer.agents.sm.delegation.logger") as mock_logger:
            await delegate_task(
                state=basic_state,
                task_type="implementation",
                task_description="Implement feature",
            )
            # Check that delegation_started was logged
            mock_logger.info.assert_called()


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestRoutingToTaskType:
    """Tests for routing_to_task_type function."""

    def test_dev_to_implementation(self) -> None:
        """Dev routing should return implementation."""
        assert routing_to_task_type("dev") == "implementation"

    def test_analyst_to_requirement_analysis(self) -> None:
        """Analyst routing should return requirement_analysis."""
        assert routing_to_task_type("analyst") == "requirement_analysis"

    def test_pm_to_story_creation(self) -> None:
        """PM routing should return story_creation."""
        assert routing_to_task_type("pm") == "story_creation"

    def test_architect_to_architecture_design(self) -> None:
        """Architect routing should return architecture_design."""
        assert routing_to_task_type("architect") == "architecture_design"

    def test_tea_to_validation(self) -> None:
        """TEA routing should return validation."""
        assert routing_to_task_type("tea") == "validation"

    def test_sm_to_orchestration(self) -> None:
        """SM routing should return orchestration."""
        assert routing_to_task_type("sm") == "orchestration"

    def test_escalate_returns_none(self) -> None:
        """Escalate routing should return None."""
        assert routing_to_task_type("escalate") is None

    def test_unknown_returns_none(self) -> None:
        """Unknown routing should return None."""
        assert routing_to_task_type("unknown") is None


# =============================================================================
# Integration Tests (Issue 6 - sm_node with delegation)
# =============================================================================


class TestSmNodeDelegationIntegration:
    """Integration tests for sm_node with delegate_task."""

    @pytest.fixture
    def analyst_state(self) -> dict[str, Any]:
        """Create state where analyst just completed, routing to PM."""
        return {
            "current_agent": "analyst",
            "messages": [],
            "decisions": [],
            "gate_blocked": False,
            "escalate_to_human": False,
            "needs_architecture": False,
        }

    @pytest.mark.asyncio
    async def test_sm_node_calls_delegate_task(self, analyst_state: dict[str, Any]) -> None:
        """sm_node should call delegate_task when routing to an agent."""
        from yolo_developer.agents.sm.node import sm_node

        result = await sm_node(analyst_state)

        # Should route to PM (natural successor of analyst)
        assert result["routing_decision"] == "pm"

        # Should include delegation result in sm_output
        sm_output = result["sm_output"]
        assert sm_output["delegation_result"] is not None
        assert sm_output["delegation_result"]["success"] is True
        assert sm_output["delegation_result"]["acknowledged"] is True
        assert sm_output["delegation_result"]["request"]["target_agent"] == "pm"
        assert sm_output["delegation_result"]["request"]["task_type"] == "story_creation"

    @pytest.mark.asyncio
    async def test_sm_node_includes_handoff_context(self, analyst_state: dict[str, Any]) -> None:
        """sm_node should include handoff_context in return dict.

        Note: With Story 10.8 integration, handoff_context is now a
        HandoffContext object, not a plain dict.
        """
        from yolo_developer.agents.sm.node import sm_node
        from yolo_developer.orchestrator.context import HandoffContext

        result = await sm_node(analyst_state)

        # Should include handoff_context
        assert "handoff_context" in result
        handoff_context = result["handoff_context"]
        assert handoff_context is not None
        # handoff_context is now a HandoffContext object (Story 10.8)
        assert isinstance(handoff_context, HandoffContext)
        assert handoff_context.source_agent == "analyst"
        assert handoff_context.target_agent == "pm"

    @pytest.mark.asyncio
    async def test_sm_node_decision_includes_delegation(
        self, analyst_state: dict[str, Any]
    ) -> None:
        """sm_node Decision record should include delegation info."""
        from yolo_developer.agents.sm.node import sm_node

        result = await sm_node(analyst_state)

        decisions = result["decisions"]
        assert len(decisions) == 1
        decision = decisions[0]
        assert "Delegated" in decision.summary
        assert "story_creation" in decision.summary
        assert "delegation" in decision.related_artifacts

    @pytest.mark.asyncio
    async def test_sm_node_no_delegation_on_escalate(self, analyst_state: dict[str, Any]) -> None:
        """sm_node should not delegate when escalating."""
        from yolo_developer.agents.sm.node import sm_node

        analyst_state["escalate_to_human"] = True
        result = await sm_node(analyst_state)

        # Should escalate, not delegate
        assert result["routing_decision"] == "escalate"
        sm_output = result["sm_output"]
        assert sm_output["delegation_result"] is None
        assert result["handoff_context"] is None

    @pytest.mark.asyncio
    async def test_sm_node_processing_notes_include_delegation(
        self, analyst_state: dict[str, Any]
    ) -> None:
        """sm_node processing_notes should mention delegation status."""
        from yolo_developer.agents.sm.node import sm_node

        result = await sm_node(analyst_state)

        sm_output = result["sm_output"]
        assert "Delegation to pm: success" in sm_output["processing_notes"]
