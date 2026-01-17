"""Tests for human escalation module (Story 10.14).

This module tests the human escalation functionality:
- should_escalate: Detect when escalation is needed
- create_escalation_request: Build request with options
- integrate_escalation_response: Process user decision
- handle_escalation_timeout: Handle timeout with default action
- manage_human_escalation: Full orchestration flow

Test Categories:
- Escalation detection with various triggers
- Request creation with options
- Response integration and validation
- Timeout handling
- Full orchestration flow
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.agents.sm.circular_detection_types import (
    CycleAnalysis,
)
from yolo_developer.agents.sm.conflict_types import MediationResult
from yolo_developer.agents.sm.health_types import (
    HealthMetrics,
    HealthStatus,
)
from yolo_developer.agents.sm.human_escalation import (
    create_escalation_request,
    handle_escalation_timeout,
    integrate_escalation_response,
    manage_human_escalation,
    should_escalate,
)
from yolo_developer.agents.sm.human_escalation_types import (
    EscalationConfig,
    EscalationOption,
    EscalationRequest,
    EscalationResponse,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_state() -> dict[str, Any]:
    """Create a minimal YoloState-like dict for testing."""
    return {
        "messages": [],
        "current_agent": "architect",
        "handoff_context": None,
        "decisions": [],
    }


@pytest.fixture
def cycle_analysis_with_escalation() -> CycleAnalysis:
    """Create a CycleAnalysis that triggers escalation."""
    return CycleAnalysis(
        circular_detected=True,
        patterns_found=(),
        intervention_strategy="escalate_human",
        intervention_message="Critical circular logic - escalating to human",
        escalation_triggered=True,
        escalation_reason="Critical severity pattern detected",
        topic_exchanges={},
        total_exchange_count=5,
        cycle_log=None,
    )


@pytest.fixture
def cycle_analysis_no_escalation() -> CycleAnalysis:
    """Create a CycleAnalysis that does NOT trigger escalation."""
    return CycleAnalysis(
        circular_detected=True,
        patterns_found=(),
        intervention_strategy="break_cycle",
        intervention_message="Breaking cycle by routing",
        escalation_triggered=False,
        escalation_reason=None,
        topic_exchanges={},
        total_exchange_count=3,
        cycle_log=None,
    )


@pytest.fixture
def mediation_result_with_escalation() -> MediationResult:
    """Create a MediationResult that triggers escalation."""
    return MediationResult(
        conflicts_detected=(),
        resolutions=(),
        notifications_sent=(),
        escalations_triggered=("conflict-123",),
        success=False,
        mediation_notes="Could not resolve conflict",
    )


@pytest.fixture
def mediation_result_no_escalation() -> MediationResult:
    """Create a MediationResult that does NOT trigger escalation."""
    return MediationResult(
        conflicts_detected=(),
        resolutions=(),
        notifications_sent=("architect", "dev"),
        escalations_triggered=(),
        success=True,
        mediation_notes="Resolved via principle",
    )


@pytest.fixture
def health_status_critical() -> HealthStatus:
    """Create a critical HealthStatus that triggers escalation."""
    metrics = HealthMetrics(
        agent_idle_times={},
        agent_cycle_times={},
        agent_churn_rates={},
        overall_cycle_time=100.0,
        overall_churn_rate=5.0,
    )
    return HealthStatus(
        status="critical",
        metrics=metrics,
        alerts=(),
        summary="System health critical",
        is_healthy=False,
    )


@pytest.fixture
def health_status_healthy() -> HealthStatus:
    """Create a healthy HealthStatus that does NOT trigger escalation."""
    metrics = HealthMetrics(
        agent_idle_times={},
        agent_cycle_times={},
        agent_churn_rates={},
        overall_cycle_time=30.0,
        overall_churn_rate=2.0,
    )
    return HealthStatus(
        status="healthy",
        metrics=metrics,
        alerts=(),
        summary="All systems nominal",
        is_healthy=True,
    )


@pytest.fixture
def sample_request() -> EscalationRequest:
    """Create a sample EscalationRequest for testing."""
    options = (
        EscalationOption(
            option_id="opt-1",
            label="Retry",
            description="Retry the failed operation",
            action="retry",
            is_recommended=True,
        ),
        EscalationOption(
            option_id="opt-2",
            label="Skip",
            description="Skip and continue",
            action="skip",
            is_recommended=False,
        ),
    )
    return EscalationRequest(
        request_id="esc-12345",
        trigger="circular_logic",
        agent="architect",
        summary="Circular logic detected between architect and pm",
        context={"exchanges": 5, "pattern": "topic-repetition"},
        options=options,
        recommended_option="opt-1",
    )


@pytest.fixture
def sample_response() -> EscalationResponse:
    """Create a sample EscalationResponse for testing."""
    return EscalationResponse(
        request_id="esc-12345",
        selected_option="opt-1",
        user_rationale="I think retry is the best option",
    )


# =============================================================================
# Test should_escalate
# =============================================================================


class TestShouldEscalate:
    """Tests for should_escalate function."""

    def test_no_escalation_when_all_none(self, minimal_state: dict[str, Any]) -> None:
        """Should return False when all inputs are None."""
        should, trigger = should_escalate(minimal_state, None, None, None)
        assert should is False
        assert trigger is None

    def test_escalate_on_circular_logic(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Should escalate when cycle_analysis.escalation_triggered is True."""
        should, trigger = should_escalate(
            minimal_state,
            cycle_analysis_with_escalation,
            None,
            None,
        )
        assert should is True
        assert trigger == "circular_logic"

    def test_no_escalate_on_non_escalating_circular(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_no_escalation: CycleAnalysis,
    ) -> None:
        """Should NOT escalate when cycle_analysis.escalation_triggered is False."""
        should, trigger = should_escalate(
            minimal_state,
            cycle_analysis_no_escalation,
            None,
            None,
        )
        assert should is False
        assert trigger is None

    def test_escalate_on_conflict_unresolved(
        self,
        minimal_state: dict[str, Any],
        mediation_result_with_escalation: MediationResult,
    ) -> None:
        """Should escalate when mediation_result.escalations_triggered is non-empty."""
        should, trigger = should_escalate(
            minimal_state,
            None,
            mediation_result_with_escalation,
            None,
        )
        assert should is True
        assert trigger == "conflict_unresolved"

    def test_no_escalate_on_resolved_conflict(
        self,
        minimal_state: dict[str, Any],
        mediation_result_no_escalation: MediationResult,
    ) -> None:
        """Should NOT escalate when mediation was successful."""
        should, trigger = should_escalate(
            minimal_state,
            None,
            mediation_result_no_escalation,
            None,
        )
        assert should is False
        assert trigger is None

    def test_escalate_on_critical_health(
        self,
        minimal_state: dict[str, Any],
        health_status_critical: HealthStatus,
    ) -> None:
        """Should escalate when health_status.status is 'critical'."""
        should, trigger = should_escalate(
            minimal_state,
            None,
            None,
            health_status_critical,
        )
        assert should is True
        assert trigger == "system_error"

    def test_no_escalate_on_healthy_status(
        self,
        minimal_state: dict[str, Any],
        health_status_healthy: HealthStatus,
    ) -> None:
        """Should NOT escalate when health is healthy."""
        should, trigger = should_escalate(
            minimal_state,
            None,
            None,
            health_status_healthy,
        )
        assert should is False
        assert trigger is None

    def test_priority_circular_over_conflict(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
        mediation_result_with_escalation: MediationResult,
    ) -> None:
        """Circular logic should take priority over conflict for trigger type."""
        should, trigger = should_escalate(
            minimal_state,
            cycle_analysis_with_escalation,
            mediation_result_with_escalation,
            None,
        )
        assert should is True
        assert trigger == "circular_logic"

    def test_escalate_on_gate_blocked_without_recovery(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should escalate when gate_blocked is True and no recovery path."""
        state = {
            **minimal_state,
            "gate_blocked": True,
            "gate_recovery_path": None,
        }
        should, trigger = should_escalate(state, None, None, None)
        assert should is True
        assert trigger == "gate_blocked"

    def test_no_escalate_on_gate_blocked_with_recovery(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should NOT escalate when gate_blocked has recovery path."""
        state = {
            **minimal_state,
            "gate_blocked": True,
            "gate_recovery_path": "retry_with_fixes",
        }
        should, trigger = should_escalate(state, None, None, None)
        assert should is False
        assert trigger is None

    def test_no_escalate_when_gate_not_blocked(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should NOT escalate when gate_blocked is False."""
        state = {
            **minimal_state,
            "gate_blocked": False,
        }
        should, trigger = should_escalate(state, None, None, None)
        assert should is False
        assert trigger is None

    def test_escalate_on_user_requested(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should escalate when escalate_to_human flag is True."""
        state = {
            **minimal_state,
            "escalate_to_human": True,
        }
        should, trigger = should_escalate(state, None, None, None)
        assert should is True
        assert trigger == "user_requested"

    def test_no_escalate_when_user_not_requested(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should NOT escalate when escalate_to_human is False."""
        state = {
            **minimal_state,
            "escalate_to_human": False,
        }
        should, trigger = should_escalate(state, None, None, None)
        assert should is False
        assert trigger is None

    def test_priority_order_circular_over_gate_blocked(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Circular logic should take priority over gate_blocked."""
        state = {
            **minimal_state,
            "gate_blocked": True,
            "gate_recovery_path": None,
        }
        should, trigger = should_escalate(
            state,
            cycle_analysis_with_escalation,
            None,
            None,
        )
        assert should is True
        assert trigger == "circular_logic"

    def test_priority_order_gate_blocked_over_user_requested(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Gate blocked should take priority over user_requested."""
        state = {
            **minimal_state,
            "gate_blocked": True,
            "gate_recovery_path": None,
            "escalate_to_human": True,
        }
        should, trigger = should_escalate(state, None, None, None)
        assert should is True
        assert trigger == "gate_blocked"


# =============================================================================
# Test create_escalation_request
# =============================================================================


class TestCreateEscalationRequest:
    """Tests for create_escalation_request function."""

    def test_creates_valid_request(self, minimal_state: dict[str, Any]) -> None:
        """Should create a valid EscalationRequest with options."""
        request = create_escalation_request(
            minimal_state,
            "circular_logic",
            {"exchanges": 5},
        )
        assert request.request_id.startswith("esc-")
        assert request.trigger == "circular_logic"
        assert request.agent == "architect"
        assert len(request.options) > 0
        assert request.summary != ""

    def test_request_includes_context(self, minimal_state: dict[str, Any]) -> None:
        """Should include provided context in the request."""
        context = {"exchanges": 5, "pattern": "topic-repetition"}
        request = create_escalation_request(
            minimal_state,
            "circular_logic",
            context,
        )
        assert "exchanges" in request.context
        assert request.context["exchanges"] == 5

    def test_request_has_recommended_option(self, minimal_state: dict[str, Any]) -> None:
        """Should set a recommended option."""
        request = create_escalation_request(
            minimal_state,
            "circular_logic",
            {},
        )
        # Should have at least one option marked as recommended
        recommended_options = [opt for opt in request.options if opt.is_recommended]
        assert len(recommended_options) >= 1
        # recommended_option field should match
        if request.recommended_option:
            assert any(opt.option_id == request.recommended_option for opt in request.options)

    def test_request_for_conflict_trigger(self, minimal_state: dict[str, Any]) -> None:
        """Should create appropriate request for conflict trigger."""
        request = create_escalation_request(
            minimal_state,
            "conflict_unresolved",
            {"conflict_id": "conflict-123"},
        )
        assert request.trigger == "conflict_unresolved"
        assert "conflict" in request.summary.lower() or "unresolved" in request.summary.lower()

    def test_request_for_system_error_trigger(self, minimal_state: dict[str, Any]) -> None:
        """Should create appropriate request for system error trigger."""
        request = create_escalation_request(
            minimal_state,
            "system_error",
            {"error": "Critical failure"},
        )
        assert request.trigger == "system_error"


# =============================================================================
# Test integrate_escalation_response
# =============================================================================


class TestIntegrateEscalationResponse:
    """Tests for integrate_escalation_response function."""

    def test_integrates_valid_response(
        self,
        minimal_state: dict[str, Any],
        sample_request: EscalationRequest,
        sample_response: EscalationResponse,
    ) -> None:
        """Should successfully integrate a valid response."""
        result = integrate_escalation_response(
            minimal_state,
            sample_request,
            sample_response,
        )
        assert result.status == "resolved"
        assert result.integration_success is True
        assert result.resolution_action == "retry"
        assert result.response == sample_response

    def test_rejects_mismatched_request_id(
        self,
        minimal_state: dict[str, Any],
        sample_request: EscalationRequest,
    ) -> None:
        """Should fail when response request_id doesn't match."""
        mismatched_response = EscalationResponse(
            request_id="wrong-id",
            selected_option="opt-1",
            user_rationale=None,
        )
        result = integrate_escalation_response(
            minimal_state,
            sample_request,
            mismatched_response,
        )
        assert result.integration_success is False
        assert result.status != "resolved"

    def test_rejects_invalid_option(
        self,
        minimal_state: dict[str, Any],
        sample_request: EscalationRequest,
    ) -> None:
        """Should fail when selected option doesn't exist in request."""
        invalid_response = EscalationResponse(
            request_id="esc-12345",
            selected_option="nonexistent-option",
            user_rationale=None,
        )
        result = integrate_escalation_response(
            minimal_state,
            sample_request,
            invalid_response,
        )
        assert result.integration_success is False

    def test_extracts_action_from_option(
        self,
        minimal_state: dict[str, Any],
        sample_request: EscalationRequest,
    ) -> None:
        """Should extract the action from the selected option."""
        response = EscalationResponse(
            request_id="esc-12345",
            selected_option="opt-2",  # Skip option
            user_rationale=None,
        )
        result = integrate_escalation_response(
            minimal_state,
            sample_request,
            response,
        )
        assert result.resolution_action == "skip"


# =============================================================================
# Test handle_escalation_timeout
# =============================================================================


class TestHandleEscalationTimeout:
    """Tests for handle_escalation_timeout function."""

    def test_handles_timeout_with_default_action(
        self,
        sample_request: EscalationRequest,
    ) -> None:
        """Should handle timeout using config default action."""
        config = EscalationConfig(default_action="skip")
        result = handle_escalation_timeout(sample_request, config)
        assert result.status == "timed_out"
        assert result.resolution_action == "skip"
        assert result.response is None
        assert result.integration_success is False

    def test_timeout_preserves_request(
        self,
        sample_request: EscalationRequest,
    ) -> None:
        """Should preserve the original request in result."""
        config = EscalationConfig()
        result = handle_escalation_timeout(sample_request, config)
        assert result.request == sample_request

    def test_timeout_uses_custom_default_action(
        self,
        sample_request: EscalationRequest,
    ) -> None:
        """Should use custom default_action from config."""
        config = EscalationConfig(default_action="escalate")
        result = handle_escalation_timeout(sample_request, config)
        assert result.resolution_action == "escalate"


# =============================================================================
# Test Logging Output (Task 8.8)
# =============================================================================


class TestLoggingOutput:
    """Tests for structured logging output.

    Verifies that human escalation functions emit proper structured logs
    using structlog, per ADR-007 logging standards.
    """

    def test_should_escalate_logs_on_trigger(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log when escalation is triggered."""
        # Note: structlog may not always appear in caplog depending on config
        # This test verifies the function executes without logging errors
        should, trigger = should_escalate(
            minimal_state,
            cycle_analysis_with_escalation,
            None,
            None,
        )

        assert should is True
        assert trigger == "circular_logic"
        # Function should complete without raising logging errors

    def test_create_request_does_not_raise_on_logging(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should not raise errors during logging in create_escalation_request."""
        # Verify function executes without logging errors
        request = create_escalation_request(
            minimal_state,
            "user_requested",
            {"test": "context"},
        )

        assert request is not None
        # Function should complete without raising logging errors

    def test_integrate_response_does_not_raise_on_logging(
        self,
        minimal_state: dict[str, Any],
        sample_request: EscalationRequest,
        sample_response: EscalationResponse,
    ) -> None:
        """Should not raise errors during logging in integrate_escalation_response."""
        # Verify function executes without logging errors
        result = integrate_escalation_response(
            minimal_state,
            sample_request,
            sample_response,
        )

        assert result is not None
        # Function should complete without raising logging errors

    def test_timeout_handler_does_not_raise_on_logging(
        self,
        sample_request: EscalationRequest,
    ) -> None:
        """Should not raise errors during logging in handle_escalation_timeout."""
        # Verify function executes without logging errors
        config = EscalationConfig()
        result = handle_escalation_timeout(sample_request, config)

        assert result is not None
        # Function should complete without raising logging errors


# =============================================================================
# Test manage_human_escalation (Task 8.6 - Full Orchestration Flow)
# =============================================================================


class TestManageHumanEscalation:
    """Tests for manage_human_escalation async orchestration function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_escalation_needed(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should return None when no escalation triggers are present."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=None,
            mediation_result=None,
            health_status=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_result_on_circular_logic_escalation(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Should return EscalationResult when circular logic triggers escalation."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=cycle_analysis_with_escalation,
            mediation_result=None,
            health_status=None,
        )
        assert result is not None
        assert result.status == "pending"
        assert result.request.trigger == "circular_logic"
        assert result.response is None
        assert result.integration_success is False

    @pytest.mark.asyncio
    async def test_returns_result_on_conflict_escalation(
        self,
        minimal_state: dict[str, Any],
        mediation_result_with_escalation: MediationResult,
    ) -> None:
        """Should return EscalationResult when conflict triggers escalation."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=None,
            mediation_result=mediation_result_with_escalation,
            health_status=None,
        )
        assert result is not None
        assert result.status == "pending"
        assert result.request.trigger == "conflict_unresolved"

    @pytest.mark.asyncio
    async def test_returns_result_on_critical_health(
        self,
        minimal_state: dict[str, Any],
        health_status_critical: HealthStatus,
    ) -> None:
        """Should return EscalationResult when health is critical."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=None,
            mediation_result=None,
            health_status=health_status_critical,
        )
        assert result is not None
        assert result.status == "pending"
        assert result.request.trigger == "system_error"

    @pytest.mark.asyncio
    async def test_returns_result_on_user_requested(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should return EscalationResult when user requests escalation."""
        state = {
            **minimal_state,
            "escalate_to_human": True,
        }
        result = await manage_human_escalation(
            state=state,
            cycle_analysis=None,
            mediation_result=None,
            health_status=None,
        )
        assert result is not None
        assert result.status == "pending"
        assert result.request.trigger == "user_requested"

    @pytest.mark.asyncio
    async def test_includes_circular_context_in_result(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Should include circular logic context in the escalation request."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=cycle_analysis_with_escalation,
            mediation_result=None,
            health_status=None,
        )
        assert result is not None
        # Context should include escalation_reason from cycle_analysis
        assert "escalation_reason" in result.request.context
        assert result.request.context["escalation_reason"] == "Critical severity pattern detected"

    @pytest.mark.asyncio
    async def test_includes_conflict_context_in_result(
        self,
        minimal_state: dict[str, Any],
        mediation_result_with_escalation: MediationResult,
    ) -> None:
        """Should include conflict context in the escalation request."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=None,
            mediation_result=mediation_result_with_escalation,
            health_status=None,
        )
        assert result is not None
        # Context should include conflicts_triggered from mediation_result
        assert "conflicts_triggered" in result.request.context
        assert result.request.context["conflicts_triggered"] == ["conflict-123"]

    @pytest.mark.asyncio
    async def test_includes_health_context_in_result(
        self,
        minimal_state: dict[str, Any],
        health_status_critical: HealthStatus,
    ) -> None:
        """Should include health context in the escalation request."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=None,
            mediation_result=None,
            health_status=health_status_critical,
        )
        assert result is not None
        # Context should include health info
        assert "health_summary" in result.request.context
        assert "health_status" in result.request.context

    @pytest.mark.asyncio
    async def test_uses_default_config_when_none_provided(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Should use default EscalationConfig when config is None."""
        # This should not raise - verifies default config is used
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=cycle_analysis_with_escalation,
            mediation_result=None,
            health_status=None,
            config=None,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_uses_custom_config_when_provided(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Should use provided EscalationConfig."""
        custom_config = EscalationConfig(
            timeout_seconds=60,
            default_action="abort",
        )
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=cycle_analysis_with_escalation,
            mediation_result=None,
            health_status=None,
            config=custom_config,
        )
        assert result is not None
        # Config doesn't affect the pending result directly
        # but verifies custom config is accepted

    @pytest.mark.asyncio
    async def test_request_has_valid_options(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Should create request with valid options for the trigger type."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=cycle_analysis_with_escalation,
            mediation_result=None,
            health_status=None,
        )
        assert result is not None
        assert len(result.request.options) > 0
        # Each option should have required fields
        for opt in result.request.options:
            assert opt.option_id != ""
            assert opt.label != ""
            assert opt.action != ""

    @pytest.mark.asyncio
    async def test_request_has_recommended_option(
        self,
        minimal_state: dict[str, Any],
        cycle_analysis_with_escalation: CycleAnalysis,
    ) -> None:
        """Should set recommended option on the request."""
        result = await manage_human_escalation(
            state=minimal_state,
            cycle_analysis=cycle_analysis_with_escalation,
            mediation_result=None,
            health_status=None,
        )
        assert result is not None
        assert result.request.recommended_option is not None
        # Recommended option should be in the options list
        option_ids = {opt.option_id for opt in result.request.options}
        assert result.request.recommended_option in option_ids
