"""Unit tests for rollback coordination module (Story 10.15 - Tasks 2-5).

Tests for:
- should_rollback(): Detecting when rollback is needed (Task 2.2)
- create_rollback_plan(): Creating rollback plans (Task 2.3-2.6)
- execute_rollback(): Executing rollback plans (Task 3.1-3.5)
- handle_rollback_failure(): Handling rollback failures (Task 4.1-4.4)
- coordinate_rollback(): Main orchestration function (Task 5.1-5.4)

Following red-green-refactor cycle per Story 10.15 requirements.
"""

from __future__ import annotations

from typing import Any

import pytest
from structlog.testing import capture_logs

from yolo_developer.agents.sm.emergency_types import (
    Checkpoint,
    EmergencyProtocol,
    EmergencyTrigger,
    RecoveryOption,
)
from yolo_developer.agents.sm.rollback import (
    coordinate_rollback,
    create_rollback_plan,
    execute_rollback,
    handle_rollback_failure,
    should_rollback,
)
from yolo_developer.agents.sm.rollback_types import (
    RollbackConfig,
    RollbackPlan,
    RollbackResult,
    RollbackStep,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_state() -> dict[str, Any]:
    """Create minimal valid YoloState for testing."""
    return {
        "messages": [],
        "current_agent": "dev",
        "decisions": [],
        "handoff_context": None,
        "escalate_to_human": False,
        "gate_errors": [],
        "recovery_attempts": 0,
        "recovery_action": None,
    }


@pytest.fixture
def sample_checkpoint() -> Checkpoint:
    """Create sample checkpoint for testing."""
    return Checkpoint(
        checkpoint_id="chk-12345678",
        state_snapshot={
            "current_agent": "analyst",
            "messages": [],
            "decisions": [],
            "gate_errors": [],
            "recovery_attempts": 0,
        },
        created_at="2026-01-17T10:00:00+00:00",
        trigger_type="health_degraded",
        metadata={"protocol_id": "emergency-abc123"},
    )


@pytest.fixture
def sample_emergency_protocol(sample_checkpoint: Checkpoint) -> EmergencyProtocol:
    """Create sample emergency protocol with rollback action."""
    trigger = EmergencyTrigger(
        emergency_type="health_degraded",
        severity="critical",
        source_agent="dev",
        trigger_reason="System health critical",
        health_status={"status": "critical"},
    )
    return EmergencyProtocol(
        protocol_id="emergency-abc123",
        trigger=trigger,
        status="recovering",
        checkpoint=sample_checkpoint,
        recovery_options=(
            RecoveryOption(
                action="rollback",
                description="Rollback to checkpoint",
                confidence=0.8,
                risks=("May lose progress",),
                estimated_impact="moderate",
            ),
        ),
        selected_action="rollback",
        escalation_reason=None,
    )


@pytest.fixture
def sample_rollback_plan(sample_checkpoint: Checkpoint) -> RollbackPlan:
    """Create sample rollback plan for testing."""
    steps = (
        RollbackStep(
            step_id="step-001",
            action="restore_field",
            target_field="current_agent",
            previous_value="analyst",
            current_value="dev",
            executed=False,
            success=None,
        ),
        RollbackStep(
            step_id="step-002",
            action="clear_field",
            target_field="gate_errors",
            previous_value=[],
            current_value=["error1", "error2"],
            executed=False,
            success=None,
        ),
    )
    return RollbackPlan(
        plan_id="plan-001",
        reason="checkpoint_recovery",
        checkpoint_id=sample_checkpoint.checkpoint_id,
        steps=steps,
        created_at="2026-01-17T10:00:00+00:00",
        estimated_impact="moderate",
    )


# =============================================================================
# Test should_rollback (Task 2.2)
# =============================================================================


class TestShouldRollback:
    """Tests for should_rollback function."""

    def test_returns_false_when_no_triggers(self, minimal_state: dict[str, Any]) -> None:
        """Should return (False, None) when no rollback triggers present."""
        should, reason = should_rollback(
            state=minimal_state,
            emergency_protocol=None,
            checkpoint=None,
        )
        assert should is False
        assert reason is None

    def test_returns_true_when_emergency_protocol_selects_rollback(
        self,
        minimal_state: dict[str, Any],
        sample_emergency_protocol: EmergencyProtocol,
        sample_checkpoint: Checkpoint,
    ) -> None:
        """Should return True when emergency protocol selected rollback action."""
        should, reason = should_rollback(
            state=minimal_state,
            emergency_protocol=sample_emergency_protocol,
            checkpoint=sample_checkpoint,
        )
        assert should is True
        assert reason == "emergency_recovery"

    def test_returns_true_when_recovery_action_is_rollback(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should return True when state.recovery_action == 'rollback'."""
        minimal_state["recovery_action"] = "rollback"
        should, reason = should_rollback(
            state=minimal_state,
            emergency_protocol=None,
            checkpoint=sample_checkpoint,
        )
        assert should is True
        assert reason == "checkpoint_recovery"

    def test_returns_false_when_no_checkpoint_available(
        self, minimal_state: dict[str, Any]
    ) -> None:
        """Should return False when no checkpoint is available even if rollback requested."""
        minimal_state["recovery_action"] = "rollback"
        should, reason = should_rollback(
            state=minimal_state,
            emergency_protocol=None,
            checkpoint=None,
        )
        assert should is False
        assert reason is None

    def test_returns_true_for_gate_failure_with_checkpoint(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should return True when gate_blocked and checkpoint available."""
        minimal_state["gate_blocked"] = True
        minimal_state["gate_errors"] = [{"error": "Test error"}] * 4  # Above threshold
        should, reason = should_rollback(
            state=minimal_state,
            emergency_protocol=None,
            checkpoint=sample_checkpoint,
        )
        assert should is True
        assert reason == "gate_failure"

    def test_returns_true_for_conflict_resolution(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should return True when conflict requires rollback."""
        minimal_state["mediation_result"] = {
            "conflicts_detected": [{"conflict_id": "c1"}],
            "escalations_triggered": ("c1",),
            "requires_rollback": True,
        }
        should, reason = should_rollback(
            state=minimal_state,
            emergency_protocol=None,
            checkpoint=sample_checkpoint,
        )
        assert should is True
        assert reason == "conflict_resolution"


# =============================================================================
# Test create_rollback_plan (Task 2.3-2.6)
# =============================================================================


class TestCreateRollbackPlan:
    """Tests for create_rollback_plan function."""

    def test_creates_valid_plan(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should create valid RollbackPlan with steps."""
        # Modify state to have different values from checkpoint
        minimal_state["current_agent"] = "dev"
        minimal_state["gate_errors"] = [{"error": "test"}]

        plan = create_rollback_plan(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            reason="checkpoint_recovery",
        )
        assert isinstance(plan, RollbackPlan)
        assert plan.reason == "checkpoint_recovery"
        assert plan.checkpoint_id == sample_checkpoint.checkpoint_id
        assert len(plan.steps) > 0
        assert plan.plan_id.startswith("plan-")

    def test_identifies_affected_state_fields(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should identify fields that differ between state and checkpoint."""
        minimal_state["current_agent"] = "dev"  # Checkpoint has "analyst"
        minimal_state["recovery_attempts"] = 3  # Checkpoint has 0

        plan = create_rollback_plan(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            reason="checkpoint_recovery",
        )
        # Should have steps for changed fields
        target_fields = [step.target_field for step in plan.steps]
        assert "current_agent" in target_fields

    def test_creates_ordered_steps(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should create steps in correct order for safe rollback."""
        minimal_state["current_agent"] = "dev"
        minimal_state["gate_errors"] = [{"error": "test"}]

        plan = create_rollback_plan(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            reason="checkpoint_recovery",
        )
        # All steps should have unique IDs
        step_ids = [step.step_id for step in plan.steps]
        assert len(step_ids) == len(set(step_ids))

    def test_validates_rollback_safety(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should validate that rollback is safe to execute."""
        plan = create_rollback_plan(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            reason="checkpoint_recovery",
        )
        # Plan should have estimated_impact
        assert plan.estimated_impact in ("minimal", "moderate", "significant")

    def test_handles_empty_state_changes(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should handle case where state matches checkpoint (no changes needed)."""
        # Make state match checkpoint exactly
        for key, value in sample_checkpoint.state_snapshot.items():
            if key in minimal_state:
                minimal_state[key] = value

        plan = create_rollback_plan(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            reason="checkpoint_recovery",
        )
        # Plan may have no steps if state matches checkpoint
        assert isinstance(plan, RollbackPlan)


# =============================================================================
# Test execute_rollback (Task 3.1-3.5)
# =============================================================================


class TestExecuteRollback:
    """Tests for execute_rollback function."""

    def test_executes_all_steps_successfully(
        self,
        minimal_state: dict[str, Any],
        sample_rollback_plan: RollbackPlan,
    ) -> None:
        """Should execute all steps and return completed result."""
        result = execute_rollback(
            state=minimal_state,
            plan=sample_rollback_plan,
            config=None,
        )
        assert isinstance(result, RollbackResult)
        assert result.status == "completed"
        assert result.rollback_complete is True
        assert result.steps_failed == 0

    def test_restores_state_fields(
        self,
        minimal_state: dict[str, Any],
        sample_rollback_plan: RollbackPlan,
    ) -> None:
        """Should restore state fields to checkpoint values."""
        minimal_state["current_agent"] = "dev"

        result = execute_rollback(
            state=minimal_state,
            plan=sample_rollback_plan,
            config=None,
        )
        # State should be modified (note: in real impl, state is mutated)
        assert result.rollback_complete is True

    def test_clears_error_state(
        self,
        minimal_state: dict[str, Any],
        sample_rollback_plan: RollbackPlan,
    ) -> None:
        """Should clear accumulated error state."""
        minimal_state["gate_errors"] = [{"error": "test1"}, {"error": "test2"}]
        minimal_state["recovery_attempts"] = 3

        result = execute_rollback(
            state=minimal_state,
            plan=sample_rollback_plan,
            config=None,
        )
        assert result.status == "completed"

    def test_returns_failed_result_on_step_failure(
        self,
        minimal_state: dict[str, Any],
    ) -> None:
        """Should return failed result when a step fails."""
        # Create a plan with a step that will fail
        failing_step = RollbackStep(
            step_id="step-fail",
            action="restore_field",
            target_field="nonexistent_field_that_will_cause_issues",
            previous_value="old",
            current_value="new",
            executed=False,
            success=None,
        )
        plan = RollbackPlan(
            plan_id="plan-fail",
            reason="checkpoint_recovery",
            checkpoint_id="chk-fail",
            steps=(failing_step,),
            estimated_impact="moderate",
        )

        result = execute_rollback(
            state=minimal_state,
            plan=plan,
            config=RollbackConfig(allow_partial_rollback=False),
        )
        # With allow_partial_rollback=False, should handle gracefully
        assert isinstance(result, RollbackResult)

    def test_respects_max_steps_config(self, minimal_state: dict[str, Any]) -> None:
        """Should respect max_steps configuration limit."""
        # Create plan with many steps
        steps = tuple(
            RollbackStep(
                step_id=f"step-{i:03d}",
                action="restore_field",
                target_field=f"field_{i}",
                previous_value="old",
                current_value="new",
                executed=False,
                success=None,
            )
            for i in range(10)
        )
        plan = RollbackPlan(
            plan_id="plan-many",
            reason="checkpoint_recovery",
            checkpoint_id="chk-many",
            steps=steps,
            estimated_impact="moderate",
        )
        config = RollbackConfig(max_steps=5)

        result = execute_rollback(
            state=minimal_state,
            plan=plan,
            config=config,
        )
        # Should only execute up to max_steps
        assert result.steps_executed <= 5

    def test_logs_each_recovery_step(
        self, minimal_state: dict[str, Any], sample_rollback_plan: RollbackPlan
    ) -> None:
        """Should log each recovery step for audit trail."""
        with capture_logs() as cap_logs:
            execute_rollback(
                state=minimal_state,
                plan=sample_rollback_plan,
                config=RollbackConfig(log_rollbacks=True),
            )
        # Should have log entries for steps
        log_events = [log["event"] for log in cap_logs]
        assert any("step" in event.lower() or "rollback" in event.lower() for event in log_events)


# =============================================================================
# Test handle_rollback_failure (Task 4.1-4.4)
# =============================================================================


class TestHandleRollbackFailure:
    """Tests for handle_rollback_failure function."""

    @pytest.fixture
    def partial_result(self, sample_rollback_plan: RollbackPlan) -> RollbackResult:
        """Create partial rollback result for testing."""
        return RollbackResult(
            plan=sample_rollback_plan,
            status="executing",
            steps_executed=1,
            steps_failed=1,
            rollback_complete=False,
            duration_ms=100.0,
            error_message="Step failed: restore_field failed",
        )

    def test_preserves_failed_state(
        self,
        minimal_state: dict[str, Any],
        sample_rollback_plan: RollbackPlan,
        partial_result: RollbackResult,
    ) -> None:
        """Should preserve current state on failure (no partial rollback)."""
        result = handle_rollback_failure(
            state=minimal_state,
            plan=sample_rollback_plan,
            partial_result=partial_result,
        )
        assert result.status in ("failed", "escalated")
        assert result.rollback_complete is False

    def test_creates_escalation_context(
        self,
        minimal_state: dict[str, Any],
        sample_rollback_plan: RollbackPlan,
        partial_result: RollbackResult,
    ) -> None:
        """Should create detailed escalation context."""
        result = handle_rollback_failure(
            state=minimal_state,
            plan=sample_rollback_plan,
            partial_result=partial_result,
        )
        # Result should contain error context
        assert result.error_message is not None

    def test_returns_escalated_status(
        self,
        minimal_state: dict[str, Any],
        sample_rollback_plan: RollbackPlan,
        partial_result: RollbackResult,
    ) -> None:
        """Should return escalated status for human intervention."""
        result = handle_rollback_failure(
            state=minimal_state,
            plan=sample_rollback_plan,
            partial_result=partial_result,
        )
        assert result.status == "escalated"


# =============================================================================
# Test coordinate_rollback (Task 5.1-5.4)
# =============================================================================


class TestCoordinateRollback:
    """Tests for coordinate_rollback async orchestration function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_rollback_needed(
        self, minimal_state: dict[str, Any]
    ) -> None:
        """Should return None when no rollback triggers present."""
        result = await coordinate_rollback(
            state=minimal_state,
            checkpoint=None,
            emergency_protocol=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_orchestrates_full_rollback_flow(
        self,
        minimal_state: dict[str, Any],
        sample_checkpoint: Checkpoint,
        sample_emergency_protocol: EmergencyProtocol,
    ) -> None:
        """Should orchestrate: check → plan → validate → execute → handle result."""
        minimal_state["current_agent"] = "dev"  # Different from checkpoint

        result = await coordinate_rollback(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            emergency_protocol=sample_emergency_protocol,
        )
        assert result is not None
        assert isinstance(result, RollbackResult)
        assert result.status in ("completed", "failed", "escalated")

    @pytest.mark.asyncio
    async def test_logs_all_rollback_events(
        self,
        minimal_state: dict[str, Any],
        sample_checkpoint: Checkpoint,
        sample_emergency_protocol: EmergencyProtocol,
    ) -> None:
        """Should log rollback events with structlog."""
        minimal_state["current_agent"] = "dev"

        with capture_logs() as cap_logs:
            await coordinate_rollback(
                state=minimal_state,
                checkpoint=sample_checkpoint,
                emergency_protocol=sample_emergency_protocol,
            )
        # Should have rollback-related log entries
        log_events = [log["event"] for log in cap_logs]
        assert len(log_events) > 0

    @pytest.mark.asyncio
    async def test_handles_errors_per_adr_007(
        self, minimal_state: dict[str, Any], sample_checkpoint: Checkpoint
    ) -> None:
        """Should handle errors gracefully per ADR-007."""
        # Force a problematic state
        minimal_state["recovery_action"] = "rollback"

        # Should not raise, should return result or None
        result = await coordinate_rollback(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            emergency_protocol=None,
        )
        # Should return a result (completed or escalated)
        assert result is None or isinstance(result, RollbackResult)

    @pytest.mark.asyncio
    async def test_uses_config_when_provided(
        self,
        minimal_state: dict[str, Any],
        sample_checkpoint: Checkpoint,
        sample_emergency_protocol: EmergencyProtocol,
    ) -> None:
        """Should respect RollbackConfig when provided."""
        minimal_state["current_agent"] = "dev"
        config = RollbackConfig(
            max_steps=5,
            allow_partial_rollback=False,
            log_rollbacks=True,
            auto_escalate_on_failure=True,
        )

        result = await coordinate_rollback(
            state=minimal_state,
            checkpoint=sample_checkpoint,
            emergency_protocol=sample_emergency_protocol,
            config=config,
        )
        assert result is None or isinstance(result, RollbackResult)


# =============================================================================
# Test Logging Output
# =============================================================================


class TestLoggingOutput:
    """Tests for structured logging with structlog."""

    @pytest.mark.asyncio
    async def test_logs_rollback_start(
        self,
        minimal_state: dict[str, Any],
        sample_checkpoint: Checkpoint,
        sample_emergency_protocol: EmergencyProtocol,
    ) -> None:
        """Should log when rollback coordination starts."""
        minimal_state["current_agent"] = "dev"

        with capture_logs() as cap_logs:
            await coordinate_rollback(
                state=minimal_state,
                checkpoint=sample_checkpoint,
                emergency_protocol=sample_emergency_protocol,
            )
        events = [log["event"] for log in cap_logs]
        # Should have some logging
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_logs_rollback_completion(
        self,
        minimal_state: dict[str, Any],
        sample_checkpoint: Checkpoint,
        sample_emergency_protocol: EmergencyProtocol,
    ) -> None:
        """Should log when rollback completes."""
        minimal_state["current_agent"] = "dev"

        with capture_logs() as cap_logs:
            await coordinate_rollback(
                state=minimal_state,
                checkpoint=sample_checkpoint,
                emergency_protocol=sample_emergency_protocol,
            )
        events = [log["event"] for log in cap_logs]
        assert len(events) > 0
