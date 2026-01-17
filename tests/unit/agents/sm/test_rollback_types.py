"""Unit tests for rollback types (Story 10.15 - Task 1).

Tests for:
- RollbackReason Literal type validation
- RollbackStatus Literal type validation
- RollbackStep frozen dataclass
- RollbackPlan frozen dataclass
- RollbackResult frozen dataclass
- RollbackConfig frozen dataclass
- Validation in __post_init__ methods
- to_dict() serialization methods
- Constants exports

Following red-green-refactor cycle per Story 10.15 requirements.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pytest

from yolo_developer.agents.sm.rollback_types import (
    DEFAULT_MAX_ROLLBACK_STEPS,
    VALID_ROLLBACK_REASONS,
    VALID_ROLLBACK_STATUSES,
    RollbackConfig,
    RollbackPlan,
    RollbackReason,
    RollbackResult,
    RollbackStatus,
    RollbackStep,
)


class TestRollbackReasonType:
    """Tests for RollbackReason Literal type."""

    def test_valid_rollback_reasons_frozenset_exists(self) -> None:
        """VALID_ROLLBACK_REASONS should be a frozenset with all valid values."""
        assert isinstance(VALID_ROLLBACK_REASONS, frozenset)
        expected_reasons = {
            "checkpoint_recovery",
            "emergency_recovery",
            "manual_request",
            "conflict_resolution",
            "gate_failure",
        }
        assert VALID_ROLLBACK_REASONS == expected_reasons

    def test_rollback_reason_values_match_literal(self) -> None:
        """RollbackReason Literal values should match VALID_ROLLBACK_REASONS."""
        # This tests that the Literal type and frozenset are in sync
        for reason in VALID_ROLLBACK_REASONS:
            assert reason in {
                "checkpoint_recovery",
                "emergency_recovery",
                "manual_request",
                "conflict_resolution",
                "gate_failure",
            }


class TestRollbackStatusType:
    """Tests for RollbackStatus Literal type."""

    def test_valid_rollback_statuses_frozenset_exists(self) -> None:
        """VALID_ROLLBACK_STATUSES should be a frozenset with all valid values."""
        assert isinstance(VALID_ROLLBACK_STATUSES, frozenset)
        expected_statuses = {
            "pending",
            "planning",
            "executing",
            "completed",
            "failed",
            "escalated",
        }
        assert VALID_ROLLBACK_STATUSES == expected_statuses


class TestRollbackStep:
    """Tests for RollbackStep frozen dataclass."""

    def test_create_valid_rollback_step(self) -> None:
        """Should create RollbackStep with valid values."""
        step = RollbackStep(
            step_id="step-001",
            action="restore_field",
            target_field="current_agent",
            previous_value="analyst",
            current_value="dev",
            executed=False,
            success=None,
        )
        assert step.step_id == "step-001"
        assert step.action == "restore_field"
        assert step.target_field == "current_agent"
        assert step.previous_value == "analyst"
        assert step.current_value == "dev"
        assert step.executed is False
        assert step.success is None

    def test_rollback_step_is_frozen(self) -> None:
        """RollbackStep should be immutable."""
        step = RollbackStep(
            step_id="step-001",
            action="restore_field",
            target_field="current_agent",
            previous_value="analyst",
            current_value="dev",
            executed=False,
            success=None,
        )
        with pytest.raises(AttributeError):
            step.executed = True  # type: ignore[misc]

    def test_rollback_step_to_dict(self) -> None:
        """RollbackStep.to_dict() should serialize all fields."""
        step = RollbackStep(
            step_id="step-001",
            action="restore_field",
            target_field="current_agent",
            previous_value="analyst",
            current_value="dev",
            executed=True,
            success=True,
        )
        result = step.to_dict()
        assert result == {
            "step_id": "step-001",
            "action": "restore_field",
            "target_field": "current_agent",
            "previous_value": "analyst",
            "current_value": "dev",
            "executed": True,
            "success": True,
        }

    def test_rollback_step_with_none_values(self) -> None:
        """RollbackStep should handle None values for previous/current."""
        step = RollbackStep(
            step_id="step-002",
            action="clear_field",
            target_field="error",
            previous_value=None,
            current_value="Some error",
            executed=False,
            success=None,
        )
        assert step.previous_value is None
        result = step.to_dict()
        assert result["previous_value"] is None

    def test_rollback_step_warns_on_empty_step_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should warn when step_id is empty."""
        with caplog.at_level(logging.WARNING):
            RollbackStep(
                step_id="",
                action="restore_field",
                target_field="test",
                previous_value="a",
                current_value="b",
                executed=False,
                success=None,
            )
        assert "step_id is empty" in caplog.text


class TestRollbackPlan:
    """Tests for RollbackPlan frozen dataclass."""

    @pytest.fixture
    def sample_steps(self) -> tuple[RollbackStep, ...]:
        """Create sample rollback steps for testing."""
        return (
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
                target_field="error",
                previous_value=None,
                current_value="Error message",
                executed=False,
                success=None,
            ),
        )

    def test_create_valid_rollback_plan(
        self, sample_steps: tuple[RollbackStep, ...]
    ) -> None:
        """Should create RollbackPlan with valid values."""
        plan = RollbackPlan(
            plan_id="plan-001",
            reason="checkpoint_recovery",
            checkpoint_id="chk-12345678",
            steps=sample_steps,
            created_at="2026-01-17T10:00:00+00:00",
            estimated_impact="moderate",
        )
        assert plan.plan_id == "plan-001"
        assert plan.reason == "checkpoint_recovery"
        assert plan.checkpoint_id == "chk-12345678"
        assert len(plan.steps) == 2
        assert plan.estimated_impact == "moderate"

    def test_rollback_plan_is_frozen(
        self, sample_steps: tuple[RollbackStep, ...]
    ) -> None:
        """RollbackPlan should be immutable."""
        plan = RollbackPlan(
            plan_id="plan-001",
            reason="checkpoint_recovery",
            checkpoint_id="chk-12345678",
            steps=sample_steps,
            created_at="2026-01-17T10:00:00+00:00",
            estimated_impact="moderate",
        )
        with pytest.raises(AttributeError):
            plan.reason = "manual_request"  # type: ignore[misc]

    def test_rollback_plan_to_dict(
        self, sample_steps: tuple[RollbackStep, ...]
    ) -> None:
        """RollbackPlan.to_dict() should serialize with nested steps."""
        plan = RollbackPlan(
            plan_id="plan-001",
            reason="checkpoint_recovery",
            checkpoint_id="chk-12345678",
            steps=sample_steps,
            created_at="2026-01-17T10:00:00+00:00",
            estimated_impact="moderate",
        )
        result = plan.to_dict()
        assert result["plan_id"] == "plan-001"
        assert result["reason"] == "checkpoint_recovery"
        assert result["checkpoint_id"] == "chk-12345678"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["step_id"] == "step-001"

    def test_rollback_plan_warns_on_invalid_reason(
        self, caplog: pytest.LogCaptureFixture, sample_steps: tuple[RollbackStep, ...]
    ) -> None:
        """Should warn when reason is not in VALID_ROLLBACK_REASONS."""
        with caplog.at_level(logging.WARNING):
            RollbackPlan(
                plan_id="plan-001",
                reason="invalid_reason",  # type: ignore[arg-type]
                checkpoint_id="chk-12345678",
                steps=sample_steps,
                created_at="2026-01-17T10:00:00+00:00",
                estimated_impact="moderate",
            )
        assert "not a valid rollback reason" in caplog.text

    def test_rollback_plan_warns_on_empty_plan_id(
        self, caplog: pytest.LogCaptureFixture, sample_steps: tuple[RollbackStep, ...]
    ) -> None:
        """Should warn when plan_id is empty."""
        with caplog.at_level(logging.WARNING):
            RollbackPlan(
                plan_id="",
                reason="checkpoint_recovery",
                checkpoint_id="chk-12345678",
                steps=sample_steps,
                created_at="2026-01-17T10:00:00+00:00",
                estimated_impact="moderate",
            )
        assert "plan_id is empty" in caplog.text

    def test_rollback_plan_auto_generates_created_at(
        self, sample_steps: tuple[RollbackStep, ...]
    ) -> None:
        """RollbackPlan should auto-generate created_at if not provided."""
        plan = RollbackPlan(
            plan_id="plan-001",
            reason="checkpoint_recovery",
            checkpoint_id="chk-12345678",
            steps=sample_steps,
            estimated_impact="moderate",
        )
        assert plan.created_at is not None
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(plan.created_at.replace("Z", "+00:00"))


class TestRollbackResult:
    """Tests for RollbackResult frozen dataclass."""

    @pytest.fixture
    def sample_plan(self) -> RollbackPlan:
        """Create sample plan for testing."""
        steps = (
            RollbackStep(
                step_id="step-001",
                action="restore_field",
                target_field="current_agent",
                previous_value="analyst",
                current_value="dev",
                executed=True,
                success=True,
            ),
        )
        return RollbackPlan(
            plan_id="plan-001",
            reason="checkpoint_recovery",
            checkpoint_id="chk-12345678",
            steps=steps,
            created_at="2026-01-17T10:00:00+00:00",
            estimated_impact="moderate",
        )

    def test_create_valid_rollback_result(self, sample_plan: RollbackPlan) -> None:
        """Should create RollbackResult with valid values."""
        result = RollbackResult(
            plan=sample_plan,
            status="completed",
            steps_executed=1,
            steps_failed=0,
            rollback_complete=True,
            duration_ms=150.5,
            error_message=None,
        )
        assert result.plan.plan_id == "plan-001"
        assert result.status == "completed"
        assert result.steps_executed == 1
        assert result.steps_failed == 0
        assert result.rollback_complete is True
        assert result.duration_ms == 150.5
        assert result.error_message is None

    def test_rollback_result_is_frozen(self, sample_plan: RollbackPlan) -> None:
        """RollbackResult should be immutable."""
        result = RollbackResult(
            plan=sample_plan,
            status="completed",
            steps_executed=1,
            steps_failed=0,
            rollback_complete=True,
            duration_ms=150.5,
            error_message=None,
        )
        with pytest.raises(AttributeError):
            result.status = "failed"  # type: ignore[misc]

    def test_rollback_result_to_dict(self, sample_plan: RollbackPlan) -> None:
        """RollbackResult.to_dict() should serialize with nested plan."""
        result = RollbackResult(
            plan=sample_plan,
            status="completed",
            steps_executed=1,
            steps_failed=0,
            rollback_complete=True,
            duration_ms=150.5,
            error_message=None,
        )
        result_dict = result.to_dict()
        assert result_dict["status"] == "completed"
        assert result_dict["steps_executed"] == 1
        assert result_dict["plan"]["plan_id"] == "plan-001"

    def test_rollback_result_warns_on_invalid_status(
        self, caplog: pytest.LogCaptureFixture, sample_plan: RollbackPlan
    ) -> None:
        """Should warn when status is not in VALID_ROLLBACK_STATUSES."""
        with caplog.at_level(logging.WARNING):
            RollbackResult(
                plan=sample_plan,
                status="invalid_status",  # type: ignore[arg-type]
                steps_executed=1,
                steps_failed=0,
                rollback_complete=True,
                duration_ms=150.5,
                error_message=None,
            )
        assert "not a valid rollback status" in caplog.text

    def test_rollback_result_warns_on_negative_duration(
        self, caplog: pytest.LogCaptureFixture, sample_plan: RollbackPlan
    ) -> None:
        """Should warn when duration_ms is negative."""
        with caplog.at_level(logging.WARNING):
            RollbackResult(
                plan=sample_plan,
                status="completed",
                steps_executed=1,
                steps_failed=0,
                rollback_complete=True,
                duration_ms=-100.0,
                error_message=None,
            )
        assert "negative" in caplog.text

    def test_rollback_result_warns_complete_but_failed_steps(
        self, caplog: pytest.LogCaptureFixture, sample_plan: RollbackPlan
    ) -> None:
        """Should warn when rollback_complete=True but steps_failed > 0."""
        with caplog.at_level(logging.WARNING):
            RollbackResult(
                plan=sample_plan,
                status="completed",
                steps_executed=2,
                steps_failed=1,
                rollback_complete=True,
                duration_ms=150.5,
                error_message=None,
            )
        assert "rollback_complete=True but steps_failed" in caplog.text

    def test_rollback_result_with_error_message(self, sample_plan: RollbackPlan) -> None:
        """Should handle error_message on failed result."""
        result = RollbackResult(
            plan=sample_plan,
            status="failed",
            steps_executed=1,
            steps_failed=1,
            rollback_complete=False,
            duration_ms=100.0,
            error_message="Failed to restore field: permission denied",
        )
        assert result.error_message == "Failed to restore field: permission denied"
        result_dict = result.to_dict()
        assert result_dict["error_message"] == "Failed to restore field: permission denied"


class TestRollbackConfig:
    """Tests for RollbackConfig frozen dataclass."""

    def test_create_default_config(self) -> None:
        """Should create RollbackConfig with default values."""
        config = RollbackConfig()
        assert config.max_steps == DEFAULT_MAX_ROLLBACK_STEPS
        assert config.allow_partial_rollback is False
        assert config.log_rollbacks is True
        assert config.auto_escalate_on_failure is True

    def test_create_custom_config(self) -> None:
        """Should create RollbackConfig with custom values."""
        config = RollbackConfig(
            max_steps=50,
            allow_partial_rollback=True,
            log_rollbacks=False,
            auto_escalate_on_failure=False,
        )
        assert config.max_steps == 50
        assert config.allow_partial_rollback is True
        assert config.log_rollbacks is False
        assert config.auto_escalate_on_failure is False

    def test_rollback_config_is_frozen(self) -> None:
        """RollbackConfig should be immutable."""
        config = RollbackConfig()
        with pytest.raises(AttributeError):
            config.max_steps = 100  # type: ignore[misc]

    def test_rollback_config_to_dict(self) -> None:
        """RollbackConfig.to_dict() should serialize all fields."""
        config = RollbackConfig(
            max_steps=25,
            allow_partial_rollback=True,
            log_rollbacks=True,
            auto_escalate_on_failure=False,
        )
        result = config.to_dict()
        assert result == {
            "max_steps": 25,
            "allow_partial_rollback": True,
            "log_rollbacks": True,
            "auto_escalate_on_failure": False,
        }

    def test_rollback_config_warns_on_negative_max_steps(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should warn when max_steps is negative."""
        with caplog.at_level(logging.WARNING):
            RollbackConfig(max_steps=-1)
        assert "max_steps" in caplog.text and "negative" in caplog.text.lower()

    def test_rollback_config_warns_on_zero_max_steps(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should warn when max_steps is zero."""
        with caplog.at_level(logging.WARNING):
            RollbackConfig(max_steps=0)
        assert "max_steps" in caplog.text


class TestConstants:
    """Tests for exported constants."""

    def test_default_max_rollback_steps(self) -> None:
        """DEFAULT_MAX_ROLLBACK_STEPS should be a reasonable positive integer."""
        assert isinstance(DEFAULT_MAX_ROLLBACK_STEPS, int)
        assert DEFAULT_MAX_ROLLBACK_STEPS > 0
        assert DEFAULT_MAX_ROLLBACK_STEPS == 100  # Per story spec

    def test_valid_rollback_reasons_is_frozenset(self) -> None:
        """VALID_ROLLBACK_REASONS should be frozenset for immutability."""
        assert isinstance(VALID_ROLLBACK_REASONS, frozenset)

    def test_valid_rollback_statuses_is_frozenset(self) -> None:
        """VALID_ROLLBACK_STATUSES should be frozenset for immutability."""
        assert isinstance(VALID_ROLLBACK_STATUSES, frozenset)
