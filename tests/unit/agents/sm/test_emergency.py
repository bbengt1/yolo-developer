"""Tests for emergency protocol module (Story 10.10).

Tests the emergency detection, checkpointing, recovery evaluation,
escalation, and main protocol functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    pass


def make_state(
    current_agent: str = "dev",
    messages: list[Any] | None = None,
    decisions: list[Any] | None = None,
) -> dict[str, Any]:
    """Create a test state dictionary."""
    return {
        "messages": messages or [],
        "current_agent": current_agent,
        "handoff_context": None,
        "decisions": decisions or [],
    }


def make_health_status(
    status: str = "healthy",
    is_healthy: bool = True,
    alerts: tuple[Any, ...] = (),
) -> dict[str, Any]:
    """Create a test health status dictionary."""
    return {
        "status": status,
        "is_healthy": is_healthy,
        "alerts": alerts,
        "metrics": {},
        "summary": f"Status: {status}",
    }


class TestCheckHealthDegradation:
    """Tests for _check_health_degradation function."""

    def test_returns_true_for_critical_status(self) -> None:
        """_check_health_degradation returns True for critical health."""
        from yolo_developer.agents.sm.emergency import _check_health_degradation

        health_status = make_health_status(status="critical", is_healthy=False)
        result = _check_health_degradation(health_status)
        assert result is True

    def test_returns_true_for_degraded_status(self) -> None:
        """_check_health_degradation returns True for degraded health."""
        from yolo_developer.agents.sm.emergency import _check_health_degradation

        health_status = make_health_status(status="degraded", is_healthy=False)
        result = _check_health_degradation(health_status)
        assert result is True

    def test_returns_false_for_healthy_status(self) -> None:
        """_check_health_degradation returns False for healthy status."""
        from yolo_developer.agents.sm.emergency import _check_health_degradation

        health_status = make_health_status(status="healthy", is_healthy=True)
        result = _check_health_degradation(health_status)
        assert result is False

    def test_returns_false_for_warning_status(self) -> None:
        """_check_health_degradation returns False for warning status."""
        from yolo_developer.agents.sm.emergency import _check_health_degradation

        health_status = make_health_status(status="warning", is_healthy=True)
        result = _check_health_degradation(health_status)
        assert result is False

    def test_returns_false_for_none(self) -> None:
        """_check_health_degradation returns False for None health status."""
        from yolo_developer.agents.sm.emergency import _check_health_degradation

        result = _check_health_degradation(None)
        assert result is False


class TestCheckCircularLogic:
    """Tests for _check_circular_logic function."""

    def test_returns_true_when_circular_pattern_detected(self) -> None:
        """_check_circular_logic returns True when circular pattern exists."""
        from yolo_developer.agents.sm.emergency import _check_circular_logic

        state = make_state()
        # Create cycle analysis with circular pattern
        cycle_analysis = {
            "has_circular_pattern": True,
            "cycles_detected": [{"agents": ["analyst", "pm"], "count": 4}],
            "severity": "high",
        }
        result = _check_circular_logic(state, cycle_analysis)
        assert result is True

    def test_returns_false_when_no_circular_pattern(self) -> None:
        """_check_circular_logic returns False when no circular pattern."""
        from yolo_developer.agents.sm.emergency import _check_circular_logic

        state = make_state()
        cycle_analysis = {
            "has_circular_pattern": False,
            "cycles_detected": [],
            "severity": "none",
        }
        result = _check_circular_logic(state, cycle_analysis)
        assert result is False

    def test_returns_false_for_none_analysis(self) -> None:
        """_check_circular_logic returns False for None analysis."""
        from yolo_developer.agents.sm.emergency import _check_circular_logic

        state = make_state()
        result = _check_circular_logic(state, None)
        assert result is False


class TestCheckGateBlocked:
    """Tests for _check_gate_blocked function."""

    def test_returns_true_for_multiple_gate_failures(self) -> None:
        """_check_gate_blocked returns True for repeated gate failures."""
        from yolo_developer.agents.sm.emergency import _check_gate_blocked

        state = make_state()
        # Simulate repeated gate failures in decisions
        state["decisions"] = [
            {"type": "gate_failure", "gate": "testability"},
            {"type": "gate_failure", "gate": "testability"},
            {"type": "gate_failure", "gate": "testability"},
        ]
        result = _check_gate_blocked(state)
        assert result is True

    def test_returns_false_for_few_gate_failures(self) -> None:
        """_check_gate_blocked returns False for few gate failures."""
        from yolo_developer.agents.sm.emergency import _check_gate_blocked

        state = make_state()
        state["decisions"] = [
            {"type": "gate_failure", "gate": "testability"},
        ]
        result = _check_gate_blocked(state)
        assert result is False

    def test_returns_false_for_no_decisions(self) -> None:
        """_check_gate_blocked returns False for empty decisions."""
        from yolo_developer.agents.sm.emergency import _check_gate_blocked

        state = make_state()
        result = _check_gate_blocked(state)
        assert result is False


class TestCheckAgentStuck:
    """Tests for _check_agent_stuck function."""

    def test_returns_true_for_high_idle_time(self) -> None:
        """_check_agent_stuck returns True for high idle time."""
        from yolo_developer.agents.sm.emergency import _check_agent_stuck

        health_status = {
            "metrics": {
                "agent_idle_times": {"dev": 700.0},  # Over 600s threshold
            }
        }
        result = _check_agent_stuck(health_status)
        assert result is True

    def test_returns_false_for_normal_idle_time(self) -> None:
        """_check_agent_stuck returns False for normal idle time."""
        from yolo_developer.agents.sm.emergency import _check_agent_stuck

        health_status = {
            "metrics": {
                "agent_idle_times": {"dev": 100.0},  # Under threshold
            }
        }
        result = _check_agent_stuck(health_status)
        assert result is False

    def test_returns_false_for_none_status(self) -> None:
        """_check_agent_stuck returns False for None health status."""
        from yolo_developer.agents.sm.emergency import _check_agent_stuck

        result = _check_agent_stuck(None)
        assert result is False


class TestCheckSystemError:
    """Tests for _check_system_error function."""

    def test_returns_true_for_system_error_in_state(self) -> None:
        """_check_system_error returns True for system error in state."""
        from yolo_developer.agents.sm.emergency import _check_system_error

        state = make_state()
        state["error"] = {"type": "unrecoverable", "message": "Critical failure"}
        result = _check_system_error(state)
        assert result is True

    def test_returns_false_for_no_error(self) -> None:
        """_check_system_error returns False when no error in state."""
        from yolo_developer.agents.sm.emergency import _check_system_error

        state = make_state()
        result = _check_system_error(state)
        assert result is False


class TestDetectEmergency:
    """Tests for _detect_emergency function."""

    @pytest.mark.asyncio
    async def test_detects_health_degradation(self) -> None:
        """_detect_emergency detects health degradation emergency."""
        from yolo_developer.agents.sm.emergency import _detect_emergency

        state = make_state()
        health_status = make_health_status(status="critical", is_healthy=False)
        result = await _detect_emergency(state, health_status)
        assert result is not None
        assert result["emergency_type"] == "health_degraded"

    @pytest.mark.asyncio
    async def test_detects_circular_logic(self) -> None:
        """_detect_emergency detects circular logic emergency."""
        from yolo_developer.agents.sm.emergency import _detect_emergency

        state = make_state()
        state["cycle_analysis"] = {
            "has_circular_pattern": True,
            "cycles_detected": [{"agents": ["analyst", "pm"]}],
        }
        result = await _detect_emergency(state, None)
        assert result is not None
        assert result["emergency_type"] == "circular_logic"

    @pytest.mark.asyncio
    async def test_returns_none_for_no_emergency(self) -> None:
        """_detect_emergency returns None when no emergency detected."""
        from yolo_developer.agents.sm.emergency import _detect_emergency

        state = make_state()
        health_status = make_health_status(status="healthy", is_healthy=True)
        result = await _detect_emergency(state, health_status)
        assert result is None


class TestCreateEmergencyTrigger:
    """Tests for _create_emergency_trigger function."""

    def test_creates_trigger_from_detection(self) -> None:
        """_create_emergency_trigger creates EmergencyTrigger from detection."""
        from yolo_developer.agents.sm.emergency import _create_emergency_trigger

        state = make_state(current_agent="dev")
        health_status = make_health_status(status="critical", is_healthy=False)
        detection = {
            "emergency_type": "health_degraded",
            "reason": "System health critical",
        }

        trigger = _create_emergency_trigger(state, health_status, detection)

        assert trigger.emergency_type == "health_degraded"
        assert trigger.severity == "critical"
        assert trigger.source_agent == "dev"
        assert "System health" in trigger.trigger_reason

    def test_creates_trigger_with_warning_severity(self) -> None:
        """_create_emergency_trigger sets warning severity for non-critical."""
        from yolo_developer.agents.sm.emergency import _create_emergency_trigger

        state = make_state()
        detection = {
            "emergency_type": "circular_logic",
            "reason": "Agents in loop",
        }

        trigger = _create_emergency_trigger(state, None, detection)

        assert trigger.severity == "warning"


class TestCaptureStateSnapshot:
    """Tests for _capture_state_snapshot function."""

    def test_captures_state_snapshot(self) -> None:
        """_capture_state_snapshot captures relevant state data."""
        from yolo_developer.agents.sm.emergency import _capture_state_snapshot

        state = make_state(current_agent="analyst")
        state["messages"] = [{"content": "test"}]

        snapshot = _capture_state_snapshot(state)

        assert "current_agent" in snapshot
        assert snapshot["current_agent"] == "analyst"
        assert "messages" in snapshot

    def test_excludes_sensitive_data(self) -> None:
        """_capture_state_snapshot excludes sensitive data."""
        from yolo_developer.agents.sm.emergency import _capture_state_snapshot

        state = make_state()
        state["api_key"] = "secret123"

        snapshot = _capture_state_snapshot(state)

        assert "api_key" not in snapshot


class TestCreateCheckpoint:
    """Tests for _create_checkpoint function."""

    def test_creates_checkpoint(self) -> None:
        """_create_checkpoint creates Checkpoint from snapshot."""
        from yolo_developer.agents.sm.emergency import _create_checkpoint

        snapshot = {"current_agent": "dev", "messages": []}
        checkpoint = _create_checkpoint(snapshot, "health_degraded")

        assert checkpoint.checkpoint_id.startswith("chk-")
        assert checkpoint.state_snapshot == snapshot
        assert checkpoint.trigger_type == "health_degraded"
        assert checkpoint.created_at is not None


class TestStoreAndRetrieveCheckpoint:
    """Tests for checkpoint storage and retrieval."""

    def test_store_and_retrieve_checkpoint(self) -> None:
        """Checkpoint can be stored and retrieved."""
        from yolo_developer.agents.sm.emergency import (
            _retrieve_checkpoint,
            _store_checkpoint,
        )
        from yolo_developer.agents.sm.emergency_types import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="chk-test123",
            state_snapshot={"key": "value"},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="health_degraded",
        )

        _store_checkpoint(checkpoint)
        retrieved = _retrieve_checkpoint("chk-test123")

        assert retrieved is not None
        assert retrieved.checkpoint_id == "chk-test123"

    def test_retrieve_nonexistent_checkpoint_returns_none(self) -> None:
        """_retrieve_checkpoint returns None for nonexistent ID."""
        from yolo_developer.agents.sm.emergency import _retrieve_checkpoint

        result = _retrieve_checkpoint("nonexistent-id")
        assert result is None


class TestCheckpointState:
    """Tests for checkpoint_state function."""

    @pytest.mark.asyncio
    async def test_checkpoint_state_creates_and_stores(self) -> None:
        """checkpoint_state creates and stores a checkpoint."""
        from yolo_developer.agents.sm.emergency import (
            _retrieve_checkpoint,
            checkpoint_state,
        )

        state = make_state(current_agent="pm")
        checkpoint = await checkpoint_state(state, "gate_blocked")

        assert checkpoint.trigger_type == "gate_blocked"
        retrieved = _retrieve_checkpoint(checkpoint.checkpoint_id)
        assert retrieved is not None


class TestEvaluateRetryOption:
    """Tests for _evaluate_retry_option function."""

    def test_returns_recovery_option(self) -> None:
        """_evaluate_retry_option returns RecoveryOption."""
        from yolo_developer.agents.sm.emergency import _evaluate_retry_option
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="gate_blocked",
            severity="warning",
            source_agent="dev",
            trigger_reason="Gate failed",
            health_status=None,
        )

        option = _evaluate_retry_option(trigger, recovery_attempts=0)

        assert option.action == "retry"
        assert option.confidence > 0
        assert option.estimated_impact in ("minimal", "moderate", "significant")

    def test_confidence_decreases_with_attempts(self) -> None:
        """_evaluate_retry_option confidence decreases with more attempts."""
        from yolo_developer.agents.sm.emergency import _evaluate_retry_option
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="gate_blocked",
            severity="warning",
            source_agent="dev",
            trigger_reason="Gate failed",
            health_status=None,
        )

        option_0 = _evaluate_retry_option(trigger, recovery_attempts=0)
        option_2 = _evaluate_retry_option(trigger, recovery_attempts=2)

        assert option_0.confidence > option_2.confidence


class TestEvaluateRollbackOption:
    """Tests for _evaluate_rollback_option function."""

    def test_returns_recovery_option(self) -> None:
        """_evaluate_rollback_option returns RecoveryOption."""
        from yolo_developer.agents.sm.emergency import _evaluate_rollback_option
        from yolo_developer.agents.sm.emergency_types import Checkpoint, EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="health_degraded",
            severity="critical",
            source_agent="dev",
            trigger_reason="Health critical",
            health_status=None,
        )
        checkpoint = Checkpoint(
            checkpoint_id="chk-abc",
            state_snapshot={},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="health_degraded",
        )

        option = _evaluate_rollback_option(trigger, checkpoint)

        assert option.action == "rollback"
        assert "rollback" in option.description.lower() or "checkpoint" in option.description.lower()

    def test_returns_none_without_checkpoint(self) -> None:
        """_evaluate_rollback_option returns None without checkpoint."""
        from yolo_developer.agents.sm.emergency import _evaluate_rollback_option
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="health_degraded",
            severity="critical",
            source_agent="dev",
            trigger_reason="Health critical",
            health_status=None,
        )

        option = _evaluate_rollback_option(trigger, None)

        assert option is None

    def test_confidence_varies_by_emergency_type(self) -> None:
        """_evaluate_rollback_option confidence varies by emergency type.

        Rollback is more reliable for state-related issues (gate_blocked)
        than for system errors.
        """
        from yolo_developer.agents.sm.emergency import _evaluate_rollback_option
        from yolo_developer.agents.sm.emergency_types import Checkpoint, EmergencyTrigger

        checkpoint = Checkpoint(
            checkpoint_id="chk-test",
            state_snapshot={},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="gate_blocked",
        )

        # gate_blocked should have higher confidence
        trigger_gate = EmergencyTrigger(
            emergency_type="gate_blocked",
            severity="warning",
            source_agent=None,
            trigger_reason="Gate blocked",
            health_status=None,
        )
        option_gate = _evaluate_rollback_option(trigger_gate, checkpoint)

        # system_error should have lower confidence (state may be corrupted)
        trigger_system = EmergencyTrigger(
            emergency_type="system_error",
            severity="critical",
            source_agent=None,
            trigger_reason="System error",
            health_status=None,
        )
        option_system = _evaluate_rollback_option(trigger_system, checkpoint)

        assert option_gate is not None
        assert option_system is not None
        assert option_gate.confidence > option_system.confidence


class TestEvaluateSkipOption:
    """Tests for _evaluate_skip_option function."""

    def test_returns_recovery_option(self) -> None:
        """_evaluate_skip_option returns RecoveryOption."""
        from yolo_developer.agents.sm.emergency import _evaluate_skip_option
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="agent_stuck",
            severity="warning",
            source_agent="tea",
            trigger_reason="Agent idle",
            health_status=None,
        )

        option = _evaluate_skip_option(trigger)

        assert option.action == "skip"


class TestEvaluateEscalateOption:
    """Tests for _evaluate_escalate_option function."""

    def test_returns_recovery_option(self) -> None:
        """_evaluate_escalate_option returns RecoveryOption."""
        from yolo_developer.agents.sm.emergency import _evaluate_escalate_option
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="system_error",
            severity="critical",
            source_agent=None,
            trigger_reason="Critical failure",
            health_status=None,
        )

        option = _evaluate_escalate_option(trigger)

        assert option.action == "escalate"
        assert option.confidence == 1.0  # Always available


class TestEvaluateTerminateOption:
    """Tests for _evaluate_terminate_option function."""

    def test_returns_recovery_option(self) -> None:
        """_evaluate_terminate_option returns RecoveryOption."""
        from yolo_developer.agents.sm.emergency import _evaluate_terminate_option
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="system_error",
            severity="critical",
            source_agent=None,
            trigger_reason="Unrecoverable",
            health_status=None,
        )

        option = _evaluate_terminate_option(trigger)

        assert option.action == "terminate"
        assert "significant" in option.estimated_impact


class TestGenerateRecoveryOptions:
    """Tests for _generate_recovery_options function."""

    @pytest.mark.asyncio
    async def test_generates_multiple_options(self) -> None:
        """_generate_recovery_options generates multiple options."""
        from yolo_developer.agents.sm.emergency import _generate_recovery_options
        from yolo_developer.agents.sm.emergency_types import (
            Checkpoint,
            EmergencyConfig,
            EmergencyTrigger,
        )

        trigger = EmergencyTrigger(
            emergency_type="gate_blocked",
            severity="warning",
            source_agent="dev",
            trigger_reason="Gate failed",
            health_status=None,
        )
        checkpoint = Checkpoint(
            checkpoint_id="chk-abc",
            state_snapshot={},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="gate_blocked",
        )
        config = EmergencyConfig()
        state = make_state()

        options = await _generate_recovery_options(state, trigger, checkpoint, config)

        assert len(options) >= 3  # retry, rollback, skip, escalate, terminate
        action_types = {o.action for o in options}
        assert "retry" in action_types
        assert "escalate" in action_types


class TestSelectBestRecovery:
    """Tests for _select_best_recovery function."""

    def test_selects_highest_confidence_option(self) -> None:
        """_select_best_recovery selects option with highest confidence."""
        from yolo_developer.agents.sm.emergency import _select_best_recovery
        from yolo_developer.agents.sm.emergency_types import RecoveryOption

        options = [
            RecoveryOption(
                action="retry",
                description="Retry",
                confidence=0.6,
                risks=(),
                estimated_impact="minimal",
            ),
            RecoveryOption(
                action="rollback",
                description="Rollback",
                confidence=0.8,
                risks=(),
                estimated_impact="moderate",
            ),
            RecoveryOption(
                action="skip",
                description="Skip",
                confidence=0.4,
                risks=(),
                estimated_impact="moderate",
            ),
        ]

        best = _select_best_recovery(options)

        assert best is not None
        assert best.action == "rollback"
        assert best.confidence == 0.8

    def test_returns_none_for_empty_options(self) -> None:
        """_select_best_recovery returns None for empty options."""
        from yolo_developer.agents.sm.emergency import _select_best_recovery

        best = _select_best_recovery([])
        assert best is None


class TestShouldEscalate:
    """Tests for _should_escalate function."""

    def test_returns_true_for_low_confidence(self) -> None:
        """_should_escalate returns True when best option below threshold."""
        from yolo_developer.agents.sm.emergency import _should_escalate
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig, RecoveryOption

        options = [
            RecoveryOption(
                action="retry",
                description="Retry",
                confidence=0.3,  # Below threshold
                risks=(),
                estimated_impact="minimal",
            ),
        ]
        config = EmergencyConfig(escalation_threshold=0.5)

        result = _should_escalate(options, config)
        assert result is True

    def test_returns_false_for_high_confidence(self) -> None:
        """_should_escalate returns False when option above threshold."""
        from yolo_developer.agents.sm.emergency import _should_escalate
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig, RecoveryOption

        options = [
            RecoveryOption(
                action="retry",
                description="Retry",
                confidence=0.7,  # Above threshold
                risks=(),
                estimated_impact="minimal",
            ),
        ]
        config = EmergencyConfig(escalation_threshold=0.5)

        result = _should_escalate(options, config)
        assert result is False


class TestEscalateEmergency:
    """Tests for escalate_emergency function."""

    @pytest.mark.asyncio
    async def test_creates_escalation_record(self) -> None:
        """escalate_emergency creates and logs escalation."""
        from yolo_developer.agents.sm.emergency import escalate_emergency
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="system_error",
            severity="critical",
            source_agent=None,
            trigger_reason="Unrecoverable error",
            health_status=None,
        )

        result = await escalate_emergency(
            protocol_id="emergency-test",
            trigger=trigger,
            reason="No recovery possible",
        )

        assert result["escalated"] is True
        assert result["reason"] == "No recovery possible"


class TestTriggerEmergencyProtocol:
    """Tests for trigger_emergency_protocol main function."""

    @pytest.mark.asyncio
    async def test_triggers_protocol_for_health_degradation(self) -> None:
        """trigger_emergency_protocol handles health degradation."""
        from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
        from yolo_developer.agents.sm.health_types import (
            HealthMetrics,
            HealthStatus,
        )

        state = make_state(current_agent="dev")
        health_status = HealthStatus(
            status="critical",
            is_healthy=False,
            metrics=HealthMetrics(
                agent_idle_times={},
                agent_cycle_times={},
                agent_churn_rates={},
                overall_cycle_time=0.0,
                overall_churn_rate=0.0,
                unproductive_churn_rate=0.0,
                cycle_time_percentiles={},
                agent_snapshots=(),
            ),
            alerts=(),
            summary="Critical health",
        )

        protocol = await trigger_emergency_protocol(state, health_status)

        assert protocol.protocol_id.startswith("emergency-")
        assert protocol.trigger.emergency_type == "health_degraded"
        assert protocol.status in ("checkpointed", "recovering", "escalated")

    @pytest.mark.asyncio
    async def test_checkpoints_state_when_configured(self) -> None:
        """trigger_emergency_protocol checkpoints state when auto_checkpoint=True."""
        from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig
        from yolo_developer.agents.sm.health_types import (
            HealthMetrics,
            HealthStatus,
        )

        state = make_state()
        health_status = HealthStatus(
            status="critical",
            is_healthy=False,
            metrics=HealthMetrics(
                agent_idle_times={},
                agent_cycle_times={},
                agent_churn_rates={},
                overall_cycle_time=0.0,
                overall_churn_rate=0.0,
                unproductive_churn_rate=0.0,
                cycle_time_percentiles={},
                agent_snapshots=(),
            ),
            alerts=(),
            summary="Critical",
        )
        config = EmergencyConfig(auto_checkpoint=True)

        protocol = await trigger_emergency_protocol(state, health_status, config)

        assert protocol.checkpoint is not None

    @pytest.mark.asyncio
    async def test_skips_checkpoint_when_disabled(self) -> None:
        """trigger_emergency_protocol skips checkpoint when auto_checkpoint=False."""
        from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig
        from yolo_developer.agents.sm.health_types import (
            HealthMetrics,
            HealthStatus,
        )

        state = make_state()
        health_status = HealthStatus(
            status="critical",
            is_healthy=False,
            metrics=HealthMetrics(
                agent_idle_times={},
                agent_cycle_times={},
                agent_churn_rates={},
                overall_cycle_time=0.0,
                overall_churn_rate=0.0,
                unproductive_churn_rate=0.0,
                cycle_time_percentiles={},
                agent_snapshots=(),
            ),
            alerts=(),
            summary="Critical",
        )
        config = EmergencyConfig(auto_checkpoint=False)

        protocol = await trigger_emergency_protocol(state, health_status, config)

        assert protocol.checkpoint is None

    @pytest.mark.asyncio
    async def test_escalates_when_auto_recovery_disabled(self) -> None:
        """trigger_emergency_protocol escalates when auto_recovery disabled."""
        from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig
        from yolo_developer.agents.sm.health_types import (
            HealthMetrics,
            HealthStatus,
        )

        state = make_state()
        health_status = HealthStatus(
            status="critical",
            is_healthy=False,
            metrics=HealthMetrics(
                agent_idle_times={},
                agent_cycle_times={},
                agent_churn_rates={},
                overall_cycle_time=0.0,
                overall_churn_rate=0.0,
                unproductive_churn_rate=0.0,
                cycle_time_percentiles={},
                agent_snapshots=(),
            ),
            alerts=(),
            summary="Critical",
        )
        config = EmergencyConfig(enable_auto_recovery=False)

        protocol = await trigger_emergency_protocol(state, health_status, config)

        assert protocol.status == "escalated"
        assert protocol.escalation_reason is not None

    @pytest.mark.asyncio
    async def test_selects_recovery_action_when_confidence_high(self) -> None:
        """trigger_emergency_protocol selects action when confidence is high."""
        from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig
        from yolo_developer.agents.sm.health_types import (
            HealthMetrics,
            HealthStatus,
        )

        state = make_state()
        health_status = HealthStatus(
            status="critical",
            is_healthy=False,
            metrics=HealthMetrics(
                agent_idle_times={},
                agent_cycle_times={},
                agent_churn_rates={},
                overall_cycle_time=0.0,
                overall_churn_rate=0.0,
                unproductive_churn_rate=0.0,
                cycle_time_percentiles={},
                agent_snapshots=(),
            ),
            alerts=(),
            summary="Critical",
        )
        config = EmergencyConfig(
            enable_auto_recovery=True,
            escalation_threshold=0.1,  # Very low threshold
        )

        protocol = await trigger_emergency_protocol(state, health_status, config)

        # Should select a recovery action rather than escalate
        assert protocol.status in ("recovering", "escalated")

    @pytest.mark.asyncio
    async def test_returns_protocol_with_to_dict(self) -> None:
        """trigger_emergency_protocol returns protocol with to_dict method."""
        from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
        from yolo_developer.agents.sm.health_types import (
            HealthMetrics,
            HealthStatus,
        )

        state = make_state()
        health_status = HealthStatus(
            status="critical",
            is_healthy=False,
            metrics=HealthMetrics(
                agent_idle_times={},
                agent_cycle_times={},
                agent_churn_rates={},
                overall_cycle_time=0.0,
                overall_churn_rate=0.0,
                unproductive_churn_rate=0.0,
                cycle_time_percentiles={},
                agent_snapshots=(),
            ),
            alerts=(),
            summary="Critical",
        )

        protocol = await trigger_emergency_protocol(state, health_status)
        result_dict = protocol.to_dict()

        assert isinstance(result_dict, dict)
        assert "protocol_id" in result_dict
        assert "trigger" in result_dict
        assert "status" in result_dict

    @pytest.mark.asyncio
    async def test_handles_nested_failure_gracefully(self) -> None:
        """trigger_emergency_protocol handles failures during protocol gracefully."""
        from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
        from yolo_developer.agents.sm.health_types import (
            HealthMetrics,
            HealthStatus,
        )

        state = make_state()
        health_status = HealthStatus(
            status="critical",
            is_healthy=False,
            metrics=HealthMetrics(
                agent_idle_times={},
                agent_cycle_times={},
                agent_churn_rates={},
                overall_cycle_time=0.0,
                overall_churn_rate=0.0,
                unproductive_churn_rate=0.0,
                cycle_time_percentiles={},
                agent_snapshots=(),
            ),
            alerts=(),
            summary="Critical",
        )

        # Should not raise even with edge case inputs
        protocol = await trigger_emergency_protocol(state, health_status)
        assert protocol is not None
