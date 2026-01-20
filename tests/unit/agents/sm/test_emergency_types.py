"""Tests for emergency protocol types module (Story 10.10).

Tests the type definitions and serialization for emergency protocols:
- EmergencyType, ProtocolStatus, RecoveryAction Literal types
- EmergencyTrigger, Checkpoint, RecoveryOption frozen dataclasses
- EmergencyProtocol, EmergencyConfig frozen dataclasses
- Constants and validation sets
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


class TestEmergencyTypeConstants:
    """Tests for emergency type constants and validation sets."""

    def test_valid_emergency_types_contains_all_types(self) -> None:
        """VALID_EMERGENCY_TYPES contains all expected emergency types."""
        from yolo_developer.agents.sm.emergency_types import VALID_EMERGENCY_TYPES

        expected = {
            "health_degraded",
            "circular_logic",
            "gate_blocked",
            "agent_stuck",
            "system_error",
        }
        assert VALID_EMERGENCY_TYPES == frozenset(expected)

    def test_valid_protocol_statuses_contains_all_statuses(self) -> None:
        """VALID_PROTOCOL_STATUSES contains all expected statuses."""
        from yolo_developer.agents.sm.emergency_types import VALID_PROTOCOL_STATUSES

        expected = {"pending", "active", "checkpointed", "recovering", "resolved", "escalated"}
        assert VALID_PROTOCOL_STATUSES == frozenset(expected)

    def test_valid_recovery_actions_contains_all_actions(self) -> None:
        """VALID_RECOVERY_ACTIONS contains all expected actions."""
        from yolo_developer.agents.sm.emergency_types import VALID_RECOVERY_ACTIONS

        expected = {"retry", "rollback", "skip", "escalate", "terminate"}
        assert VALID_RECOVERY_ACTIONS == frozenset(expected)

    def test_default_max_recovery_attempts(self) -> None:
        """DEFAULT_MAX_RECOVERY_ATTEMPTS is 3."""
        from yolo_developer.agents.sm.emergency_types import DEFAULT_MAX_RECOVERY_ATTEMPTS

        assert DEFAULT_MAX_RECOVERY_ATTEMPTS == 3

    def test_default_escalation_threshold(self) -> None:
        """DEFAULT_ESCALATION_THRESHOLD is 0.5."""
        from yolo_developer.agents.sm.emergency_types import DEFAULT_ESCALATION_THRESHOLD

        assert DEFAULT_ESCALATION_THRESHOLD == 0.5


class TestEmergencyTrigger:
    """Tests for EmergencyTrigger dataclass."""

    def test_create_emergency_trigger(self) -> None:
        """EmergencyTrigger can be created with required fields."""
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="health_degraded",
            severity="critical",
            source_agent="dev",
            trigger_reason="System health fell below critical thresholds",
            health_status={"status": "critical"},
        )
        assert trigger.emergency_type == "health_degraded"
        assert trigger.severity == "critical"
        assert trigger.source_agent == "dev"
        assert trigger.trigger_reason == "System health fell below critical thresholds"
        assert trigger.health_status == {"status": "critical"}
        assert trigger.detected_at is not None

    def test_emergency_trigger_is_frozen(self) -> None:
        """EmergencyTrigger is immutable."""
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="health_degraded",
            severity="critical",
            source_agent=None,
            trigger_reason="Test",
            health_status=None,
        )
        with pytest.raises(AttributeError):
            trigger.severity = "warning"  # type: ignore[misc]

    def test_emergency_trigger_to_dict(self) -> None:
        """EmergencyTrigger.to_dict() returns expected dictionary."""
        from yolo_developer.agents.sm.emergency_types import EmergencyTrigger

        trigger = EmergencyTrigger(
            emergency_type="circular_logic",
            severity="warning",
            source_agent="analyst",
            trigger_reason="Agents in ping-pong loop",
            health_status=None,
            detected_at="2026-01-16T10:00:00+00:00",
        )
        result = trigger.to_dict()
        assert result == {
            "emergency_type": "circular_logic",
            "severity": "warning",
            "source_agent": "analyst",
            "trigger_reason": "Agents in ping-pong loop",
            "health_status": None,
            "detected_at": "2026-01-16T10:00:00+00:00",
        }

    def test_emergency_trigger_all_types(self) -> None:
        """EmergencyTrigger accepts all valid emergency types."""
        from yolo_developer.agents.sm.emergency_types import VALID_EMERGENCY_TYPES, EmergencyTrigger

        for emergency_type in VALID_EMERGENCY_TYPES:
            trigger = EmergencyTrigger(
                emergency_type=emergency_type,  # type: ignore[arg-type]
                severity="critical",
                source_agent=None,
                trigger_reason="Test",
                health_status=None,
            )
            assert trigger.emergency_type == emergency_type


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_create_checkpoint(self) -> None:
        """Checkpoint can be created with required fields."""
        from yolo_developer.agents.sm.emergency_types import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="chk-12345678",
            state_snapshot={"current_agent": "dev", "messages": []},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="health_degraded",
        )
        assert checkpoint.checkpoint_id == "chk-12345678"
        assert checkpoint.state_snapshot == {"current_agent": "dev", "messages": []}
        assert checkpoint.created_at == "2026-01-16T10:00:00+00:00"
        assert checkpoint.trigger_type == "health_degraded"
        assert checkpoint.metadata == {}

    def test_checkpoint_with_metadata(self) -> None:
        """Checkpoint can include optional metadata."""
        from yolo_developer.agents.sm.emergency_types import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="chk-12345678",
            state_snapshot={},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="system_error",
            metadata={"recovery_attempt": 1, "error": "Connection timeout"},
        )
        assert checkpoint.metadata == {"recovery_attempt": 1, "error": "Connection timeout"}

    def test_checkpoint_is_frozen(self) -> None:
        """Checkpoint is immutable."""
        from yolo_developer.agents.sm.emergency_types import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="chk-12345678",
            state_snapshot={},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="health_degraded",
        )
        with pytest.raises(AttributeError):
            checkpoint.checkpoint_id = "new-id"  # type: ignore[misc]

    def test_checkpoint_to_dict(self) -> None:
        """Checkpoint.to_dict() returns expected dictionary."""
        from yolo_developer.agents.sm.emergency_types import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="chk-abc",
            state_snapshot={"key": "value"},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="gate_blocked",
            metadata={"attempt": 2},
        )
        result = checkpoint.to_dict()
        assert result == {
            "checkpoint_id": "chk-abc",
            "state_snapshot": {"key": "value"},
            "created_at": "2026-01-16T10:00:00+00:00",
            "trigger_type": "gate_blocked",
            "metadata": {"attempt": 2},
        }


class TestRecoveryOption:
    """Tests for RecoveryOption dataclass."""

    def test_create_recovery_option(self) -> None:
        """RecoveryOption can be created with required fields."""
        from yolo_developer.agents.sm.emergency_types import RecoveryOption

        option = RecoveryOption(
            action="retry",
            description="Retry the failed operation",
            confidence=0.8,
            risks=("May fail again",),
            estimated_impact="minimal",
        )
        assert option.action == "retry"
        assert option.description == "Retry the failed operation"
        assert option.confidence == 0.8
        assert option.risks == ("May fail again",)
        assert option.estimated_impact == "minimal"

    def test_recovery_option_is_frozen(self) -> None:
        """RecoveryOption is immutable."""
        from yolo_developer.agents.sm.emergency_types import RecoveryOption

        option = RecoveryOption(
            action="retry",
            description="Test",
            confidence=0.5,
            risks=(),
            estimated_impact="minimal",
        )
        with pytest.raises(AttributeError):
            option.confidence = 0.9  # type: ignore[misc]

    def test_recovery_option_to_dict(self) -> None:
        """RecoveryOption.to_dict() returns expected dictionary."""
        from yolo_developer.agents.sm.emergency_types import RecoveryOption

        option = RecoveryOption(
            action="rollback",
            description="Rollback to checkpoint",
            confidence=0.7,
            risks=("Data loss", "State inconsistency"),
            estimated_impact="moderate",
        )
        result = option.to_dict()
        assert result == {
            "action": "rollback",
            "description": "Rollback to checkpoint",
            "confidence": 0.7,
            "risks": ["Data loss", "State inconsistency"],
            "estimated_impact": "moderate",
        }

    def test_recovery_option_all_actions(self) -> None:
        """RecoveryOption accepts all valid recovery actions."""
        from yolo_developer.agents.sm.emergency_types import VALID_RECOVERY_ACTIONS, RecoveryOption

        for action in VALID_RECOVERY_ACTIONS:
            option = RecoveryOption(
                action=action,  # type: ignore[arg-type]
                description="Test",
                confidence=0.5,
                risks=(),
                estimated_impact="minimal",
            )
            assert option.action == action


class TestEmergencyProtocol:
    """Tests for EmergencyProtocol dataclass."""

    def test_create_emergency_protocol_minimal(self) -> None:
        """EmergencyProtocol can be created with minimal fields."""
        from yolo_developer.agents.sm.emergency_types import (
            EmergencyProtocol,
            EmergencyTrigger,
        )

        trigger = EmergencyTrigger(
            emergency_type="health_degraded",
            severity="critical",
            source_agent=None,
            trigger_reason="Test",
            health_status=None,
        )
        protocol = EmergencyProtocol(
            protocol_id="emergency-12345678",
            trigger=trigger,
            status="pending",
            checkpoint=None,
            recovery_options=(),
            selected_action=None,
            escalation_reason=None,
        )
        assert protocol.protocol_id == "emergency-12345678"
        assert protocol.trigger == trigger
        assert protocol.status == "pending"
        assert protocol.checkpoint is None
        assert protocol.recovery_options == ()
        assert protocol.selected_action is None
        assert protocol.escalation_reason is None
        assert protocol.created_at is not None
        assert protocol.resolved_at is None

    def test_create_emergency_protocol_full(self) -> None:
        """EmergencyProtocol can be created with all fields."""
        from yolo_developer.agents.sm.emergency_types import (
            Checkpoint,
            EmergencyProtocol,
            EmergencyTrigger,
            RecoveryOption,
        )

        trigger = EmergencyTrigger(
            emergency_type="circular_logic",
            severity="warning",
            source_agent="analyst",
            trigger_reason="Loop detected",
            health_status={"status": "degraded"},
        )
        checkpoint = Checkpoint(
            checkpoint_id="chk-abc",
            state_snapshot={"key": "value"},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="circular_logic",
        )
        option = RecoveryOption(
            action="skip",
            description="Skip the step",
            confidence=0.6,
            risks=("May miss important work",),
            estimated_impact="moderate",
        )
        protocol = EmergencyProtocol(
            protocol_id="emergency-abc",
            trigger=trigger,
            status="recovering",
            checkpoint=checkpoint,
            recovery_options=(option,),
            selected_action="skip",
            escalation_reason=None,
            created_at="2026-01-16T10:00:00+00:00",
            resolved_at="2026-01-16T10:05:00+00:00",
        )
        assert protocol.checkpoint == checkpoint
        assert protocol.recovery_options == (option,)
        assert protocol.selected_action == "skip"
        assert protocol.resolved_at == "2026-01-16T10:05:00+00:00"

    def test_emergency_protocol_is_frozen(self) -> None:
        """EmergencyProtocol is immutable."""
        from yolo_developer.agents.sm.emergency_types import (
            EmergencyProtocol,
            EmergencyTrigger,
        )

        trigger = EmergencyTrigger(
            emergency_type="health_degraded",
            severity="critical",
            source_agent=None,
            trigger_reason="Test",
            health_status=None,
        )
        protocol = EmergencyProtocol(
            protocol_id="emergency-12345678",
            trigger=trigger,
            status="pending",
            checkpoint=None,
            recovery_options=(),
            selected_action=None,
            escalation_reason=None,
        )
        with pytest.raises(AttributeError):
            protocol.status = "resolved"  # type: ignore[misc]

    def test_emergency_protocol_to_dict(self) -> None:
        """EmergencyProtocol.to_dict() returns expected dictionary."""
        from yolo_developer.agents.sm.emergency_types import (
            Checkpoint,
            EmergencyProtocol,
            EmergencyTrigger,
            RecoveryOption,
        )

        trigger = EmergencyTrigger(
            emergency_type="agent_stuck",
            severity="critical",
            source_agent="dev",
            trigger_reason="Agent idle too long",
            health_status=None,
            detected_at="2026-01-16T10:00:00+00:00",
        )
        checkpoint = Checkpoint(
            checkpoint_id="chk-123",
            state_snapshot={"agent": "dev"},
            created_at="2026-01-16T10:00:00+00:00",
            trigger_type="agent_stuck",
            metadata={"idle_time": 600},
        )
        option = RecoveryOption(
            action="escalate",
            description="Escalate to human",
            confidence=1.0,
            risks=(),
            estimated_impact="significant",
        )
        protocol = EmergencyProtocol(
            protocol_id="emergency-xyz",
            trigger=trigger,
            status="escalated",
            checkpoint=checkpoint,
            recovery_options=(option,),
            selected_action=None,
            escalation_reason="No automatic recovery possible",
            created_at="2026-01-16T10:00:00+00:00",
            resolved_at=None,
        )
        result = protocol.to_dict()
        assert result["protocol_id"] == "emergency-xyz"
        assert result["trigger"]["emergency_type"] == "agent_stuck"
        assert result["status"] == "escalated"
        assert result["checkpoint"]["checkpoint_id"] == "chk-123"
        assert len(result["recovery_options"]) == 1
        assert result["recovery_options"][0]["action"] == "escalate"
        assert result["selected_action"] is None
        assert result["escalation_reason"] == "No automatic recovery possible"

    def test_emergency_protocol_to_dict_without_checkpoint(self) -> None:
        """EmergencyProtocol.to_dict() handles None checkpoint."""
        from yolo_developer.agents.sm.emergency_types import (
            EmergencyProtocol,
            EmergencyTrigger,
        )

        trigger = EmergencyTrigger(
            emergency_type="system_error",
            severity="critical",
            source_agent=None,
            trigger_reason="Test",
            health_status=None,
            detected_at="2026-01-16T10:00:00+00:00",
        )
        protocol = EmergencyProtocol(
            protocol_id="emergency-test",
            trigger=trigger,
            status="pending",
            checkpoint=None,
            recovery_options=(),
            selected_action=None,
            escalation_reason=None,
            created_at="2026-01-16T10:00:00+00:00",
        )
        result = protocol.to_dict()
        assert result["checkpoint"] is None


class TestEmergencyConfig:
    """Tests for EmergencyConfig dataclass."""

    def test_create_emergency_config_defaults(self) -> None:
        """EmergencyConfig has expected default values."""
        from yolo_developer.agents.sm.emergency_types import (
            DEFAULT_ESCALATION_THRESHOLD,
            DEFAULT_MAX_RECOVERY_ATTEMPTS,
            EmergencyConfig,
        )

        config = EmergencyConfig()
        assert config.auto_checkpoint is True
        assert config.max_recovery_attempts == DEFAULT_MAX_RECOVERY_ATTEMPTS
        assert config.escalation_threshold == DEFAULT_ESCALATION_THRESHOLD
        assert config.enable_auto_recovery is True

    def test_create_emergency_config_custom(self) -> None:
        """EmergencyConfig can be created with custom values."""
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig

        config = EmergencyConfig(
            auto_checkpoint=False,
            max_recovery_attempts=5,
            escalation_threshold=0.7,
            enable_auto_recovery=False,
        )
        assert config.auto_checkpoint is False
        assert config.max_recovery_attempts == 5
        assert config.escalation_threshold == 0.7
        assert config.enable_auto_recovery is False

    def test_emergency_config_is_frozen(self) -> None:
        """EmergencyConfig is immutable."""
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig

        config = EmergencyConfig()
        with pytest.raises(AttributeError):
            config.max_recovery_attempts = 10  # type: ignore[misc]

    def test_emergency_config_to_dict(self) -> None:
        """EmergencyConfig.to_dict() returns expected dictionary."""
        from yolo_developer.agents.sm.emergency_types import EmergencyConfig

        config = EmergencyConfig(
            auto_checkpoint=True,
            max_recovery_attempts=3,
            escalation_threshold=0.5,
            enable_auto_recovery=True,
        )
        result = config.to_dict()
        assert result == {
            "auto_checkpoint": True,
            "max_recovery_attempts": 3,
            "escalation_threshold": 0.5,
            "enable_auto_recovery": True,
        }


class TestEmergencyTypesExports:
    """Tests for module exports."""

    def test_all_types_importable(self) -> None:
        """All expected types are importable from emergency_types."""
        from yolo_developer.agents.sm.emergency_types import (
            DEFAULT_MAX_RECOVERY_ATTEMPTS,
            VALID_EMERGENCY_TYPES,
            VALID_PROTOCOL_STATUSES,
            VALID_RECOVERY_ACTIONS,
        )

        # Just verify they're importable and are the right type
        assert DEFAULT_MAX_RECOVERY_ATTEMPTS == 3
        assert isinstance(VALID_EMERGENCY_TYPES, frozenset)
        assert isinstance(VALID_PROTOCOL_STATUSES, frozenset)
        assert isinstance(VALID_RECOVERY_ACTIONS, frozenset)

    def test_literal_types_available(self) -> None:
        """Literal types are available for type hints."""
        from yolo_developer.agents.sm.emergency_types import (
            EmergencyType,
            ProtocolStatus,
            RecoveryAction,
        )

        # These are type aliases, just verify they exist
        assert EmergencyType is not None
        assert ProtocolStatus is not None
        assert RecoveryAction is not None
