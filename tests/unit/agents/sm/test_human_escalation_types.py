"""Tests for human escalation type definitions (Story 10.14).

This module tests the type definitions used by the human escalation module:
- EscalationTrigger Literal type validation
- EscalationStatus Literal type validation
- EscalationOption frozen dataclass
- EscalationRequest frozen dataclass
- EscalationResponse frozen dataclass
- EscalationResult frozen dataclass
- EscalationConfig frozen dataclass
- Constants and validation

Test Categories:
- Valid value tests: Confirm types accept valid values
- Invalid value tests: Confirm warnings logged for invalid values
- Edge case tests: Empty strings, boundary values
- Serialization tests: to_dict() methods
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from yolo_developer.agents.sm.human_escalation_types import (
    DEFAULT_ESCALATION_TIMEOUT_SECONDS,
    DEFAULT_LOG_ESCALATIONS,
    DEFAULT_MAX_PENDING,
    MAX_DURATION_MS,
    MIN_DURATION_MS,
    VALID_ESCALATION_STATUSES,
    VALID_ESCALATION_TRIGGERS,
    EscalationConfig,
    EscalationOption,
    EscalationRequest,
    EscalationResponse,
    EscalationResult,
)


class TestConstants:
    """Tests for module constants."""

    def test_default_timeout_seconds_is_positive(self) -> None:
        """Default timeout should be a positive value."""
        assert DEFAULT_ESCALATION_TIMEOUT_SECONDS > 0

    def test_default_log_escalations_is_bool(self) -> None:
        """Default log escalations should be a boolean."""
        assert isinstance(DEFAULT_LOG_ESCALATIONS, bool)

    def test_default_max_pending_is_positive(self) -> None:
        """Default max pending should be a positive value."""
        assert DEFAULT_MAX_PENDING > 0

    def test_valid_escalation_triggers_is_frozenset(self) -> None:
        """Valid triggers should be a frozenset."""
        assert isinstance(VALID_ESCALATION_TRIGGERS, frozenset)
        assert len(VALID_ESCALATION_TRIGGERS) > 0

    def test_valid_escalation_triggers_contains_expected_values(self) -> None:
        """Valid triggers should include expected values."""
        expected = {
            "circular_logic",
            "conflict_unresolved",
            "gate_blocked",
            "system_error",
            "agent_stuck",
            "user_requested",
        }
        assert expected == VALID_ESCALATION_TRIGGERS

    def test_valid_escalation_statuses_is_frozenset(self) -> None:
        """Valid statuses should be a frozenset."""
        assert isinstance(VALID_ESCALATION_STATUSES, frozenset)
        assert len(VALID_ESCALATION_STATUSES) > 0

    def test_valid_escalation_statuses_contains_expected_values(self) -> None:
        """Valid statuses should include expected values."""
        expected = {"pending", "presented", "resolved", "timed_out", "cancelled"}
        assert expected == VALID_ESCALATION_STATUSES


class TestEscalationOption:
    """Tests for EscalationOption dataclass."""

    def test_create_valid_option(self) -> None:
        """Should create a valid escalation option."""
        option = EscalationOption(
            option_id="opt-1",
            label="Retry",
            description="Retry the failed operation",
            action="retry",
            is_recommended=True,
        )
        assert option.option_id == "opt-1"
        assert option.label == "Retry"
        assert option.description == "Retry the failed operation"
        assert option.action == "retry"
        assert option.is_recommended is True

    def test_option_is_frozen(self) -> None:
        """Option should be immutable."""
        option = EscalationOption(
            option_id="opt-1",
            label="Retry",
            description="Retry the failed operation",
            action="retry",
            is_recommended=False,
        )
        with pytest.raises(AttributeError):
            option.label = "Changed"  # type: ignore[misc]

    def test_option_to_dict(self) -> None:
        """to_dict should serialize all fields."""
        option = EscalationOption(
            option_id="opt-1",
            label="Retry",
            description="Retry the failed operation",
            action="retry",
            is_recommended=True,
        )
        result = option.to_dict()
        assert result == {
            "option_id": "opt-1",
            "label": "Retry",
            "description": "Retry the failed operation",
            "action": "retry",
            "is_recommended": True,
        }

    def test_option_empty_id_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty option_id should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationOption(
                option_id="",
                label="Retry",
                description="Test",
                action="retry",
                is_recommended=False,
            )
        assert "option_id is empty" in caplog.text

    def test_option_empty_label_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty label should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationOption(
                option_id="opt-1",
                label="",
                description="Test",
                action="retry",
                is_recommended=False,
            )
        assert "label is empty" in caplog.text


class TestEscalationRequest:
    """Tests for EscalationRequest dataclass."""

    def test_create_valid_request(self) -> None:
        """Should create a valid escalation request."""
        option = EscalationOption(
            option_id="opt-1",
            label="Retry",
            description="Retry",
            action="retry",
            is_recommended=True,
        )
        request = EscalationRequest(
            request_id="esc-12345",
            trigger="circular_logic",
            agent="architect",
            summary="Circular logic detected between architect and pm",
            context={"exchanges": 5},
            options=(option,),
            recommended_option="opt-1",
        )
        assert request.request_id == "esc-12345"
        assert request.trigger == "circular_logic"
        assert request.agent == "architect"
        assert len(request.options) == 1
        assert request.recommended_option == "opt-1"

    def test_request_is_frozen(self) -> None:
        """Request should be immutable."""
        request = EscalationRequest(
            request_id="esc-12345",
            trigger="circular_logic",
            agent="architect",
            summary="Test",
            context={},
            options=(),
            recommended_option=None,
        )
        with pytest.raises(AttributeError):
            request.agent = "pm"  # type: ignore[misc]

    def test_request_auto_generates_timestamp(self) -> None:
        """Request should auto-generate created_at timestamp."""
        request = EscalationRequest(
            request_id="esc-12345",
            trigger="circular_logic",
            agent="architect",
            summary="Test",
            context={},
            options=(),
            recommended_option=None,
        )
        # Verify it's a valid ISO timestamp
        parsed = datetime.fromisoformat(request.created_at)
        assert parsed.tzinfo == timezone.utc

    def test_request_to_dict(self) -> None:
        """to_dict should serialize all fields including options."""
        option = EscalationOption(
            option_id="opt-1",
            label="Retry",
            description="Retry",
            action="retry",
            is_recommended=True,
        )
        request = EscalationRequest(
            request_id="esc-12345",
            trigger="circular_logic",
            agent="architect",
            summary="Test summary",
            context={"key": "value"},
            options=(option,),
            recommended_option="opt-1",
        )
        result = request.to_dict()
        assert result["request_id"] == "esc-12345"
        assert result["trigger"] == "circular_logic"
        assert result["agent"] == "architect"
        assert result["summary"] == "Test summary"
        assert result["context"] == {"key": "value"}
        assert len(result["options"]) == 1
        assert result["options"][0]["option_id"] == "opt-1"
        assert result["recommended_option"] == "opt-1"
        assert "created_at" in result

    def test_request_invalid_trigger_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Invalid trigger should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationRequest(
                request_id="esc-12345",
                trigger="invalid_trigger",  # type: ignore[arg-type]
                agent="architect",
                summary="Test",
                context={},
                options=(),
                recommended_option=None,
            )
        assert "not a valid trigger" in caplog.text

    def test_request_empty_id_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty request_id should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationRequest(
                request_id="",
                trigger="circular_logic",
                agent="architect",
                summary="Test",
                context={},
                options=(),
                recommended_option=None,
            )
        assert "request_id is empty" in caplog.text


class TestEscalationResponse:
    """Tests for EscalationResponse dataclass."""

    def test_create_valid_response(self) -> None:
        """Should create a valid escalation response."""
        response = EscalationResponse(
            request_id="esc-12345",
            selected_option="opt-1",
            user_rationale="User decided to retry",
        )
        assert response.request_id == "esc-12345"
        assert response.selected_option == "opt-1"
        assert response.user_rationale == "User decided to retry"

    def test_response_is_frozen(self) -> None:
        """Response should be immutable."""
        response = EscalationResponse(
            request_id="esc-12345",
            selected_option="opt-1",
            user_rationale=None,
        )
        with pytest.raises(AttributeError):
            response.selected_option = "opt-2"  # type: ignore[misc]

    def test_response_auto_generates_timestamp(self) -> None:
        """Response should auto-generate responded_at timestamp."""
        response = EscalationResponse(
            request_id="esc-12345",
            selected_option="opt-1",
            user_rationale=None,
        )
        parsed = datetime.fromisoformat(response.responded_at)
        assert parsed.tzinfo == timezone.utc

    def test_response_to_dict(self) -> None:
        """to_dict should serialize all fields."""
        response = EscalationResponse(
            request_id="esc-12345",
            selected_option="opt-1",
            user_rationale="Because reasons",
        )
        result = response.to_dict()
        assert result["request_id"] == "esc-12345"
        assert result["selected_option"] == "opt-1"
        assert result["user_rationale"] == "Because reasons"
        assert "responded_at" in result

    def test_response_empty_request_id_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty request_id should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationResponse(
                request_id="",
                selected_option="opt-1",
                user_rationale=None,
            )
        assert "request_id is empty" in caplog.text

    def test_response_empty_selected_option_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty selected_option should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationResponse(
                request_id="esc-12345",
                selected_option="",
                user_rationale=None,
            )
        assert "selected_option is empty" in caplog.text


class TestEscalationResult:
    """Tests for EscalationResult dataclass."""

    @pytest.fixture
    def sample_request(self) -> EscalationRequest:
        """Create a sample request for testing."""
        option = EscalationOption(
            option_id="opt-1",
            label="Retry",
            description="Retry",
            action="retry",
            is_recommended=True,
        )
        return EscalationRequest(
            request_id="esc-12345",
            trigger="circular_logic",
            agent="architect",
            summary="Test",
            context={},
            options=(option,),
            recommended_option="opt-1",
        )

    @pytest.fixture
    def sample_response(self) -> EscalationResponse:
        """Create a sample response for testing."""
        return EscalationResponse(
            request_id="esc-12345",
            selected_option="opt-1",
            user_rationale="Test rationale",
        )

    def test_create_valid_result(
        self,
        sample_request: EscalationRequest,
        sample_response: EscalationResponse,
    ) -> None:
        """Should create a valid escalation result."""
        result = EscalationResult(
            request=sample_request,
            response=sample_response,
            status="resolved",
            resolution_action="retry",
            integration_success=True,
            duration_ms=150.0,
        )
        assert result.request == sample_request
        assert result.response == sample_response
        assert result.status == "resolved"
        assert result.resolution_action == "retry"
        assert result.integration_success is True
        assert result.duration_ms == 150.0

    def test_result_is_frozen(
        self,
        sample_request: EscalationRequest,
        sample_response: EscalationResponse,
    ) -> None:
        """Result should be immutable."""
        result = EscalationResult(
            request=sample_request,
            response=sample_response,
            status="resolved",
            resolution_action="retry",
            integration_success=True,
            duration_ms=150.0,
        )
        with pytest.raises(AttributeError):
            result.status = "pending"  # type: ignore[misc]

    def test_result_to_dict(
        self,
        sample_request: EscalationRequest,
        sample_response: EscalationResponse,
    ) -> None:
        """to_dict should serialize all fields including nested objects."""
        result = EscalationResult(
            request=sample_request,
            response=sample_response,
            status="resolved",
            resolution_action="retry",
            integration_success=True,
            duration_ms=150.0,
        )
        d = result.to_dict()
        assert d["status"] == "resolved"
        assert d["resolution_action"] == "retry"
        assert d["integration_success"] is True
        assert d["duration_ms"] == 150.0
        assert d["request"]["request_id"] == "esc-12345"
        assert d["response"]["selected_option"] == "opt-1"

    def test_result_invalid_status_warns(
        self,
        sample_request: EscalationRequest,
        sample_response: EscalationResponse,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Invalid status should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationResult(
                request=sample_request,
                response=sample_response,
                status="invalid",  # type: ignore[arg-type]
                resolution_action="retry",
                integration_success=True,
                duration_ms=150.0,
            )
        assert "not a valid status" in caplog.text

    def test_result_negative_duration_warns(
        self,
        sample_request: EscalationRequest,
        sample_response: EscalationResponse,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Negative duration_ms should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationResult(
                request=sample_request,
                response=sample_response,
                status="resolved",
                resolution_action="retry",
                integration_success=True,
                duration_ms=-10.0,
            )
        assert "duration_ms" in caplog.text and "negative" in caplog.text.lower()

    def test_result_with_none_response(
        self,
        sample_request: EscalationRequest,
    ) -> None:
        """Result should allow None response for timeouts."""
        result = EscalationResult(
            request=sample_request,
            response=None,
            status="timed_out",
            resolution_action="skip",
            integration_success=False,
            duration_ms=30000.0,
        )
        assert result.response is None
        assert result.status == "timed_out"

    def test_result_to_dict_with_none_response(
        self,
        sample_request: EscalationRequest,
    ) -> None:
        """to_dict should handle None response."""
        result = EscalationResult(
            request=sample_request,
            response=None,
            status="timed_out",
            resolution_action="skip",
            integration_success=False,
            duration_ms=30000.0,
        )
        d = result.to_dict()
        assert d["response"] is None
        assert d["status"] == "timed_out"


class TestEscalationConfig:
    """Tests for EscalationConfig dataclass."""

    def test_create_with_defaults(self) -> None:
        """Should create config with default values."""
        config = EscalationConfig()
        assert config.timeout_seconds == DEFAULT_ESCALATION_TIMEOUT_SECONDS
        assert config.default_action == "skip"
        assert config.log_escalations == DEFAULT_LOG_ESCALATIONS
        assert config.max_pending == DEFAULT_MAX_PENDING

    def test_create_with_custom_values(self) -> None:
        """Should create config with custom values."""
        config = EscalationConfig(
            timeout_seconds=60,
            default_action="escalate",
            log_escalations=False,
            max_pending=10,
        )
        assert config.timeout_seconds == 60
        assert config.default_action == "escalate"
        assert config.log_escalations is False
        assert config.max_pending == 10

    def test_config_is_frozen(self) -> None:
        """Config should be immutable."""
        config = EscalationConfig()
        with pytest.raises(AttributeError):
            config.timeout_seconds = 100  # type: ignore[misc]

    def test_config_to_dict(self) -> None:
        """to_dict should serialize all fields."""
        config = EscalationConfig(
            timeout_seconds=60,
            default_action="retry",
            log_escalations=True,
            max_pending=5,
        )
        result = config.to_dict()
        assert result == {
            "timeout_seconds": 60,
            "default_action": "retry",
            "log_escalations": True,
            "max_pending": 5,
        }

    def test_config_negative_timeout_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Negative timeout should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationConfig(timeout_seconds=-10)
        assert "timeout_seconds" in caplog.text

    def test_config_zero_max_pending_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Zero max_pending should log warning."""
        with caplog.at_level(logging.WARNING):
            EscalationConfig(max_pending=0)
        assert "max_pending" in caplog.text


class TestDurationBounds:
    """Tests for duration bounds constants."""

    def test_min_duration_is_zero(self) -> None:
        """Min duration should be zero."""
        assert MIN_DURATION_MS == 0.0

    def test_max_duration_is_reasonable(self) -> None:
        """Max duration should be a reasonable large value."""
        # 1 hour in milliseconds
        assert MAX_DURATION_MS >= 3_600_000
