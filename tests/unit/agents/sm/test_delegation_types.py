"""Tests for delegation types (Story 10.4).

Tests the frozen dataclass types used by the delegation module:
- TaskType literal type
- DelegationRequest dataclass
- DelegationResult dataclass
- DelegationConfig dataclass
- Agent expertise mappings

All tests verify immutability, serialization, and default values.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from yolo_developer.agents.sm.delegation_types import (
    DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    AGENT_EXPERTISE,
    TASK_TO_AGENT,
    VALID_TASK_TYPES,
    TaskType,
    DelegationConfig,
    DelegationRequest,
    DelegationResult,
)


class TestTaskType:
    """Tests for TaskType literal and mappings."""

    def test_valid_task_types_contains_all_expected(self) -> None:
        """VALID_TASK_TYPES should contain all task types."""
        expected = {
            "requirement_analysis",
            "story_creation",
            "architecture_design",
            "implementation",
            "validation",
            "orchestration",
        }
        assert VALID_TASK_TYPES == expected

    def test_task_to_agent_mapping_complete(self) -> None:
        """TASK_TO_AGENT should map every task type to an agent."""
        for task_type in VALID_TASK_TYPES:
            assert task_type in TASK_TO_AGENT
            assert isinstance(TASK_TO_AGENT[task_type], str)

    def test_agent_expertise_mapping_complete(self) -> None:
        """AGENT_EXPERTISE should define expertise for all agents."""
        expected_agents = {"analyst", "pm", "architect", "dev", "tea", "sm"}
        assert set(AGENT_EXPERTISE.keys()) == expected_agents

    def test_agent_expertise_covers_all_task_types(self) -> None:
        """Every task type should be mapped to exactly one agent."""
        all_task_types: set[str] = set()
        for task_types in AGENT_EXPERTISE.values():
            all_task_types.update(task_types)
        assert all_task_types == VALID_TASK_TYPES


class TestDelegationConfig:
    """Tests for DelegationConfig dataclass."""

    def test_default_values(self) -> None:
        """DelegationConfig should have sensible defaults."""
        config = DelegationConfig()
        assert config.acknowledgment_timeout_seconds == DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS
        assert config.max_retry_attempts == DEFAULT_MAX_RETRY_ATTEMPTS
        assert config.allow_self_delegation is False

    def test_custom_values(self) -> None:
        """DelegationConfig should accept custom values."""
        config = DelegationConfig(
            acknowledgment_timeout_seconds=60.0,
            max_retry_attempts=5,
            allow_self_delegation=True,
        )
        assert config.acknowledgment_timeout_seconds == 60.0
        assert config.max_retry_attempts == 5
        assert config.allow_self_delegation is True

    def test_frozen(self) -> None:
        """DelegationConfig should be immutable."""
        config = DelegationConfig()
        with pytest.raises(AttributeError):
            config.max_retry_attempts = 10  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict should return complete dictionary representation."""
        config = DelegationConfig(
            acknowledgment_timeout_seconds=45.0,
            max_retry_attempts=2,
            allow_self_delegation=True,
        )
        result = config.to_dict()
        assert result == {
            "acknowledgment_timeout_seconds": 45.0,
            "max_retry_attempts": 2,
            "allow_self_delegation": True,
        }


class TestDelegationRequest:
    """Tests for DelegationRequest dataclass."""

    def test_required_fields(self) -> None:
        """DelegationRequest should require task_type, description, source, target, context."""
        request = DelegationRequest(
            task_type="implementation",
            task_description="Implement feature X",
            source_agent="sm",
            target_agent="dev",
            context={"story_id": "10-4"},
        )
        assert request.task_type == "implementation"
        assert request.task_description == "Implement feature X"
        assert request.source_agent == "sm"
        assert request.target_agent == "dev"
        assert request.context == {"story_id": "10-4"}

    def test_default_priority(self) -> None:
        """DelegationRequest should default to normal priority."""
        request = DelegationRequest(
            task_type="validation",
            task_description="Validate tests",
            source_agent="sm",
            target_agent="tea",
            context={},
        )
        assert request.priority == "normal"

    def test_custom_priority(self) -> None:
        """DelegationRequest should accept custom priority."""
        request = DelegationRequest(
            task_type="implementation",
            task_description="Critical fix",
            source_agent="sm",
            target_agent="dev",
            context={},
            priority="critical",
        )
        assert request.priority == "critical"

    def test_created_at_auto_generated(self) -> None:
        """DelegationRequest should auto-generate created_at timestamp."""
        before = datetime.now(timezone.utc).isoformat()
        request = DelegationRequest(
            task_type="implementation",
            task_description="Test",
            source_agent="sm",
            target_agent="dev",
            context={},
        )
        after = datetime.now(timezone.utc).isoformat()
        assert before <= request.created_at <= after

    def test_frozen(self) -> None:
        """DelegationRequest should be immutable."""
        request = DelegationRequest(
            task_type="implementation",
            task_description="Test",
            source_agent="sm",
            target_agent="dev",
            context={},
        )
        with pytest.raises(AttributeError):
            request.task_type = "validation"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict should return complete dictionary representation."""
        request = DelegationRequest(
            task_type="story_creation",
            task_description="Create user story",
            source_agent="sm",
            target_agent="pm",
            context={"requirements": ["req-1"]},
            priority="high",
        )
        result = request.to_dict()
        assert result["task_type"] == "story_creation"
        assert result["task_description"] == "Create user story"
        assert result["source_agent"] == "sm"
        assert result["target_agent"] == "pm"
        assert result["context"] == {"requirements": ["req-1"]}
        assert result["priority"] == "high"
        assert "created_at" in result


class TestDelegationResult:
    """Tests for DelegationResult dataclass."""

    @pytest.fixture
    def sample_request(self) -> DelegationRequest:
        """Create a sample request for testing."""
        return DelegationRequest(
            task_type="implementation",
            task_description="Implement feature",
            source_agent="sm",
            target_agent="dev",
            context={"story": "10-4"},
        )

    def test_successful_result(self, sample_request: DelegationRequest) -> None:
        """DelegationResult should represent successful delegation."""
        result = DelegationResult(
            request=sample_request,
            success=True,
            acknowledged=True,
            acknowledgment_timestamp="2026-01-12T12:00:00+00:00",
        )
        assert result.success is True
        assert result.acknowledged is True
        assert result.acknowledgment_timestamp == "2026-01-12T12:00:00+00:00"
        assert result.error_message is None

    def test_failed_result(self, sample_request: DelegationRequest) -> None:
        """DelegationResult should represent failed delegation."""
        result = DelegationResult(
            request=sample_request,
            success=False,
            acknowledged=False,
            error_message="Agent unavailable",
        )
        assert result.success is False
        assert result.acknowledged is False
        assert result.error_message == "Agent unavailable"

    def test_handoff_context(self, sample_request: DelegationRequest) -> None:
        """DelegationResult should include handoff context."""
        handoff = {
            "source_agent": "sm",
            "target_agent": "dev",
            "task_summary": "Implement feature",
        }
        result = DelegationResult(
            request=sample_request,
            success=True,
            acknowledged=True,
            handoff_context=handoff,
        )
        assert result.handoff_context == handoff

    def test_frozen(self, sample_request: DelegationRequest) -> None:
        """DelegationResult should be immutable."""
        result = DelegationResult(
            request=sample_request,
            success=True,
            acknowledged=True,
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_to_dict(self, sample_request: DelegationRequest) -> None:
        """to_dict should return complete dictionary representation."""
        result = DelegationResult(
            request=sample_request,
            success=True,
            acknowledged=True,
            acknowledgment_timestamp="2026-01-12T12:00:00+00:00",
            handoff_context={"key": "value"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["acknowledged"] is True
        assert d["acknowledgment_timestamp"] == "2026-01-12T12:00:00+00:00"
        assert d["handoff_context"] == {"key": "value"}
        assert "request" in d
        assert d["request"]["task_type"] == "implementation"


class TestConstants:
    """Tests for module constants."""

    def test_default_timeout_is_positive(self) -> None:
        """DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS should be positive."""
        assert DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS > 0

    def test_default_retries_is_positive(self) -> None:
        """DEFAULT_MAX_RETRY_ATTEMPTS should be positive."""
        assert DEFAULT_MAX_RETRY_ATTEMPTS > 0

    def test_task_to_agent_is_valid(self) -> None:
        """TASK_TO_AGENT values should be valid agent names."""
        valid_agents = {"analyst", "pm", "architect", "dev", "tea", "sm"}
        for agent in TASK_TO_AGENT.values():
            assert agent in valid_agents
