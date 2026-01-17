"""Tests for audit type definitions (Story 11.1 - Task 1).

Tests type validation, JSON serialization, and frozen dataclass immutability
for Decision, AgentIdentity, DecisionContext, and related types.
"""

from __future__ import annotations

import json
from typing import Any

import pytest


class TestDecisionType:
    """Tests for DecisionType literal type."""

    def test_valid_decision_types_constant_exists(self) -> None:
        """VALID_DECISION_TYPES constant should be exported."""
        from yolo_developer.audit.types import VALID_DECISION_TYPES

        assert isinstance(VALID_DECISION_TYPES, frozenset)
        assert "requirement_analysis" in VALID_DECISION_TYPES
        assert "story_creation" in VALID_DECISION_TYPES
        assert "architecture_choice" in VALID_DECISION_TYPES
        assert "implementation_choice" in VALID_DECISION_TYPES
        assert "test_strategy" in VALID_DECISION_TYPES
        assert "orchestration" in VALID_DECISION_TYPES
        assert "quality_gate" in VALID_DECISION_TYPES
        assert "escalation" in VALID_DECISION_TYPES


class TestDecisionSeverity:
    """Tests for DecisionSeverity literal type."""

    def test_valid_decision_severities_constant_exists(self) -> None:
        """VALID_DECISION_SEVERITIES constant should be exported."""
        from yolo_developer.audit.types import VALID_DECISION_SEVERITIES

        assert isinstance(VALID_DECISION_SEVERITIES, frozenset)
        assert "info" in VALID_DECISION_SEVERITIES
        assert "warning" in VALID_DECISION_SEVERITIES
        assert "critical" in VALID_DECISION_SEVERITIES


class TestAgentIdentity:
    """Tests for AgentIdentity frozen dataclass."""

    def test_create_agent_identity(self) -> None:
        """Should create AgentIdentity with required fields."""
        from yolo_developer.audit.types import AgentIdentity

        agent = AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-123",
        )

        assert agent.agent_name == "analyst"
        assert agent.agent_type == "analyst"
        assert agent.session_id == "session-123"

    def test_agent_identity_is_frozen(self) -> None:
        """AgentIdentity should be immutable."""
        from yolo_developer.audit.types import AgentIdentity

        agent = AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-123",
        )

        with pytest.raises(AttributeError):
            agent.agent_name = "pm"  # type: ignore[misc]

    def test_agent_identity_to_dict(self) -> None:
        """to_dict should produce JSON-serializable output."""
        from yolo_developer.audit.types import AgentIdentity

        agent = AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-123",
        )

        result = agent.to_dict()

        assert isinstance(result, dict)
        assert result["agent_name"] == "analyst"
        assert result["agent_type"] == "analyst"
        assert result["session_id"] == "session-123"

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_agent_identity_empty_name_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty agent_name should log warning."""
        from yolo_developer.audit.types import AgentIdentity

        AgentIdentity(
            agent_name="",
            agent_type="analyst",
            session_id="session-123",
        )

        assert any("empty" in record.message.lower() for record in caplog.records)

    def test_agent_identity_empty_type_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty agent_type should log warning."""
        from yolo_developer.audit.types import AgentIdentity

        AgentIdentity(
            agent_name="analyst",
            agent_type="",
            session_id="session-123",
        )

        assert any(
            "agent_type" in record.message.lower() and "empty" in record.message.lower()
            for record in caplog.records
        )

    def test_agent_identity_empty_session_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty session_id should log warning."""
        from yolo_developer.audit.types import AgentIdentity

        AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="",
        )

        assert any(
            "session_id" in record.message.lower() and "empty" in record.message.lower()
            for record in caplog.records
        )


class TestDecisionContext:
    """Tests for DecisionContext frozen dataclass."""

    def test_create_decision_context_minimal(self) -> None:
        """Should create DecisionContext with optional fields as None."""
        from yolo_developer.audit.types import DecisionContext

        context = DecisionContext()

        assert context.sprint_id is None
        assert context.story_id is None
        assert context.artifact_id is None
        assert context.parent_decision_id is None

    def test_create_decision_context_full(self) -> None:
        """Should create DecisionContext with all fields."""
        from yolo_developer.audit.types import DecisionContext

        context = DecisionContext(
            sprint_id="sprint-1",
            story_id="1-2-auth",
            artifact_id="req-001",
            parent_decision_id="dec-000",
        )

        assert context.sprint_id == "sprint-1"
        assert context.story_id == "1-2-auth"
        assert context.artifact_id == "req-001"
        assert context.parent_decision_id == "dec-000"

    def test_decision_context_is_frozen(self) -> None:
        """DecisionContext should be immutable."""
        from yolo_developer.audit.types import DecisionContext

        context = DecisionContext(sprint_id="sprint-1")

        with pytest.raises(AttributeError):
            context.sprint_id = "sprint-2"  # type: ignore[misc]

    def test_decision_context_to_dict(self) -> None:
        """to_dict should produce JSON-serializable output."""
        from yolo_developer.audit.types import DecisionContext

        context = DecisionContext(
            sprint_id="sprint-1",
            story_id="1-2-auth",
        )

        result = context.to_dict()

        assert isinstance(result, dict)
        assert result["sprint_id"] == "sprint-1"
        assert result["story_id"] == "1-2-auth"
        assert result["artifact_id"] is None
        assert result["parent_decision_id"] is None

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None


class TestDecision:
    """Tests for Decision frozen dataclass."""

    @pytest.fixture
    def sample_agent(self) -> Any:
        """Create sample AgentIdentity."""
        from yolo_developer.audit.types import AgentIdentity

        return AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-123",
        )

    @pytest.fixture
    def sample_context(self) -> Any:
        """Create sample DecisionContext."""
        from yolo_developer.audit.types import DecisionContext

        return DecisionContext(
            sprint_id="sprint-1",
            story_id="1-2-auth",
        )

    def test_create_decision(self, sample_agent: Any, sample_context: Any) -> None:
        """Should create Decision with all required fields."""
        from yolo_developer.audit.types import Decision

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Crystallized requirement: OAuth2 authentication",
            rationale="Industry standard security",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )

        assert decision.id == "dec-001"
        assert decision.decision_type == "requirement_analysis"
        assert decision.content == "Crystallized requirement: OAuth2 authentication"
        assert decision.rationale == "Industry standard security"
        assert decision.agent == sample_agent
        assert decision.context == sample_context
        assert decision.timestamp == "2026-01-17T12:00:00Z"
        assert decision.metadata == {}
        assert decision.severity == "info"

    def test_create_decision_with_metadata(self, sample_agent: Any, sample_context: Any) -> None:
        """Should create Decision with optional metadata."""
        from yolo_developer.audit.types import Decision

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Decision content",
            rationale="Decision rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
            metadata={"source": "seed.md"},
            severity="warning",
        )

        assert decision.metadata == {"source": "seed.md"}
        assert decision.severity == "warning"

    def test_decision_is_frozen(self, sample_agent: Any, sample_context: Any) -> None:
        """Decision should be immutable."""
        from yolo_developer.audit.types import Decision

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )

        with pytest.raises(AttributeError):
            decision.content = "New content"  # type: ignore[misc]

    def test_decision_to_dict(self, sample_agent: Any, sample_context: Any) -> None:
        """to_dict should produce JSON-serializable output."""
        from yolo_developer.audit.types import Decision

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Decision content",
            rationale="Decision rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
            metadata={"source": "seed.md"},
        )

        result = decision.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == "dec-001"
        assert result["decision_type"] == "requirement_analysis"
        assert result["content"] == "Decision content"
        assert result["rationale"] == "Decision rationale"
        assert result["timestamp"] == "2026-01-17T12:00:00Z"
        assert result["metadata"] == {"source": "seed.md"}
        assert isinstance(result["agent"], dict)
        assert isinstance(result["context"], dict)

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_decision_invalid_type_warns(
        self, sample_agent: Any, sample_context: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid decision_type should log warning."""
        from yolo_developer.audit.types import Decision

        Decision(
            id="dec-001",
            decision_type="invalid_type",  # type: ignore[arg-type]
            content="Content",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )

        assert any(
            "invalid" in record.message.lower() or "decision_type" in record.message.lower()
            for record in caplog.records
        )

    def test_decision_empty_content_warns(
        self, sample_agent: Any, sample_context: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty content should log warning."""
        from yolo_developer.audit.types import Decision

        Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )

        assert any(
            "empty" in record.message.lower() or "content" in record.message.lower()
            for record in caplog.records
        )

    def test_decision_empty_rationale_warns(
        self, sample_agent: Any, sample_context: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty rationale should log warning."""
        from yolo_developer.audit.types import Decision

        Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Content",
            rationale="",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )

        assert any(
            "empty" in record.message.lower() or "rationale" in record.message.lower()
            for record in caplog.records
        )

    def test_decision_empty_id_warns(
        self, sample_agent: Any, sample_context: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty id should log warning."""
        from yolo_developer.audit.types import Decision

        Decision(
            id="",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )

        assert any(
            "id" in record.message.lower() and "empty" in record.message.lower()
            for record in caplog.records
        )

    def test_decision_empty_timestamp_warns(
        self, sample_agent: Any, sample_context: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty timestamp should log warning."""
        from yolo_developer.audit.types import Decision

        Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="",
        )

        assert any(
            "timestamp" in record.message.lower() and "empty" in record.message.lower()
            for record in caplog.records
        )

    def test_decision_invalid_severity_warns(
        self, sample_agent: Any, sample_context: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid severity should log warning."""
        from yolo_developer.audit.types import Decision

        Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
            severity="invalid_severity",  # type: ignore[arg-type]
        )

        assert any("severity" in record.message.lower() for record in caplog.records)
