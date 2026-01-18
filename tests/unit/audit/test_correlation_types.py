"""Tests for correlation type definitions (Story 11.5 - Task 1).

Tests the correlation types for cross-agent decision correlation.

References:
    - FR85: System can correlate decisions across agent boundaries
    - AC #1: Decision chains are identified
    - AC #2: Cause-effect relationships are shown
"""

from __future__ import annotations

import json

import pytest


class TestCorrelationType:
    """Tests for CorrelationType literal type."""

    def test_valid_correlation_types_constant_exists(self) -> None:
        """Test VALID_CORRELATION_TYPES constant exists."""
        from yolo_developer.audit.correlation_types import VALID_CORRELATION_TYPES

        assert VALID_CORRELATION_TYPES is not None
        assert isinstance(VALID_CORRELATION_TYPES, frozenset)

    def test_valid_correlation_types_contains_expected_values(self) -> None:
        """Test VALID_CORRELATION_TYPES contains all expected values."""
        from yolo_developer.audit.correlation_types import VALID_CORRELATION_TYPES

        expected = {"causal", "temporal", "session", "artifact"}
        assert VALID_CORRELATION_TYPES == expected


class TestDecisionChain:
    """Tests for DecisionChain dataclass."""

    def test_decision_chain_exists(self) -> None:
        """Test DecisionChain class exists."""
        from yolo_developer.audit.correlation_types import DecisionChain

        assert DecisionChain is not None

    def test_decision_chain_creation(self) -> None:
        """Test DecisionChain can be created with required fields."""
        from yolo_developer.audit.correlation_types import DecisionChain

        chain = DecisionChain(
            id="chain-001",
            decisions=("dec-001", "dec-002", "dec-003"),
            chain_type="session",
            created_at="2026-01-18T12:00:00Z",
        )

        assert chain.id == "chain-001"
        assert chain.decisions == ("dec-001", "dec-002", "dec-003")
        assert chain.chain_type == "session"
        assert chain.created_at == "2026-01-18T12:00:00Z"

    def test_decision_chain_with_metadata(self) -> None:
        """Test DecisionChain with optional metadata."""
        from yolo_developer.audit.correlation_types import DecisionChain

        chain = DecisionChain(
            id="chain-001",
            decisions=("dec-001",),
            chain_type="causal",
            created_at="2026-01-18T12:00:00Z",
            metadata={"source": "auto-correlation"},
        )

        assert chain.metadata == {"source": "auto-correlation"}

    def test_decision_chain_is_frozen(self) -> None:
        """Test DecisionChain is immutable (frozen dataclass)."""
        from yolo_developer.audit.correlation_types import DecisionChain

        chain = DecisionChain(
            id="chain-001",
            decisions=("dec-001",),
            chain_type="session",
            created_at="2026-01-18T12:00:00Z",
        )

        with pytest.raises(AttributeError):
            chain.id = "new-id"  # type: ignore[misc]

    def test_decision_chain_to_dict(self) -> None:
        """Test DecisionChain to_dict produces JSON-serializable output."""
        from yolo_developer.audit.correlation_types import DecisionChain

        chain = DecisionChain(
            id="chain-001",
            decisions=("dec-001", "dec-002"),
            chain_type="temporal",
            created_at="2026-01-18T12:00:00Z",
            metadata={"key": "value"},
        )

        result = chain.to_dict()

        assert result["id"] == "chain-001"
        assert result["decisions"] == ["dec-001", "dec-002"]
        assert result["chain_type"] == "temporal"
        assert result["created_at"] == "2026-01-18T12:00:00Z"
        assert result["metadata"] == {"key": "value"}

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_decision_chain_empty_id_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty id logs warning."""
        from yolo_developer.audit.correlation_types import DecisionChain

        DecisionChain(
            id="",
            decisions=("dec-001",),
            chain_type="session",
            created_at="2026-01-18T12:00:00Z",
        )

        assert "DecisionChain id is empty" in caplog.text

    def test_decision_chain_invalid_type_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test invalid chain_type logs warning."""
        from yolo_developer.audit.correlation_types import DecisionChain

        DecisionChain(
            id="chain-001",
            decisions=("dec-001",),
            chain_type="invalid",  # type: ignore[arg-type]
            created_at="2026-01-18T12:00:00Z",
        )

        assert "is not a valid correlation type" in caplog.text

    def test_decision_chain_empty_decisions_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty decisions tuple logs warning."""
        from yolo_developer.audit.correlation_types import DecisionChain

        DecisionChain(
            id="chain-001",
            decisions=(),
            chain_type="session",
            created_at="2026-01-18T12:00:00Z",
        )

        assert "DecisionChain decisions is empty" in caplog.text


class TestCausalRelation:
    """Tests for CausalRelation dataclass."""

    def test_causal_relation_exists(self) -> None:
        """Test CausalRelation class exists."""
        from yolo_developer.audit.correlation_types import CausalRelation

        assert CausalRelation is not None

    def test_causal_relation_creation(self) -> None:
        """Test CausalRelation can be created with required fields."""
        from yolo_developer.audit.correlation_types import CausalRelation

        relation = CausalRelation(
            id="rel-001",
            cause_decision_id="dec-001",
            effect_decision_id="dec-002",
            relation_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        assert relation.id == "rel-001"
        assert relation.cause_decision_id == "dec-001"
        assert relation.effect_decision_id == "dec-002"
        assert relation.relation_type == "derives_from"
        assert relation.created_at == "2026-01-18T12:00:00Z"

    def test_causal_relation_with_evidence(self) -> None:
        """Test CausalRelation with evidence field."""
        from yolo_developer.audit.correlation_types import CausalRelation

        relation = CausalRelation(
            id="rel-001",
            cause_decision_id="dec-001",
            effect_decision_id="dec-002",
            relation_type="triggers",
            evidence="Parent decision ID reference in context",
            created_at="2026-01-18T12:00:00Z",
        )

        assert relation.evidence == "Parent decision ID reference in context"

    def test_causal_relation_is_frozen(self) -> None:
        """Test CausalRelation is immutable (frozen dataclass)."""
        from yolo_developer.audit.correlation_types import CausalRelation

        relation = CausalRelation(
            id="rel-001",
            cause_decision_id="dec-001",
            effect_decision_id="dec-002",
            relation_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        with pytest.raises(AttributeError):
            relation.id = "new-id"  # type: ignore[misc]

    def test_causal_relation_to_dict(self) -> None:
        """Test CausalRelation to_dict produces JSON-serializable output."""
        from yolo_developer.audit.correlation_types import CausalRelation

        relation = CausalRelation(
            id="rel-001",
            cause_decision_id="dec-001",
            effect_decision_id="dec-002",
            relation_type="triggers",
            evidence="Test evidence",
            created_at="2026-01-18T12:00:00Z",
        )

        result = relation.to_dict()

        assert result["id"] == "rel-001"
        assert result["cause_decision_id"] == "dec-001"
        assert result["effect_decision_id"] == "dec-002"
        assert result["relation_type"] == "triggers"
        assert result["evidence"] == "Test evidence"
        assert result["created_at"] == "2026-01-18T12:00:00Z"

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_causal_relation_empty_id_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty id logs warning."""
        from yolo_developer.audit.correlation_types import CausalRelation

        CausalRelation(
            id="",
            cause_decision_id="dec-001",
            effect_decision_id="dec-002",
            relation_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        assert "CausalRelation id is empty" in caplog.text

    def test_causal_relation_empty_cause_id_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty cause_decision_id logs warning."""
        from yolo_developer.audit.correlation_types import CausalRelation

        CausalRelation(
            id="rel-001",
            cause_decision_id="",
            effect_decision_id="dec-002",
            relation_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        assert "CausalRelation cause_decision_id is empty" in caplog.text


class TestAgentTransition:
    """Tests for AgentTransition dataclass."""

    def test_agent_transition_exists(self) -> None:
        """Test AgentTransition class exists."""
        from yolo_developer.audit.correlation_types import AgentTransition

        assert AgentTransition is not None

    def test_agent_transition_creation(self) -> None:
        """Test AgentTransition can be created with required fields."""
        from yolo_developer.audit.correlation_types import AgentTransition

        transition = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )

        assert transition.id == "trans-001"
        assert transition.from_agent == "analyst"
        assert transition.to_agent == "pm"
        assert transition.decision_id == "dec-001"
        assert transition.timestamp == "2026-01-18T12:00:00Z"

    def test_agent_transition_with_context(self) -> None:
        """Test AgentTransition with optional context."""
        from yolo_developer.audit.correlation_types import AgentTransition

        transition = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
            context={"session_id": "session-123", "sprint_id": "sprint-1"},
        )

        assert transition.context == {"session_id": "session-123", "sprint_id": "sprint-1"}

    def test_agent_transition_is_frozen(self) -> None:
        """Test AgentTransition is immutable (frozen dataclass)."""
        from yolo_developer.audit.correlation_types import AgentTransition

        transition = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )

        with pytest.raises(AttributeError):
            transition.id = "new-id"  # type: ignore[misc]

    def test_agent_transition_to_dict(self) -> None:
        """Test AgentTransition to_dict produces JSON-serializable output."""
        from yolo_developer.audit.correlation_types import AgentTransition

        transition = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
            context={"key": "value"},
        )

        result = transition.to_dict()

        assert result["id"] == "trans-001"
        assert result["from_agent"] == "analyst"
        assert result["to_agent"] == "pm"
        assert result["decision_id"] == "dec-001"
        assert result["timestamp"] == "2026-01-18T12:00:00Z"
        assert result["context"] == {"key": "value"}

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_agent_transition_empty_id_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty id logs warning."""
        from yolo_developer.audit.correlation_types import AgentTransition

        AgentTransition(
            id="",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )

        assert "AgentTransition id is empty" in caplog.text

    def test_agent_transition_empty_from_agent_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty from_agent logs warning."""
        from yolo_developer.audit.correlation_types import AgentTransition

        AgentTransition(
            id="trans-001",
            from_agent="",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )

        assert "AgentTransition from_agent is empty" in caplog.text

    def test_agent_transition_empty_to_agent_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty to_agent logs warning."""
        from yolo_developer.audit.correlation_types import AgentTransition

        AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )

        assert "AgentTransition to_agent is empty" in caplog.text


class TestDefaultValues:
    """Tests for default values in correlation types."""

    def test_decision_chain_default_metadata(self) -> None:
        """Test DecisionChain has default empty metadata."""
        from yolo_developer.audit.correlation_types import DecisionChain

        chain = DecisionChain(
            id="chain-001",
            decisions=("dec-001",),
            chain_type="session",
            created_at="2026-01-18T12:00:00Z",
        )

        assert chain.metadata == {}

    def test_causal_relation_default_evidence(self) -> None:
        """Test CausalRelation has default empty evidence."""
        from yolo_developer.audit.correlation_types import CausalRelation

        relation = CausalRelation(
            id="rel-001",
            cause_decision_id="dec-001",
            effect_decision_id="dec-002",
            relation_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        assert relation.evidence == ""

    def test_agent_transition_default_context(self) -> None:
        """Test AgentTransition has default empty context."""
        from yolo_developer.audit.correlation_types import AgentTransition

        transition = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )

        assert transition.context == {}

    def test_timeline_entry_default_agent_transition(self) -> None:
        """Test TimelineEntry has default None agent_transition."""
        from yolo_developer.audit.correlation_types import TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        entry = TimelineEntry(
            decision=decision,
            sequence_number=1,
            timestamp="2026-01-18T12:00:00Z",
        )

        assert entry.agent_transition is None

    def test_timeline_entry_default_previous_agent(self) -> None:
        """Test TimelineEntry has default None previous_agent."""
        from yolo_developer.audit.correlation_types import TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        entry = TimelineEntry(
            decision=decision,
            sequence_number=1,
            timestamp="2026-01-18T12:00:00Z",
        )

        assert entry.previous_agent is None


class TestTimelineEntry:
    """Tests for TimelineEntry dataclass."""

    def test_timeline_entry_exists(self) -> None:
        """Test TimelineEntry class exists."""
        from yolo_developer.audit.correlation_types import TimelineEntry

        assert TimelineEntry is not None

    def test_timeline_entry_creation(self) -> None:
        """Test TimelineEntry can be created with required fields."""
        from yolo_developer.audit.correlation_types import TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test decision",
            rationale="Test rationale",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        entry = TimelineEntry(
            decision=decision,
            sequence_number=1,
            timestamp="2026-01-18T12:00:00Z",
        )

        assert entry.decision.id == "dec-001"
        assert entry.sequence_number == 1
        assert entry.timestamp == "2026-01-18T12:00:00Z"

    def test_timeline_entry_with_agent_transition(self) -> None:
        """Test TimelineEntry with agent_transition and previous_agent."""
        from yolo_developer.audit.correlation_types import AgentTransition, TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="pm", agent_type="pm", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        transition = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )
        entry = TimelineEntry(
            decision=decision,
            sequence_number=2,
            timestamp="2026-01-18T12:00:00Z",
            agent_transition=transition,
            previous_agent="analyst",
        )

        assert entry.agent_transition is not None
        assert entry.agent_transition.from_agent == "analyst"
        assert entry.previous_agent == "analyst"

    def test_timeline_entry_is_frozen(self) -> None:
        """Test TimelineEntry is immutable (frozen dataclass)."""
        from yolo_developer.audit.correlation_types import TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        entry = TimelineEntry(
            decision=decision,
            sequence_number=1,
            timestamp="2026-01-18T12:00:00Z",
        )

        with pytest.raises(AttributeError):
            entry.sequence_number = 2  # type: ignore[misc]

    def test_timeline_entry_to_dict(self) -> None:
        """Test TimelineEntry to_dict produces JSON-serializable output."""
        from yolo_developer.audit.correlation_types import AgentTransition, TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="pm", agent_type="pm", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        transition = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T12:00:00Z",
        )
        entry = TimelineEntry(
            decision=decision,
            sequence_number=1,
            timestamp="2026-01-18T12:00:00Z",
            agent_transition=transition,
            previous_agent="analyst",
        )

        result = entry.to_dict()

        assert result["sequence_number"] == 1
        assert result["timestamp"] == "2026-01-18T12:00:00Z"
        assert result["agent_transition"] is not None
        assert result["agent_transition"]["from_agent"] == "analyst"
        assert result["previous_agent"] == "analyst"

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_timeline_entry_to_dict_without_transition(self) -> None:
        """Test TimelineEntry to_dict without agent_transition."""
        from yolo_developer.audit.correlation_types import TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        entry = TimelineEntry(
            decision=decision,
            sequence_number=1,
            timestamp="2026-01-18T12:00:00Z",
        )

        result = entry.to_dict()

        assert result["agent_transition"] is None
        assert result["previous_agent"] is None

    def test_timeline_entry_negative_sequence_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test sequence_number < 1 logs warning."""
        from yolo_developer.audit.correlation_types import TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )

        TimelineEntry(
            decision=decision,
            sequence_number=0,
            timestamp="2026-01-18T12:00:00Z",
        )

        assert "sequence_number=0 is less than 1" in caplog.text

    def test_timeline_entry_empty_timestamp_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test empty timestamp logs warning."""
        from yolo_developer.audit.correlation_types import TimelineEntry
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )

        TimelineEntry(
            decision=decision,
            sequence_number=1,
            timestamp="",
        )

        assert "TimelineEntry timestamp is empty" in caplog.text
