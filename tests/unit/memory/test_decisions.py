"""Unit tests for Decision data structures.

Tests the Decision, DecisionType, DecisionResult, and DecisionFilter
dataclasses used for historical decision queries.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from yolo_developer.memory.decisions import (
    VALID_AGENT_TYPES,
    Decision,
    DecisionFilter,
    DecisionResult,
    DecisionType,
    validate_agent_type,
)


class TestAgentTypeValidation:
    """Tests for agent_type validation."""

    def test_valid_agent_types_defined(self) -> None:
        """Verify all expected agent types are in VALID_AGENT_TYPES."""
        expected = {"Analyst", "PM", "Architect", "Dev", "SM", "TEA"}
        assert VALID_AGENT_TYPES == frozenset(expected)

    def test_validate_agent_type_accepts_valid_types(self) -> None:
        """validate_agent_type returns the agent type for valid inputs."""
        for agent_type in VALID_AGENT_TYPES:
            assert validate_agent_type(agent_type) == agent_type

    def test_validate_agent_type_rejects_invalid_type(self) -> None:
        """validate_agent_type raises ValueError for invalid agent types."""
        with pytest.raises(ValueError, match="Invalid agent_type"):
            validate_agent_type("InvalidAgent")

    def test_decision_validates_agent_type(self) -> None:
        """Decision raises ValueError for invalid agent_type."""
        with pytest.raises(ValueError, match="Invalid agent_type"):
            Decision(
                id="dec-invalid",
                agent_type="InvalidAgent",
                context="Test context",
                rationale="Test rationale",
            )


class TestDecisionType:
    """Tests for DecisionType enum."""

    def test_decision_types_exist(self) -> None:
        """Verify all expected decision types are defined."""
        assert DecisionType.REQUIREMENT_CLARIFICATION is not None
        assert DecisionType.STORY_PRIORITIZATION is not None
        assert DecisionType.ARCHITECTURE_CHOICE is not None
        assert DecisionType.IMPLEMENTATION_APPROACH is not None
        assert DecisionType.TEST_STRATEGY is not None
        assert DecisionType.CONFLICT_RESOLUTION is not None

    def test_decision_type_values(self) -> None:
        """Verify decision type enum values are strings."""
        assert DecisionType.REQUIREMENT_CLARIFICATION.value == "requirement_clarification"
        assert DecisionType.STORY_PRIORITIZATION.value == "story_prioritization"
        assert DecisionType.ARCHITECTURE_CHOICE.value == "architecture_choice"
        assert DecisionType.IMPLEMENTATION_APPROACH.value == "implementation_approach"


class TestDecision:
    """Tests for Decision dataclass."""

    def test_decision_creation_minimal(self) -> None:
        """Create a Decision with minimal required fields."""
        decision = Decision(
            id="dec-001",
            agent_type="Analyst",
            context="User wants to add dark mode",
            rationale="Dark mode improves accessibility and reduces eye strain",
        )

        assert decision.id == "dec-001"
        assert decision.agent_type == "Analyst"
        assert decision.context == "User wants to add dark mode"
        assert decision.rationale == "Dark mode improves accessibility and reduces eye strain"
        assert decision.outcome is None
        assert decision.artifact_type is None
        assert decision.artifact_ids == ()
        assert decision.decision_type is None

    def test_decision_creation_full(self) -> None:
        """Create a Decision with all fields specified."""
        timestamp = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        decision = Decision(
            id="dec-002",
            agent_type="Architect",
            context="Choosing between REST and GraphQL for API",
            rationale="REST is simpler and aligns with team expertise",
            outcome="REST API implemented successfully",
            decision_type=DecisionType.ARCHITECTURE_CHOICE,
            artifact_type="design",
            artifact_ids=("adr-001", "story-005"),
            timestamp=timestamp,
        )

        assert decision.id == "dec-002"
        assert decision.agent_type == "Architect"
        assert decision.outcome == "REST API implemented successfully"
        assert decision.decision_type == DecisionType.ARCHITECTURE_CHOICE
        assert decision.artifact_type == "design"
        assert decision.artifact_ids == ("adr-001", "story-005")
        assert decision.timestamp == timestamp

    def test_decision_is_immutable(self) -> None:
        """Verify Decision is frozen (immutable)."""
        decision = Decision(
            id="dec-003",
            agent_type="PM",
            context="Story prioritization",
            rationale="Higher business value first",
        )

        with pytest.raises(AttributeError):
            decision.rationale = "Changed rationale"  # type: ignore[misc]

    def test_decision_default_timestamp(self) -> None:
        """Verify Decision gets a default timestamp."""
        decision = Decision(
            id="dec-004",
            agent_type="Dev",
            context="Implementation choice",
            rationale="Use existing library",
        )

        assert decision.timestamp is not None
        assert decision.timestamp.tzinfo == timezone.utc

    def test_decision_to_embedding_text(self) -> None:
        """Test embedding text generation for semantic search."""
        decision = Decision(
            id="dec-005",
            agent_type="Architect",
            context="Database choice for user data storage",
            rationale="PostgreSQL for ACID compliance and complex queries",
            decision_type=DecisionType.ARCHITECTURE_CHOICE,
        )

        embedding_text = decision.to_embedding_text()

        assert "Architect" in embedding_text
        assert "Database choice for user data storage" in embedding_text
        assert "PostgreSQL for ACID compliance" in embedding_text
        assert "architecture_choice" in embedding_text

    def test_decision_to_embedding_text_with_outcome(self) -> None:
        """Test embedding text includes outcome when present."""
        decision = Decision(
            id="dec-006",
            agent_type="TEA",
            context="Test strategy for authentication",
            rationale="Use integration tests for auth flow",
            outcome="All auth tests passing with 95% coverage",
            decision_type=DecisionType.TEST_STRATEGY,
        )

        embedding_text = decision.to_embedding_text()

        assert "TEA" in embedding_text
        assert "Test strategy for authentication" in embedding_text
        assert "95% coverage" in embedding_text


class TestDecisionResult:
    """Tests for DecisionResult dataclass."""

    def test_decision_result_creation(self) -> None:
        """Create a DecisionResult wrapping a Decision."""
        decision = Decision(
            id="dec-007",
            agent_type="PM",
            context="Feature prioritization",
            rationale="Core features first",
        )

        result = DecisionResult(decision=decision, similarity=0.85)

        assert result.decision == decision
        assert result.similarity == 0.85

    def test_decision_result_is_immutable(self) -> None:
        """Verify DecisionResult is frozen."""
        decision = Decision(
            id="dec-008",
            agent_type="SM",
            context="Sprint planning",
            rationale="Two-week sprints",
        )
        result = DecisionResult(decision=decision, similarity=0.9)

        with pytest.raises(AttributeError):
            result.similarity = 0.5  # type: ignore[misc]


class TestDecisionFilter:
    """Tests for DecisionFilter dataclass."""

    def test_filter_creation_empty(self) -> None:
        """Create an empty filter (no restrictions)."""
        filter_obj = DecisionFilter()

        assert filter_obj.agent_type is None
        assert filter_obj.time_range_start is None
        assert filter_obj.time_range_end is None
        assert filter_obj.artifact_type is None
        assert filter_obj.decision_type is None

    def test_filter_creation_with_agent_type(self) -> None:
        """Create a filter for specific agent type."""
        filter_obj = DecisionFilter(agent_type="Architect")

        assert filter_obj.agent_type == "Architect"

    def test_filter_creation_with_time_range(self) -> None:
        """Create a filter with time range."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 31, tzinfo=timezone.utc)

        filter_obj = DecisionFilter(time_range_start=start, time_range_end=end)

        assert filter_obj.time_range_start == start
        assert filter_obj.time_range_end == end

    def test_filter_creation_with_artifact_type(self) -> None:
        """Create a filter for specific artifact type."""
        filter_obj = DecisionFilter(artifact_type="code")

        assert filter_obj.artifact_type == "code"

    def test_filter_creation_full(self) -> None:
        """Create a filter with all fields."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 31, tzinfo=timezone.utc)

        filter_obj = DecisionFilter(
            agent_type="Dev",
            time_range_start=start,
            time_range_end=end,
            artifact_type="code",
            decision_type=DecisionType.IMPLEMENTATION_APPROACH,
        )

        assert filter_obj.agent_type == "Dev"
        assert filter_obj.time_range_start == start
        assert filter_obj.time_range_end == end
        assert filter_obj.artifact_type == "code"
        assert filter_obj.decision_type == DecisionType.IMPLEMENTATION_APPROACH

    def test_filter_to_chromadb_where(self) -> None:
        """Test conversion to ChromaDB where clause."""
        filter_obj = DecisionFilter(agent_type="Architect", artifact_type="design")

        where_clause = filter_obj.to_chromadb_where()

        assert where_clause is not None
        assert "$and" in where_clause
        conditions = where_clause["$and"]
        assert {"agent_type": {"$eq": "Architect"}} in conditions
        assert {"artifact_type": {"$eq": "design"}} in conditions

    def test_filter_to_chromadb_where_empty(self) -> None:
        """Empty filter returns None for where clause."""
        filter_obj = DecisionFilter()

        where_clause = filter_obj.to_chromadb_where()

        assert where_clause is None

    def test_filter_to_chromadb_where_single_condition(self) -> None:
        """Single condition filter returns simple where clause."""
        filter_obj = DecisionFilter(agent_type="PM")

        where_clause = filter_obj.to_chromadb_where()

        assert where_clause == {"agent_type": {"$eq": "PM"}}

    def test_filter_to_chromadb_where_with_time_range(self) -> None:
        """Time range filter uses $gte and $lte operators."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 31, tzinfo=timezone.utc)
        filter_obj = DecisionFilter(time_range_start=start, time_range_end=end)

        where_clause = filter_obj.to_chromadb_where()

        assert where_clause is not None
        assert "$and" in where_clause
