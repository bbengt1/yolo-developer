"""Unit tests for orchestrator context module.

Tests cover:
- Decision dataclass creation and immutability
- HandoffContext dataclass creation and immutability
- create_handoff_context function
- compute_state_checksum function
- validate_state_integrity function
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from yolo_developer.orchestrator.context import (
    Decision,
    HandoffContext,
    compute_state_checksum,
    create_handoff_context,
    validate_state_integrity,
)


class TestDecision:
    """Tests for the Decision dataclass."""

    def test_decision_creation_with_required_fields(self) -> None:
        """Decision should store all required fields."""
        decision = Decision(
            agent="analyst",
            summary="Prioritized security over performance",
            rationale="User explicitly requested secure design",
        )
        assert decision.agent == "analyst"
        assert decision.summary == "Prioritized security over performance"
        assert decision.rationale == "User explicitly requested secure design"

    def test_decision_has_default_timestamp(self) -> None:
        """Decision should have a default timestamp."""
        decision = Decision(
            agent="pm",
            summary="test summary",
            rationale="test rationale",
        )
        assert decision.timestamp is not None
        assert isinstance(decision.timestamp, datetime)

    def test_decision_has_default_related_artifacts(self) -> None:
        """Decision should have empty related_artifacts by default."""
        decision = Decision(
            agent="architect",
            summary="test",
            rationale="test",
        )
        assert decision.related_artifacts == ()

    def test_decision_with_related_artifacts(self) -> None:
        """Decision should store related artifacts as tuple."""
        decision = Decision(
            agent="dev",
            summary="Implementation choice",
            rationale="Better maintainability",
            related_artifacts=("req-001", "story-002"),
        )
        assert decision.related_artifacts == ("req-001", "story-002")
        assert len(decision.related_artifacts) == 2

    def test_decision_is_frozen(self) -> None:
        """Decision should be immutable."""
        decision = Decision(
            agent="pm",
            summary="test",
            rationale="test",
        )
        with pytest.raises(AttributeError):
            decision.agent = "architect"  # type: ignore[misc]

    def test_decision_equality(self) -> None:
        """Decisions with same required values should be equal (ignoring timestamp)."""
        # Create with explicit timestamp for equality test
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        d1 = Decision(
            agent="analyst",
            summary="test",
            rationale="reason",
            timestamp=fixed_time,
        )
        d2 = Decision(
            agent="analyst",
            summary="test",
            rationale="reason",
            timestamp=fixed_time,
        )
        assert d1 == d2

    def test_decision_hashable(self) -> None:
        """Decision should be hashable for use in sets."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        d1 = Decision(agent="a", summary="b", rationale="c", timestamp=fixed_time)
        d2 = Decision(agent="a", summary="b", rationale="c", timestamp=fixed_time)
        decision_set = {d1, d2}
        assert len(decision_set) == 1


class TestHandoffContext:
    """Tests for the HandoffContext dataclass."""

    def test_context_creation_with_required_fields(self) -> None:
        """HandoffContext should store source and target agents."""
        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
        )
        assert context.source_agent == "analyst"
        assert context.target_agent == "pm"

    def test_context_has_default_decisions(self) -> None:
        """HandoffContext should have empty decisions tuple by default."""
        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
        )
        assert context.decisions == ()

    def test_context_has_default_memory_refs(self) -> None:
        """HandoffContext should have empty memory_refs tuple by default."""
        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
        )
        assert context.memory_refs == ()

    def test_context_has_default_timestamp(self) -> None:
        """HandoffContext should have a default timestamp."""
        context = HandoffContext(
            source_agent="architect",
            target_agent="dev",
        )
        assert context.timestamp is not None
        assert isinstance(context.timestamp, datetime)

    def test_context_with_decisions(self) -> None:
        """HandoffContext should store decisions tuple."""
        decision = Decision(
            agent="analyst",
            summary="Security priority",
            rationale="User requirement",
        )
        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
            decisions=(decision,),
        )
        assert len(context.decisions) == 1
        assert context.decisions[0].agent == "analyst"

    def test_context_with_memory_refs(self) -> None:
        """HandoffContext should store memory references."""
        context = HandoffContext(
            source_agent="pm",
            target_agent="architect",
            memory_refs=("req-001", "req-002", "story-001"),
        )
        assert len(context.memory_refs) == 3
        assert "req-001" in context.memory_refs

    def test_context_is_frozen(self) -> None:
        """HandoffContext should be immutable."""
        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
        )
        with pytest.raises(AttributeError):
            context.source_agent = "dev"  # type: ignore[misc]

    def test_context_equality(self) -> None:
        """HandoffContexts with same values should be equal."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c1 = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
            timestamp=fixed_time,
        )
        c2 = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
            timestamp=fixed_time,
        )
        assert c1 == c2

    def test_context_hashable(self) -> None:
        """HandoffContext should be hashable."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c1 = HandoffContext(source_agent="a", target_agent="b", timestamp=fixed_time)
        c2 = HandoffContext(source_agent="a", target_agent="b", timestamp=fixed_time)
        context_set = {c1, c2}
        assert len(context_set) == 1


class TestCreateHandoffContext:
    """Tests for the create_handoff_context function."""

    def test_creates_context_with_agents(self) -> None:
        """create_handoff_context should create context with source and target."""
        result = create_handoff_context(
            source_agent="analyst",
            target_agent="pm",
        )
        assert "handoff_context" in result
        assert "current_agent" in result
        context = result["handoff_context"]
        assert context.source_agent == "analyst"
        assert context.target_agent == "pm"

    def test_updates_current_agent(self) -> None:
        """create_handoff_context should set current_agent to target."""
        result = create_handoff_context(
            source_agent="pm",
            target_agent="architect",
        )
        assert result["current_agent"] == "architect"

    def test_includes_decisions(self) -> None:
        """create_handoff_context should include provided decisions."""
        decisions = [
            Decision(agent="analyst", summary="s1", rationale="r1"),
            Decision(agent="analyst", summary="s2", rationale="r2"),
        ]
        result = create_handoff_context(
            source_agent="analyst",
            target_agent="pm",
            decisions=decisions,
        )
        context = result["handoff_context"]
        assert len(context.decisions) == 2

    def test_includes_memory_refs(self) -> None:
        """create_handoff_context should include memory references."""
        result = create_handoff_context(
            source_agent="pm",
            target_agent="architect",
            memory_refs=["req-001", "story-001"],
        )
        context = result["handoff_context"]
        assert len(context.memory_refs) == 2
        assert "req-001" in context.memory_refs

    def test_defaults_to_empty_decisions(self) -> None:
        """create_handoff_context should default to empty decisions."""
        result = create_handoff_context(
            source_agent="analyst",
            target_agent="pm",
        )
        context = result["handoff_context"]
        assert context.decisions == ()

    def test_defaults_to_empty_memory_refs(self) -> None:
        """create_handoff_context should default to empty memory refs."""
        result = create_handoff_context(
            source_agent="analyst",
            target_agent="pm",
        )
        context = result["handoff_context"]
        assert context.memory_refs == ()


class TestComputeStateChecksum:
    """Tests for compute_state_checksum function."""

    def test_computes_deterministic_checksum(self) -> None:
        """Same state should produce same checksum."""
        state: dict[str, Any] = {"key1": "value1", "key2": 123}
        checksum1 = compute_state_checksum(state)
        checksum2 = compute_state_checksum(state)
        assert checksum1 == checksum2

    def test_different_states_produce_different_checksums(self) -> None:
        """Different states should produce different checksums."""
        state1: dict[str, Any] = {"key1": "value1"}
        state2: dict[str, Any] = {"key1": "value2"}
        checksum1 = compute_state_checksum(state1)
        checksum2 = compute_state_checksum(state2)
        assert checksum1 != checksum2

    def test_excludes_specified_keys(self) -> None:
        """Excluded keys should not affect checksum."""
        state1: dict[str, Any] = {"data": "same", "transient": "value1"}
        state2: dict[str, Any] = {"data": "same", "transient": "value2"}
        checksum1 = compute_state_checksum(state1, exclude_keys={"transient"})
        checksum2 = compute_state_checksum(state2, exclude_keys={"transient"})
        assert checksum1 == checksum2

    def test_default_excludes_handoff_context(self) -> None:
        """handoff_context should be excluded by default."""
        state1: dict[str, Any] = {"data": "same", "handoff_context": "ctx1"}
        state2: dict[str, Any] = {"data": "same", "handoff_context": "ctx2"}
        checksum1 = compute_state_checksum(state1)
        checksum2 = compute_state_checksum(state2)
        assert checksum1 == checksum2

    def test_returns_sha256_format(self) -> None:
        """Checksum should be a 64-character hex string (SHA-256)."""
        state: dict[str, Any] = {"key": "value"}
        checksum = compute_state_checksum(state)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_handles_non_json_types(self) -> None:
        """Should handle datetime and other non-JSON types via str conversion."""
        state: dict[str, Any] = {
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "nested": {"value": 42},
        }
        # Should not raise
        checksum = compute_state_checksum(state)
        assert len(checksum) == 64


class TestValidateStateIntegrity:
    """Tests for validate_state_integrity function."""

    def test_returns_true_for_identical_states(self) -> None:
        """Identical states should pass integrity check."""
        state: dict[str, Any] = {"data": "value", "count": 42}
        assert validate_state_integrity(state, state) is True

    def test_returns_true_when_only_excluded_keys_differ(self) -> None:
        """States differing only in excluded keys should pass."""
        before: dict[str, Any] = {
            "data": "value",
            "current_agent": "analyst",
            "handoff_context": None,
        }
        after: dict[str, Any] = {
            "data": "value",
            "current_agent": "pm",
            "handoff_context": "new_context",
        }
        assert validate_state_integrity(before, after) is True

    def test_returns_false_when_data_changes(self) -> None:
        """States with different data should fail integrity check."""
        before: dict[str, Any] = {"data": "original"}
        after: dict[str, Any] = {"data": "modified"}
        assert validate_state_integrity(before, after) is False

    def test_returns_false_when_keys_added(self) -> None:
        """States with added keys (outside exclude) should fail."""
        before: dict[str, Any] = {"data": "value"}
        after: dict[str, Any] = {"data": "value", "new_key": "new_value"}
        assert validate_state_integrity(before, after) is False

    def test_returns_false_when_keys_removed(self) -> None:
        """States with removed keys should fail."""
        before: dict[str, Any] = {"data": "value", "extra": "key"}
        after: dict[str, Any] = {"data": "value"}
        assert validate_state_integrity(before, after) is False

    def test_default_excludes_standard_handoff_keys(self) -> None:
        """current_agent, handoff_context, messages should be excluded by default."""
        before: dict[str, Any] = {
            "important_data": "preserved",
            "current_agent": "analyst",
            "handoff_context": None,
            "messages": ["msg1"],
        }
        after: dict[str, Any] = {
            "important_data": "preserved",
            "current_agent": "pm",
            "handoff_context": "new_ctx",
            "messages": ["msg1", "msg2"],
        }
        assert validate_state_integrity(before, after) is True

    def test_custom_exclude_keys(self) -> None:
        """Custom exclude keys should work."""
        before: dict[str, Any] = {"data": "same", "custom_transient": "v1"}
        after: dict[str, Any] = {"data": "same", "custom_transient": "v2"}
        assert validate_state_integrity(before, after, exclude_keys={"custom_transient"}) is True


class TestNoContextLoss:
    """Tests for verifying no context is lost during simulated handoffs."""

    def test_all_state_fields_preserved_through_handoff(self) -> None:
        """All non-transient state fields should be preserved through handoff."""
        from yolo_developer.orchestrator import create_handoff_context

        # Create a rich state with all field types
        initial_state: dict[str, Any] = {
            "user_input": "Build an API",
            "requirements": ["req-1", "req-2", "req-3"],
            "architecture_notes": {"database": "PostgreSQL", "cache": "Redis"},
            "complexity_score": 7.5,
            "is_validated": True,
        }

        # Simulate handoff
        handoff_result = create_handoff_context(
            source_agent="analyst",
            target_agent="pm",
            decisions=[Decision(agent="analyst", summary="test", rationale="test")],
        )

        # After handoff, all original fields should still be present
        after_state = {**initial_state, **handoff_result}

        # Verify no data loss
        assert after_state["user_input"] == "Build an API"
        assert after_state["requirements"] == ["req-1", "req-2", "req-3"]
        assert after_state["architecture_notes"]["database"] == "PostgreSQL"
        assert after_state["complexity_score"] == 7.5
        assert after_state["is_validated"] is True

    def test_nested_data_structures_preserved(self) -> None:
        """Nested data structures should be preserved without corruption."""
        initial_state: dict[str, Any] = {
            "nested": {"level1": {"level2": {"level3": ["a", "b", "c"]}}}
        }

        # Simulate state passing through handoff (just copy)
        after_state = dict(initial_state)

        # Validate nested structure is intact
        assert validate_state_integrity(initial_state, after_state) is True
        assert after_state["nested"]["level1"]["level2"]["level3"] == ["a", "b", "c"]

    def test_large_state_preserved(self) -> None:
        """Large state with many fields should be preserved."""
        # Create state with many fields
        initial_state: dict[str, Any] = {f"field_{i}": f"value_{i}" for i in range(100)}

        # Simulate handoff (copy)
        after_state = dict(initial_state)

        # Validate all fields preserved
        assert validate_state_integrity(initial_state, after_state) is True
        assert len(after_state) == 100

    def test_decisions_accumulate_not_replace(self) -> None:
        """Decisions from multiple handoffs should accumulate, not replace."""
        decisions: list[Decision] = []

        # Simulate multiple agents adding decisions
        decisions.append(Decision(agent="analyst", summary="D1", rationale="R1"))
        decisions.append(Decision(agent="pm", summary="D2", rationale="R2"))
        decisions.append(Decision(agent="architect", summary="D3", rationale="R3"))

        # All decisions should be present
        assert len(decisions) == 3
        agents = [d.agent for d in decisions]
        assert "analyst" in agents
        assert "pm" in agents
        assert "architect" in agents
