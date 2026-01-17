"""Tests for DecisionStore protocol and DecisionFilters (Story 11.1 - Task 2).

Tests the protocol definition and filter dataclass.
"""

from __future__ import annotations

import json

import pytest


class TestDecisionFilters:
    """Tests for DecisionFilters frozen dataclass."""

    def test_create_decision_filters_empty(self) -> None:
        """Should create DecisionFilters with all None defaults."""
        from yolo_developer.audit.store import DecisionFilters

        filters = DecisionFilters()

        assert filters.agent_name is None
        assert filters.decision_type is None
        assert filters.start_time is None
        assert filters.end_time is None
        assert filters.sprint_id is None
        assert filters.story_id is None

    def test_create_decision_filters_full(self) -> None:
        """Should create DecisionFilters with all fields."""
        from yolo_developer.audit.store import DecisionFilters

        filters = DecisionFilters(
            agent_name="analyst",
            decision_type="requirement_analysis",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-31T23:59:59Z",
            sprint_id="sprint-1",
            story_id="1-2-auth",
        )

        assert filters.agent_name == "analyst"
        assert filters.decision_type == "requirement_analysis"
        assert filters.start_time == "2026-01-01T00:00:00Z"
        assert filters.end_time == "2026-01-31T23:59:59Z"
        assert filters.sprint_id == "sprint-1"
        assert filters.story_id == "1-2-auth"

    def test_decision_filters_is_frozen(self) -> None:
        """DecisionFilters should be immutable."""
        from yolo_developer.audit.store import DecisionFilters

        filters = DecisionFilters(agent_name="analyst")

        with pytest.raises(AttributeError):
            filters.agent_name = "pm"  # type: ignore[misc]

    def test_decision_filters_to_dict(self) -> None:
        """to_dict should produce JSON-serializable output."""
        from yolo_developer.audit.store import DecisionFilters

        filters = DecisionFilters(
            agent_name="analyst",
            sprint_id="sprint-1",
        )

        result = filters.to_dict()

        assert isinstance(result, dict)
        assert result["agent_name"] == "analyst"
        assert result["sprint_id"] == "sprint-1"
        assert result["decision_type"] is None

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None


class TestDecisionStoreProtocol:
    """Tests for DecisionStore Protocol."""

    def test_decision_store_protocol_exists(self) -> None:
        """DecisionStore Protocol should be importable."""
        from yolo_developer.audit.store import DecisionStore

        # Protocol should be importable
        assert DecisionStore is not None

    def test_decision_store_has_required_methods(self) -> None:
        """DecisionStore should define required methods."""

        from yolo_developer.audit.store import DecisionStore

        # Check that Protocol has the expected methods
        # We can verify by checking if these are defined
        assert hasattr(DecisionStore, "log_decision")
        assert hasattr(DecisionStore, "get_decision")
        assert hasattr(DecisionStore, "get_decisions")
        assert hasattr(DecisionStore, "get_decision_count")
