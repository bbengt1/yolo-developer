"""Tests for cost store protocol and filters (Story 11.6).

Tests for CostFilters dataclass and CostStore protocol definition.
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.cost_store import CostFilters, CostStore
from yolo_developer.audit.cost_types import CostAggregation, CostRecord


class TestCostFilters:
    """Tests for CostFilters dataclass."""

    def test_create_empty_filters(self) -> None:
        """Test creating CostFilters with no arguments."""
        filters = CostFilters()

        assert filters.agent_name is None
        assert filters.story_id is None
        assert filters.sprint_id is None
        assert filters.session_id is None
        assert filters.model is None
        assert filters.tier is None
        assert filters.start_time is None
        assert filters.end_time is None

    def test_create_filters_with_agent(self) -> None:
        """Test creating CostFilters with agent_name."""
        filters = CostFilters(agent_name="analyst")

        assert filters.agent_name == "analyst"
        assert filters.story_id is None

    def test_create_filters_with_all_fields(self) -> None:
        """Test creating CostFilters with all fields."""
        filters = CostFilters(
            agent_name="analyst",
            story_id="1-2-user-auth",
            sprint_id="sprint-1",
            session_id="session-123",
            model="gpt-4o-mini",
            tier="routine",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-18T23:59:59Z",
        )

        assert filters.agent_name == "analyst"
        assert filters.story_id == "1-2-user-auth"
        assert filters.sprint_id == "sprint-1"
        assert filters.session_id == "session-123"
        assert filters.model == "gpt-4o-mini"
        assert filters.tier == "routine"
        assert filters.start_time == "2026-01-01T00:00:00Z"
        assert filters.end_time == "2026-01-18T23:59:59Z"

    def test_filters_is_frozen(self) -> None:
        """Test that CostFilters is immutable."""
        filters = CostFilters(agent_name="analyst")

        with pytest.raises(AttributeError):
            filters.agent_name = "pm"  # type: ignore[misc]

    def test_filters_to_dict(self) -> None:
        """Test CostFilters to_dict serialization."""
        filters = CostFilters(
            agent_name="analyst",
            story_id="1-2-user-auth",
            session_id="session-123",
        )

        result = filters.to_dict()

        assert result == {
            "agent_name": "analyst",
            "story_id": "1-2-user-auth",
            "sprint_id": None,
            "session_id": "session-123",
            "model": None,
            "tier": None,
            "start_time": None,
            "end_time": None,
        }


class TestCostStoreProtocol:
    """Tests for CostStore protocol definition."""

    def test_cost_store_is_protocol(self) -> None:
        """Test that CostStore is a protocol."""
        assert hasattr(CostStore, "__protocol_attrs__") or hasattr(CostStore, "_is_protocol")

    def test_cost_store_has_required_methods(self) -> None:
        """Test that CostStore protocol has required methods."""
        assert hasattr(CostStore, "store_cost")
        assert hasattr(CostStore, "get_cost")
        assert hasattr(CostStore, "get_costs")
        assert hasattr(CostStore, "get_aggregation")
        assert hasattr(CostStore, "get_grouped_aggregation")

    def test_cost_store_is_runtime_checkable(self) -> None:
        """Test that CostStore can be used with isinstance."""

        # Create a minimal implementation
        class MockCostStore:
            async def store_cost(self, record: CostRecord) -> None:
                pass

            async def get_cost(self, cost_id: str) -> CostRecord | None:
                return None

            async def get_costs(self, filters: CostFilters | None = None) -> list[CostRecord]:
                return []

            async def get_aggregation(self, filters: CostFilters | None = None) -> CostAggregation:
                return CostAggregation(
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    call_count=0,
                    models=(),
                )

            async def get_grouped_aggregation(
                self, group_by: str, filters: CostFilters | None = None
            ) -> dict[str, CostAggregation]:
                return {}

        store = MockCostStore()
        assert isinstance(store, CostStore)

    def test_non_conforming_class_fails_isinstance(self) -> None:
        """Test that non-conforming class fails isinstance check."""

        class NotACostStore:
            pass

        obj = NotACostStore()
        assert not isinstance(obj, CostStore)
