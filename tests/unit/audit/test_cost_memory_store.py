"""Tests for in-memory cost store implementation (Story 11.6).

Tests for InMemoryCostStore class.
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.cost_memory_store import InMemoryCostStore
from yolo_developer.audit.cost_store import CostFilters
from yolo_developer.audit.cost_types import CostRecord, TokenUsage


def _make_usage(prompt: int = 100, completion: int = 50) -> TokenUsage:
    """Helper to create a TokenUsage instance."""
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _make_record(
    cost_id: str = "cost-001",
    model: str = "gpt-4o-mini",
    tier: str = "routine",
    cost_usd: float = 0.0015,
    agent_name: str = "analyst",
    session_id: str = "session-123",
    story_id: str | None = None,
    sprint_id: str | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    timestamp: str = "2026-01-18T12:00:00Z",
) -> CostRecord:
    """Helper to create a CostRecord instance."""
    return CostRecord(
        id=cost_id,
        timestamp=timestamp,
        model=model,
        tier=tier,
        token_usage=_make_usage(prompt_tokens, completion_tokens),
        cost_usd=cost_usd,
        agent_name=agent_name,
        session_id=session_id,
        story_id=story_id,
        sprint_id=sprint_id,
    )


class TestInMemoryCostStoreBasics:
    """Basic tests for InMemoryCostStore."""

    @pytest.mark.asyncio
    async def test_store_and_get_cost(self) -> None:
        """Test storing and retrieving a cost record."""
        store = InMemoryCostStore()
        record = _make_record(cost_id="cost-001")

        await store.store_cost(record)
        result = await store.get_cost("cost-001")

        assert result is not None
        assert result.id == "cost-001"
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_get_nonexistent_cost(self) -> None:
        """Test getting a cost record that doesn't exist."""
        store = InMemoryCostStore()

        result = await store.get_cost("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_costs(self) -> None:
        """Test getting all cost records."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001"))
        await store.store_cost(_make_record(cost_id="cost-002"))
        await store.store_cost(_make_record(cost_id="cost-003"))

        costs = await store.get_costs()

        assert len(costs) == 3
        cost_ids = {c.id for c in costs}
        assert cost_ids == {"cost-001", "cost-002", "cost-003"}


class TestInMemoryCostStoreFiltering:
    """Tests for filtering cost records."""

    @pytest.mark.asyncio
    async def test_filter_by_agent_name(self) -> None:
        """Test filtering costs by agent name."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", agent_name="analyst"))
        await store.store_cost(_make_record(cost_id="cost-002", agent_name="pm"))
        await store.store_cost(_make_record(cost_id="cost-003", agent_name="analyst"))

        filters = CostFilters(agent_name="analyst")
        costs = await store.get_costs(filters)

        assert len(costs) == 2
        assert all(c.agent_name == "analyst" for c in costs)

    @pytest.mark.asyncio
    async def test_filter_by_story_id(self) -> None:
        """Test filtering costs by story ID."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", story_id="1-2-auth"))
        await store.store_cost(_make_record(cost_id="cost-002", story_id="1-3-profile"))
        await store.store_cost(_make_record(cost_id="cost-003", story_id="1-2-auth"))

        filters = CostFilters(story_id="1-2-auth")
        costs = await store.get_costs(filters)

        assert len(costs) == 2
        assert all(c.story_id == "1-2-auth" for c in costs)

    @pytest.mark.asyncio
    async def test_filter_by_session_id(self) -> None:
        """Test filtering costs by session ID."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", session_id="sess-a"))
        await store.store_cost(_make_record(cost_id="cost-002", session_id="sess-b"))

        filters = CostFilters(session_id="sess-a")
        costs = await store.get_costs(filters)

        assert len(costs) == 1
        assert costs[0].session_id == "sess-a"

    @pytest.mark.asyncio
    async def test_filter_by_model(self) -> None:
        """Test filtering costs by model."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", model="gpt-4o-mini"))
        await store.store_cost(_make_record(cost_id="cost-002", model="claude-sonnet"))

        filters = CostFilters(model="gpt-4o-mini")
        costs = await store.get_costs(filters)

        assert len(costs) == 1
        assert costs[0].model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_filter_by_tier(self) -> None:
        """Test filtering costs by tier."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", tier="routine"))
        await store.store_cost(_make_record(cost_id="cost-002", tier="complex"))
        await store.store_cost(_make_record(cost_id="cost-003", tier="routine"))

        filters = CostFilters(tier="routine")
        costs = await store.get_costs(filters)

        assert len(costs) == 2
        assert all(c.tier == "routine" for c in costs)

    @pytest.mark.asyncio
    async def test_filter_by_time_range(self) -> None:
        """Test filtering costs by time range."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", timestamp="2026-01-01T10:00:00Z"))
        await store.store_cost(_make_record(cost_id="cost-002", timestamp="2026-01-15T10:00:00Z"))
        await store.store_cost(_make_record(cost_id="cost-003", timestamp="2026-01-20T10:00:00Z"))

        filters = CostFilters(
            start_time="2026-01-10T00:00:00Z",
            end_time="2026-01-18T00:00:00Z",
        )
        costs = await store.get_costs(filters)

        assert len(costs) == 1
        assert costs[0].id == "cost-002"

    @pytest.mark.asyncio
    async def test_filter_multiple_criteria(self) -> None:
        """Test filtering with multiple criteria."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", agent_name="analyst", tier="routine"))
        await store.store_cost(_make_record(cost_id="cost-002", agent_name="analyst", tier="complex"))
        await store.store_cost(_make_record(cost_id="cost-003", agent_name="pm", tier="routine"))

        filters = CostFilters(agent_name="analyst", tier="routine")
        costs = await store.get_costs(filters)

        assert len(costs) == 1
        assert costs[0].id == "cost-001"


class TestInMemoryCostStoreAggregation:
    """Tests for cost aggregation."""

    @pytest.mark.asyncio
    async def test_get_aggregation_empty(self) -> None:
        """Test aggregation with no records."""
        store = InMemoryCostStore()

        agg = await store.get_aggregation()

        assert agg.total_prompt_tokens == 0
        assert agg.total_completion_tokens == 0
        assert agg.total_tokens == 0
        assert agg.total_cost_usd == 0.0
        assert agg.call_count == 0
        assert agg.models == ()

    @pytest.mark.asyncio
    async def test_get_aggregation_single_record(self) -> None:
        """Test aggregation with a single record."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(
            cost_id="cost-001",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.0015,
            model="gpt-4o-mini",
        ))

        agg = await store.get_aggregation()

        assert agg.total_prompt_tokens == 100
        assert agg.total_completion_tokens == 50
        assert agg.total_tokens == 150
        assert agg.total_cost_usd == pytest.approx(0.0015)
        assert agg.call_count == 1
        assert agg.models == ("gpt-4o-mini",)

    @pytest.mark.asyncio
    async def test_get_aggregation_multiple_records(self) -> None:
        """Test aggregation with multiple records."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(
            cost_id="cost-001",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.0015,
            model="gpt-4o-mini",
        ))
        await store.store_cost(_make_record(
            cost_id="cost-002",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.015,
            model="claude-sonnet",
        ))
        await store.store_cost(_make_record(
            cost_id="cost-003",
            prompt_tokens=50,
            completion_tokens=25,
            cost_usd=0.0008,
            model="gpt-4o-mini",
        ))

        agg = await store.get_aggregation()

        assert agg.total_prompt_tokens == 350
        assert agg.total_completion_tokens == 175
        assert agg.total_tokens == 525
        assert agg.total_cost_usd == pytest.approx(0.0173)
        assert agg.call_count == 3
        # Models should be unique
        assert set(agg.models) == {"gpt-4o-mini", "claude-sonnet"}

    @pytest.mark.asyncio
    async def test_get_aggregation_with_filters(self) -> None:
        """Test aggregation with filters applied."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(
            cost_id="cost-001",
            agent_name="analyst",
            cost_usd=0.001,
        ))
        await store.store_cost(_make_record(
            cost_id="cost-002",
            agent_name="pm",
            cost_usd=0.002,
        ))
        await store.store_cost(_make_record(
            cost_id="cost-003",
            agent_name="analyst",
            cost_usd=0.003,
        ))

        filters = CostFilters(agent_name="analyst")
        agg = await store.get_aggregation(filters)

        assert agg.call_count == 2
        assert agg.total_cost_usd == pytest.approx(0.004)


class TestInMemoryCostStoreGroupedAggregation:
    """Tests for grouped aggregation."""

    @pytest.mark.asyncio
    async def test_grouped_aggregation_by_agent(self) -> None:
        """Test grouped aggregation by agent."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", agent_name="analyst", cost_usd=0.001))
        await store.store_cost(_make_record(cost_id="cost-002", agent_name="pm", cost_usd=0.002))
        await store.store_cost(_make_record(cost_id="cost-003", agent_name="analyst", cost_usd=0.003))

        result = await store.get_grouped_aggregation("agent")

        assert len(result) == 2
        assert "analyst" in result
        assert "pm" in result
        assert result["analyst"].call_count == 2
        assert result["analyst"].total_cost_usd == pytest.approx(0.004)
        assert result["pm"].call_count == 1
        assert result["pm"].total_cost_usd == pytest.approx(0.002)

    @pytest.mark.asyncio
    async def test_grouped_aggregation_by_story(self) -> None:
        """Test grouped aggregation by story."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", story_id="1-2-auth", cost_usd=0.001))
        await store.store_cost(_make_record(cost_id="cost-002", story_id="1-3-profile", cost_usd=0.002))
        await store.store_cost(_make_record(cost_id="cost-003", story_id=None, cost_usd=0.003))

        result = await store.get_grouped_aggregation("story")

        # None story_id should be in a "None" group or excluded
        assert "1-2-auth" in result
        assert "1-3-profile" in result
        assert result["1-2-auth"].call_count == 1
        assert result["1-3-profile"].call_count == 1

    @pytest.mark.asyncio
    async def test_grouped_aggregation_by_model(self) -> None:
        """Test grouped aggregation by model."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", model="gpt-4o-mini", cost_usd=0.001))
        await store.store_cost(_make_record(cost_id="cost-002", model="claude-sonnet", cost_usd=0.01))
        await store.store_cost(_make_record(cost_id="cost-003", model="gpt-4o-mini", cost_usd=0.002))

        result = await store.get_grouped_aggregation("model")

        assert len(result) == 2
        assert result["gpt-4o-mini"].call_count == 2
        assert result["claude-sonnet"].call_count == 1

    @pytest.mark.asyncio
    async def test_grouped_aggregation_by_tier(self) -> None:
        """Test grouped aggregation by tier."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", tier="routine", cost_usd=0.001))
        await store.store_cost(_make_record(cost_id="cost-002", tier="complex", cost_usd=0.01))
        await store.store_cost(_make_record(cost_id="cost-003", tier="routine", cost_usd=0.002))

        result = await store.get_grouped_aggregation("tier")

        assert len(result) == 2
        assert result["routine"].call_count == 2
        assert result["complex"].call_count == 1

    @pytest.mark.asyncio
    async def test_grouped_aggregation_empty(self) -> None:
        """Test grouped aggregation with no records."""
        store = InMemoryCostStore()

        result = await store.get_grouped_aggregation("agent")

        assert result == {}

    @pytest.mark.asyncio
    async def test_grouped_aggregation_with_filters(self) -> None:
        """Test grouped aggregation with filters applied."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", agent_name="analyst", tier="routine"))
        await store.store_cost(_make_record(cost_id="cost-002", agent_name="pm", tier="routine"))
        await store.store_cost(_make_record(cost_id="cost-003", agent_name="analyst", tier="complex"))

        filters = CostFilters(tier="routine")
        result = await store.get_grouped_aggregation("agent", filters)

        # Only routine tier should be included
        assert len(result) == 2
        assert result["analyst"].call_count == 1
        assert result["pm"].call_count == 1

    @pytest.mark.asyncio
    async def test_grouped_aggregation_invalid_group_by(self) -> None:
        """Test grouped aggregation with invalid group_by returns empty dict."""
        store = InMemoryCostStore()
        await store.store_cost(_make_record(cost_id="cost-001", agent_name="analyst"))
        await store.store_cost(_make_record(cost_id="cost-002", agent_name="pm"))

        # Invalid group_by should return empty dict (no records match None key)
        result = await store.get_grouped_aggregation("invalid_dimension")

        assert result == {}
