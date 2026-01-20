"""Tests for cost tracking service (Story 11.6).

Tests for CostTrackingService class and get_cost_tracking_service factory function.
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.cost_memory_store import InMemoryCostStore
from yolo_developer.audit.cost_service import (
    CostTrackingService,
    get_cost_tracking_service,
)


class TestCostTrackingServiceRecording:
    """Tests for recording LLM calls."""

    @pytest.mark.asyncio
    async def test_record_llm_call_basic(self) -> None:
        """Test recording a basic LLM call."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        record = await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        assert record.model == "gpt-4o-mini"
        assert record.tier == "routine"
        assert record.token_usage.prompt_tokens == 100
        assert record.token_usage.completion_tokens == 50
        assert record.token_usage.total_tokens == 150
        assert record.cost_usd == 0.0015
        assert record.agent_name == "analyst"
        assert record.session_id == "session-123"
        assert record.id  # Should have generated UUID
        assert record.timestamp  # Should have generated timestamp

    @pytest.mark.asyncio
    async def test_record_llm_call_with_optional_fields(self) -> None:
        """Test recording an LLM call with optional fields."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        record = await service.record_llm_call(
            model="claude-sonnet-4-20250514",
            tier="complex",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.015,
            agent_name="pm",
            session_id="session-456",
            story_id="1-2-user-auth",
            sprint_id="sprint-1",
            metadata={"feature": "login"},
        )

        assert record.story_id == "1-2-user-auth"
        assert record.sprint_id == "sprint-1"
        assert record.metadata == {"feature": "login"}

    @pytest.mark.asyncio
    async def test_record_llm_call_stores_in_store(self) -> None:
        """Test that recorded call is stored in the cost store."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        record = await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        # Verify record is in store
        retrieved = await store.get_cost(record.id)
        assert retrieved is not None
        assert retrieved.id == record.id

    @pytest.mark.asyncio
    async def test_record_llm_call_disabled(self) -> None:
        """Test that recording is skipped when disabled."""
        store = InMemoryCostStore()
        service = CostTrackingService(store, enabled=False)

        record = await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        # Should still return a record but not store it
        assert record is not None
        retrieved = await store.get_cost(record.id)
        assert retrieved is None


class TestCostTrackingServiceAgentCosts:
    """Tests for getting agent cost breakdown."""

    @pytest.mark.asyncio
    async def test_get_agent_costs_empty(self) -> None:
        """Test getting agent costs with no records."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        result = await service.get_agent_costs()

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_agent_costs_grouped(self) -> None:
        """Test getting agent costs grouped by agent."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            agent_name="analyst",
            session_id="sess-1",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.002,
            agent_name="pm",
            session_id="sess-1",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.003,
            agent_name="analyst",
            session_id="sess-1",
        )

        result = await service.get_agent_costs()

        assert len(result) == 2
        assert "analyst" in result
        assert "pm" in result
        assert result["analyst"].call_count == 2
        assert result["analyst"].total_cost_usd == pytest.approx(0.004)
        assert result["pm"].call_count == 1


class TestCostTrackingServiceStoryCosts:
    """Tests for getting story cost breakdown."""

    @pytest.mark.asyncio
    async def test_get_story_costs_empty(self) -> None:
        """Test getting story costs with no records."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        result = await service.get_story_costs()

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_story_costs_grouped(self) -> None:
        """Test getting story costs grouped by story."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            agent_name="analyst",
            session_id="sess-1",
            story_id="1-2-auth",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.002,
            agent_name="pm",
            session_id="sess-1",
            story_id="1-3-profile",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.003,
            agent_name="dev",
            session_id="sess-1",
            story_id="1-2-auth",
        )

        result = await service.get_story_costs()

        assert len(result) == 2
        assert "1-2-auth" in result
        assert "1-3-profile" in result
        assert result["1-2-auth"].call_count == 2


class TestCostTrackingServiceTotals:
    """Tests for getting session and sprint totals."""

    @pytest.mark.asyncio
    async def test_get_session_total(self) -> None:
        """Test getting total costs for a session."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            agent_name="analyst",
            session_id="sess-a",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.002,
            agent_name="pm",
            session_id="sess-b",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.003,
            agent_name="dev",
            session_id="sess-a",
        )

        result = await service.get_session_total("sess-a")

        assert result.call_count == 2
        assert result.total_cost_usd == pytest.approx(0.004)

    @pytest.mark.asyncio
    async def test_get_sprint_total(self) -> None:
        """Test getting total costs for a sprint."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            agent_name="analyst",
            session_id="sess-1",
            sprint_id="sprint-1",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.002,
            agent_name="pm",
            session_id="sess-1",
            sprint_id="sprint-2",
        )
        await service.record_llm_call(
            model="gpt-4o-mini",
            tier="routine",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.003,
            agent_name="dev",
            session_id="sess-1",
            sprint_id="sprint-1",
        )

        result = await service.get_sprint_total("sprint-1")

        assert result.call_count == 2
        assert result.total_cost_usd == pytest.approx(0.004)

    @pytest.mark.asyncio
    async def test_get_session_total_empty(self) -> None:
        """Test getting session total with no matching records."""
        store = InMemoryCostStore()
        service = CostTrackingService(store)

        result = await service.get_session_total("nonexistent")

        assert result.call_count == 0
        assert result.total_cost_usd == 0.0


class TestCostTrackingServiceFactory:
    """Tests for factory function."""

    def test_get_cost_tracking_service_default(self) -> None:
        """Test factory with default parameters."""
        store = InMemoryCostStore()
        service = get_cost_tracking_service(store)

        assert isinstance(service, CostTrackingService)

    def test_get_cost_tracking_service_disabled(self) -> None:
        """Test factory with tracking disabled."""
        store = InMemoryCostStore()
        service = get_cost_tracking_service(store, enabled=False)

        assert isinstance(service, CostTrackingService)

    def test_get_cost_tracking_service_default_store(self) -> None:
        """Test factory creates default store if not provided."""
        service = get_cost_tracking_service()

        assert isinstance(service, CostTrackingService)
