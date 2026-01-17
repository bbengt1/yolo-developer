"""Tests for InMemoryDecisionStore (Story 11.1 - Task 3).

Tests the in-memory implementation of the DecisionStore protocol.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


@pytest.fixture
def sample_agent() -> Any:
    """Create sample AgentIdentity."""
    from yolo_developer.audit.types import AgentIdentity

    return AgentIdentity(
        agent_name="analyst",
        agent_type="analyst",
        session_id="session-123",
    )


@pytest.fixture
def sample_context() -> Any:
    """Create sample DecisionContext."""
    from yolo_developer.audit.types import DecisionContext

    return DecisionContext(
        sprint_id="sprint-1",
        story_id="1-2-auth",
    )


@pytest.fixture
def sample_decision(sample_agent: Any, sample_context: Any) -> Any:
    """Create sample Decision."""
    from yolo_developer.audit.types import Decision

    return Decision(
        id="dec-001",
        decision_type="requirement_analysis",
        content="OAuth2 authentication required",
        rationale="Industry standard security",
        agent=sample_agent,
        context=sample_context,
        timestamp="2026-01-17T12:00:00Z",
    )


class TestInMemoryDecisionStore:
    """Tests for InMemoryDecisionStore."""

    @pytest.mark.asyncio
    async def test_log_decision_returns_id(self, sample_decision: Any) -> None:
        """log_decision should return the decision ID."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()
        decision_id = await store.log_decision(sample_decision)

        assert decision_id == "dec-001"

    @pytest.mark.asyncio
    async def test_get_decision_retrieves_correct_decision(self, sample_decision: Any) -> None:
        """get_decision should retrieve the correct decision by ID."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()
        await store.log_decision(sample_decision)

        retrieved = await store.get_decision("dec-001")

        assert retrieved is not None
        assert retrieved.id == "dec-001"
        assert retrieved.content == "OAuth2 authentication required"
        assert retrieved.rationale == "Industry standard security"

    @pytest.mark.asyncio
    async def test_get_decision_returns_none_for_missing(self) -> None:
        """get_decision should return None for non-existent ID."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()

        retrieved = await store.get_decision("non-existent")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_decisions_returns_chronological_order(
        self, sample_agent: Any, sample_context: Any
    ) -> None:
        """get_decisions should return decisions in chronological order."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.types import Decision

        store = InMemoryDecisionStore()

        # Add decisions with different timestamps (out of order)
        dec3 = Decision(
            id="dec-003",
            decision_type="requirement_analysis",
            content="Third",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T14:00:00Z",
        )
        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="First",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T10:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="requirement_analysis",
            content="Second",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )

        await store.log_decision(dec3)
        await store.log_decision(dec1)
        await store.log_decision(dec2)

        decisions = await store.get_decisions()

        assert len(decisions) == 3
        assert decisions[0].id == "dec-001"  # Earliest
        assert decisions[1].id == "dec-002"
        assert decisions[2].id == "dec-003"  # Latest

    @pytest.mark.asyncio
    async def test_get_decisions_with_agent_filter(self, sample_context: Any) -> None:
        """get_decisions should filter by agent_name."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.store import DecisionFilters
        from yolo_developer.audit.types import AgentIdentity, Decision

        store = InMemoryDecisionStore()

        analyst_agent = AgentIdentity("analyst", "analyst", "session-1")
        pm_agent = AgentIdentity("pm", "pm", "session-2")

        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Analyst decision",
            rationale="Rationale",
            agent=analyst_agent,
            context=sample_context,
            timestamp="2026-01-17T10:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="story_creation",
            content="PM decision",
            rationale="Rationale",
            agent=pm_agent,
            context=sample_context,
            timestamp="2026-01-17T11:00:00Z",
        )

        await store.log_decision(dec1)
        await store.log_decision(dec2)

        filters = DecisionFilters(agent_name="analyst")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 1
        assert decisions[0].id == "dec-001"

    @pytest.mark.asyncio
    async def test_get_decisions_with_decision_type_filter(
        self, sample_agent: Any, sample_context: Any
    ) -> None:
        """get_decisions should filter by decision_type."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.store import DecisionFilters
        from yolo_developer.audit.types import Decision

        store = InMemoryDecisionStore()

        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Requirement decision",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T10:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="architecture_choice",
            content="Architecture decision",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T11:00:00Z",
        )

        await store.log_decision(dec1)
        await store.log_decision(dec2)

        filters = DecisionFilters(decision_type="architecture_choice")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 1
        assert decisions[0].id == "dec-002"

    @pytest.mark.asyncio
    async def test_get_decisions_with_time_range_filter(
        self, sample_agent: Any, sample_context: Any
    ) -> None:
        """get_decisions should filter by time range."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.store import DecisionFilters
        from yolo_developer.audit.types import Decision

        store = InMemoryDecisionStore()

        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Early decision",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T08:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="requirement_analysis",
            content="Middle decision",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T12:00:00Z",
        )
        dec3 = Decision(
            id="dec-003",
            decision_type="requirement_analysis",
            content="Late decision",
            rationale="Rationale",
            agent=sample_agent,
            context=sample_context,
            timestamp="2026-01-17T18:00:00Z",
        )

        await store.log_decision(dec1)
        await store.log_decision(dec2)
        await store.log_decision(dec3)

        # Filter for middle of the day only
        filters = DecisionFilters(
            start_time="2026-01-17T10:00:00Z",
            end_time="2026-01-17T14:00:00Z",
        )
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 1
        assert decisions[0].id == "dec-002"

    @pytest.mark.asyncio
    async def test_get_decisions_with_sprint_filter(self, sample_agent: Any) -> None:
        """get_decisions should filter by sprint_id."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.store import DecisionFilters
        from yolo_developer.audit.types import Decision, DecisionContext

        store = InMemoryDecisionStore()

        ctx1 = DecisionContext(sprint_id="sprint-1")
        ctx2 = DecisionContext(sprint_id="sprint-2")

        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Sprint 1 decision",
            rationale="Rationale",
            agent=sample_agent,
            context=ctx1,
            timestamp="2026-01-17T10:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="requirement_analysis",
            content="Sprint 2 decision",
            rationale="Rationale",
            agent=sample_agent,
            context=ctx2,
            timestamp="2026-01-17T11:00:00Z",
        )

        await store.log_decision(dec1)
        await store.log_decision(dec2)

        filters = DecisionFilters(sprint_id="sprint-1")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 1
        assert decisions[0].id == "dec-001"

    @pytest.mark.asyncio
    async def test_get_decisions_with_story_filter(self, sample_agent: Any) -> None:
        """get_decisions should filter by story_id."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.store import DecisionFilters
        from yolo_developer.audit.types import Decision, DecisionContext

        store = InMemoryDecisionStore()

        ctx1 = DecisionContext(story_id="1-2-auth")
        ctx2 = DecisionContext(story_id="1-3-profile")

        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Auth story decision",
            rationale="Rationale",
            agent=sample_agent,
            context=ctx1,
            timestamp="2026-01-17T10:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="requirement_analysis",
            content="Profile story decision",
            rationale="Rationale",
            agent=sample_agent,
            context=ctx2,
            timestamp="2026-01-17T11:00:00Z",
        )

        await store.log_decision(dec1)
        await store.log_decision(dec2)

        filters = DecisionFilters(story_id="1-2-auth")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 1
        assert decisions[0].id == "dec-001"

    @pytest.mark.asyncio
    async def test_get_decision_count(self, sample_decision: Any) -> None:
        """get_decision_count should return correct count."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()

        assert await store.get_decision_count() == 0

        await store.log_decision(sample_decision)

        assert await store.get_decision_count() == 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self, sample_agent: Any, sample_context: Any) -> None:
        """Store should handle concurrent access safely."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.types import Decision

        store = InMemoryDecisionStore()

        async def log_decision(i: int) -> None:
            dec = Decision(
                id=f"dec-{i:03d}",
                decision_type="requirement_analysis",
                content=f"Decision {i}",
                rationale="Rationale",
                agent=sample_agent,
                context=sample_context,
                timestamp=f"2026-01-17T{i:02d}:00:00Z",
            )
            await store.log_decision(dec)

        # Log 100 decisions concurrently
        await asyncio.gather(*[log_decision(i) for i in range(100)])

        # Should have all 100 decisions
        assert await store.get_decision_count() == 100

        # All should be retrievable
        decisions = await store.get_decisions()
        assert len(decisions) == 100


class TestInMemoryDecisionStoreProtocolCompliance:
    """Test that InMemoryDecisionStore implements DecisionStore protocol."""

    def test_implements_protocol(self) -> None:
        """InMemoryDecisionStore should be a DecisionStore."""
        from yolo_developer.audit.memory_store import InMemoryDecisionStore
        from yolo_developer.audit.store import DecisionStore

        store = InMemoryDecisionStore()

        # runtime_checkable Protocol allows isinstance check
        assert isinstance(store, DecisionStore)
