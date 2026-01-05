"""Unit tests for DecisionQueryEngine.

Tests the high-level query interface for historical decision queries.
"""

from __future__ import annotations

import pytest

from yolo_developer.memory.decision_queries import DecisionQueryEngine
from yolo_developer.memory.decision_store import ChromaDecisionStore
from yolo_developer.memory.decisions import (
    Decision,
    DecisionFilter,
    DecisionResult,
    DecisionType,
)


@pytest.fixture
def decision_store(tmp_path: pytest.fixture) -> ChromaDecisionStore:
    """Create a decision store with test data."""
    store = ChromaDecisionStore(
        persist_directory=str(tmp_path),
        project_id="test",
    )
    return store


@pytest.fixture
async def populated_store(decision_store: ChromaDecisionStore) -> ChromaDecisionStore:
    """Populate store with test decisions."""
    decisions = [
        Decision(
            id="dec-001",
            agent_type="Architect",
            context="Choosing PostgreSQL for user data storage",
            rationale="ACID compliance and complex query support needed",
            outcome="Successfully deployed with good performance",
            decision_type=DecisionType.ARCHITECTURE_CHOICE,
            artifact_type="design",
            artifact_ids=("adr-001",),
        ),
        Decision(
            id="dec-002",
            agent_type="Architect",
            context="Using Redis for session caching",
            rationale="Fast in-memory storage for session data",
            decision_type=DecisionType.ARCHITECTURE_CHOICE,
            artifact_type="design",
        ),
        Decision(
            id="dec-003",
            agent_type="PM",
            context="Prioritizing authentication stories first",
            rationale="Security foundation before other features",
            decision_type=DecisionType.STORY_PRIORITIZATION,
            artifact_type="story",
            artifact_ids=("story-001", "story-002"),
        ),
        Decision(
            id="dec-004",
            agent_type="Dev",
            context="Using pytest fixtures for test isolation",
            rationale="Better test organization and reusability",
            decision_type=DecisionType.TEST_STRATEGY,
            artifact_type="code",
        ),
        Decision(
            id="dec-005",
            agent_type="TEA",
            context="Setting 90% coverage target for core modules",
            rationale="Balance between coverage and development speed",
            decision_type=DecisionType.TEST_STRATEGY,
            artifact_type="requirement",
        ),
    ]

    for decision in decisions:
        await decision_store.store_decision(decision)

    return decision_store


class TestDecisionQueryEngineInit:
    """Tests for DecisionQueryEngine initialization."""

    def test_init_with_store(self, decision_store: ChromaDecisionStore) -> None:
        """Engine initializes with a decision store."""
        engine = DecisionQueryEngine(store=decision_store)

        assert engine._store is decision_store

    def test_init_creates_store_if_not_provided(self, tmp_path: pytest.fixture) -> None:
        """Engine creates store when not provided."""
        engine = DecisionQueryEngine(
            persist_directory=str(tmp_path),
            project_id="my-project",
        )

        assert engine._store is not None


class TestFindSimilarDecisions:
    """Tests for find_similar_decisions method."""

    @pytest.mark.asyncio
    async def test_find_similar_returns_results(self, populated_store: ChromaDecisionStore) -> None:
        """Find returns semantically similar decisions."""
        engine = DecisionQueryEngine(store=populated_store)

        results = await engine.find_similar_decisions(
            context="database selection for storing user information",
            k=3,
        )

        assert len(results) >= 1
        assert all(isinstance(r, DecisionResult) for r in results)
        # First result should be about database (PostgreSQL)
        assert (
            "database" in results[0].decision.context.lower()
            or "postgresql" in results[0].decision.context.lower()
        )

    @pytest.mark.asyncio
    async def test_find_similar_respects_k(self, populated_store: ChromaDecisionStore) -> None:
        """Find returns at most k results."""
        engine = DecisionQueryEngine(store=populated_store)

        results = await engine.find_similar_decisions(context="technical decision", k=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_find_similar_empty_store(self, decision_store: ChromaDecisionStore) -> None:
        """Find returns empty list on empty store."""
        engine = DecisionQueryEngine(store=decision_store)

        results = await engine.find_similar_decisions(context="anything")

        assert results == []


class TestGetAgentDecisionHistory:
    """Tests for get_agent_decision_history method."""

    @pytest.mark.asyncio
    async def test_get_history_by_agent(self, populated_store: ChromaDecisionStore) -> None:
        """Get history returns decisions for specific agent."""
        engine = DecisionQueryEngine(store=populated_store)

        decisions = await engine.get_agent_decision_history(
            agent_type="Architect",
            limit=10,
        )

        assert len(decisions) == 2
        assert all(d.agent_type == "Architect" for d in decisions)

    @pytest.mark.asyncio
    async def test_get_history_respects_limit(self, populated_store: ChromaDecisionStore) -> None:
        """Get history returns at most limit results."""
        engine = DecisionQueryEngine(store=populated_store)

        decisions = await engine.get_agent_decision_history(
            agent_type="Architect",
            limit=1,
        )

        assert len(decisions) <= 1

    @pytest.mark.asyncio
    async def test_get_history_no_results(self, populated_store: ChromaDecisionStore) -> None:
        """Get history returns empty list for unknown agent."""
        engine = DecisionQueryEngine(store=populated_store)

        decisions = await engine.get_agent_decision_history(
            agent_type="UnknownAgent",
            limit=10,
        )

        assert decisions == []


class TestSearchWithFilters:
    """Tests for search_with_filters method."""

    @pytest.mark.asyncio
    async def test_search_with_agent_filter(self, populated_store: ChromaDecisionStore) -> None:
        """Search with agent filter returns matching results."""
        engine = DecisionQueryEngine(store=populated_store)

        filters = DecisionFilter(agent_type="PM")
        results = await engine.search_with_filters(
            query="prioritization",
            filters=filters,
        )

        assert len(results) >= 1
        assert all(r.decision.agent_type == "PM" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_artifact_type_filter(
        self, populated_store: ChromaDecisionStore
    ) -> None:
        """Search with artifact type filter returns matching results."""
        engine = DecisionQueryEngine(store=populated_store)

        filters = DecisionFilter(artifact_type="design")
        results = await engine.search_with_filters(
            query="architecture",
            filters=filters,
        )

        assert all(r.decision.artifact_type == "design" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_decision_type_filter(
        self, populated_store: ChromaDecisionStore
    ) -> None:
        """Search with decision type filter returns matching results."""
        engine = DecisionQueryEngine(store=populated_store)

        filters = DecisionFilter(decision_type=DecisionType.TEST_STRATEGY)
        results = await engine.search_with_filters(
            query="testing",
            filters=filters,
        )

        assert all(r.decision.decision_type == DecisionType.TEST_STRATEGY for r in results)

    @pytest.mark.asyncio
    async def test_search_with_combined_filters(self, populated_store: ChromaDecisionStore) -> None:
        """Search with multiple filters applies all conditions."""
        engine = DecisionQueryEngine(store=populated_store)

        filters = DecisionFilter(
            agent_type="Architect",
            artifact_type="design",
        )
        results = await engine.search_with_filters(
            query="storage",
            filters=filters,
        )

        assert all(
            r.decision.agent_type == "Architect" and r.decision.artifact_type == "design"
            for r in results
        )

    @pytest.mark.asyncio
    async def test_search_without_filters(self, populated_store: ChromaDecisionStore) -> None:
        """Search without filters returns all matching results."""
        engine = DecisionQueryEngine(store=populated_store)

        results = await engine.search_with_filters(
            query="decision",
            filters=None,
        )

        assert len(results) >= 1


class TestRecordDecision:
    """Tests for record_decision method."""

    @pytest.mark.asyncio
    async def test_record_decision_stores_and_returns_id(
        self, decision_store: ChromaDecisionStore
    ) -> None:
        """Recording a decision stores it and returns ID."""
        engine = DecisionQueryEngine(store=decision_store)

        decision_id = await engine.record_decision(
            agent_type="SM",
            context="Sprint planning approach",
            rationale="Two-week sprints for faster feedback",
            decision_type=DecisionType.STORY_PRIORITIZATION,
        )

        assert decision_id is not None

        # Verify it was stored
        decisions = await decision_store.get_decisions_by_agent("SM", limit=10)
        assert len(decisions) == 1
        assert decisions[0].context == "Sprint planning approach"

    @pytest.mark.asyncio
    async def test_record_decision_with_outcome(self, decision_store: ChromaDecisionStore) -> None:
        """Recording a decision with outcome stores outcome."""
        engine = DecisionQueryEngine(store=decision_store)

        decision_id = await engine.record_decision(
            agent_type="Dev",
            context="Code formatting",
            rationale="Use black for consistency",
            outcome="Team adopted successfully",
        )

        decision = await decision_store.get_decision_by_id(decision_id)
        assert decision is not None
        assert decision.outcome == "Team adopted successfully"

    @pytest.mark.asyncio
    async def test_record_decision_generates_id(self, decision_store: ChromaDecisionStore) -> None:
        """Recording a decision generates a unique ID."""
        engine = DecisionQueryEngine(store=decision_store)

        id1 = await engine.record_decision(
            agent_type="Analyst",
            context="Requirement clarification",
            rationale="Split ambiguous requirement",
        )

        id2 = await engine.record_decision(
            agent_type="Analyst",
            context="Another clarification",
            rationale="Different approach",
        )

        assert id1 != id2
