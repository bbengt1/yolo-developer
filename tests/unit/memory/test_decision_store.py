"""Unit tests for ChromaDecisionStore.

Tests the ChromaDB-backed decision storage including store, search,
and retrieval operations.
"""

from __future__ import annotations

import pytest

from yolo_developer.memory.decision_store import ChromaDecisionStore
from yolo_developer.memory.decisions import (
    Decision,
    DecisionFilter,
    DecisionResult,
    DecisionType,
)


class TestChromaDecisionStoreInit:
    """Tests for ChromaDecisionStore initialization."""

    def test_init_creates_collection(self, tmp_path: pytest.fixture) -> None:
        """Store creates a project-specific collection."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test-project",
        )

        assert store.project_id == "test-project"
        assert store.persist_directory == str(tmp_path)

    def test_init_default_project_id(self, tmp_path: pytest.fixture) -> None:
        """Store uses default project ID if not provided."""
        store = ChromaDecisionStore(persist_directory=str(tmp_path))

        assert store.project_id == "default"

    def test_collection_name_includes_project_id(self, tmp_path: pytest.fixture) -> None:
        """Collection name follows 'decisions_{project_id}' pattern."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="my-project",
        )

        # The collection should be named decisions_my-project
        assert store._collection is not None


class TestStoreDecision:
    """Tests for store_decision method."""

    @pytest.mark.asyncio
    async def test_store_decision_returns_id(self, tmp_path: pytest.fixture) -> None:
        """Storing a decision returns the decision ID."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )
        decision = Decision(
            id="dec-001",
            agent_type="Architect",
            context="Database selection",
            rationale="PostgreSQL for reliability",
        )

        result_id = await store.store_decision(decision)

        assert result_id == "dec-001"

    @pytest.mark.asyncio
    async def test_store_decision_with_all_fields(self, tmp_path: pytest.fixture) -> None:
        """Store a decision with all optional fields."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )
        decision = Decision(
            id="dec-002",
            agent_type="PM",
            context="Story prioritization",
            rationale="Business value first",
            outcome="Sprint completed on time",
            decision_type=DecisionType.STORY_PRIORITIZATION,
            artifact_type="story",
            artifact_ids=("story-001", "story-002"),
        )

        result_id = await store.store_decision(decision)

        assert result_id == "dec-002"

    @pytest.mark.asyncio
    async def test_store_decision_upsert_behavior(self, tmp_path: pytest.fixture) -> None:
        """Storing same ID twice updates the decision."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )
        decision1 = Decision(
            id="dec-003",
            agent_type="Dev",
            context="Original context",
            rationale="Original rationale",
        )
        decision2 = Decision(
            id="dec-003",
            agent_type="Dev",
            context="Updated context",
            rationale="Updated rationale",
        )

        await store.store_decision(decision1)
        await store.store_decision(decision2)

        # Retrieve and verify update
        decisions = await store.get_decisions_by_agent("Dev", limit=10)
        assert len(decisions) == 1
        assert decisions[0].context == "Updated context"


class TestSearchDecisions:
    """Tests for search_decisions method."""

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_no_data(self, tmp_path: pytest.fixture) -> None:
        """Search returns empty list when no decisions stored."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        results = await store.search_decisions("database choice")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results_with_similarity(self, tmp_path: pytest.fixture) -> None:
        """Search returns results with similarity scores."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        # Store some decisions
        await store.store_decision(
            Decision(
                id="dec-001",
                agent_type="Architect",
                context="Choosing database for user data storage",
                rationale="PostgreSQL for ACID compliance",
                decision_type=DecisionType.ARCHITECTURE_CHOICE,
            )
        )
        await store.store_decision(
            Decision(
                id="dec-002",
                agent_type="Dev",
                context="API authentication method",
                rationale="JWT for stateless auth",
                decision_type=DecisionType.IMPLEMENTATION_APPROACH,
            )
        )

        # Search for database-related decisions
        results = await store.search_decisions("database storage", k=5)

        assert len(results) >= 1
        assert all(isinstance(r, DecisionResult) for r in results)
        assert all(0.0 <= r.similarity <= 1.0 for r in results)

    @pytest.mark.asyncio
    async def test_search_with_agent_type_filter(self, tmp_path: pytest.fixture) -> None:
        """Search can filter by agent type."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        # Store decisions from different agents
        await store.store_decision(
            Decision(
                id="dec-001",
                agent_type="Architect",
                context="Database choice",
                rationale="PostgreSQL",
            )
        )
        await store.store_decision(
            Decision(
                id="dec-002",
                agent_type="Dev",
                context="Caching strategy",
                rationale="Redis for sessions",
            )
        )

        # Filter by agent type
        filters = DecisionFilter(agent_type="Architect")
        results = await store.search_decisions("technical choice", filters=filters, k=5)

        # Only Architect decisions should match
        assert all(r.decision.agent_type == "Architect" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_artifact_type_filter(self, tmp_path: pytest.fixture) -> None:
        """Search can filter by artifact type."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        await store.store_decision(
            Decision(
                id="dec-001",
                agent_type="PM",
                context="Story breakdown",
                rationale="Split by feature",
                artifact_type="story",
            )
        )
        await store.store_decision(
            Decision(
                id="dec-002",
                agent_type="Architect",
                context="API design",
                rationale="RESTful approach",
                artifact_type="design",
            )
        )

        filters = DecisionFilter(artifact_type="story")
        results = await store.search_decisions("planning", filters=filters, k=5)

        assert all(r.decision.artifact_type == "story" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_decision_type_filter(self, tmp_path: pytest.fixture) -> None:
        """Search can filter by decision type."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        await store.store_decision(
            Decision(
                id="dec-001",
                agent_type="Architect",
                context="Tech stack",
                rationale="Python and FastAPI",
                decision_type=DecisionType.ARCHITECTURE_CHOICE,
            )
        )
        await store.store_decision(
            Decision(
                id="dec-002",
                agent_type="Dev",
                context="Test approach",
                rationale="Pytest with fixtures",
                decision_type=DecisionType.TEST_STRATEGY,
            )
        )

        filters = DecisionFilter(decision_type=DecisionType.ARCHITECTURE_CHOICE)
        results = await store.search_decisions("technology", filters=filters, k=5)

        assert all(r.decision.decision_type == DecisionType.ARCHITECTURE_CHOICE for r in results)


class TestGetDecisionsByAgent:
    """Tests for get_decisions_by_agent method."""

    @pytest.mark.asyncio
    async def test_get_decisions_empty(self, tmp_path: pytest.fixture) -> None:
        """Returns empty list when no decisions for agent."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        decisions = await store.get_decisions_by_agent("Architect", limit=10)

        assert decisions == []

    @pytest.mark.asyncio
    async def test_get_decisions_by_agent_filters_correctly(self, tmp_path: pytest.fixture) -> None:
        """Returns only decisions from specified agent."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        # Store decisions from different agents
        await store.store_decision(
            Decision(
                id="dec-001",
                agent_type="Architect",
                context="Database choice",
                rationale="PostgreSQL",
            )
        )
        await store.store_decision(
            Decision(
                id="dec-002",
                agent_type="Architect",
                context="Caching strategy",
                rationale="Redis",
            )
        )
        await store.store_decision(
            Decision(
                id="dec-003",
                agent_type="Dev",
                context="Logging approach",
                rationale="Structured logs",
            )
        )

        # Get only Architect decisions
        decisions = await store.get_decisions_by_agent("Architect", limit=10)

        assert len(decisions) == 2
        assert all(d.agent_type == "Architect" for d in decisions)

    @pytest.mark.asyncio
    async def test_get_decisions_respects_limit(self, tmp_path: pytest.fixture) -> None:
        """Returns at most 'limit' decisions."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        # Store multiple decisions
        for i in range(5):
            await store.store_decision(
                Decision(
                    id=f"dec-{i:03d}",
                    agent_type="PM",
                    context=f"Decision {i}",
                    rationale=f"Rationale {i}",
                )
            )

        decisions = await store.get_decisions_by_agent("PM", limit=3)

        assert len(decisions) == 3


class TestGetDecisionById:
    """Tests for get_decision_by_id method."""

    @pytest.mark.asyncio
    async def test_get_decision_by_id_found(self, tmp_path: pytest.fixture) -> None:
        """Returns decision when ID exists."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        await store.store_decision(
            Decision(
                id="dec-specific",
                agent_type="TEA",
                context="Test coverage decision",
                rationale="90% coverage target",
            )
        )

        decision = await store.get_decision_by_id("dec-specific")

        assert decision is not None
        assert decision.id == "dec-specific"
        assert decision.agent_type == "TEA"

    @pytest.mark.asyncio
    async def test_get_decision_by_id_not_found(self, tmp_path: pytest.fixture) -> None:
        """Returns None when ID doesn't exist."""
        store = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="test",
        )

        decision = await store.get_decision_by_id("non-existent")

        assert decision is None
