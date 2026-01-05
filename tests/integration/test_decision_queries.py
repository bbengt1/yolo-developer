"""Integration tests for decision queries.

Tests the full decision lifecycle and performance requirements.
"""

from __future__ import annotations

import time

import pytest

from yolo_developer.memory.decision_queries import DecisionQueryEngine
from yolo_developer.memory.decision_store import ChromaDecisionStore
from yolo_developer.memory.decisions import (
    Decision,
    DecisionFilter,
    DecisionType,
)


@pytest.fixture
def decision_store(tmp_path: pytest.fixture) -> ChromaDecisionStore:
    """Create a real ChromaDB decision store."""
    return ChromaDecisionStore(
        persist_directory=str(tmp_path / "decisions"),
        project_id="integration-test",
    )


@pytest.fixture
def query_engine(decision_store: ChromaDecisionStore) -> DecisionQueryEngine:
    """Create a query engine with the test store."""
    return DecisionQueryEngine(store=decision_store)


class TestDecisionLifecycle:
    """Tests for full decision lifecycle: store -> query -> retrieve."""

    @pytest.mark.asyncio
    async def test_store_query_retrieve_cycle(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test complete lifecycle of storing, querying, and retrieving decisions."""
        # Step 1: Store decisions
        decisions_to_store = [
            Decision(
                id="lifecycle-001",
                agent_type="Architect",
                context="Selecting PostgreSQL for relational data storage",
                rationale="ACID compliance and strong query support",
                outcome="Database performing well under load",
                decision_type=DecisionType.ARCHITECTURE_CHOICE,
                artifact_type="design",
                artifact_ids=("adr-001",),
            ),
            Decision(
                id="lifecycle-002",
                agent_type="PM",
                context="Prioritizing security features in sprint 1",
                rationale="Foundation for all other features",
                decision_type=DecisionType.STORY_PRIORITIZATION,
                artifact_type="story",
            ),
            Decision(
                id="lifecycle-003",
                agent_type="Dev",
                context="Implementing retry logic with exponential backoff",
                rationale="Handle transient failures gracefully",
                decision_type=DecisionType.IMPLEMENTATION_APPROACH,
                artifact_type="code",
            ),
        ]

        for decision in decisions_to_store:
            await decision_store.store_decision(decision)

        # Step 2: Query by semantic similarity
        results = await query_engine.find_similar_decisions(
            context="database selection for persistent storage",
            k=3,
        )

        assert len(results) >= 1
        # PostgreSQL decision should be most similar
        assert any("PostgreSQL" in r.decision.context for r in results)

        # Step 3: Retrieve by ID
        decision = await decision_store.get_decision_by_id("lifecycle-001")
        assert decision is not None
        assert decision.agent_type == "Architect"
        assert decision.outcome == "Database performing well under load"

        # Step 4: Retrieve by agent type
        architect_decisions = await query_engine.get_agent_decision_history(
            agent_type="Architect",
            limit=10,
        )
        assert len(architect_decisions) == 1
        assert architect_decisions[0].id == "lifecycle-001"

    @pytest.mark.asyncio
    async def test_update_decision_through_upsert(
        self,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that storing same ID updates the decision."""
        # Store initial decision
        initial = Decision(
            id="upsert-001",
            agent_type="TEA",
            context="Initial test strategy",
            rationale="80% coverage target",
        )
        await decision_store.store_decision(initial)

        # Update with new information
        updated = Decision(
            id="upsert-001",
            agent_type="TEA",
            context="Updated test strategy",
            rationale="90% coverage target after review",
            outcome="Coverage improved to 92%",
        )
        await decision_store.store_decision(updated)

        # Retrieve and verify update
        decision = await decision_store.get_decision_by_id("upsert-001")
        assert decision is not None
        assert decision.context == "Updated test strategy"
        assert decision.rationale == "90% coverage target after review"
        assert decision.outcome == "Coverage improved to 92%"


class TestFilteringCapabilities:
    """Tests for filtering by agent type, time range, and artifact type."""

    @pytest.mark.asyncio
    async def test_filter_by_agent_type(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test filtering decisions by agent type (AC4)."""
        # Store decisions from different agents
        agents = ["Analyst", "PM", "Architect", "Dev", "SM", "TEA"]
        for i, agent in enumerate(agents):
            await decision_store.store_decision(
                Decision(
                    id=f"agent-{i:03d}",
                    agent_type=agent,
                    context=f"Decision by {agent}",
                    rationale=f"Rationale from {agent}",
                )
            )

        # Filter by each agent type
        for agent in agents:
            filters = DecisionFilter(agent_type=agent)
            results = await query_engine.search_with_filters(
                query="decision",
                filters=filters,
            )
            assert all(r.decision.agent_type == agent for r in results)

    @pytest.mark.asyncio
    async def test_filter_by_artifact_type(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test filtering decisions by artifact type (AC4)."""
        artifact_types = ["requirement", "story", "design", "code"]

        for i, artifact_type in enumerate(artifact_types):
            await decision_store.store_decision(
                Decision(
                    id=f"artifact-{i:03d}",
                    agent_type="Dev",
                    context=f"Decision about {artifact_type}",
                    rationale=f"Handling {artifact_type} artifacts",
                    artifact_type=artifact_type,
                )
            )

        # Filter by each artifact type
        for artifact_type in artifact_types:
            filters = DecisionFilter(artifact_type=artifact_type)
            results = await query_engine.search_with_filters(
                query="artifact",
                filters=filters,
            )
            assert all(r.decision.artifact_type == artifact_type for r in results)

    @pytest.mark.asyncio
    async def test_filter_by_decision_type(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test filtering by decision type (AC4)."""
        decision_types = [
            DecisionType.REQUIREMENT_CLARIFICATION,
            DecisionType.STORY_PRIORITIZATION,
            DecisionType.ARCHITECTURE_CHOICE,
            DecisionType.IMPLEMENTATION_APPROACH,
            DecisionType.TEST_STRATEGY,
            DecisionType.CONFLICT_RESOLUTION,
        ]

        for i, dtype in enumerate(decision_types):
            await decision_store.store_decision(
                Decision(
                    id=f"dtype-{i:03d}",
                    agent_type="SM",
                    context=f"Decision of type {dtype.value}",
                    rationale=f"Handling {dtype.value}",
                    decision_type=dtype,
                )
            )

        # Filter by each decision type
        for dtype in decision_types:
            filters = DecisionFilter(decision_type=dtype)
            results = await query_engine.search_with_filters(
                query="decision type",
                filters=filters,
            )
            assert all(r.decision.decision_type == dtype for r in results)

    @pytest.mark.asyncio
    async def test_combined_filters(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test combining multiple filter conditions."""
        # Store diverse decisions
        await decision_store.store_decision(
            Decision(
                id="combo-001",
                agent_type="Architect",
                context="Database architecture",
                rationale="Scalability concerns",
                decision_type=DecisionType.ARCHITECTURE_CHOICE,
                artifact_type="design",
            )
        )
        await decision_store.store_decision(
            Decision(
                id="combo-002",
                agent_type="Architect",
                context="API design pattern",
                rationale="REST for simplicity",
                decision_type=DecisionType.IMPLEMENTATION_APPROACH,
                artifact_type="design",
            )
        )
        await decision_store.store_decision(
            Decision(
                id="combo-003",
                agent_type="Dev",
                context="Code implementation",
                rationale="Following patterns",
                decision_type=DecisionType.IMPLEMENTATION_APPROACH,
                artifact_type="code",
            )
        )

        # Apply combined filters
        filters = DecisionFilter(
            agent_type="Architect",
            decision_type=DecisionType.ARCHITECTURE_CHOICE,
        )
        results = await query_engine.search_with_filters(
            query="architecture",
            filters=filters,
        )

        assert len(results) == 1
        assert results[0].decision.id == "combo-001"


class TestSemanticSimilarity:
    """Tests for semantic similarity matching."""

    @pytest.mark.asyncio
    async def test_semantic_matching_returns_relevant_results(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that semantic search returns contextually relevant decisions (AC2)."""
        # Store decisions with distinct contexts
        await decision_store.store_decision(
            Decision(
                id="sem-001",
                agent_type="Architect",
                context="Choosing PostgreSQL for transactional data with ACID requirements",
                rationale="Strong consistency guarantees and mature ecosystem",
            )
        )
        await decision_store.store_decision(
            Decision(
                id="sem-002",
                agent_type="Dev",
                context="Implementing user authentication with JWT tokens",
                rationale="Stateless authentication for scalability",
            )
        )
        await decision_store.store_decision(
            Decision(
                id="sem-003",
                agent_type="TEA",
                context="Setting up pytest fixtures for database testing",
                rationale="Isolated test environment for each test case",
            )
        )

        # Query for database-related decisions
        results = await query_engine.find_similar_decisions(
            context="SQL database selection for storing user records",
            k=3,
        )

        # PostgreSQL decision should be most relevant
        assert len(results) >= 1
        top_result = results[0]
        assert (
            "PostgreSQL" in top_result.decision.context
            or "database" in top_result.decision.context.lower()
        )

    @pytest.mark.asyncio
    async def test_similarity_scores_are_normalized(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that similarity scores are between 0 and 1 (AC2)."""
        await decision_store.store_decision(
            Decision(
                id="score-001",
                agent_type="PM",
                context="Sprint planning decision",
                rationale="Velocity-based planning",
            )
        )

        results = await query_engine.find_similar_decisions(
            context="planning sprints",
            k=5,
        )

        assert len(results) >= 1
        for result in results:
            assert 0.0 <= result.similarity <= 1.0

    @pytest.mark.asyncio
    async def test_results_ranked_by_relevance(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that results are ordered by relevance score (AC2)."""
        # Store multiple decisions
        for i in range(5):
            await decision_store.store_decision(
                Decision(
                    id=f"rank-{i:03d}",
                    agent_type="Dev",
                    context=f"Decision number {i}",
                    rationale=f"Rationale {i}",
                )
            )

        results = await query_engine.find_similar_decisions(
            context="development decision",
            k=5,
        )

        # Verify results are ordered by similarity (descending)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].similarity >= results[i + 1].similarity


class TestPerformanceRequirements:
    """Tests for performance requirements (AC5: < 500ms retrieval)."""

    @pytest.mark.asyncio
    async def test_query_performance_under_500ms(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that query retrieval is under 500ms (AC5)."""
        # Store a reasonable number of decisions
        for i in range(50):
            await decision_store.store_decision(
                Decision(
                    id=f"perf-{i:03d}",
                    agent_type=["Analyst", "PM", "Architect", "Dev", "SM", "TEA"][i % 6],
                    context=f"Performance test decision {i}",
                    rationale=f"Testing query performance with decision {i}",
                    decision_type=DecisionType.IMPLEMENTATION_APPROACH,
                )
            )

        # Measure query time
        start = time.perf_counter()
        results = await query_engine.find_similar_decisions(
            context="implementation approach for testing",
            k=10,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Query took {elapsed_ms:.2f}ms, exceeds 500ms limit"
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_filtered_query_performance_under_500ms(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that filtered queries are under 500ms (AC5)."""
        # Store decisions
        for i in range(50):
            await decision_store.store_decision(
                Decision(
                    id=f"fperf-{i:03d}",
                    agent_type="Architect" if i % 3 == 0 else "Dev",
                    context=f"Filtered performance test {i}",
                    rationale=f"Testing filtered query performance {i}",
                    artifact_type="design" if i % 2 == 0 else "code",
                )
            )

        # Measure filtered query time
        filters = DecisionFilter(agent_type="Architect", artifact_type="design")
        start = time.perf_counter()
        _results = await query_engine.search_with_filters(
            query="architecture design",
            filters=filters,
            k=10,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Filtered query took {elapsed_ms:.2f}ms, exceeds 500ms limit"

    @pytest.mark.asyncio
    async def test_agent_history_retrieval_performance(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that agent history retrieval is under 500ms (AC5)."""
        # Store decisions from one agent
        for i in range(30):
            await decision_store.store_decision(
                Decision(
                    id=f"hist-{i:03d}",
                    agent_type="Architect",
                    context=f"Architect decision {i}",
                    rationale=f"Rationale {i}",
                )
            )

        # Measure retrieval time
        start = time.perf_counter()
        decisions = await query_engine.get_agent_decision_history(
            agent_type="Architect",
            limit=10,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"History retrieval took {elapsed_ms:.2f}ms, exceeds 500ms limit"
        assert len(decisions) <= 10


class TestDecisionRationalePreservation:
    """Tests for decision rationale inclusion (AC3)."""

    @pytest.mark.asyncio
    async def test_full_rationale_preserved(
        self,
        decision_store: ChromaDecisionStore,
        query_engine: DecisionQueryEngine,
    ) -> None:
        """Test that full decision rationale is preserved in results (AC3)."""
        long_rationale = """
        After extensive analysis of the requirements and constraints, we determined
        that PostgreSQL would be the optimal choice for several reasons:
        1. ACID compliance ensures data integrity for financial transactions
        2. Strong support for complex queries with CTEs and window functions
        3. Excellent ecosystem with tools like pgAdmin and pg_dump
        4. Proven track record in production environments at scale
        5. Active community and long-term support guarantees
        """.strip()

        await decision_store.store_decision(
            Decision(
                id="rationale-001",
                agent_type="Architect",
                context="Database selection for financial application",
                rationale=long_rationale,
            )
        )

        # Retrieve and verify rationale is complete
        decision = await decision_store.get_decision_by_id("rationale-001")
        assert decision is not None
        assert decision.rationale == long_rationale

    @pytest.mark.asyncio
    async def test_outcome_recorded_when_available(
        self,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that outcome is recorded when available (AC3)."""
        await decision_store.store_decision(
            Decision(
                id="outcome-001",
                agent_type="Dev",
                context="Caching strategy selection",
                rationale="Redis for low-latency cache",
                outcome="Cache hit rate improved to 95%, response time reduced by 40%",
            )
        )

        decision = await decision_store.get_decision_by_id("outcome-001")
        assert decision is not None
        assert decision.outcome == "Cache hit rate improved to 95%, response time reduced by 40%"

    @pytest.mark.asyncio
    async def test_context_preserved_in_search_results(
        self,
        query_engine: DecisionQueryEngine,
        decision_store: ChromaDecisionStore,
    ) -> None:
        """Test that original context is preserved in search results (AC3)."""
        original_context = "Implementing message queue for async processing of user notifications"

        await decision_store.store_decision(
            Decision(
                id="context-001",
                agent_type="Architect",
                context=original_context,
                rationale="RabbitMQ for reliable message delivery",
            )
        )

        results = await query_engine.find_similar_decisions(
            context="async message processing",
            k=5,
        )

        assert len(results) >= 1
        found = any(r.decision.context == original_context for r in results)
        assert found, "Original context not preserved in search results"


class TestProjectIsolation:
    """Tests for project isolation via collection naming."""

    @pytest.mark.asyncio
    async def test_decisions_isolated_by_project(
        self,
        tmp_path: pytest.fixture,
    ) -> None:
        """Test that decisions are isolated between projects (AC5)."""
        # Create stores for two different projects
        store_a = ChromaDecisionStore(
            persist_directory=str(tmp_path / "shared"),
            project_id="project-a",
        )
        store_b = ChromaDecisionStore(
            persist_directory=str(tmp_path / "shared"),
            project_id="project-b",
        )

        # Store decision in project A
        await store_a.store_decision(
            Decision(
                id="isolated-001",
                agent_type="Dev",
                context="Project A decision",
                rationale="Only visible in project A",
            )
        )

        # Store decision in project B
        await store_b.store_decision(
            Decision(
                id="isolated-002",
                agent_type="Dev",
                context="Project B decision",
                rationale="Only visible in project B",
            )
        )

        # Verify isolation
        a_decisions = await store_a.get_decisions_by_agent("Dev", limit=10)
        b_decisions = await store_b.get_decisions_by_agent("Dev", limit=10)

        assert len(a_decisions) == 1
        assert a_decisions[0].id == "isolated-001"
        assert len(b_decisions) == 1
        assert b_decisions[0].id == "isolated-002"

        # Verify cross-project query doesn't leak
        decision_a = await store_a.get_decision_by_id("isolated-002")
        decision_b = await store_b.get_decision_by_id("isolated-001")

        assert decision_a is None, "Project B decision should not be visible in Project A"
        assert decision_b is None, "Project A decision should not be visible in Project B"
