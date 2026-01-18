"""Tests for AuditFilterService (Story 11.7).

Tests cover:
- filter_decisions() with various filters
- filter_artifacts() with type and time filters
- filter_costs() with agent and session filters
- filter_all() combined results
- filter service with None cost_store
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.filter_service import (
    AuditFilterService,
    get_audit_filter_service,
)
from yolo_developer.audit.filter_types import AuditFilters
from yolo_developer.audit.memory_store import InMemoryDecisionStore
from yolo_developer.audit.traceability_memory_store import InMemoryTraceabilityStore
from yolo_developer.audit.traceability_types import TraceableArtifact
from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext


def _make_decision(
    decision_id: str,
    agent_name: str = "analyst",
    decision_type: str = "requirement_analysis",
    timestamp: str = "2026-01-18T12:00:00Z",
    sprint_id: str | None = None,
    story_id: str | None = None,
) -> Decision:
    """Helper to create test decisions."""
    return Decision(
        id=decision_id,
        decision_type=decision_type,  # type: ignore[arg-type]
        content=f"Test decision {decision_id}",
        rationale="Test rationale",
        agent=AgentIdentity(
            agent_name=agent_name,
            agent_type=agent_name,
            session_id="session-123",
        ),
        context=DecisionContext(
            sprint_id=sprint_id,
            story_id=story_id,
        ),
        timestamp=timestamp,
    )


def _make_artifact(
    artifact_id: str,
    artifact_type: str = "requirement",
    created_at: str = "2026-01-18T12:00:00Z",
) -> TraceableArtifact:
    """Helper to create test artifacts."""
    return TraceableArtifact(
        id=artifact_id,
        artifact_type=artifact_type,
        name=f"Test {artifact_type} {artifact_id}",
        description=f"Description for {artifact_id}",
        created_at=created_at,
    )


class TestAuditFilterServiceCreation:
    """Tests for AuditFilterService creation and factory function."""

    def test_create_service_with_all_stores(self) -> None:
        """Test creating service with all stores."""
        from yolo_developer.audit.cost_memory_store import InMemoryCostStore

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        cost_store = InMemoryCostStore()

        service = AuditFilterService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            cost_store=cost_store,
        )

        assert service._decision_store is decision_store
        assert service._traceability_store is traceability_store
        assert service._cost_store is cost_store

    def test_create_service_without_cost_store(self) -> None:
        """Test creating service without cost store."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        service = AuditFilterService(
            decision_store=decision_store,
            traceability_store=traceability_store,
        )

        assert service._decision_store is decision_store
        assert service._traceability_store is traceability_store
        assert service._cost_store is None

    def test_factory_function(self) -> None:
        """Test get_audit_filter_service factory function."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        service = get_audit_filter_service(
            decision_store=decision_store,
            traceability_store=traceability_store,
        )

        assert isinstance(service, AuditFilterService)


class TestFilterDecisions:
    """Tests for filter_decisions method."""

    @pytest.mark.asyncio
    async def test_filter_decisions_by_agent_name(self) -> None:
        """Test filtering decisions by agent name."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        # Add decisions from different agents
        await decision_store.log_decision(_make_decision("dec-1", agent_name="analyst"))
        await decision_store.log_decision(_make_decision("dec-2", agent_name="pm"))
        await decision_store.log_decision(_make_decision("dec-3", agent_name="analyst"))

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(agent_name="analyst")

        results = await service.filter_decisions(filters)

        assert len(results) == 2
        assert all(d.agent.agent_name == "analyst" for d in results)

    @pytest.mark.asyncio
    async def test_filter_decisions_by_time_range(self) -> None:
        """Test filtering decisions by time range."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        await decision_store.log_decision(_make_decision("dec-1", timestamp="2026-01-01T00:00:00Z"))
        await decision_store.log_decision(_make_decision("dec-2", timestamp="2026-01-15T00:00:00Z"))
        await decision_store.log_decision(_make_decision("dec-3", timestamp="2026-01-31T00:00:00Z"))

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(
            start_time="2026-01-10T00:00:00Z",
            end_time="2026-01-20T00:00:00Z",
        )

        results = await service.filter_decisions(filters)

        assert len(results) == 1
        assert results[0].id == "dec-2"

    @pytest.mark.asyncio
    async def test_filter_decisions_by_decision_type(self) -> None:
        """Test filtering decisions by decision type."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        await decision_store.log_decision(
            _make_decision("dec-1", decision_type="requirement_analysis")
        )
        await decision_store.log_decision(
            _make_decision("dec-2", decision_type="architecture_choice")
        )

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(decision_type="architecture_choice")

        results = await service.filter_decisions(filters)

        assert len(results) == 1
        assert results[0].decision_type == "architecture_choice"

    @pytest.mark.asyncio
    async def test_filter_decisions_empty_filters_returns_all(self) -> None:
        """Test that empty filters return all decisions."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        await decision_store.log_decision(_make_decision("dec-1"))
        await decision_store.log_decision(_make_decision("dec-2"))

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters()

        results = await service.filter_decisions(filters)

        assert len(results) == 2


class TestFilterArtifacts:
    """Tests for filter_artifacts method."""

    @pytest.mark.asyncio
    async def test_filter_artifacts_by_type(self) -> None:
        """Test filtering artifacts by artifact type."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        await traceability_store.register_artifact(
            _make_artifact("req-1", artifact_type="requirement")
        )
        await traceability_store.register_artifact(_make_artifact("story-1", artifact_type="story"))
        await traceability_store.register_artifact(
            _make_artifact("req-2", artifact_type="requirement")
        )

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(artifact_type="requirement")

        results = await service.filter_artifacts(filters)

        assert len(results) == 2
        assert all(a.artifact_type == "requirement" for a in results)

    @pytest.mark.asyncio
    async def test_filter_artifacts_by_time_range(self) -> None:
        """Test filtering artifacts by time range."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        await traceability_store.register_artifact(
            _make_artifact("req-1", created_at="2026-01-01T00:00:00Z")
        )
        await traceability_store.register_artifact(
            _make_artifact("req-2", created_at="2026-01-15T00:00:00Z")
        )
        await traceability_store.register_artifact(
            _make_artifact("req-3", created_at="2026-01-31T00:00:00Z")
        )

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(
            start_time="2026-01-10T00:00:00Z",
            end_time="2026-01-20T00:00:00Z",
        )

        results = await service.filter_artifacts(filters)

        assert len(results) == 1
        assert results[0].id == "req-2"

    @pytest.mark.asyncio
    async def test_filter_artifacts_combined_filters(self) -> None:
        """Test filtering artifacts with combined filters."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        await traceability_store.register_artifact(
            _make_artifact("req-1", artifact_type="requirement", created_at="2026-01-01T00:00:00Z")
        )
        await traceability_store.register_artifact(
            _make_artifact("req-2", artifact_type="requirement", created_at="2026-01-15T00:00:00Z")
        )
        await traceability_store.register_artifact(
            _make_artifact("story-1", artifact_type="story", created_at="2026-01-15T00:00:00Z")
        )

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(
            artifact_type="requirement",
            start_time="2026-01-10T00:00:00Z",
        )

        results = await service.filter_artifacts(filters)

        assert len(results) == 1
        assert results[0].id == "req-2"
        assert results[0].artifact_type == "requirement"


class TestFilterCosts:
    """Tests for filter_costs method."""

    @pytest.mark.asyncio
    async def test_filter_costs_without_cost_store_returns_empty(self) -> None:
        """Test that filter_costs returns empty list when no cost store."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        service = AuditFilterService(decision_store, traceability_store, cost_store=None)
        filters = AuditFilters(agent_name="analyst")

        results = await service.filter_costs(filters)

        assert results == []

    @pytest.mark.asyncio
    async def test_filter_costs_by_agent_name(self) -> None:
        """Test filtering costs by agent name."""
        from yolo_developer.audit.cost_memory_store import InMemoryCostStore
        from yolo_developer.audit.cost_types import CostRecord, TokenUsage

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        cost_store = InMemoryCostStore()

        # Add cost records for different agents
        cost1 = CostRecord(
            id="cost-1",
            timestamp="2026-01-18T12:00:00Z",
            agent_name="analyst",
            session_id="session-123",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost_usd=0.001,
        )
        cost2 = CostRecord(
            id="cost-2",
            timestamp="2026-01-18T12:00:00Z",
            agent_name="pm",
            session_id="session-123",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost_usd=0.001,
        )

        await cost_store.store_cost(cost1)
        await cost_store.store_cost(cost2)

        service = AuditFilterService(decision_store, traceability_store, cost_store)
        filters = AuditFilters(agent_name="analyst")

        results = await service.filter_costs(filters)

        assert len(results) == 1
        assert results[0].agent_name == "analyst"


class TestFilterAll:
    """Tests for filter_all method."""

    @pytest.mark.asyncio
    async def test_filter_all_returns_combined_results(self) -> None:
        """Test that filter_all returns results from all stores."""
        from yolo_developer.audit.cost_memory_store import InMemoryCostStore
        from yolo_developer.audit.cost_types import CostRecord, TokenUsage

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        cost_store = InMemoryCostStore()

        # Add test data
        await decision_store.log_decision(_make_decision("dec-1", agent_name="analyst"))
        await traceability_store.register_artifact(_make_artifact("req-1"))
        await cost_store.store_cost(
            CostRecord(
                id="cost-1",
                timestamp="2026-01-18T12:00:00Z",
                agent_name="analyst",
                session_id="session-123",
                model="gpt-4o-mini",
                tier="routine",
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                cost_usd=0.001,
            )
        )

        service = AuditFilterService(decision_store, traceability_store, cost_store)
        filters = AuditFilters()

        results = await service.filter_all(filters)

        assert "decisions" in results
        assert "artifacts" in results
        assert "costs" in results
        assert "filters_applied" in results
        assert len(results["decisions"]) == 1
        assert len(results["artifacts"]) == 1
        assert len(results["costs"]) == 1

    @pytest.mark.asyncio
    async def test_filter_all_includes_filters_applied(self) -> None:
        """Test that filter_all includes filters_applied in results."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(
            agent_name="analyst",
            artifact_type="requirement",
        )

        results = await service.filter_all(filters)

        assert results["filters_applied"]["agent_name"] == "analyst"
        assert results["filters_applied"]["artifact_type"] == "requirement"

    @pytest.mark.asyncio
    async def test_filter_all_with_combined_filters(self) -> None:
        """Test filter_all with combined filters applied."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        # Add data for different agents
        await decision_store.log_decision(_make_decision("dec-1", agent_name="analyst"))
        await decision_store.log_decision(_make_decision("dec-2", agent_name="pm"))
        await traceability_store.register_artifact(
            _make_artifact("req-1", artifact_type="requirement")
        )
        await traceability_store.register_artifact(_make_artifact("story-1", artifact_type="story"))

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(
            agent_name="analyst",
            artifact_type="requirement",
        )

        results = await service.filter_all(filters)

        assert len(results["decisions"]) == 1
        assert results["decisions"][0].agent.agent_name == "analyst"
        assert len(results["artifacts"]) == 1
        assert results["artifacts"][0].artifact_type == "requirement"

    @pytest.mark.asyncio
    async def test_filter_all_empty_results(self) -> None:
        """Test filter_all with no matching data."""
        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        service = AuditFilterService(decision_store, traceability_store)
        filters = AuditFilters(agent_name="nonexistent")

        results = await service.filter_all(filters)

        assert results["decisions"] == []
        assert results["artifacts"] == []
        assert results["costs"] == []
