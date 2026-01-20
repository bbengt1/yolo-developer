"""Tests for audit view service (Story 11.3).

Tests cover:
- AuditViewService class
- view_decisions retrieves and formats decisions
- view_decision retrieves single decision
- view_trace_chain navigates and formats trace chain
- view_coverage generates coverage report
- view_summary generates summary statistics
- get_audit_view_service factory function
- Filters and format options
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.memory_store import InMemoryDecisionStore
from yolo_developer.audit.plain_formatter import PlainAuditFormatter
from yolo_developer.audit.store import DecisionFilters
from yolo_developer.audit.traceability_memory_store import InMemoryTraceabilityStore
from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
from yolo_developer.audit.types import (
    AgentIdentity,
    Decision,
    DecisionContext,
)


@pytest.fixture
def decision_store() -> InMemoryDecisionStore:
    """Create an in-memory decision store."""
    return InMemoryDecisionStore()


@pytest.fixture
def traceability_store() -> InMemoryTraceabilityStore:
    """Create an in-memory traceability store."""
    return InMemoryTraceabilityStore()


@pytest.fixture
def formatter() -> PlainAuditFormatter:
    """Create a plain text formatter."""
    return PlainAuditFormatter()


@pytest.fixture
def sample_decision() -> Decision:
    """Create a sample decision."""
    return Decision(
        id="dec-001",
        decision_type="requirement_analysis",
        content="OAuth2 authentication required",
        rationale="Industry standard security",
        agent=AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-123",
        ),
        context=DecisionContext(sprint_id="sprint-1"),
        timestamp="2026-01-18T10:00:00Z",
        severity="critical",
    )


@pytest.fixture
def sample_artifact() -> TraceableArtifact:
    """Create a sample artifact."""
    return TraceableArtifact(
        id="req-001",
        artifact_type="requirement",
        name="User Authentication",
        description="Users must authenticate",
        created_at="2026-01-18T09:00:00Z",
    )


class TestAuditViewService:
    """Tests for AuditViewService class."""

    def test_view_service_exists(self) -> None:
        """Test that AuditViewService class exists."""
        from yolo_developer.audit.view import AuditViewService

        assert AuditViewService is not None

    def test_view_service_constructor(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
    ) -> None:
        """Test AuditViewService constructor."""
        from yolo_developer.audit.view import AuditViewService

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        assert service._decision_store is decision_store
        assert service._traceability_store is traceability_store
        assert service._formatter is formatter

    def test_view_service_default_formatter(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
    ) -> None:
        """Test AuditViewService uses default formatter when none provided."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter
        from yolo_developer.audit.view import AuditViewService

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
        )
        assert isinstance(service._formatter, PlainAuditFormatter)


class TestViewDecisions:
    """Tests for view_decisions method."""

    @pytest.mark.asyncio
    async def test_view_decisions_returns_string(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
    ) -> None:
        """Test that view_decisions returns a string."""
        from yolo_developer.audit.view import AuditViewService

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_decisions()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_view_decisions_with_data(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
        sample_decision: Decision,
    ) -> None:
        """Test view_decisions with stored decisions."""
        from yolo_developer.audit.view import AuditViewService

        await decision_store.log_decision(sample_decision)

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_decisions()
        assert "dec-001" in result

    @pytest.mark.asyncio
    async def test_view_decisions_with_filters(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
    ) -> None:
        """Test view_decisions with filters applied."""
        from yolo_developer.audit.view import AuditViewService

        # Store multiple decisions
        agent = AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1")
        context = DecisionContext(sprint_id="sprint-1")

        await decision_store.log_decision(
            Decision(
                id="dec-analyst",
                decision_type="requirement_analysis",
                content="Analyst decision",
                rationale="Rationale",
                agent=agent,
                context=context,
                timestamp="2026-01-18T10:00:00Z",
            )
        )

        pm_agent = AgentIdentity(agent_name="pm", agent_type="pm", session_id="s1")
        await decision_store.log_decision(
            Decision(
                id="dec-pm",
                decision_type="story_creation",
                content="PM decision",
                rationale="Rationale",
                agent=pm_agent,
                context=context,
                timestamp="2026-01-18T11:00:00Z",
            )
        )

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )

        # Filter to only analyst decisions
        filters = DecisionFilters(agent_name="analyst")
        result = await service.view_decisions(filters=filters)
        assert "dec-analyst" in result
        assert "dec-pm" not in result


class TestViewDecision:
    """Tests for view_decision method."""

    @pytest.mark.asyncio
    async def test_view_decision_returns_string(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
        sample_decision: Decision,
    ) -> None:
        """Test that view_decision returns a string."""
        from yolo_developer.audit.view import AuditViewService

        await decision_store.log_decision(sample_decision)

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_decision("dec-001")
        assert isinstance(result, str)
        assert "dec-001" in result

    @pytest.mark.asyncio
    async def test_view_decision_not_found(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
    ) -> None:
        """Test view_decision with non-existent ID."""
        from yolo_developer.audit.view import AuditViewService

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_decision("nonexistent")
        assert "not found" in result.lower()


class TestViewTraceChain:
    """Tests for view_trace_chain method."""

    @pytest.mark.asyncio
    async def test_view_trace_chain_returns_string(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
        sample_artifact: TraceableArtifact,
    ) -> None:
        """Test that view_trace_chain returns a string."""
        from yolo_developer.audit.view import AuditViewService

        await traceability_store.register_artifact(sample_artifact)

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_trace_chain("req-001", "downstream")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_view_trace_chain_with_chain(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
    ) -> None:
        """Test view_trace_chain with connected artifacts."""
        from yolo_developer.audit.view import AuditViewService

        # Create artifacts
        req = TraceableArtifact(
            id="req-001",
            artifact_type="requirement",
            name="User Auth",
            description="Auth requirement",
            created_at="2026-01-18T09:00:00Z",
        )
        story = TraceableArtifact(
            id="story-001",
            artifact_type="story",
            name="Login Story",
            description="Login story",
            created_at="2026-01-18T10:00:00Z",
        )

        await traceability_store.register_artifact(req)
        await traceability_store.register_artifact(story)

        # Create link
        link = TraceLink(
            id="link-001",
            source_id="story-001",
            source_type="story",
            target_id="req-001",
            target_type="requirement",
            link_type="derives_from",
            created_at="2026-01-18T10:00:00Z",
        )
        await traceability_store.create_link(link)

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_trace_chain("req-001", "downstream")
        assert "User Auth" in result


class TestViewCoverage:
    """Tests for view_coverage method."""

    @pytest.mark.asyncio
    async def test_view_coverage_returns_string(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
    ) -> None:
        """Test that view_coverage returns a string."""
        from yolo_developer.audit.view import AuditViewService

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_coverage()
        assert isinstance(result, str)


class TestViewSummary:
    """Tests for view_summary method."""

    @pytest.mark.asyncio
    async def test_view_summary_returns_string(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
        sample_decision: Decision,
    ) -> None:
        """Test that view_summary returns a string."""
        from yolo_developer.audit.view import AuditViewService

        await decision_store.log_decision(sample_decision)

        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        result = await service.view_summary()
        assert isinstance(result, str)


class TestFactoryFunction:
    """Tests for get_audit_view_service factory function."""

    def test_factory_function_exists(self) -> None:
        """Test that get_audit_view_service function exists."""
        from yolo_developer.audit.view import get_audit_view_service

        assert get_audit_view_service is not None

    def test_factory_creates_service(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
    ) -> None:
        """Test that factory creates AuditViewService."""
        from yolo_developer.audit.view import AuditViewService, get_audit_view_service

        service = get_audit_view_service(
            decision_store=decision_store,
            traceability_store=traceability_store,
        )
        assert isinstance(service, AuditViewService)

    def test_factory_with_custom_formatter(
        self,
        decision_store: InMemoryDecisionStore,
        traceability_store: InMemoryTraceabilityStore,
        formatter: PlainAuditFormatter,
    ) -> None:
        """Test that factory accepts custom formatter."""
        from yolo_developer.audit.view import get_audit_view_service

        service = get_audit_view_service(
            decision_store=decision_store,
            traceability_store=traceability_store,
            formatter=formatter,
        )
        assert service._formatter is formatter
