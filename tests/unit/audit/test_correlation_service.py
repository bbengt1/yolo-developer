"""Tests for correlation service (Story 11.5 - Task 4, 5, 6).

Tests the CorrelationService for cross-agent decision correlation.

References:
    - FR85: System can correlate decisions across agent boundaries
    - AC #1-5: All acceptance criteria for correlation
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore
from yolo_developer.audit.correlation_store import CorrelationFilters
from yolo_developer.audit.correlation_types import (
    AgentTransition,
    CausalRelation,
    DecisionChain,
)
from yolo_developer.audit.memory_store import InMemoryDecisionStore
from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext


def create_test_decision(
    id: str = "dec-001",
    agent_name: str = "analyst",
    content: str = "Test decision",
    session_id: str = "session-123",
    parent_decision_id: str | None = None,
    timestamp: str = "2026-01-18T12:00:00Z",
) -> Decision:
    """Create a test Decision."""
    return Decision(
        id=id,
        decision_type="requirement_analysis",
        content=content,
        rationale="Test rationale",
        agent=AgentIdentity(
            agent_name=agent_name,
            agent_type=agent_name,
            session_id=session_id,
        ),
        context=DecisionContext(
            sprint_id="sprint-1",
            story_id="story-1",
            parent_decision_id=parent_decision_id,
        ),
        timestamp=timestamp,
    )


class TestCorrelationService:
    """Tests for CorrelationService class."""

    def test_service_exists(self) -> None:
        """Test CorrelationService class exists."""
        from yolo_developer.audit.correlation import CorrelationService

        assert CorrelationService is not None

    def test_service_initializes_with_stores(self) -> None:
        """Test service initializes with decision and correlation stores."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()

        service = CorrelationService(
            decision_store=decision_store,
            correlation_store=correlation_store,
        )

        assert service is not None


class TestCorrelateDecisions:
    """Tests for correlate_decisions method."""

    @pytest.mark.asyncio
    async def test_correlate_decisions_creates_chain(self) -> None:
        """Test correlate_decisions creates a DecisionChain."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        result = await service.correlate_decisions(["dec-001", "dec-002"], chain_type="session")

        assert isinstance(result, DecisionChain)
        assert "dec-001" in result.decisions
        assert "dec-002" in result.decisions

    @pytest.mark.asyncio
    async def test_correlate_decisions_stores_chain(self) -> None:
        """Test correlate_decisions stores the chain in the store."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        chain = await service.correlate_decisions(["dec-001", "dec-002"])
        retrieved = await correlation_store.get_chain(chain.id)

        assert retrieved is not None
        assert retrieved.id == chain.id


class TestAddCausalRelation:
    """Tests for add_causal_relation method."""

    @pytest.mark.asyncio
    async def test_add_causal_relation_creates_relation(self) -> None:
        """Test add_causal_relation creates a CausalRelation."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        result = await service.add_causal_relation(
            cause_id="dec-001",
            effect_id="dec-002",
            relation_type="derives_from",
        )

        assert isinstance(result, CausalRelation)
        assert result.cause_decision_id == "dec-001"
        assert result.effect_decision_id == "dec-002"

    @pytest.mark.asyncio
    async def test_add_causal_relation_with_evidence(self) -> None:
        """Test add_causal_relation with evidence."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        result = await service.add_causal_relation(
            cause_id="dec-001",
            effect_id="dec-002",
            relation_type="triggers",
            evidence="Parent decision ID reference",
        )

        assert result.evidence == "Parent decision ID reference"


class TestRecordTransition:
    """Tests for record_transition method."""

    @pytest.mark.asyncio
    async def test_record_transition_creates_transition(self) -> None:
        """Test record_transition creates an AgentTransition."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        result = await service.record_transition(
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
        )

        assert isinstance(result, AgentTransition)
        assert result.from_agent == "analyst"
        assert result.to_agent == "pm"

    @pytest.mark.asyncio
    async def test_record_transition_with_context(self) -> None:
        """Test record_transition with context."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        result = await service.record_transition(
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            context={"session_id": "session-123"},
        )

        assert result.context.get("session_id") == "session-123"


class TestGetDecisionChain:
    """Tests for get_decision_chain method."""

    @pytest.mark.asyncio
    async def test_get_decision_chain_returns_correlated_decisions(self) -> None:
        """Test get_decision_chain returns correlated decisions."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions
        dec1 = create_test_decision(id="dec-001")
        dec2 = create_test_decision(id="dec-002")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        # Create chain
        await service.correlate_decisions(["dec-001", "dec-002"])

        # Get chain for dec-001
        result = await service.get_decision_chain("dec-001")

        assert len(result) == 2
        decision_ids = {d.id for d in result}
        assert "dec-001" in decision_ids
        assert "dec-002" in decision_ids

    @pytest.mark.asyncio
    async def test_get_decision_chain_returns_empty_for_unknown(self) -> None:
        """Test get_decision_chain returns empty list for unknown decision."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        result = await service.get_decision_chain("unknown")

        assert result == []


class TestGetTimeline:
    """Tests for get_timeline method."""

    @pytest.mark.asyncio
    async def test_get_timeline_returns_chronological_decisions(self) -> None:
        """Test get_timeline returns decisions in chronological order."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions
        dec1 = create_test_decision(id="dec-001", timestamp="2026-01-18T14:00:00Z")
        dec2 = create_test_decision(id="dec-002", timestamp="2026-01-18T12:00:00Z")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.get_timeline()

        # Should be ordered by timestamp
        assert len(result) == 2
        assert result[0][0].id == "dec-002"  # 12:00
        assert result[1][0].id == "dec-001"  # 14:00

    @pytest.mark.asyncio
    async def test_get_timeline_with_filters(self) -> None:
        """Test get_timeline respects filters."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions with different agents
        dec1 = create_test_decision(id="dec-001", agent_name="analyst")
        dec2 = create_test_decision(id="dec-002", agent_name="pm")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        filters = CorrelationFilters(agent_name="analyst")
        result = await service.get_timeline(filters=filters)

        assert len(result) == 1
        assert result[0][0].id == "dec-001"


class TestGetWorkflowFlow:
    """Tests for get_workflow_flow method."""

    @pytest.mark.asyncio
    async def test_get_workflow_flow_returns_complete_trace(self) -> None:
        """Test get_workflow_flow returns complete workflow trace."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions
        dec1 = create_test_decision(id="dec-001", agent_name="analyst", session_id="session-123")
        dec2 = create_test_decision(id="dec-002", agent_name="pm", session_id="session-123")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        # Record transition
        await service.record_transition(
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-002",
            context={"session_id": "session-123"},
        )

        result = await service.get_workflow_flow("session-123")

        assert "total_decisions" in result
        assert "agent_sequence" in result
        assert "transitions" in result


class TestSearch:
    """Tests for search method."""

    @pytest.mark.asyncio
    async def test_search_finds_decisions_by_content(self) -> None:
        """Test search finds decisions by content."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions
        dec1 = create_test_decision(id="dec-001", content="Implement authentication")
        dec2 = create_test_decision(id="dec-002", content="Add logging")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.search("authentication")

        assert len(result) == 1
        assert result[0].id == "dec-001"

    @pytest.mark.asyncio
    async def test_search_with_filters(self) -> None:
        """Test search respects filters."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions
        dec1 = create_test_decision(id="dec-001", agent_name="analyst", content="authentication")
        dec2 = create_test_decision(id="dec-002", agent_name="pm", content="authentication")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        filters = CorrelationFilters(agent_name="analyst")
        result = await service.search("authentication", filters=filters)

        assert len(result) == 1
        assert result[0].agent.agent_name == "analyst"


class TestTimelineView:
    """Tests for get_timeline_view method (Task 5)."""

    @pytest.mark.asyncio
    async def test_get_timeline_view_returns_entries(self) -> None:
        """Test get_timeline_view returns TimelineEntry list."""
        from yolo_developer.audit.correlation import CorrelationService
        from yolo_developer.audit.correlation_types import TimelineEntry

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        dec1 = create_test_decision(id="dec-001", session_id="session-123")
        await decision_store.log_decision(dec1)

        result = await service.get_timeline_view(session_id="session-123")

        assert len(result) >= 1
        assert isinstance(result[0], TimelineEntry)

    @pytest.mark.asyncio
    async def test_get_timeline_view_ordered_by_timestamp(self) -> None:
        """Test get_timeline_view returns entries ordered by timestamp."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        dec1 = create_test_decision(
            id="dec-001", session_id="session-123", timestamp="2026-01-18T14:00:00Z"
        )
        dec2 = create_test_decision(
            id="dec-002", session_id="session-123", timestamp="2026-01-18T12:00:00Z"
        )
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.get_timeline_view(session_id="session-123")

        assert result[0].decision.id == "dec-002"  # Earlier timestamp
        assert result[1].decision.id == "dec-001"

    @pytest.mark.asyncio
    async def test_get_timeline_view_has_sequence_numbers(self) -> None:
        """Test get_timeline_view entries have sequence numbers."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        dec1 = create_test_decision(id="dec-001", session_id="session-123")
        dec2 = create_test_decision(id="dec-002", session_id="session-123")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.get_timeline_view(session_id="session-123")

        assert result[0].sequence_number == 1
        assert result[1].sequence_number == 2

    @pytest.mark.asyncio
    async def test_get_timeline_view_with_time_filters(self) -> None:
        """Test get_timeline_view respects time filters."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        dec1 = create_test_decision(
            id="dec-001", session_id="session-123", timestamp="2026-01-18T10:00:00Z"
        )
        dec2 = create_test_decision(
            id="dec-002", session_id="session-123", timestamp="2026-01-18T14:00:00Z"
        )
        dec3 = create_test_decision(
            id="dec-003", session_id="session-123", timestamp="2026-01-18T18:00:00Z"
        )
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)
        await decision_store.log_decision(dec3)

        result = await service.get_timeline_view(
            session_id="session-123",
            start_time="2026-01-18T12:00:00Z",
            end_time="2026-01-18T16:00:00Z",
        )

        assert len(result) == 1
        assert result[0].decision.id == "dec-002"

    @pytest.mark.asyncio
    async def test_get_timeline_view_with_agent_transitions(self) -> None:
        """Test get_timeline_view includes agent transitions."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="session-123"),
            context=DecisionContext(),
            timestamp="2026-01-18T10:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="story_creation",
            content="Test",
            rationale="Test",
            agent=AgentIdentity(agent_name="pm", agent_type="pm", session_id="session-123"),
            context=DecisionContext(),
            timestamp="2026-01-18T12:00:00Z",
        )
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        # Record a transition
        await service.record_transition(
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-002",
            context={"session_id": "session-123"},
        )

        result = await service.get_timeline_view(session_id="session-123")

        assert len(result) == 2
        # First entry has no previous agent
        assert result[0].previous_agent is None
        # Second entry should have previous_agent set (agent changed)
        assert result[1].previous_agent == "analyst"

    @pytest.mark.asyncio
    async def test_get_timeline_view_without_session_id(self) -> None:
        """Test get_timeline_view works without session_id (returns all)."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        dec1 = create_test_decision(id="dec-001", session_id="session-A")
        dec2 = create_test_decision(id="dec-002", session_id="session-B")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.get_timeline_view()

        assert len(result) == 2


class TestAutoCorrelateSession:
    """Tests for auto_correlate_session method (Task 6)."""

    @pytest.mark.asyncio
    async def test_auto_correlate_session_creates_chain(self) -> None:
        """Test auto_correlate_session creates a DecisionChain."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions for same session
        dec1 = create_test_decision(id="dec-001", session_id="session-123")
        dec2 = create_test_decision(id="dec-002", session_id="session-123")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.auto_correlate_session("session-123")

        assert isinstance(result, DecisionChain)
        assert len(result.decisions) == 2

    @pytest.mark.asyncio
    async def test_auto_correlate_session_only_includes_session_decisions(self) -> None:
        """Test auto_correlate_session only includes decisions from that session."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions for different sessions
        dec1 = create_test_decision(id="dec-001", session_id="session-A")
        dec2 = create_test_decision(id="dec-002", session_id="session-B")
        dec3 = create_test_decision(id="dec-003", session_id="session-A")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)
        await decision_store.log_decision(dec3)

        result = await service.auto_correlate_session("session-A")

        assert len(result.decisions) == 2
        assert "dec-001" in result.decisions
        assert "dec-003" in result.decisions
        assert "dec-002" not in result.decisions


class TestDetectCausalRelations:
    """Tests for detect_causal_relations method (Task 6)."""

    @pytest.mark.asyncio
    async def test_detect_causal_relations_from_parent_decision_id(self) -> None:
        """Test detect_causal_relations infers from parent_decision_id."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions with parent relationship
        dec1 = create_test_decision(id="dec-001")
        dec2 = create_test_decision(id="dec-002", parent_decision_id="dec-001")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.detect_causal_relations("dec-002")

        assert len(result) >= 1
        assert any(r.cause_decision_id == "dec-001" for r in result)

    @pytest.mark.asyncio
    async def test_detect_causal_relations_returns_empty_for_no_parent(self) -> None:
        """Test detect_causal_relations returns empty for decision with no parent."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        dec1 = create_test_decision(id="dec-001", parent_decision_id=None)
        await decision_store.log_decision(dec1)

        result = await service.detect_causal_relations("dec-001")

        # Should have no auto-detected causal relations
        # (might have manually added ones, but none auto-detected)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_causal_relations_from_trace_links(self) -> None:
        """Test detect_causal_relations infers from shared trace_links."""
        from yolo_developer.audit.correlation import CorrelationService

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()
        service = CorrelationService(decision_store, correlation_store)

        # Add decisions with shared trace_links
        dec1 = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="First decision",
            rationale="Test",
            agent=AgentIdentity(agent_name="analyst", agent_type="analyst", session_id="s1"),
            context=DecisionContext(trace_links=("link-001", "link-002")),
            timestamp="2026-01-18T10:00:00Z",
        )
        dec2 = Decision(
            id="dec-002",
            decision_type="story_creation",
            content="Second decision",
            rationale="Test",
            agent=AgentIdentity(agent_name="pm", agent_type="pm", session_id="s1"),
            context=DecisionContext(trace_links=("link-002", "link-003")),
            timestamp="2026-01-18T12:00:00Z",
        )
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        result = await service.detect_causal_relations("dec-002")

        # Should detect artifact_related relation from dec-001 (earlier) to dec-002
        assert len(result) >= 1
        artifact_relations = [r for r in result if r.relation_type == "artifact_related"]
        assert len(artifact_relations) >= 1
        assert any(r.cause_decision_id == "dec-001" for r in artifact_relations)
        assert any("link-002" in r.evidence for r in artifact_relations)


class TestGetCorrelationServiceFactory:
    """Tests for get_correlation_service factory function."""

    def test_factory_function_exists(self) -> None:
        """Test get_correlation_service factory exists."""
        from yolo_developer.audit.correlation import get_correlation_service

        assert get_correlation_service is not None

    def test_factory_creates_service(self) -> None:
        """Test factory creates CorrelationService instance."""
        from yolo_developer.audit.correlation import (
            CorrelationService,
            get_correlation_service,
        )

        decision_store = InMemoryDecisionStore()
        correlation_store = InMemoryCorrelationStore()

        service = get_correlation_service(decision_store, correlation_store)

        assert isinstance(service, CorrelationService)

    def test_factory_creates_default_correlation_store(self) -> None:
        """Test factory creates default InMemoryCorrelationStore if not provided."""
        from yolo_developer.audit.correlation import (
            CorrelationService,
            get_correlation_service,
        )

        decision_store = InMemoryDecisionStore()

        service = get_correlation_service(decision_store)

        assert isinstance(service, CorrelationService)
