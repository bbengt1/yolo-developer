"""Tests for in-memory correlation store (Story 11.5 - Task 3).

Tests the InMemoryCorrelationStore implementation.

References:
    - FR85: System can correlate decisions across agent boundaries
    - AC #1: Decision chains are identified
    - AC #4: Correlations are searchable
"""

from __future__ import annotations

import asyncio

import pytest

from yolo_developer.audit.correlation_types import (
    AgentTransition,
    CausalRelation,
    DecisionChain,
)


def create_test_chain(
    id: str = "chain-001",
    decisions: tuple[str, ...] = ("dec-001", "dec-002"),
    chain_type: str = "session",
) -> DecisionChain:
    """Create a test DecisionChain."""
    return DecisionChain(
        id=id,
        decisions=decisions,
        chain_type=chain_type,  # type: ignore[arg-type]
        created_at="2026-01-18T12:00:00Z",
    )


def create_test_relation(
    id: str = "rel-001",
    cause_id: str = "dec-001",
    effect_id: str = "dec-002",
) -> CausalRelation:
    """Create a test CausalRelation."""
    return CausalRelation(
        id=id,
        cause_decision_id=cause_id,
        effect_decision_id=effect_id,
        relation_type="derives_from",
        created_at="2026-01-18T12:00:00Z",
    )


def create_test_transition(
    id: str = "trans-001",
    from_agent: str = "analyst",
    to_agent: str = "pm",
    session_id: str = "session-123",
) -> AgentTransition:
    """Create a test AgentTransition."""
    return AgentTransition(
        id=id,
        from_agent=from_agent,
        to_agent=to_agent,
        decision_id="dec-001",
        timestamp="2026-01-18T12:00:00Z",
        context={"session_id": session_id},
    )


class TestInMemoryCorrelationStore:
    """Tests for InMemoryCorrelationStore class."""

    def test_store_exists(self) -> None:
        """Test InMemoryCorrelationStore class exists."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        assert InMemoryCorrelationStore is not None

    def test_store_implements_protocol(self) -> None:
        """Test InMemoryCorrelationStore implements CorrelationStore protocol."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore
        from yolo_developer.audit.correlation_store import CorrelationStore

        store = InMemoryCorrelationStore()
        assert isinstance(store, CorrelationStore)


class TestStoreChain:
    """Tests for store_chain method."""

    @pytest.mark.asyncio
    async def test_store_chain_returns_id(self) -> None:
        """Test store_chain returns chain ID."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        chain = create_test_chain()

        result = await store.store_chain(chain)

        assert result == "chain-001"

    @pytest.mark.asyncio
    async def test_store_chain_persists_chain(self) -> None:
        """Test store_chain persists the chain for retrieval."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        chain = create_test_chain()

        await store.store_chain(chain)
        retrieved = await store.get_chain("chain-001")

        assert retrieved is not None
        assert retrieved.id == "chain-001"


class TestStoreCausalRelation:
    """Tests for store_causal_relation method."""

    @pytest.mark.asyncio
    async def test_store_causal_relation_returns_id(self) -> None:
        """Test store_causal_relation returns relation ID."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        relation = create_test_relation()

        result = await store.store_causal_relation(relation)

        assert result == "rel-001"

    @pytest.mark.asyncio
    async def test_store_causal_relation_persists(self) -> None:
        """Test store_causal_relation persists for retrieval."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        relation = create_test_relation()

        await store.store_causal_relation(relation)
        relations = await store.get_causal_relations("dec-001")

        assert len(relations) == 1
        assert relations[0].id == "rel-001"


class TestStoreTransition:
    """Tests for store_transition method."""

    @pytest.mark.asyncio
    async def test_store_transition_returns_id(self) -> None:
        """Test store_transition returns transition ID."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        transition = create_test_transition()

        result = await store.store_transition(transition)

        assert result == "trans-001"

    @pytest.mark.asyncio
    async def test_store_transition_persists(self) -> None:
        """Test store_transition persists for retrieval."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        transition = create_test_transition(session_id="session-123")

        await store.store_transition(transition)
        transitions = await store.get_transitions_by_session("session-123")

        assert len(transitions) == 1
        assert transitions[0].id == "trans-001"


class TestGetChain:
    """Tests for get_chain method."""

    @pytest.mark.asyncio
    async def test_get_chain_returns_none_for_unknown(self) -> None:
        """Test get_chain returns None for unknown chain."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()

        result = await store.get_chain("unknown")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_chain_retrieves_correct_chain(self) -> None:
        """Test get_chain retrieves the correct chain."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        chain1 = create_test_chain(id="chain-001")
        chain2 = create_test_chain(id="chain-002")

        await store.store_chain(chain1)
        await store.store_chain(chain2)

        result = await store.get_chain("chain-002")

        assert result is not None
        assert result.id == "chain-002"


class TestGetChainsForDecision:
    """Tests for get_chains_for_decision method."""

    @pytest.mark.asyncio
    async def test_get_chains_for_decision_returns_empty_for_unknown(self) -> None:
        """Test get_chains_for_decision returns empty list for unknown decision."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()

        result = await store.get_chains_for_decision("unknown")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_chains_for_decision_retrieves_containing_chains(self) -> None:
        """Test get_chains_for_decision retrieves chains containing the decision."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        chain1 = create_test_chain(id="chain-001", decisions=("dec-001", "dec-002"))
        chain2 = create_test_chain(id="chain-002", decisions=("dec-002", "dec-003"))
        chain3 = create_test_chain(id="chain-003", decisions=("dec-004",))

        await store.store_chain(chain1)
        await store.store_chain(chain2)
        await store.store_chain(chain3)

        result = await store.get_chains_for_decision("dec-002")

        assert len(result) == 2
        chain_ids = {c.id for c in result}
        assert "chain-001" in chain_ids
        assert "chain-002" in chain_ids


class TestGetCausalRelations:
    """Tests for get_causal_relations method."""

    @pytest.mark.asyncio
    async def test_get_causal_relations_returns_empty_for_unknown(self) -> None:
        """Test get_causal_relations returns empty list for unknown decision."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()

        result = await store.get_causal_relations("unknown")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_causal_relations_retrieves_as_cause(self) -> None:
        """Test get_causal_relations retrieves relations where decision is cause."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        relation = create_test_relation(id="rel-001", cause_id="dec-001", effect_id="dec-002")

        await store.store_causal_relation(relation)
        result = await store.get_causal_relations("dec-001")

        assert len(result) == 1
        assert result[0].cause_decision_id == "dec-001"

    @pytest.mark.asyncio
    async def test_get_causal_relations_retrieves_as_effect(self) -> None:
        """Test get_causal_relations retrieves relations where decision is effect."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        relation = create_test_relation(id="rel-001", cause_id="dec-001", effect_id="dec-002")

        await store.store_causal_relation(relation)
        result = await store.get_causal_relations("dec-002")

        assert len(result) == 1
        assert result[0].effect_decision_id == "dec-002"


class TestGetTransitionsBySession:
    """Tests for get_transitions_by_session method."""

    @pytest.mark.asyncio
    async def test_get_transitions_by_session_returns_empty_for_unknown(self) -> None:
        """Test get_transitions_by_session returns empty list for unknown session."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()

        result = await store.get_transitions_by_session("unknown")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_transitions_by_session_retrieves_correct_session(self) -> None:
        """Test get_transitions_by_session retrieves transitions for correct session."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        trans1 = create_test_transition(id="trans-001", session_id="session-A")
        trans2 = create_test_transition(id="trans-002", session_id="session-B")
        trans3 = create_test_transition(id="trans-003", session_id="session-A")

        await store.store_transition(trans1)
        await store.store_transition(trans2)
        await store.store_transition(trans3)

        result = await store.get_transitions_by_session("session-A")

        assert len(result) == 2
        trans_ids = {t.id for t in result}
        assert "trans-001" in trans_ids
        assert "trans-003" in trans_ids

    @pytest.mark.asyncio
    async def test_get_transitions_by_session_ordered_by_timestamp(self) -> None:
        """Test get_transitions_by_session returns transitions ordered by timestamp."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()

        trans1 = AgentTransition(
            id="trans-001",
            from_agent="analyst",
            to_agent="pm",
            decision_id="dec-001",
            timestamp="2026-01-18T14:00:00Z",
            context={"session_id": "session-A"},
        )
        trans2 = AgentTransition(
            id="trans-002",
            from_agent="pm",
            to_agent="architect",
            decision_id="dec-002",
            timestamp="2026-01-18T12:00:00Z",
            context={"session_id": "session-A"},
        )

        await store.store_transition(trans1)
        await store.store_transition(trans2)

        result = await store.get_transitions_by_session("session-A")

        # Should be ordered by timestamp (earliest first)
        assert result[0].id == "trans-002"  # 12:00
        assert result[1].id == "trans-001"  # 14:00


class TestSearchCorrelations:
    """Tests for search_correlations method."""

    @pytest.mark.asyncio
    async def test_search_correlations_no_filters_returns_all(self) -> None:
        """Test search_correlations with no filters returns all chains."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()
        chain1 = create_test_chain(id="chain-001")
        chain2 = create_test_chain(id="chain-002")

        await store.store_chain(chain1)
        await store.store_chain(chain2)

        result = await store.search_correlations()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_search_correlations_filter_by_chain_type(self) -> None:
        """Test search_correlations filters by chain_type."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore
        from yolo_developer.audit.correlation_store import CorrelationFilters

        store = InMemoryCorrelationStore()
        chain1 = create_test_chain(id="chain-001", chain_type="session")
        chain2 = create_test_chain(id="chain-002", chain_type="causal")

        await store.store_chain(chain1)
        await store.store_chain(chain2)

        filters = CorrelationFilters(chain_type="causal")
        result = await store.search_correlations(filters)

        assert len(result) == 1
        assert result[0].id == "chain-002"

    @pytest.mark.asyncio
    async def test_search_correlations_filter_by_time_range(self) -> None:
        """Test search_correlations filters by time range."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore
        from yolo_developer.audit.correlation_store import CorrelationFilters

        store = InMemoryCorrelationStore()
        chain1 = DecisionChain(
            id="chain-001",
            decisions=("dec-001",),
            chain_type="session",
            created_at="2026-01-18T10:00:00Z",
        )
        chain2 = DecisionChain(
            id="chain-002",
            decisions=("dec-002",),
            chain_type="session",
            created_at="2026-01-18T15:00:00Z",
        )

        await store.store_chain(chain1)
        await store.store_chain(chain2)

        filters = CorrelationFilters(
            start_time="2026-01-18T12:00:00Z",
            end_time="2026-01-18T18:00:00Z",
        )
        result = await store.search_correlations(filters)

        assert len(result) == 1
        assert result[0].id == "chain-002"


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_store_chain(self) -> None:
        """Test concurrent store_chain operations are thread-safe."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()

        async def store_chain(i: int) -> str:
            chain = create_test_chain(id=f"chain-{i:03d}")
            return await store.store_chain(chain)

        # Store 100 chains concurrently
        tasks = [store_chain(i) for i in range(100)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 100
        assert len(set(results)) == 100  # All unique IDs

        # Verify all chains stored
        all_chains = await store.search_correlations()
        assert len(all_chains) == 100

    @pytest.mark.asyncio
    async def test_concurrent_store_and_get(self) -> None:
        """Test concurrent store and get operations are thread-safe."""
        from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

        store = InMemoryCorrelationStore()

        # Pre-populate some chains
        for i in range(50):
            chain = create_test_chain(id=f"chain-{i:03d}")
            await store.store_chain(chain)

        async def get_chain(i: int) -> DecisionChain | None:
            return await store.get_chain(f"chain-{i:03d}")

        # Get chains concurrently
        tasks = [get_chain(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should be found
        assert all(r is not None for r in results)
