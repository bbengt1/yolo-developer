"""In-memory correlation store implementation (Story 11.5).

This module provides an in-memory implementation of the CorrelationStore protocol.
It's suitable for testing and single-session use cases.

The implementation uses thread-safe storage with threading.Lock for concurrent access.

Example:
    >>> from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore
    >>> from yolo_developer.audit.correlation_types import DecisionChain
    >>>
    >>> store = InMemoryCorrelationStore()
    >>> chain = DecisionChain(
    ...     id="chain-001",
    ...     decisions=("dec-001", "dec-002"),
    ...     chain_type="session",
    ...     created_at="2026-01-18T12:00:00Z",
    ... )
    >>> await store.store_chain(chain)
    'chain-001'

References:
    - FR85: System can correlate decisions across agent boundaries
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolo_developer.audit.correlation_store import CorrelationFilters
    from yolo_developer.audit.correlation_types import (
        AgentTransition,
        CausalRelation,
        DecisionChain,
    )


class InMemoryCorrelationStore:
    """In-memory implementation of the CorrelationStore protocol.

    Stores correlation data in memory with thread-safe access. Data includes
    decision chains, causal relations, and agent transitions.

    This implementation is suitable for:
    - Unit and integration testing
    - Single-session use cases
    - Development and debugging

    For persistent storage, use a file-based or database implementation.

    Attributes:
        _chains: Internal storage mapping chain IDs to DecisionChain objects.
        _relations: Internal storage mapping relation IDs to CausalRelation objects.
        _transitions: Internal storage mapping transition IDs to AgentTransition objects.
        _decision_to_chains: Index mapping decision IDs to chain IDs for efficient lookup.
        _lock: Threading lock for concurrent access safety.

    Example:
        >>> store = InMemoryCorrelationStore()
        >>> await store.store_chain(chain)
        >>> chains = await store.search_correlations()
    """

    def __init__(self) -> None:
        """Initialize the in-memory correlation store."""
        self._chains: dict[str, DecisionChain] = {}
        self._relations: dict[str, CausalRelation] = {}
        self._transitions: dict[str, AgentTransition] = {}
        self._decision_to_chains: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    async def store_chain(self, chain: DecisionChain) -> str:
        """Store a decision chain and return its ID.

        Thread-safe operation that stores the chain in memory and updates
        the decision-to-chains index for efficient lookup.

        Args:
            chain: The DecisionChain to store.

        Returns:
            The chain ID.
        """
        with self._lock:
            self._chains[chain.id] = chain
            # Update decision-to-chains index
            for decision_id in chain.decisions:
                if decision_id not in self._decision_to_chains:
                    self._decision_to_chains[decision_id] = set()
                self._decision_to_chains[decision_id].add(chain.id)
        return chain.id

    async def store_causal_relation(self, relation: CausalRelation) -> str:
        """Store a causal relation and return its ID.

        Thread-safe operation that stores the causal relation in memory.

        Args:
            relation: The CausalRelation to store.

        Returns:
            The relation ID.
        """
        with self._lock:
            self._relations[relation.id] = relation
        return relation.id

    async def store_transition(self, transition: AgentTransition) -> str:
        """Store an agent transition and return its ID.

        Thread-safe operation that stores the transition in memory.

        Args:
            transition: The AgentTransition to store.

        Returns:
            The transition ID.
        """
        with self._lock:
            self._transitions[transition.id] = transition
        return transition.id

    async def get_chain(self, chain_id: str) -> DecisionChain | None:
        """Retrieve a chain by its ID.

        Thread-safe operation that looks up a chain by ID.

        Args:
            chain_id: The ID of the chain to retrieve.

        Returns:
            The DecisionChain if found, None otherwise.
        """
        with self._lock:
            return self._chains.get(chain_id)

    async def get_chains_for_decision(self, decision_id: str) -> list[DecisionChain]:
        """Get all chains containing a specific decision.

        Uses the decision-to-chains index for efficient lookup.

        Args:
            decision_id: The ID of the decision.

        Returns:
            List of DecisionChains containing the decision.
        """
        with self._lock:
            chain_ids = self._decision_to_chains.get(decision_id, set())
            return [self._chains[cid] for cid in chain_ids if cid in self._chains]

    async def get_causal_relations(self, decision_id: str) -> list[CausalRelation]:
        """Get causal relations for a decision (as cause or effect).

        Returns relations where the decision is either the cause or effect.

        Args:
            decision_id: The ID of the decision.

        Returns:
            List of CausalRelations involving the decision.
        """
        with self._lock:
            return [
                r
                for r in self._relations.values()
                if r.cause_decision_id == decision_id or r.effect_decision_id == decision_id
            ]

    async def get_transitions_by_session(self, session_id: str) -> list[AgentTransition]:
        """Get all transitions in a session.

        Returns transitions where the session_id is in the context,
        ordered by timestamp.

        Args:
            session_id: The session ID to query.

        Returns:
            List of AgentTransitions in the session, ordered by timestamp.
        """
        with self._lock:
            transitions = [
                t for t in self._transitions.values() if t.context.get("session_id") == session_id
            ]
        # Sort by timestamp (outside lock for efficiency)
        transitions.sort(key=lambda t: t.timestamp)
        return transitions

    async def search_correlations(
        self,
        filters: CorrelationFilters | None = None,
    ) -> list[DecisionChain]:
        """Search chains with optional filters.

        Applies filters to find matching chains.

        Args:
            filters: Optional filters to apply. None returns all chains.

        Returns:
            List of matching DecisionChains.
        """
        with self._lock:
            chains = list(self._chains.values())

        # Apply filters if provided
        if filters is not None:
            chains = self._apply_filters(chains, filters)

        return chains

    def _apply_filters(
        self,
        chains: list[DecisionChain],
        filters: CorrelationFilters,
    ) -> list[DecisionChain]:
        """Apply filters to a list of chains.

        Args:
            chains: List of chains to filter.
            filters: Filters to apply.

        Returns:
            Filtered list of chains.
        """
        result = chains

        # Filter by chain_type
        if filters.chain_type is not None:
            result = [c for c in result if c.chain_type == filters.chain_type]

        # Filter by start_time (inclusive)
        if filters.start_time is not None:
            result = [c for c in result if c.created_at >= filters.start_time]

        # Filter by end_time (inclusive)
        if filters.end_time is not None:
            result = [c for c in result if c.created_at <= filters.end_time]

        return result
