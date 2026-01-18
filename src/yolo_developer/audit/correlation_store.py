"""Correlation store protocol and filters (Story 11.5).

This module defines the CorrelationStore protocol and CorrelationFilters dataclass
for storing and retrieving correlation data.

The Protocol pattern enables:
- Easy mocking in tests
- Future implementations (JSON file, SQLite, Neo4j, etc.)
- Dependency injection

Example:
    >>> from yolo_developer.audit.correlation_store import CorrelationStore, CorrelationFilters
    >>>
    >>> # CorrelationStore is a Protocol - implementations must provide these methods
    >>> class MyStore(CorrelationStore):
    ...     async def store_chain(self, chain: DecisionChain) -> str: ...
    ...     async def get_chain(self, chain_id: str) -> DecisionChain | None: ...
    ...     # ... other methods

References:
    - FR85: System can correlate decisions across agent boundaries
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from yolo_developer.audit.correlation_types import (
        AgentTransition,
        CausalRelation,
        DecisionChain,
    )


# =============================================================================
# Filter Dataclass (Subtask 2.3)
# =============================================================================


@dataclass(frozen=True)
class CorrelationFilters:
    """Filters for querying correlations from the store.

    All fields are optional; None means no filtering on that field.

    Attributes:
        agent_name: Filter by agent name
        session_id: Filter by session ID
        start_time: Filter correlations after this timestamp (inclusive)
        end_time: Filter correlations before this timestamp (inclusive)
        chain_type: Filter by chain type (causal, temporal, session, artifact)

    Example:
        >>> filters = CorrelationFilters(
        ...     agent_name="analyst",
        ...     session_id="session-123",
        ... )
        >>> chains = await store.search_correlations(filters)
    """

    agent_name: str | None = None
    session_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    chain_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the filters.
        """
        return {
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "chain_type": self.chain_type,
        }


# =============================================================================
# Store Protocol (Subtasks 2.1, 2.2)
# =============================================================================


@runtime_checkable
class CorrelationStore(Protocol):
    """Protocol for correlation storage implementations.

    Defines the interface for storing and retrieving correlation data.
    Implementations can use various backends (memory, file, database, graph DB).

    Methods:
        store_chain: Store a decision chain and return its ID
        store_causal_relation: Store a causal relation and return its ID
        store_transition: Store an agent transition and return its ID
        get_chain: Retrieve a chain by ID
        get_chains_for_decision: Get all chains containing a specific decision
        get_causal_relations: Get causal relations for a decision
        get_transitions_by_session: Get all transitions in a session
        search_correlations: Search chains with optional filters

    Example:
        >>> class InMemoryCorrelationStore:
        ...     async def store_chain(self, chain: DecisionChain) -> str:
        ...         # Store chain
        ...         return chain.id
    """

    async def store_chain(self, chain: DecisionChain) -> str:
        """Store a decision chain and return its ID.

        Args:
            chain: The DecisionChain to store.

        Returns:
            The chain ID.
        """
        ...

    async def store_causal_relation(self, relation: CausalRelation) -> str:
        """Store a causal relation and return its ID.

        Args:
            relation: The CausalRelation to store.

        Returns:
            The relation ID.
        """
        ...

    async def store_transition(self, transition: AgentTransition) -> str:
        """Store an agent transition and return its ID.

        Args:
            transition: The AgentTransition to store.

        Returns:
            The transition ID.
        """
        ...

    async def get_chain(self, chain_id: str) -> DecisionChain | None:
        """Retrieve a chain by its ID.

        Args:
            chain_id: The ID of the chain to retrieve.

        Returns:
            The DecisionChain if found, None otherwise.
        """
        ...

    async def get_chains_for_decision(self, decision_id: str) -> list[DecisionChain]:
        """Get all chains containing a specific decision.

        Args:
            decision_id: The ID of the decision.

        Returns:
            List of DecisionChains containing the decision.
        """
        ...

    async def get_causal_relations(self, decision_id: str) -> list[CausalRelation]:
        """Get causal relations for a decision (as cause or effect).

        Args:
            decision_id: The ID of the decision.

        Returns:
            List of CausalRelations involving the decision.
        """
        ...

    async def get_transitions_by_session(self, session_id: str) -> list[AgentTransition]:
        """Get all transitions in a session.

        Args:
            session_id: The session ID to query.

        Returns:
            List of AgentTransitions in the session, ordered by timestamp.
        """
        ...

    async def search_correlations(
        self,
        filters: CorrelationFilters | None = None,
    ) -> list[DecisionChain]:
        """Search chains with optional filters.

        Args:
            filters: Optional filters to apply. None returns all chains.

        Returns:
            List of matching DecisionChains.
        """
        ...
