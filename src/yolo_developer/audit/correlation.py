"""Correlation service for cross-agent decision correlation (Story 11.5).

This module provides the CorrelationService for correlating decisions across agents:

- CorrelationService: Main service for correlation operations
- get_correlation_service: Factory function for creating service instances

Example:
    >>> from yolo_developer.audit.correlation import get_correlation_service
    >>> from yolo_developer.audit.memory_store import InMemoryDecisionStore
    >>>
    >>> decision_store = InMemoryDecisionStore()
    >>> service = get_correlation_service(decision_store)
    >>> chain = await service.correlate_decisions(["dec-001", "dec-002"])

References:
    - FR85: System can correlate decisions across agent boundaries
    - AC #1: Decision chains are identified
    - AC #2: Causal relationships are tracked
    - AC #3: Agent transitions are recorded
    - AC #4: Correlations are searchable
    - AC #5: Timeline view is available
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore
from yolo_developer.audit.correlation_store import CorrelationFilters
from yolo_developer.audit.correlation_types import (
    AgentTransition,
    CausalRelation,
    DecisionChain,
    TimelineEntry,
)
from yolo_developer.audit.store import DecisionFilters

if TYPE_CHECKING:
    from yolo_developer.audit.correlation_store import CorrelationStore
    from yolo_developer.audit.store import DecisionStore
    from yolo_developer.audit.types import Decision

logger = structlog.get_logger(__name__)


class CorrelationService:
    """Service for correlating decisions across agents.

    Provides methods for creating and querying decision correlations,
    including decision chains, causal relations, and agent transitions.

    Attributes:
        _decision_store: Store for decision records
        _correlation_store: Store for correlation data

    Example:
        >>> service = CorrelationService(decision_store, correlation_store)
        >>> chain = await service.correlate_decisions(["dec-001", "dec-002"])
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        correlation_store: CorrelationStore,
    ) -> None:
        """Initialize the correlation service.

        Args:
            decision_store: Store for decision records
            correlation_store: Store for correlation data
        """
        self._decision_store = decision_store
        self._correlation_store = correlation_store

    async def correlate_decisions(
        self,
        decision_ids: list[str],
        chain_type: str = "session",
    ) -> DecisionChain:
        """Create a correlation chain for the given decisions.

        Args:
            decision_ids: List of decision IDs to correlate
            chain_type: Type of correlation (default: "session")

        Returns:
            The created DecisionChain
        """
        chain_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        chain = DecisionChain(
            id=chain_id,
            decisions=tuple(decision_ids),
            chain_type=chain_type,  # type: ignore[arg-type]
            created_at=timestamp,
        )

        await self._correlation_store.store_chain(chain)

        logger.info(
            "created_decision_chain",
            chain_id=chain_id,
            decision_count=len(decision_ids),
            chain_type=chain_type,
        )

        return chain

    async def add_causal_relation(
        self,
        cause_id: str,
        effect_id: str,
        relation_type: str,
        evidence: str = "",
    ) -> CausalRelation:
        """Add a causal relation between two decisions.

        Args:
            cause_id: ID of the cause decision
            effect_id: ID of the effect decision
            relation_type: Type of relation (e.g., "derives_from", "triggers")
            evidence: Explanation of why this relationship exists

        Returns:
            The created CausalRelation
        """
        relation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        relation = CausalRelation(
            id=relation_id,
            cause_decision_id=cause_id,
            effect_decision_id=effect_id,
            relation_type=relation_type,
            evidence=evidence,
            created_at=timestamp,
        )

        await self._correlation_store.store_causal_relation(relation)

        logger.info(
            "added_causal_relation",
            relation_id=relation_id,
            cause_id=cause_id,
            effect_id=effect_id,
            relation_type=relation_type,
        )

        return relation

    async def record_transition(
        self,
        from_agent: str,
        to_agent: str,
        decision_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentTransition:
        """Record an agent transition.

        Args:
            from_agent: Name of the agent handing off work
            to_agent: Name of the agent receiving work
            decision_id: ID of the decision triggering the transition
            context: Additional context data

        Returns:
            The created AgentTransition
        """
        transition_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        transition = AgentTransition(
            id=transition_id,
            from_agent=from_agent,
            to_agent=to_agent,
            decision_id=decision_id,
            timestamp=timestamp,
            context=context or {},
        )

        await self._correlation_store.store_transition(transition)

        logger.info(
            "recorded_agent_transition",
            transition_id=transition_id,
            from_agent=from_agent,
            to_agent=to_agent,
            decision_id=decision_id,
        )

        return transition

    async def get_decision_chain(self, decision_id: str) -> list[Decision]:
        """Get all decisions in chains containing the given decision.

        Args:
            decision_id: ID of the decision to get chain for

        Returns:
            List of Decision objects in the chain
        """
        chains = await self._correlation_store.get_chains_for_decision(decision_id)

        if not chains:
            return []

        # Collect all unique decision IDs from all chains
        all_decision_ids: set[str] = set()
        for chain in chains:
            all_decision_ids.update(chain.decisions)

        # Retrieve all decisions
        decisions: list[Decision] = []
        for dec_id in all_decision_ids:
            decision = await self._decision_store.get_decision(dec_id)
            if decision is not None:
                decisions.append(decision)

        return decisions

    async def get_timeline(
        self,
        filters: CorrelationFilters | None = None,
    ) -> list[tuple[Decision, list[DecisionChain]]]:
        """Get a timeline of decisions with their correlations.

        Args:
            filters: Optional filters to apply

        Returns:
            List of (Decision, [DecisionChain]) tuples ordered by timestamp
        """
        # Convert CorrelationFilters to DecisionFilters
        decision_filters: DecisionFilters | None = None
        if filters is not None:
            decision_filters = DecisionFilters(
                agent_name=filters.agent_name,
                start_time=filters.start_time,
                end_time=filters.end_time,
            )

        decisions = await self._decision_store.get_decisions(decision_filters)

        # Sort by timestamp
        decisions.sort(key=lambda d: d.timestamp)

        # Get chains for each decision
        result: list[tuple[Decision, list[DecisionChain]]] = []
        for decision in decisions:
            chains = await self._correlation_store.get_chains_for_decision(decision.id)
            result.append((decision, chains))

        return result

    async def get_workflow_flow(self, session_id: str) -> dict[str, Any]:
        """Get complete workflow trace for a session.

        Args:
            session_id: Session ID to get workflow flow for

        Returns:
            Dictionary with workflow information
        """
        # Get all transitions in the session
        transitions = await self._correlation_store.get_transitions_by_session(session_id)

        # Get all decisions for the session
        decisions = await self._decision_store.get_decisions()
        session_decisions = [d for d in decisions if d.agent.session_id == session_id]

        # Build agent sequence from decisions
        agent_sequence: list[str] = []
        for decision in sorted(session_decisions, key=lambda d: d.timestamp):
            if not agent_sequence or agent_sequence[-1] != decision.agent.agent_name:
                agent_sequence.append(decision.agent.agent_name)

        return {
            "session_id": session_id,
            "total_decisions": len(session_decisions),
            "agent_sequence": agent_sequence,
            "transitions": [t.to_dict() for t in transitions],
        }

    async def search(
        self,
        query: str,
        filters: CorrelationFilters | None = None,
    ) -> list[Decision]:
        """Search decisions by content.

        Args:
            query: Search string to find in decision content
            filters: Optional filters to apply

        Returns:
            List of matching Decision objects
        """
        # Convert CorrelationFilters to DecisionFilters
        decision_filters: DecisionFilters | None = None
        if filters is not None:
            decision_filters = DecisionFilters(
                agent_name=filters.agent_name,
                start_time=filters.start_time,
                end_time=filters.end_time,
            )

        decisions = await self._decision_store.get_decisions(decision_filters)

        # Filter by content
        query_lower = query.lower()
        return [d for d in decisions if query_lower in d.content.lower()]

    async def get_timeline_view(
        self,
        session_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[TimelineEntry]:
        """Get a timeline view for a session.

        Args:
            session_id: Session ID to get timeline for (optional)
            start_time: Filter decisions after this timestamp (optional)
            end_time: Filter decisions before this timestamp (optional)

        Returns:
            List of TimelineEntry objects ordered by timestamp
        """
        decisions = await self._decision_store.get_decisions()

        # Filter by session_id if provided
        if session_id is not None:
            decisions = [d for d in decisions if d.agent.session_id == session_id]

        # Filter by time range if provided
        if start_time is not None:
            decisions = [d for d in decisions if d.timestamp >= start_time]
        if end_time is not None:
            decisions = [d for d in decisions if d.timestamp <= end_time]

        # Sort by timestamp
        decisions.sort(key=lambda d: d.timestamp)

        # Get transitions for the session if available
        transitions_by_decision: dict[str, AgentTransition] = {}
        if session_id is not None:
            transitions = await self._correlation_store.get_transitions_by_session(session_id)
            for t in transitions:
                transitions_by_decision[t.decision_id] = t

        # Create timeline entries with agent transition information
        entries: list[TimelineEntry] = []
        previous_agent: str | None = None

        for i, decision in enumerate(decisions, start=1):
            current_agent = decision.agent.agent_name

            # Check if this decision has an associated transition
            transition = transitions_by_decision.get(decision.id)

            # Determine if there's an agent change
            agent_changed = previous_agent is not None and previous_agent != current_agent

            entry = TimelineEntry(
                decision=decision,
                sequence_number=i,
                timestamp=decision.timestamp,
                agent_transition=transition,
                previous_agent=previous_agent if agent_changed else None,
            )
            entries.append(entry)

            # Track previous agent for next iteration
            previous_agent = current_agent

        return entries

    async def auto_correlate_session(self, session_id: str) -> DecisionChain:
        """Automatically correlate all decisions in a session.

        Args:
            session_id: Session ID to correlate

        Returns:
            The created DecisionChain
        """
        decisions = await self._decision_store.get_decisions()

        # Filter by session_id
        session_decisions = [d for d in decisions if d.agent.session_id == session_id]

        decision_ids = [d.id for d in session_decisions]

        logger.info(
            "auto_correlating_session",
            session_id=session_id,
            decision_count=len(decision_ids),
        )

        return await self.correlate_decisions(decision_ids, chain_type="session")

    async def detect_causal_relations(
        self,
        decision_id: str,
    ) -> list[CausalRelation]:
        """Detect causal relations for a decision based on parent_decision_id and trace_links.

        Args:
            decision_id: ID of the decision to detect relations for

        Returns:
            List of detected CausalRelation objects
        """
        decision = await self._decision_store.get_decision(decision_id)

        if decision is None:
            return []

        relations: list[CausalRelation] = []

        # Check for parent_decision_id
        parent_id = decision.context.parent_decision_id
        if parent_id is not None:
            # Create a causal relation from parent to this decision
            relation = await self.add_causal_relation(
                cause_id=parent_id,
                effect_id=decision_id,
                relation_type="derives_from",
                evidence="Parent decision ID reference",
            )
            relations.append(relation)

            logger.info(
                "detected_causal_relation",
                decision_id=decision_id,
                parent_id=parent_id,
            )

        # Check for trace_links - correlate with other decisions sharing same links
        trace_links = decision.context.trace_links
        if trace_links:
            # Get all decisions to find ones sharing trace_links
            all_decisions = await self._decision_store.get_decisions()

            for other_decision in all_decisions:
                # Skip self
                if other_decision.id == decision_id:
                    continue

                # Check for shared trace_links
                other_links = other_decision.context.trace_links
                shared_links = set(trace_links) & set(other_links)

                if shared_links:
                    # Create artifact-based causal relation
                    # Earlier decision is cause, later is effect
                    if other_decision.timestamp < decision.timestamp:
                        relation = await self.add_causal_relation(
                            cause_id=other_decision.id,
                            effect_id=decision_id,
                            relation_type="artifact_related",
                            evidence=f"Shared trace links: {', '.join(sorted(shared_links))}",
                        )
                        relations.append(relation)

                        logger.info(
                            "detected_trace_link_relation",
                            decision_id=decision_id,
                            related_id=other_decision.id,
                            shared_links=list(shared_links),
                        )

        return relations


def get_correlation_service(
    decision_store: DecisionStore,
    correlation_store: CorrelationStore | None = None,
) -> CorrelationService:
    """Factory function to create a CorrelationService instance.

    Args:
        decision_store: Store for decision records
        correlation_store: Store for correlation data (default: InMemoryCorrelationStore)

    Returns:
        Configured CorrelationService instance
    """
    if correlation_store is None:
        correlation_store = InMemoryCorrelationStore()

    return CorrelationService(decision_store, correlation_store)
