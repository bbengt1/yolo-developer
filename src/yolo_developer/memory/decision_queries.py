"""Decision query engine for historical decision queries.

This module provides the DecisionQueryEngine class as a high-level interface
for querying agent decisions. It wraps the ChromaDecisionStore with convenient
methods for common query patterns.

Example:
    >>> from yolo_developer.memory.decision_queries import DecisionQueryEngine
    >>> from yolo_developer.memory.decisions import DecisionFilter
    >>>
    >>> engine = DecisionQueryEngine(
    ...     persist_directory=".yolo/decisions",
    ...     project_id="my-project",
    ... )
    >>> results = await engine.find_similar_decisions(
    ...     context="database selection for user data",
    ...     k=5,
    ... )
    >>> for result in results:
    ...     print(f"{result.decision.agent_type}: {result.similarity:.2f}")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from yolo_developer.memory.decision_store import ChromaDecisionStore
from yolo_developer.memory.decisions import (
    Decision,
    DecisionFilter,
    DecisionResult,
    DecisionType,
)

logger = logging.getLogger(__name__)


class DecisionQueryEngine:
    """High-level interface for decision queries.

    Provides convenient methods for finding similar decisions, retrieving
    agent history, and searching with filters. Wraps ChromaDecisionStore
    for common query patterns.

    Example:
        >>> engine = DecisionQueryEngine(
        ...     persist_directory=".yolo/decisions",
        ...     project_id="my-project",
        ... )
        >>> results = await engine.find_similar_decisions("database choice", k=3)
    """

    def __init__(
        self,
        store: ChromaDecisionStore | None = None,
        persist_directory: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Initialize the query engine.

        Args:
            store: Existing ChromaDecisionStore to use. If not provided,
                creates a new store with persist_directory and project_id.
            persist_directory: Directory for ChromaDB persistence.
                Required if store is not provided.
            project_id: Project identifier for isolation.

        Raises:
            ValueError: If neither store nor persist_directory is provided.
        """
        if store is not None:
            self._store = store
        elif persist_directory is not None:
            self._store = ChromaDecisionStore(
                persist_directory=persist_directory,
                project_id=project_id,
            )
        else:
            raise ValueError("Either 'store' or 'persist_directory' must be provided")

        logger.debug(
            "Initialized DecisionQueryEngine",
            extra={"project_id": self._store.project_id},
        )

    async def find_similar_decisions(
        self,
        context: str,
        k: int = 5,
    ) -> list[DecisionResult]:
        """Find decisions similar to the given context.

        Performs semantic similarity search to find decisions that match
        the given context description. Useful for learning from past
        decisions in similar situations.

        Args:
            context: Description of the current situation or problem.
            k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of DecisionResult instances ordered by similarity (highest first).

        Example:
            >>> results = await engine.find_similar_decisions(
            ...     context="choosing a database for user profile storage",
            ...     k=3,
            ... )
            >>> for r in results:
            ...     print(f"Similarity: {r.similarity:.2f} - {r.decision.context}")
        """
        return await self._store.search_decisions(query=context, k=k)

    async def get_agent_decision_history(
        self,
        agent_type: str,
        limit: int = 10,
    ) -> list[Decision]:
        """Get decision history for a specific agent type.

        Retrieves past decisions made by an agent type, useful for
        understanding an agent's decision patterns and learning from
        previous approaches.

        Args:
            agent_type: Agent type (Analyst, PM, Architect, Dev, SM, TEA).
            limit: Maximum number of decisions to return. Defaults to 10.

        Returns:
            List of Decision instances ordered by timestamp (newest first).

        Example:
            >>> decisions = await engine.get_agent_decision_history("Architect")
            >>> for d in decisions:
            ...     print(f"{d.timestamp.date()}: {d.context[:50]}...")
        """
        return await self._store.get_decisions_by_agent(
            agent_type=agent_type,
            limit=limit,
        )

    async def search_with_filters(
        self,
        query: str,
        filters: DecisionFilter | None = None,
        k: int = 10,
    ) -> list[DecisionResult]:
        """Search decisions with semantic matching and metadata filters.

        Combines semantic similarity search with metadata filtering for
        more targeted queries. Supports filtering by agent type, time range,
        artifact type, and decision type.

        Args:
            query: Search query for semantic matching.
            filters: Optional DecisionFilter for metadata filtering.
            k: Maximum number of results to return. Defaults to 10.

        Returns:
            List of DecisionResult instances ordered by similarity.

        Example:
            >>> filters = DecisionFilter(
            ...     agent_type="Architect",
            ...     artifact_type="design",
            ... )
            >>> results = await engine.search_with_filters(
            ...     query="database architecture",
            ...     filters=filters,
            ... )
        """
        return await self._store.search_decisions(
            query=query,
            filters=filters,
            k=k,
        )

    async def record_decision(
        self,
        agent_type: str,
        context: str,
        rationale: str,
        decision_type: DecisionType | None = None,
        outcome: str | None = None,
        artifact_type: str | None = None,
        artifact_ids: tuple[str, ...] = (),
        decision_id: str | None = None,
    ) -> str:
        """Record a new decision made by an agent.

        Convenience method for storing decisions with automatic ID generation.
        Use this when recording decisions during agent execution.

        Args:
            agent_type: Type of agent making the decision.
            context: Description of the situation requiring a decision.
            rationale: Reasoning behind the decision.
            decision_type: Category of the decision.
            outcome: Result of the decision (if known).
            artifact_type: Type of artifact involved.
            artifact_ids: Related artifact IDs.
            decision_id: Optional custom ID. If not provided, generates a UUID.

        Returns:
            The ID of the stored decision.

        Example:
            >>> decision_id = await engine.record_decision(
            ...     agent_type="Architect",
            ...     context="Choosing message queue",
            ...     rationale="RabbitMQ for reliable delivery",
            ...     decision_type=DecisionType.ARCHITECTURE_CHOICE,
            ... )
        """
        if decision_id is None:
            decision_id = f"dec-{uuid.uuid4().hex[:12]}"

        decision = Decision(
            id=decision_id,
            agent_type=agent_type,
            context=context,
            rationale=rationale,
            outcome=outcome,
            decision_type=decision_type,
            artifact_type=artifact_type,
            artifact_ids=artifact_ids,
            timestamp=datetime.now(timezone.utc),
        )

        stored_id = await self._store.store_decision(decision)

        logger.info(
            "Recorded decision",
            extra={
                "decision_id": stored_id,
                "agent_type": agent_type,
                "decision_type": decision_type.value if decision_type else None,
            },
        )

        return stored_id
