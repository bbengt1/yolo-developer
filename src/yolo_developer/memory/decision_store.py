"""Decision storage for historical decision queries.

This module provides the ChromaDecisionStore class for storing and retrieving
agent decisions using ChromaDB as the backend. Decisions are stored with
embeddings for semantic search and isolated per project.

Example:
    >>> from yolo_developer.memory.decision_store import ChromaDecisionStore
    >>> from yolo_developer.memory.decisions import Decision, DecisionType
    >>>
    >>> store = ChromaDecisionStore(
    ...     persist_directory=".yolo/decisions",
    ...     project_id="my-project",
    ... )
    >>> decision = Decision(
    ...     id="dec-001",
    ...     agent_type="Architect",
    ...     context="Choosing database",
    ...     rationale="PostgreSQL for reliability",
    ...     decision_type=DecisionType.ARCHITECTURE_CHOICE,
    ... )
    >>> decision_id = await store.store_decision(decision)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, cast

import chromadb
from chromadb.config import Settings
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.memory.decisions import (
    Decision,
    DecisionFilter,
    DecisionResult,
    DecisionType,
)

logger = logging.getLogger(__name__)

# Default project ID if not specified
DEFAULT_PROJECT_ID = "default"

# Retry configuration for ChromaDB operations
RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 1  # seconds
RETRY_MAX_WAIT = 10  # seconds


class ChromaDecisionStore:
    """ChromaDB-backed decision storage.

    Stores agent decisions with embeddings for semantic search.
    Decisions are isolated per project using collection naming.

    Attributes:
        project_id: Identifier for project isolation.
        persist_directory: Directory for ChromaDB persistence.

    Example:
        >>> store = ChromaDecisionStore(
        ...     persist_directory=".yolo/decisions",
        ...     project_id="my-project",
        ... )
        >>> decisions = await store.get_decisions_by_agent("Architect", limit=5)
    """

    def __init__(
        self,
        persist_directory: str,
        project_id: str | None = None,
    ) -> None:
        """Initialize the decision store.

        Args:
            persist_directory: Directory for ChromaDB persistence.
            project_id: Identifier for project isolation. Defaults to "default".
        """
        self.persist_directory = persist_directory
        self.project_id = project_id or DEFAULT_PROJECT_ID

        # Initialize ChromaDB client
        self._client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                is_persistent=True,
                anonymized_telemetry=False,
            )
        )

        # Get or create collection for this project's decisions
        collection_name = f"decisions_{self.project_id}"
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.debug(
            "Initialized decision store",
            extra={
                "project_id": self.project_id,
                "persist_directory": persist_directory,
                "collection_name": collection_name,
            },
        )

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    )
    async def store_decision(self, decision: Decision) -> str:
        """Store an agent decision and return its ID.

        Stores the decision with its vector embedding for semantic search.
        Uses upsert behavior - updates existing decision if same ID exists.

        Args:
            decision: The Decision instance to store.

        Returns:
            The unique identifier of the stored decision.

        Example:
            >>> decision = Decision(
            ...     id="dec-001",
            ...     agent_type="Architect",
            ...     context="Database selection",
            ...     rationale="PostgreSQL for ACID compliance",
            ... )
            >>> decision_id = await store.store_decision(decision)
        """
        # Generate embedding text
        embedding_text = decision.to_embedding_text()

        # Prepare metadata - all values must be string, int, float, or bool for ChromaDB
        metadata: dict[str, Any] = {
            "agent_type": decision.agent_type,
            "context": decision.context,
            "rationale": decision.rationale,
            "timestamp": decision.timestamp.isoformat(),
        }

        if decision.outcome:
            metadata["outcome"] = decision.outcome

        if decision.decision_type:
            metadata["decision_type"] = decision.decision_type.value

        if decision.artifact_type:
            metadata["artifact_type"] = decision.artifact_type

        if decision.artifact_ids:
            # Store as comma-separated string
            metadata["artifact_ids"] = ",".join(decision.artifact_ids)

        # Store in ChromaDB (upsert behavior)
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[decision.id],
            documents=[embedding_text],
            metadatas=[metadata],
        )

        logger.debug(
            "Stored decision",
            extra={
                "decision_id": decision.id,
                "agent_type": decision.agent_type,
                "decision_type": decision.decision_type.value if decision.decision_type else None,
            },
        )

        return decision.id

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    )
    async def search_decisions(
        self,
        query: str,
        filters: DecisionFilter | None = None,
        k: int = 5,
    ) -> list[DecisionResult]:
        """Search for decisions by semantic similarity with optional filters.

        Performs semantic similarity search on decision context and rationale.
        Results can be filtered by agent type, time range, and artifact type.

        Args:
            query: Search query for semantic matching.
            filters: Optional DecisionFilter for metadata filtering.
            k: Maximum number of results to return.

        Returns:
            List of DecisionResult instances ordered by similarity (highest first).

        Example:
            >>> filters = DecisionFilter(agent_type="Architect")
            >>> results = await store.search_decisions(
            ...     "database choice for user data",
            ...     filters=filters,
            ...     k=5,
            ... )
        """
        # Check if collection has any items
        count = await asyncio.to_thread(self._collection.count)
        if count == 0:
            return []

        # Build where filter from DecisionFilter
        where_clause = filters.to_chromadb_where() if filters else None

        # Query ChromaDB
        try:
            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[query],
                n_results=min(k, count),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(
                "Decision search failed",
                extra={"query": query, "error": str(e)},
            )
            return []

        # Convert results to DecisionResult instances
        return self._convert_query_results(cast(dict[str, Any], results))

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    )
    async def get_decisions_by_agent(
        self,
        agent_type: str,
        limit: int = 10,
    ) -> list[Decision]:
        """Get decisions made by a specific agent type.

        Retrieves decisions filtered by agent type, ordered by timestamp
        (most recent first).

        Args:
            agent_type: Agent type filter (Analyst, PM, Architect, Dev, SM, TEA).
            limit: Maximum number of decisions to return.

        Returns:
            List of Decision instances ordered by timestamp (newest first).

        Example:
            >>> decisions = await store.get_decisions_by_agent("Architect", limit=5)
            >>> for dec in decisions:
            ...     print(f"{dec.id}: {dec.context[:50]}...")
        """
        # Check if collection has any items
        count = await asyncio.to_thread(self._collection.count)
        if count == 0:
            return []

        # Query by metadata filter
        try:
            results = await asyncio.to_thread(
                self._collection.get,
                where={"agent_type": agent_type},
                include=["metadatas"],
                limit=limit,
            )
        except Exception as e:
            logger.warning(
                "Get decisions by agent failed",
                extra={"agent_type": agent_type, "error": str(e)},
            )
            return []

        if not results or not results.get("ids"):
            return []

        decisions: list[Decision] = []
        raw_metadatas = results.get("metadatas")
        metadatas: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], raw_metadatas) if raw_metadatas else []
        )

        for i, decision_id in enumerate(results["ids"]):
            metadata = metadatas[i] if i < len(metadatas) else {}
            try:
                decision = self._metadata_to_decision(decision_id, metadata)
                decisions.append(decision)
            except Exception as e:
                logger.warning(
                    "Failed to reconstruct decision",
                    extra={"decision_id": decision_id, "error": str(e)},
                )

        # Sort by timestamp (newest first)
        decisions.sort(key=lambda d: d.timestamp, reverse=True)

        return decisions[:limit]

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    )
    async def get_decision_by_id(self, decision_id: str) -> Decision | None:
        """Get a specific decision by its ID.

        Args:
            decision_id: The unique identifier of the decision.

        Returns:
            The Decision if found, None otherwise.

        Example:
            >>> decision = await store.get_decision_by_id("dec-001")
            >>> if decision:
            ...     print(f"Found: {decision.context}")
        """
        try:
            results = await asyncio.to_thread(
                self._collection.get,
                ids=[decision_id],
                include=["metadatas"],
            )
        except Exception as e:
            logger.warning(
                "Get decision by ID failed",
                extra={"decision_id": decision_id, "error": str(e)},
            )
            return None

        if not results or not results.get("ids") or not results["ids"]:
            return None

        raw_metadatas = results.get("metadatas")
        metadatas: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], raw_metadatas) if raw_metadatas else []
        )

        if not metadatas:
            return None

        try:
            return self._metadata_to_decision(decision_id, metadatas[0])
        except Exception as e:
            logger.warning(
                "Failed to reconstruct decision",
                extra={"decision_id": decision_id, "error": str(e)},
            )
            return None

    def _convert_query_results(self, results: dict[str, Any]) -> list[DecisionResult]:
        """Convert ChromaDB query results to DecisionResult instances.

        Args:
            results: Raw results from ChromaDB query.

        Returns:
            List of DecisionResult instances.
        """
        decision_results: list[DecisionResult] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        raw_metadatas = results.get("metadatas")
        metadatas: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], raw_metadatas[0]) if raw_metadatas else []
        )
        raw_distances = results.get("distances")
        distances: list[float] = raw_distances[0] if raw_distances else []

        for i, decision_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0.0

            # Convert distance to similarity (ChromaDB returns distances)
            # For cosine distance: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - distance))

            try:
                decision = self._metadata_to_decision(decision_id, metadata)
                decision_results.append(DecisionResult(decision=decision, similarity=similarity))
            except Exception as e:
                logger.warning(
                    "Failed to reconstruct decision from metadata",
                    extra={"decision_id": decision_id, "error": str(e)},
                )

        return decision_results

    def _metadata_to_decision(self, decision_id: str, metadata: dict[str, Any]) -> Decision:
        """Convert ChromaDB metadata to a Decision.

        Args:
            decision_id: The decision ID.
            metadata: Metadata dict from ChromaDB.

        Returns:
            Reconstructed Decision instance.
        """
        # Parse artifact_ids from comma-separated string
        artifact_ids_str = metadata.get("artifact_ids", "")
        artifact_ids = tuple(artifact_ids_str.split(",")) if artifact_ids_str else ()

        # Parse timestamp
        timestamp_str = metadata.get("timestamp", "")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Parse decision_type with error handling for invalid values
        decision_type_str = metadata.get("decision_type")
        decision_type: DecisionType | None = None
        if decision_type_str:
            try:
                decision_type = DecisionType(decision_type_str)
            except ValueError:
                logger.warning(
                    "Invalid decision_type in stored decision",
                    extra={"decision_id": decision_id, "decision_type": decision_type_str},
                )
                decision_type = None

        return Decision(
            id=decision_id,
            agent_type=metadata["agent_type"],
            context=metadata["context"],
            rationale=metadata["rationale"],
            outcome=metadata.get("outcome"),
            decision_type=decision_type,
            artifact_type=metadata.get("artifact_type"),
            artifact_ids=artifact_ids,
            timestamp=timestamp,
        )
