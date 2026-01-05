"""ChromaDB vector storage implementation for YOLO Developer.

This module provides the ChromaMemory class, which implements the MemoryStore
protocol using ChromaDB for vector embedding storage and similarity search.

ChromaDB is used in embedded mode with local persistence, enabling semantic
similarity search without external infrastructure dependencies.

Example:
    >>> from yolo_developer.memory import ChromaMemory
    >>>
    >>> memory = ChromaMemory(persist_directory=".yolo/memory")
    >>> await memory.store_embedding(
    ...     key="req-001",
    ...     content="User authentication via OAuth2",
    ...     metadata={"type": "requirement"}
    ... )
    >>> results = await memory.search_similar("OAuth login", k=5)
    >>> for result in results:
    ...     print(f"{result.key}: {result.score:.2f}")

Security Note:
    The persist_directory should be within the project's .yolo directory
    to ensure proper isolation between projects.

Thread Safety:
    This class uses an asyncio.Lock to ensure thread-safe access to the
    ChromaDB collection in concurrent async contexts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

import chromadb
from chromadb.api.types import Metadatas, QueryResult
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yolo_developer.memory.protocol import MemoryResult
from yolo_developer.orchestrator.context import Decision

if TYPE_CHECKING:
    from yolo_developer.memory.graph import JSONGraphStore

logger = logging.getLogger(__name__)


class ChromaDBError(Exception):
    """Wrapper for ChromaDB errors with additional context.

    Provides descriptive error messages for permanent failures after
    retry attempts are exhausted.

    Attributes:
        operation: The ChromaDB operation that failed.
        original_error: The underlying exception from ChromaDB.
    """

    def __init__(self, message: str, operation: str, original_error: Exception) -> None:
        """Initialize ChromaDBError with context.

        Args:
            message: Descriptive error message.
            operation: The operation that failed (e.g., "upsert", "query").
            original_error: The underlying ChromaDB exception.
        """
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for observability."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "ChromaDB operation retry attempt %d: %s",
        retry_state.attempt_number,
        str(exception) if exception else "unknown error",
        extra={
            "attempt": retry_state.attempt_number,
            "exception_type": type(exception).__name__ if exception else None,
        },
    )


# Transient errors that should be retried (disk I/O, connection issues)
# OSError covers file system errors, RuntimeError covers ChromaDB internal issues
_TRANSIENT_EXCEPTIONS = (OSError, RuntimeError, ConnectionError, TimeoutError)

# Retry decorator for ChromaDB operations: 3 attempts with exponential backoff
# Only retries on transient errors, not programming errors (ValueError, TypeError)
_chromadb_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=2.0),
    retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
    before_sleep=_log_retry_attempt,
    reraise=True,
)


class ChromaMemory:
    """ChromaDB implementation of MemoryStore for vector embeddings.

    Uses ChromaDB's PersistentClient for local storage with automatic
    embedding generation via the default embedding function. Supports
    semantic similarity search and upsert behavior for content updates.

    Can optionally integrate with JSONGraphStore for relationship storage.

    Attributes:
        client: ChromaDB PersistentClient instance.
        collection: ChromaDB collection for storing embeddings.

    Example:
        >>> memory = ChromaMemory(persist_directory="/tmp/test_memory")
        >>> await memory.store_embedding(
        ...     key="doc-1",
        ...     content="Python is a programming language",
        ...     metadata={"type": "fact"}
        ... )
        >>> results = await memory.search_similar("programming", k=3)
        >>>
        >>> # With graph storage integration
        >>> from yolo_developer.memory import JSONGraphStore
        >>> graph = JSONGraphStore(persist_path="/tmp/graph.json")
        >>> memory = ChromaMemory("/tmp/test", graph_store=graph)
        >>> await memory.store_relationship("story-001", "req-001", "implements")
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "yolo_memory",
        graph_store: JSONGraphStore | None = None,
    ) -> None:
        """Initialize ChromaMemory with persistent storage.

        Args:
            persist_directory: Directory path for ChromaDB persistence.
                Data will be stored in this directory and survive restarts.
            collection_name: Name for the ChromaDB collection.
                Defaults to "yolo_memory".
            graph_store: Optional JSONGraphStore for relationship storage.
                If provided, store_relationship will delegate to this store.
                If not provided, store_relationship logs a warning.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        self._graph_store = graph_store
        self._lock = asyncio.Lock()  # Protect concurrent access to collection

    @_chromadb_retry
    def _upsert(
        self,
        key: str,
        content: str,
        metadatas: Metadatas | None,
    ) -> None:
        """Upsert with retry logic for transient errors."""
        self.collection.upsert(
            ids=[key],
            documents=[content],
            metadatas=metadatas,
        )

    @_chromadb_retry
    def _query(
        self,
        query: str,
        n_results: int,
    ) -> QueryResult:
        """Query with retry logic for transient errors."""
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    @_chromadb_retry
    def _count(self) -> int:
        """Count with retry logic for transient errors."""
        return self.collection.count()

    async def store_embedding(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store content with vector embedding for similarity search.

        Uses ChromaDB's upsert operation for idempotent storage - if a key
        already exists, its content and metadata will be updated.

        Args:
            key: Unique identifier for this content. Used for updates and deletion.
            content: Text content to embed and store. ChromaDB will generate
                the embedding automatically using its default embedding function.
            metadata: Additional data to store alongside the embedding.
                Must be a dict with string keys. Empty dict is allowed.

        Note:
            ChromaDB operations are synchronous under the hood, but this method
            is async to satisfy the MemoryStore protocol. For embedded mode,
            this is acceptable as operations are fast and local.
        """
        # ChromaDB requires metadatas to be None or non-empty
        # Handle empty dict by passing None
        metadatas: Metadatas | None = None
        if metadata:
            # Convert Any values to ChromaDB-compatible types
            metadatas = cast(Metadatas, [dict(metadata)])

        # Use lock to protect concurrent access, retry-wrapped for transient errors
        async with self._lock:
            self._upsert(key, content, metadatas)

    async def search_similar(
        self,
        query: str,
        k: int = 5,
    ) -> list[MemoryResult]:
        """Search for content similar to the query.

        Performs semantic similarity search using vector embeddings.
        ChromaDB generates an embedding for the query and finds the
        k most similar stored documents.

        Args:
            query: Text to find similar content for. Will be embedded
                and compared against stored embeddings.
            k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of MemoryResult objects ordered by similarity (highest first).
            Returns empty list if no documents are stored.

        Note:
            Scores are converted from ChromaDB's cosine distance (0-2 range,
            lower = more similar) to similarity (higher = more similar).
        """
        # Use lock to protect concurrent access
        async with self._lock:
            # Handle empty collection case (uses retry-wrapped method)
            count = self._count()
            if count == 0:
                return []

            # Use retry-wrapped method for transient error handling
            results = self._query(query, min(k, count))

        # Convert to MemoryResult objects
        memory_results: list[MemoryResult] = []

        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, id_ in enumerate(ids):
                # Convert cosine distance to similarity score
                # Cosine distance is in range [0, 2], we convert to similarity
                distance = distances[i] if i < len(distances) else 0.0
                score = 1.0 - (distance / 2.0)  # Normalize to [0, 1]

                content = documents[i] if i < len(documents) else ""
                raw_metadata = metadatas[i] if i < len(metadatas) else {}

                # Handle None metadata from ChromaDB and convert to dict[str, Any]
                result_metadata: dict[str, Any] = {}
                if raw_metadata is not None:
                    result_metadata = dict(raw_metadata)

                memory_results.append(
                    MemoryResult(
                        key=id_,
                        content=content,
                        score=score,
                        metadata=result_metadata,
                    )
                )

        return memory_results

    async def store_relationship(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        """Store a relationship between two entities.

        Delegates to the configured JSONGraphStore if available.
        If no graph store is configured, logs a warning.

        Args:
            source: The source entity identifier (e.g., "story-001").
            target: The target entity identifier (e.g., "req-001").
            relation: The type of relationship (e.g., "implements", "depends_on").

        Example:
            >>> from yolo_developer.memory import ChromaMemory, JSONGraphStore
            >>> graph = JSONGraphStore(persist_path=".yolo/memory/graph.json")
            >>> memory = ChromaMemory(".yolo/memory", graph_store=graph)
            >>> await memory.store_relationship("story-001", "req-001", "implements")
        """
        if self._graph_store is not None:
            await self._graph_store.store_relationship(source, target, relation)
        else:
            logger.warning(
                "store_relationship called but no graph_store configured",
                extra={
                    "source": source,
                    "target": target,
                    "relation": relation,
                },
            )

    async def store_decision(
        self,
        decision: Decision,
    ) -> str:
        """Store a decision for later semantic retrieval.

        Stores the decision content with its metadata for semantic search.
        If a graph store is configured, also stores relationships to
        related artifacts.

        Args:
            decision: The Decision dataclass instance to store.

        Returns:
            The key used to store the decision (format: decision-{agent}-{timestamp}).

        Example:
            >>> from yolo_developer.orchestrator import Decision
            >>> from yolo_developer.memory import ChromaMemory
            >>>
            >>> memory = ChromaMemory(persist_directory=".yolo/memory")
            >>> decision = Decision(
            ...     agent="analyst",
            ...     summary="Selected REST API",
            ...     rationale="Simpler for MVP",
            ... )
            >>> key = await memory.store_decision(decision)
            >>> print(key)
            decision-analyst-2024-01-01T12:00:00+00:00
        """
        # Ensure we have a proper Decision instance
        if not isinstance(decision, Decision):
            raise TypeError(f"Expected Decision, got {type(decision).__name__}")

        # Generate unique key based on agent and timestamp
        key = f"decision-{decision.agent}-{decision.timestamp.isoformat()}"

        # Create content from summary and rationale for semantic search
        content = f"{decision.summary}: {decision.rationale}"

        # Build metadata
        metadata: dict[str, Any] = {
            "type": "decision",
            "agent": decision.agent,
            "timestamp": decision.timestamp.isoformat(),
        }

        # Store related artifacts as comma-separated string (ChromaDB doesn't support lists)
        if decision.related_artifacts:
            metadata["related_artifacts"] = ",".join(decision.related_artifacts)

        # Store the embedding
        await self.store_embedding(key=key, content=content, metadata=metadata)

        # Store relationships in graph if configured
        if self._graph_store is not None and decision.related_artifacts:
            for artifact in decision.related_artifacts:
                await self._graph_store.store_relationship(
                    source=key,
                    target=artifact,
                    relation="relates_to",
                )

        logger.debug(
            "Stored decision",
            extra={
                "key": key,
                "agent": decision.agent,
                "related_artifacts": decision.related_artifacts,
            },
        )

        return key

    async def query_decisions(
        self,
        query: str,
        agent: str | None = None,
        k: int = 5,
    ) -> list[MemoryResult]:
        """Query decisions semantically.

        Searches for decisions matching the query using semantic similarity.
        Results can be filtered by agent and are limited to k results.

        Args:
            query: Semantic search query text.
            agent: Optional filter to only return decisions by this agent.
            k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of matching MemoryResult objects, filtered to only decisions.

        Example:
            >>> from yolo_developer.memory import ChromaMemory
            >>>
            >>> memory = ChromaMemory(persist_directory=".yolo/memory")
            >>> results = await memory.query_decisions("database choice", agent="architect")
            >>> for result in results:
            ...     print(f"{result.key}: {result.metadata['agent']}")
        """
        # Over-fetch to account for filtering, but cap to prevent excessive queries
        # Multiplier of 3 handles typical filtering, cap of 100 prevents abuse
        fetch_k = min(k * 3, max(k + 50, 100))
        results = await self.search_similar(query, k=fetch_k)

        # Filter to only decisions
        decision_results = [r for r in results if r.metadata.get("type") == "decision"]

        # Filter by agent if specified
        if agent is not None:
            decision_results = [r for r in decision_results if r.metadata.get("agent") == agent]

        # Return up to k results
        return decision_results[:k]
