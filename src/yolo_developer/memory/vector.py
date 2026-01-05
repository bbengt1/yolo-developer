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
"""

from __future__ import annotations

import logging
from typing import Any, cast

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
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "yolo_memory",
    ) -> None:
        """Initialize ChromaMemory with persistent storage.

        Args:
            persist_directory: Directory path for ChromaDB persistence.
                Data will be stored in this directory and survive restarts.
            collection_name: Name for the ChromaDB collection.
                Defaults to "yolo_memory".
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

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

        # Use retry-wrapped method for transient error handling
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

        This is a stub implementation for MVP. Graph relationship storage
        will be implemented in Story 2.3 using JSON-based graph storage.

        Args:
            source: The source entity identifier (e.g., "story-001").
            target: The target entity identifier (e.g., "req-001").
            relation: The type of relationship (e.g., "implements", "depends_on").

        Note:
            This method is a no-op in the current implementation.
            Full relationship storage will be added in Story 2.3.
        """
        # No-op for MVP - graph storage implemented in Story 2.3
        pass
