"""JSON-based graph storage for YOLO Developer.

This module provides the JSONGraphStore class for storing relationships between
artifacts in a JSON file. It supports querying by source, target, relation type,
and transitive queries to find related nodes within a specified depth.

Example:
    >>> from yolo_developer.memory import JSONGraphStore
    >>>
    >>> store = JSONGraphStore(persist_path=".yolo/memory/graph.json")
    >>> await store.store_relationship("story-001", "req-001", "implements")
    >>> results = await store.get_relationships(source="story-001")
    >>> for result in results:
    ...     print(f"{result.source} -> {result.target}: {result.relation}")

Security Note:
    The persist_path should be within the project's .yolo directory
    to ensure proper isolation between projects.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class JSONGraphError(Exception):
    """Wrapper for JSON graph errors with additional context.

    Provides descriptive error messages for permanent failures after
    retry attempts are exhausted.

    Attributes:
        operation: The graph operation that failed.
        original_error: The underlying exception.
    """

    def __init__(self, message: str, operation: str, original_error: Exception) -> None:
        """Initialize JSONGraphError with context.

        Args:
            message: Descriptive error message.
            operation: The operation that failed (e.g., "load", "save").
            original_error: The underlying exception.
        """
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for observability."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "JSON graph operation retry attempt %d: %s",
        retry_state.attempt_number,
        str(exception) if exception else "unknown error",
        extra={
            "attempt": retry_state.attempt_number,
            "exception_type": type(exception).__name__ if exception else None,
        },
    )


# Transient errors that should be retried (file I/O issues)
# Note: IOError is alias for OSError in Python 3, PermissionError is subclass of OSError
_TRANSIENT_EXCEPTIONS = (OSError, TimeoutError)

# Retry decorator for file I/O operations: 3 attempts with exponential backoff
_file_io_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=2.0),
    retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
    before_sleep=_log_retry_attempt,
    reraise=True,
)


@dataclass(frozen=True)
class Relationship:
    """A directed edge in the relationship graph.

    Represents a relationship between two entities (source -> target)
    with a specified relation type.

    Attributes:
        source: The source entity identifier (e.g., "story-001").
        target: The target entity identifier (e.g., "req-001").
        relation: The type of relationship (e.g., "implements", "depends_on").

    Example:
        >>> edge = Relationship(
        ...     source="story-001",
        ...     target="req-001",
        ...     relation="implements"
        ... )
        >>> edge.source
        'story-001'
    """

    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class RelationshipResult:
    """Result from a relationship query.

    Represents a single query result containing the relationship details
    and optional path information for transitive queries.

    Attributes:
        source: The source entity identifier.
        target: The target entity identifier.
        relation: The type of relationship.
        path: Optional path from origin to this relationship (for transitive queries).

    Example:
        >>> result = RelationshipResult(
        ...     source="story-001",
        ...     target="req-001",
        ...     relation="implements"
        ... )
        >>> result.relation
        'implements'
    """

    source: str
    target: str
    relation: str
    path: list[str] | None = None


class JSONGraphStore:
    """JSON-based graph storage for artifact relationships.

    Stores relationships as a list of edges in a JSON file. Supports queries
    by source, target, relation type, and transitive queries to find all
    nodes reachable within a specified depth.

    Uses asyncio.Lock for concurrent access protection and tenacity for
    retry logic on file I/O operations.

    Attributes:
        persist_path: Path to the JSON file for persistence.

    Example:
        >>> store = JSONGraphStore(persist_path="/tmp/graph.json")
        >>> await store.store_relationship("story-001", "req-001", "implements")
        >>> results = await store.get_relationships(source="story-001")
        >>> for r in results:
        ...     print(f"{r.source} -> {r.target}")
        story-001 -> req-001
    """

    def __init__(self, persist_path: str) -> None:
        """Initialize JSONGraphStore with persistent storage.

        Args:
            persist_path: Path to the JSON file for persistence.
                The file will be created if it doesn't exist.
                Parent directories will be created as needed.
        """
        self.persist_path = Path(persist_path)
        self._edges: set[Relationship] = set()
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        """Load existing graph from JSON file.

        Reads the JSON file and populates the internal edge set.
        Handles missing files gracefully (starts with empty graph).
        Handles JSON parse errors gracefully (logs warning, starts with empty graph).
        """
        if not self.persist_path.exists():
            logger.debug("graph_file_not_found", extra={"path": str(self.persist_path)})
            self._edges = set()
            return

        try:
            self._load_from_file()
        except json.JSONDecodeError as e:
            logger.warning(
                "graph_json_parse_failed",
                extra={"path": str(self.persist_path), "error": str(e)},
            )
            self._edges = set()
        except KeyError as e:
            logger.warning(
                "graph_data_invalid",
                extra={"path": str(self.persist_path), "error": str(e)},
            )
            self._edges = set()

    @_file_io_retry
    def _load_from_file(self) -> None:
        """Load graph data from file with retry logic."""
        data = json.loads(self.persist_path.read_text())
        self._edges = {
            Relationship(e["source"], e["target"], e["relation"]) for e in data.get("edges", [])
        }

    @_file_io_retry
    def _save(self) -> None:
        """Persist graph to JSON file.

        Creates parent directories if they don't exist.
        Writes the graph as a JSON object with an "edges" array.
        """
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"edges": [asdict(e) for e in self._edges]}
        self.persist_path.write_text(json.dumps(data, indent=2))

    async def store_relationship(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        """Store a relationship between two entities.

        Creates a directed edge from source to target with the specified
        relation type. Duplicate relationships are ignored (idempotent).

        Uses asyncio.Lock for concurrent access protection.

        Args:
            source: The source entity identifier (e.g., "story-001").
            target: The target entity identifier (e.g., "req-001").
            relation: The type of relationship (e.g., "implements", "depends_on").

        Example:
            >>> await store.store_relationship(
            ...     source="story-001",
            ...     target="req-001",
            ...     relation="implements"
            ... )
        """
        async with self._lock:
            edge = Relationship(source, target, relation)
            if edge not in self._edges:
                self._edges.add(edge)
                self._save()

    async def get_relationships(
        self,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> list[RelationshipResult]:
        """Query relationships by optional filters.

        Returns all relationships matching the specified filters.
        Multiple filters are combined with AND logic.

        Uses asyncio.Lock to prevent race conditions with concurrent writes.

        Args:
            source: Filter by source entity (optional).
            target: Filter by target entity (optional).
            relation: Filter by relation type (optional).

        Returns:
            List of RelationshipResult objects matching the filters.
            Returns all relationships if no filters are specified.

        Example:
            >>> # Get all relationships from story-001
            >>> results = await store.get_relationships(source="story-001")
            >>>
            >>> # Get all "implements" relationships
            >>> results = await store.get_relationships(relation="implements")
            >>>
            >>> # Get all relationships (no filters)
            >>> results = await store.get_relationships()
        """
        async with self._lock:
            results: list[RelationshipResult] = []
            for edge in self._edges:
                if source is not None and edge.source != source:
                    continue
                if target is not None and edge.target != target:
                    continue
                if relation is not None and edge.relation != relation:
                    continue
                results.append(
                    RelationshipResult(
                        source=edge.source,
                        target=edge.target,
                        relation=edge.relation,
                    )
                )
            return results

    async def get_related(
        self,
        node: str,
        depth: int = 1,
    ) -> list[str]:
        """Find all nodes reachable from node within depth.

        Performs a breadth-first search to find all nodes connected
        to the starting node through outgoing edges.

        Uses asyncio.Lock to prevent race conditions with concurrent writes.

        Args:
            node: The starting node identifier.
            depth: Maximum depth to traverse (default: 1).
                depth=1 returns direct neighbors only.
                depth=2 returns neighbors and their neighbors.

        Returns:
            List of node identifiers reachable within the specified depth.
            Does not include the starting node.
            Handles cycles gracefully (each node visited at most once).

        Note:
            Path information is not included in results. Only node identifiers
            are returned. For path tracking, use get_relationships() to
            reconstruct paths manually.

        Example:
            >>> # A -> B -> C chain
            >>> await store.store_relationship("A", "B", "links")
            >>> await store.store_relationship("B", "C", "links")
            >>>
            >>> # depth=1 returns only B
            >>> await store.get_related("A", depth=1)
            ['B']
            >>>
            >>> # depth=2 returns B and C
            >>> await store.get_related("A", depth=2)
            ['B', 'C']
        """
        async with self._lock:
            visited: set[str] = set()
            queue: deque[tuple[str, int]] = deque([(node, 0)])

            while queue:
                current, current_depth = queue.popleft()
                if current_depth > depth:
                    continue
                if current in visited and current != node:
                    continue
                visited.add(current)

                # Find all adjacent nodes (outgoing edges)
                for edge in self._edges:
                    if edge.source == current and edge.target not in visited:
                        queue.append((edge.target, current_depth + 1))

            # Remove the starting node from results
            visited.discard(node)
            return list(visited)
