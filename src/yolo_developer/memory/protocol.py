"""Memory store protocol and types for YOLO Developer.

This module defines the MemoryStore protocol and supporting types for the
memory abstraction layer. The protocol pattern enables different storage
backends (ChromaDB, Neo4j, mock) to be used interchangeably without
requiring inheritance.

Example:
    >>> from yolo_developer.memory import MemoryStore, MemoryResult
    >>>
    >>> class MyMemoryStore:
    ...     async def store_embedding(self, key: str, content: str, metadata: dict) -> None:
    ...         # Implementation
    ...         pass
    ...     async def search_similar(self, query: str, k: int = 5) -> list[MemoryResult]:
    ...         return []
    ...     async def store_relationship(self, source: str, target: str, relation: str) -> None:
    ...         pass
    >>>
    >>> # MyMemoryStore is compatible with MemoryStore protocol
    >>> memory: MemoryStore = MyMemoryStore()

Security Note:
    Memory storage may contain sensitive project data. Implementations should
    ensure proper access controls and data isolation between projects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from yolo_developer.memory.patterns import CodePattern, PatternResult, PatternType


@dataclass(frozen=True)
class MemoryResult:
    """Result from a memory similarity search.

    Represents a single search result containing the matched content,
    its similarity score, and associated metadata.

    Attributes:
        key: Unique identifier for the stored content.
        content: The actual text content that was stored and matched.
        score: Similarity score between 0.0 and 1.0 (higher is more similar).
        metadata: Additional metadata stored alongside the content.

    Example:
        >>> result = MemoryResult(
        ...     key="req-001",
        ...     content="User authentication via OAuth2",
        ...     score=0.95,
        ...     metadata={"type": "requirement", "source": "prd.md"}
        ... )
        >>> result.score
        0.95
    """

    key: str
    content: str
    score: float
    metadata: dict[str, Any]


class MemoryStore(Protocol):
    """Protocol for memory storage backends.

    This protocol defines the interface for memory storage implementations.
    Any class implementing these methods is considered compatible, enabling
    different backends (ChromaDB, Neo4j, mock) to be used interchangeably.

    All methods are async to support non-blocking I/O operations.

    The protocol defines three core methods:
    - store_embedding: Store content with vector embedding for similarity search
    - search_similar: Find content similar to a query string
    - store_relationship: Store relationships between entities (graph storage)

    Example:
        >>> class MockMemory:
        ...     async def store_embedding(
        ...         self, key: str, content: str, metadata: dict[str, Any]
        ...     ) -> None:
        ...         pass  # Store implementation
        ...
        ...     async def search_similar(
        ...         self, query: str, k: int = 5
        ...     ) -> list[MemoryResult]:
        ...         return []  # Search implementation
        ...
        ...     async def store_relationship(
        ...         self, source: str, target: str, relation: str
        ...     ) -> None:
        ...         pass  # Relationship storage
        >>>
        >>> # MockMemory is compatible with MemoryStore protocol
        >>> memory: MemoryStore = MockMemory()
    """

    async def store_embedding(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store content with vector embedding for similarity search.

        Stores the content along with its vector embedding for later
        similarity-based retrieval. The embedding is generated internally
        by the implementation.

        Args:
            key: Unique identifier for this content. Used for updates and deletion.
            content: Text content to embed and store. Will be converted to a
                vector embedding for similarity search.
            metadata: Additional data to store alongside the embedding.
                Common keys include "type", "source", "timestamp".

        Example:
            >>> await memory.store_embedding(
            ...     key="req-001",
            ...     content="User must authenticate via OAuth2",
            ...     metadata={"type": "requirement", "source": "prd.md"}
            ... )
        """
        ...

    async def search_similar(
        self,
        query: str,
        k: int = 5,
    ) -> list[MemoryResult]:
        """Search for content similar to the query.

        Performs semantic similarity search using vector embeddings.
        Returns the top k most similar results ordered by similarity score.

        Args:
            query: Text to find similar content for. Will be converted to
                a vector embedding and compared against stored embeddings.
            k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of MemoryResult objects ordered by similarity (highest first).
            May return fewer than k results if not enough content is stored.

        Example:
            >>> results = await memory.search_similar("OAuth authentication", k=3)
            >>> for result in results:
            ...     print(f"{result.key}: {result.score:.2f}")
            req-001: 0.95
            req-007: 0.82
            req-012: 0.71
        """
        ...

    async def store_relationship(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        """Store a relationship between two entities.

        Creates a directed edge in the relationship graph from source
        to target with the specified relation type.

        Args:
            source: The source entity identifier (e.g., "story-001").
            target: The target entity identifier (e.g., "req-001").
            relation: The type of relationship (e.g., "implements", "depends_on").

        Example:
            >>> await memory.store_relationship(
            ...     source="story-001",
            ...     target="req-001",
            ...     relation="implements"
            ... )
        """
        ...

    async def store_pattern(self, pattern: CodePattern) -> str:
        """Store a code pattern and return its ID.

        Stores the pattern with its vector embedding for semantic search.
        Uses upsert behavior - updates existing pattern if same ID exists.

        Args:
            pattern: The CodePattern instance to store.

        Returns:
            The unique identifier assigned to the pattern.

        Example:
            >>> pattern = CodePattern(
            ...     pattern_type=PatternType.NAMING_FUNCTION,
            ...     name="function_naming",
            ...     value="snake_case",
            ...     confidence=0.95,
            ... )
            >>> pattern_id = await memory.store_pattern(pattern)
        """
        ...

    async def search_patterns(
        self,
        query: str = "",
        pattern_type: PatternType | None = None,
        k: int = 5,
    ) -> list[PatternResult]:
        """Search for patterns by semantic similarity.

        Args:
            query: Search query for semantic matching.
            pattern_type: Optional filter by pattern type.
            k: Maximum number of results to return.

        Returns:
            List of PatternResult instances ordered by similarity.
        """
        ...

    async def get_patterns_by_type(
        self,
        pattern_type: PatternType,
    ) -> list[CodePattern]:
        """Get all patterns of a specific type.

        Args:
            pattern_type: The type of patterns to retrieve.

        Returns:
            List of CodePattern instances of the specified type.
        """
        ...
