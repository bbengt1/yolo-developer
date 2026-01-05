"""Pattern storage for code pattern learning.

This module provides the ChromaPatternStore class for storing and retrieving
learned code patterns using ChromaDB as the backend. Patterns are stored with
embeddings for semantic search and isolated per project.

Example:
    >>> from yolo_developer.memory.pattern_store import ChromaPatternStore
    >>> from yolo_developer.memory.patterns import CodePattern, PatternType
    >>>
    >>> store = ChromaPatternStore(
    ...     persist_directory=".yolo/patterns",
    ...     project_id="my-project",
    ... )
    >>> pattern = CodePattern(
    ...     pattern_type=PatternType.NAMING_FUNCTION,
    ...     name="function_naming",
    ...     value="snake_case",
    ...     confidence=0.95,
    ... )
    >>> pattern_id = await store.store_pattern(pattern)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, cast

import chromadb

from yolo_developer.memory.isolation import DEFAULT_PROJECT_ID, validate_project_id
from yolo_developer.memory.patterns import CodePattern, PatternResult, PatternType

logger = logging.getLogger(__name__)


class ChromaPatternStore:
    """ChromaDB-backed pattern storage.

    Stores code patterns with embeddings for semantic search.
    Patterns are isolated per project using collection naming.

    Attributes:
        project_id: Identifier for project isolation.
        persist_directory: Directory for ChromaDB persistence.

    Example:
        >>> store = ChromaPatternStore(
        ...     persist_directory=".yolo/patterns",
        ...     project_id="my-project",
        ... )
        >>> patterns = await store.get_patterns_by_type(PatternType.NAMING_FUNCTION)
    """

    def __init__(
        self,
        persist_directory: str,
        project_id: str | None = None,
    ) -> None:
        """Initialize the pattern store.

        Args:
            persist_directory: Directory for ChromaDB persistence.
            project_id: Identifier for project isolation. Defaults to "default".

        Raises:
            InvalidProjectIdError: If project_id is invalid.
        """
        self.persist_directory = persist_directory
        # Validate project_id (raises InvalidProjectIdError if invalid)
        effective_project_id = project_id if project_id is not None else DEFAULT_PROJECT_ID
        self.project_id = validate_project_id(effective_project_id)

        # Initialize ChromaDB client with PersistentClient (same as ChromaMemory)
        self._client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection for this project's patterns
        collection_name = f"patterns_{self.project_id}"
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.debug(
            "Initialized pattern store",
            extra={
                "project_id": self.project_id,
                "persist_directory": persist_directory,
                "collection_name": collection_name,
            },
        )

    async def store_pattern(self, pattern: CodePattern) -> str:
        """Store a code pattern and return its ID.

        Args:
            pattern: The code pattern to store.

        Returns:
            The unique ID assigned to the pattern.
        """
        # Generate a unique ID based on pattern type and name
        pattern_id = f"{pattern.pattern_type.value}_{pattern.name}"

        # Generate embedding text
        embedding_text = pattern.to_embedding_text()

        # Prepare metadata
        metadata: dict[str, Any] = {
            "pattern_type": pattern.pattern_type.value,
            "name": pattern.name,
            "value": pattern.value,
            "confidence": pattern.confidence,
            "examples": ",".join(pattern.examples[:10]),
            "source_files": ",".join(pattern.source_files[:10]),
            "created_at": pattern.created_at.isoformat(),
        }

        # Store in ChromaDB (upsert behavior)
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[pattern_id],
            documents=[embedding_text],
            metadatas=[metadata],
        )

        logger.debug(
            "Stored pattern",
            extra={
                "pattern_id": pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "value": pattern.value,
            },
        )

        return pattern_id

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
        # Check if collection has any items
        count = await asyncio.to_thread(self._collection.count)
        if count == 0:
            return []

        # Build where filter for pattern type
        where_filter: dict[str, str] | None = None
        if pattern_type is not None:
            where_filter = {"pattern_type": pattern_type.value}

        # Query ChromaDB
        try:
            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[query] if query else None,
                n_results=min(k, count),
                where=where_filter,  # type: ignore[arg-type]
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(
                "Pattern search failed",
                extra={"query": query, "error": str(e)},
            )
            return []

        # Convert results to PatternResult instances
        pattern_results: list[PatternResult] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        raw_metadatas = results.get("metadatas")
        metadatas: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], raw_metadatas[0]) if raw_metadatas else []
        )
        raw_distances = results.get("distances")
        distances: list[float] = raw_distances[0] if raw_distances else []

        for i, pattern_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0.0

            # Convert distance to similarity (ChromaDB returns distances)
            # For cosine distance: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - distance))

            # Reconstruct CodePattern from metadata
            try:
                pattern = self._metadata_to_pattern(metadata)
                pattern_results.append(PatternResult(pattern=pattern, similarity=similarity))
            except Exception as e:
                logger.warning(
                    "Failed to reconstruct pattern from metadata",
                    extra={"pattern_id": pattern_id, "error": str(e)},
                )

        return pattern_results

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
        # Check if collection has any items
        count = await asyncio.to_thread(self._collection.count)
        if count == 0:
            return []

        # Query by metadata filter
        try:
            results = await asyncio.to_thread(
                self._collection.get,
                where={"pattern_type": pattern_type.value},
                include=["metadatas"],
            )
        except Exception as e:
            logger.warning(
                "Get patterns by type failed",
                extra={"pattern_type": pattern_type.value, "error": str(e)},
            )
            return []

        if not results or not results.get("ids"):
            return []

        patterns: list[CodePattern] = []
        raw_metadatas = results.get("metadatas")
        metadatas: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], raw_metadatas) if raw_metadatas else []
        )

        for i, _pattern_id in enumerate(results["ids"]):
            metadata = metadatas[i] if i < len(metadatas) else {}
            try:
                pattern = self._metadata_to_pattern(metadata)
                patterns.append(pattern)
            except Exception as e:
                logger.warning(
                    "Failed to reconstruct pattern",
                    extra={"error": str(e)},
                )

        return patterns

    def _metadata_to_pattern(self, metadata: dict[str, Any]) -> CodePattern:
        """Convert ChromaDB metadata to a CodePattern.

        Args:
            metadata: Metadata dict from ChromaDB.

        Returns:
            Reconstructed CodePattern instance.
        """
        # Parse examples and source_files from comma-separated strings
        examples_str = metadata.get("examples", "")
        examples = tuple(examples_str.split(",")) if examples_str else ()

        source_files_str = metadata.get("source_files", "")
        source_files = tuple(source_files_str.split(",")) if source_files_str else ()

        # Parse created_at
        created_at_str = metadata.get("created_at", "")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except ValueError:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

        return CodePattern(
            pattern_type=PatternType(metadata["pattern_type"]),
            name=metadata["name"],
            value=metadata["value"],
            confidence=float(metadata["confidence"]),
            examples=examples,
            source_files=source_files,
            created_at=created_at,
        )
