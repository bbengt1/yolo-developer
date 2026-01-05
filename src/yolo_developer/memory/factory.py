"""Memory factory for project-isolated store creation.

This module provides the MemoryFactory class, which creates all memory stores
with consistent project_id isolation. It serves as the single entry point for
creating vector, graph, pattern, and decision stores.

Example:
    >>> from yolo_developer.memory import MemoryFactory
    >>>
    >>> factory = MemoryFactory(
    ...     project_id="my-project",
    ...     base_directory=".yolo/memory",
    ... )
    >>> vector_store = factory.create_vector_store()
    >>> graph_store = factory.create_graph_store()
    >>> pattern_store = factory.create_pattern_store()
    >>> decision_store = factory.create_decision_store()
    >>>
    >>> # Get all stores at once
    >>> stores = factory.get_all_stores()

Security Note:
    All stores created by the factory use the same project_id for isolation.
    This ensures data from one project cannot leak into another.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from yolo_developer.memory.isolation import (
    DEFAULT_PROJECT_ID,
    validate_project_id,
)

if TYPE_CHECKING:
    from yolo_developer.memory.decision_store import ChromaDecisionStore
    from yolo_developer.memory.graph import JSONGraphStore
    from yolo_developer.memory.pattern_store import ChromaPatternStore
    from yolo_developer.memory.vector import ChromaMemory


class StoreDict(TypedDict):
    """Type dictionary for get_all_stores return value."""

    vector: ChromaMemory
    graph: JSONGraphStore
    pattern: ChromaPatternStore
    decision: ChromaDecisionStore


class MemoryFactory:
    """Factory for creating project-isolated memory stores.

    Creates all memory stores (vector, graph, pattern, decision) with consistent
    project_id isolation. All stores created by a single factory instance share
    the same project_id, ensuring complete isolation between projects.

    Attributes:
        project_id: The validated project identifier.
        base_directory: Base directory for all memory stores.

    Example:
        >>> factory = MemoryFactory(
        ...     project_id="my-project",
        ...     base_directory=".yolo/memory",
        ... )
        >>> vector_store = factory.create_vector_store()
        >>> # Collection name will be "yolo_memory_my-project"
        >>>
        >>> graph_store = factory.create_graph_store()
        >>> # File path will be ".yolo/memory/my-project/graph.json"
    """

    def __init__(
        self,
        base_directory: str,
        project_id: str | None = None,
    ) -> None:
        """Initialize MemoryFactory with project isolation.

        Args:
            base_directory: Base directory for all memory stores.
                Vector stores use this directly, graph stores create
                project subdirectories.
            project_id: Optional project identifier for isolation.
                If not provided, uses DEFAULT_PROJECT_ID ("default").
                Must be 1-64 characters, alphanumeric with hyphens/underscores.

        Raises:
            InvalidProjectIdError: If project_id is invalid.

        Example:
            >>> # With explicit project ID
            >>> factory = MemoryFactory(
            ...     project_id="my-project",
            ...     base_directory=".yolo/memory",
            ... )
            >>>
            >>> # With default project ID
            >>> factory = MemoryFactory(base_directory=".yolo/memory")
            >>> factory.project_id
            'default'
        """
        # Use default if not specified
        effective_project_id = project_id if project_id is not None else DEFAULT_PROJECT_ID

        # Validate project ID (raises InvalidProjectIdError if invalid)
        self._project_id = validate_project_id(effective_project_id)
        self._base_directory = base_directory

    @property
    def project_id(self) -> str:
        """Get the validated project ID."""
        return self._project_id

    @property
    def base_directory(self) -> str:
        """Get the base directory path."""
        return self._base_directory

    def create_vector_store(self, collection_name: str = "yolo_memory") -> ChromaMemory:
        """Create a project-isolated ChromaMemory vector store.

        Creates a ChromaMemory instance with the factory's project_id.
        The collection name will be `{collection_name}_{project_id}`.

        Args:
            collection_name: Base name for the ChromaDB collection.
                Defaults to "yolo_memory".

        Returns:
            ChromaMemory instance with project isolation.

        Example:
            >>> factory = MemoryFactory(
            ...     project_id="my-project",
            ...     base_directory=".yolo/memory",
            ... )
            >>> store = factory.create_vector_store()
            >>> # Collection name: "yolo_memory_my-project"
        """
        from yolo_developer.memory.vector import ChromaMemory

        return ChromaMemory(
            persist_directory=self._base_directory,
            collection_name=collection_name,
            project_id=self._project_id,
        )

    def create_graph_store(self) -> JSONGraphStore:
        """Create a project-isolated JSONGraphStore.

        Creates a JSONGraphStore instance with the file path scoped
        to the project: `{base_directory}/{project_id}/graph.json`.

        Returns:
            JSONGraphStore instance with project isolation.

        Example:
            >>> factory = MemoryFactory(
            ...     project_id="my-project",
            ...     base_directory=".yolo/memory",
            ... )
            >>> store = factory.create_graph_store()
            >>> # File path: ".yolo/memory/my-project/graph.json"
        """
        from yolo_developer.memory.graph import JSONGraphStore

        # Create project-scoped path
        graph_path = Path(self._base_directory) / self._project_id / "graph.json"

        return JSONGraphStore(persist_path=str(graph_path))

    def create_pattern_store(self) -> ChromaPatternStore:
        """Create a project-isolated ChromaPatternStore.

        Creates a ChromaPatternStore instance with the factory's project_id.
        The collection name will be `patterns_{project_id}`.

        Returns:
            ChromaPatternStore instance with project isolation.

        Example:
            >>> factory = MemoryFactory(
            ...     project_id="my-project",
            ...     base_directory=".yolo/memory",
            ... )
            >>> store = factory.create_pattern_store()
            >>> # Collection name: "patterns_my-project"
        """
        from yolo_developer.memory.pattern_store import ChromaPatternStore

        return ChromaPatternStore(
            persist_directory=self._base_directory,
            project_id=self._project_id,
        )

    def create_decision_store(self) -> ChromaDecisionStore:
        """Create a project-isolated ChromaDecisionStore.

        Creates a ChromaDecisionStore instance with the factory's project_id.
        The collection name will be `decisions_{project_id}`.

        Returns:
            ChromaDecisionStore instance with project isolation.

        Example:
            >>> factory = MemoryFactory(
            ...     project_id="my-project",
            ...     base_directory=".yolo/memory",
            ... )
            >>> store = factory.create_decision_store()
            >>> # Collection name: "decisions_my-project"
        """
        from yolo_developer.memory.decision_store import ChromaDecisionStore

        return ChromaDecisionStore(
            persist_directory=self._base_directory,
            project_id=self._project_id,
        )

    def get_all_stores(self) -> StoreDict:
        """Create all stores at once with consistent project isolation.

        Convenience method that creates all four store types with the
        same project_id, ensuring consistent isolation.

        Returns:
            StoreDict with keys: "vector", "graph", "pattern", "decision".
            Each value is the corresponding store instance.

        Example:
            >>> factory = MemoryFactory(
            ...     project_id="my-project",
            ...     base_directory=".yolo/memory",
            ... )
            >>> stores = factory.get_all_stores()
            >>> vector_store = stores["vector"]
            >>> graph_store = stores["graph"]
        """
        return {
            "vector": self.create_vector_store(),
            "graph": self.create_graph_store(),
            "pattern": self.create_pattern_store(),
            "decision": self.create_decision_store(),
        }
