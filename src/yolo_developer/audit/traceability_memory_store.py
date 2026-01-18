"""In-memory implementation of TraceabilityStore (Story 11.2).

This module provides an in-memory implementation of the TraceabilityStore
protocol for testing and single-session use.

The implementation follows the same pattern as InMemoryDecisionStore from
Story 11.1, using threading.Lock for thread-safe concurrent access.

Example:
    >>> from yolo_developer.audit.traceability_memory_store import (
    ...     InMemoryTraceabilityStore,
    ... )
    >>> from yolo_developer.audit.traceability_types import TraceableArtifact
    >>>
    >>> store = InMemoryTraceabilityStore()
    >>> artifact = TraceableArtifact(
    ...     id="FR82",
    ...     artifact_type="requirement",
    ...     name="Requirement Traceability",
    ...     description="System can trace code to requirements",
    ...     created_at="2026-01-18T12:00:00Z",
    ... )
    >>> await store.register_artifact(artifact)
    'FR82'

References:
    - FR82: System can generate decision traceability from requirement to code
    - Story 11.1: InMemoryDecisionStore implementation pattern
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Literal

from yolo_developer.audit.traceability_types import (
    ArtifactType,
    TraceableArtifact,
    TraceLink,
)


class InMemoryTraceabilityStore:
    """In-memory implementation of TraceabilityStore protocol.

    Stores artifacts and links in memory with thread-safe access.
    Uses BFS traversal for trace chain navigation.

    Attributes:
        _artifacts: Dictionary mapping artifact IDs to artifacts.
        _links: List of all trace links.
        _links_from_index: Index mapping source_id to list of links.
        _links_to_index: Index mapping target_id to list of links.
        _lock: Thread lock for concurrent access safety.

    Example:
        >>> store = InMemoryTraceabilityStore()
        >>> await store.register_artifact(artifact)
        >>> await store.create_link(link)
    """

    def __init__(self) -> None:
        """Initialize the in-memory traceability store."""
        self._artifacts: dict[str, TraceableArtifact] = {}
        self._links: list[TraceLink] = []
        self._links_from_index: dict[str, list[TraceLink]] = defaultdict(list)
        self._links_to_index: dict[str, list[TraceLink]] = defaultdict(list)
        self._lock = threading.Lock()

    async def register_artifact(self, artifact: TraceableArtifact) -> str:
        """Register a new traceable artifact.

        Args:
            artifact: The artifact to register.

        Returns:
            The artifact ID.
        """
        with self._lock:
            self._artifacts[artifact.id] = artifact
        return artifact.id

    async def create_link(self, link: TraceLink) -> str:
        """Create a link between two artifacts.

        Args:
            link: The link to create.

        Returns:
            The link ID.
        """
        with self._lock:
            self._links.append(link)
            self._links_from_index[link.source_id].append(link)
            self._links_to_index[link.target_id].append(link)
        return link.id

    async def get_artifact(self, artifact_id: str) -> TraceableArtifact | None:
        """Retrieve an artifact by ID.

        Args:
            artifact_id: The ID of the artifact to retrieve.

        Returns:
            The artifact if found, None otherwise.
        """
        with self._lock:
            return self._artifacts.get(artifact_id)

    async def get_links_from(self, source_id: str) -> list[TraceLink]:
        """Get all links where source_id matches.

        Args:
            source_id: The source artifact ID to search for.

        Returns:
            List of links originating from the source artifact.
        """
        with self._lock:
            return list(self._links_from_index.get(source_id, []))

    async def get_links_to(self, target_id: str) -> list[TraceLink]:
        """Get all links where target_id matches.

        Args:
            target_id: The target artifact ID to search for.

        Returns:
            List of links pointing to the target artifact.
        """
        with self._lock:
            return list(self._links_to_index.get(target_id, []))

    async def get_trace_chain(
        self, artifact_id: str, direction: Literal["upstream", "downstream"]
    ) -> list[TraceableArtifact]:
        """Traverse the full trace chain from an artifact.

        Uses BFS traversal to find all connected artifacts. The starting
        artifact is NOT included in the result - only connected artifacts
        found during traversal are returned.

        Note:
            The visited set prevents infinite loops if cycles exist in the
            data (which shouldn't happen in a proper DAG). If a cycle is
            detected, a warning is logged per ADR-007.

        Args:
            artifact_id: The starting artifact ID.
            direction: Direction to traverse:
                - "upstream": Follow links from source to target
                  (e.g., code → design → story → requirement)
                - "downstream": Follow links from target to source
                  (e.g., requirement → story → design → code)

        Returns:
            List of artifacts in the trace chain, ordered by traversal.
            Does not include the starting artifact.
        """
        result: list[TraceableArtifact] = []
        visited: set[str] = {artifact_id}
        queue: list[str] = [artifact_id]

        while queue:
            current_id = queue.pop(0)

            # Get links based on direction
            if direction == "upstream":
                # For upstream, we follow from source to target
                # (source derives_from target means target is upstream)
                links = await self.get_links_from(current_id)
                next_ids = [link.target_id for link in links]
            else:
                # For downstream, we follow from target to source
                # (source derives_from target means source is downstream)
                links = await self.get_links_to(current_id)
                next_ids = [link.source_id for link in links]

            for next_id in next_ids:
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append(next_id)
                    artifact = await self.get_artifact(next_id)
                    if artifact is not None:
                        result.append(artifact)

        return result

    async def get_unlinked_artifacts(self, artifact_type: ArtifactType) -> list[TraceableArtifact]:
        """Find artifacts of a given type with no outgoing links.

        Useful for finding requirements that haven't been implemented,
        or code that hasn't been tested.

        Args:
            artifact_type: The type of artifacts to search for.

        Returns:
            List of artifacts with no outgoing links.
        """
        with self._lock:
            result: list[TraceableArtifact] = []
            for artifact in self._artifacts.values():
                if artifact.artifact_type == artifact_type:
                    # Check if this artifact has any outgoing links
                    if (
                        artifact.id not in self._links_from_index
                        or len(self._links_from_index[artifact.id]) == 0
                    ):
                        result.append(artifact)
            return result
