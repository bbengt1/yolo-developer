"""TraceabilityStore protocol definition (Story 11.2).

This module defines the protocol for traceability stores, allowing
different implementations (in-memory, file-based, database, etc.).

The protocol follows the same pattern as DecisionStore from Story 11.1.

Example:
    >>> from yolo_developer.audit.traceability_store import TraceabilityStore
    >>>
    >>> class MyStore(TraceabilityStore):
    ...     async def register_artifact(self, artifact: TraceableArtifact) -> str:
    ...         # Implementation
    ...         return artifact.id
    ...     # ... other methods

References:
    - FR82: System can generate decision traceability from requirement to code
    - Story 11.1: DecisionStore protocol pattern
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from yolo_developer.audit.traceability_types import (
        ArtifactType,
        TraceableArtifact,
        TraceLink,
    )


@runtime_checkable
class TraceabilityStore(Protocol):
    """Protocol for traceability stores.

    Defines the interface for storing and querying traceable artifacts
    and the links between them. Implementations may use different
    storage backends (in-memory, file, database).

    Methods:
        register_artifact: Register a new traceable artifact
        create_link: Create a link between two artifacts
        get_artifact: Retrieve an artifact by ID
        get_links_from: Get all links where source_id matches
        get_links_to: Get all links where target_id matches
        get_trace_chain: Traverse the full trace chain
        get_unlinked_artifacts: Find artifacts with no outgoing links
    """

    async def register_artifact(self, artifact: TraceableArtifact) -> str:
        """Register a new traceable artifact.

        Args:
            artifact: The artifact to register.

        Returns:
            The artifact ID.
        """
        ...

    async def create_link(self, link: TraceLink) -> str:
        """Create a link between two artifacts.

        Args:
            link: The link to create.

        Returns:
            The link ID.
        """
        ...

    async def get_artifact(self, artifact_id: str) -> TraceableArtifact | None:
        """Retrieve an artifact by ID.

        Args:
            artifact_id: The ID of the artifact to retrieve.

        Returns:
            The artifact if found, None otherwise.
        """
        ...

    async def get_links_from(self, source_id: str) -> list[TraceLink]:
        """Get all links where source_id matches.

        Args:
            source_id: The source artifact ID to search for.

        Returns:
            List of links originating from the source artifact.
        """
        ...

    async def get_links_to(self, target_id: str) -> list[TraceLink]:
        """Get all links where target_id matches.

        Args:
            target_id: The target artifact ID to search for.

        Returns:
            List of links pointing to the target artifact.
        """
        ...

    async def get_trace_chain(
        self, artifact_id: str, direction: Literal["upstream", "downstream"]
    ) -> list[TraceableArtifact]:
        """Traverse the full trace chain from an artifact.

        Args:
            artifact_id: The starting artifact ID.
            direction: Direction to traverse:
                - "upstream": Follow links to find source artifacts
                  (e.g., code → design → story → requirement)
                - "downstream": Follow links to find target artifacts
                  (e.g., requirement → story → design → code)

        Returns:
            List of artifacts in the trace chain, ordered by traversal.
        """
        ...

    async def get_unlinked_artifacts(self, artifact_type: ArtifactType) -> list[TraceableArtifact]:
        """Find artifacts of a given type with no outgoing links.

        Useful for finding requirements that haven't been implemented,
        or code that hasn't been tested.

        Args:
            artifact_type: The type of artifacts to search for.

        Returns:
            List of artifacts with no outgoing links.
        """
        ...
