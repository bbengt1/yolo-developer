"""TraceabilityService for high-level traceability operations (Story 11.2).

This module provides a high-level service for creating and querying
traceability chains between requirements, stories, designs, code, and tests.

Example:
    >>> from yolo_developer.audit.traceability import (
    ...     TraceabilityService,
    ...     get_traceability_service,
    ... )
    >>>
    >>> service = get_traceability_service()
    >>> await service.trace_requirement("FR82", "Traceability", "Description")
    >>> await service.trace_story("story-1", "FR82", "Story Name", "Description")
    >>> requirement = await service.get_requirement_for_code("code-id")

References:
    - FR82: System can generate decision traceability from requirement to code
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from yolo_developer.audit.traceability_memory_store import InMemoryTraceabilityStore
from yolo_developer.audit.traceability_store import TraceabilityStore
from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink

logger = structlog.get_logger(__name__)


class TraceabilityService:
    """High-level service for traceability operations.

    Provides convenient methods for creating and navigating the
    traceability chain:

    Requirement → Story → Design Decision → Code → Test

    Attributes:
        _store: The underlying traceability store.

    Example:
        >>> service = TraceabilityService(InMemoryTraceabilityStore())
        >>> await service.trace_requirement("FR82", "Name", "Desc")
        >>> await service.trace_story("story-1", "FR82", "Name", "Desc")
    """

    def __init__(self, store: TraceabilityStore) -> None:
        """Initialize the traceability service.

        Args:
            store: The traceability store to use for persistence.
        """
        self._store = store

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()

    def _generate_link_id(self) -> str:
        """Generate a unique link ID."""
        return f"link-{uuid.uuid4()}"

    async def trace_requirement(
        self,
        requirement_id: str,
        name: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a requirement artifact.

        Args:
            requirement_id: Unique identifier for the requirement (e.g., "FR82").
            name: Human-readable name of the requirement.
            description: Detailed description of the requirement.
            metadata: Optional additional metadata.

        Returns:
            The requirement ID.
        """
        artifact = TraceableArtifact(
            id=requirement_id,
            artifact_type="requirement",
            name=name,
            description=description,
            created_at=self._get_timestamp(),
            metadata=metadata or {},
        )

        await self._store.register_artifact(artifact)

        logger.info(
            "traced_requirement",
            requirement_id=requirement_id,
            name=name,
        )

        return requirement_id

    async def trace_story(
        self,
        story_id: str,
        requirement_id: str,
        name: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a story artifact and link it to a requirement.

        Args:
            story_id: Unique identifier for the story.
            requirement_id: ID of the requirement this story derives from.
            name: Human-readable name of the story.
            description: Detailed description of the story.
            metadata: Optional additional metadata.

        Returns:
            The story ID.
        """
        artifact = TraceableArtifact(
            id=story_id,
            artifact_type="story",
            name=name,
            description=description,
            created_at=self._get_timestamp(),
            metadata=metadata or {},
        )

        await self._store.register_artifact(artifact)

        # Create link: story derives_from requirement
        link = TraceLink(
            id=self._generate_link_id(),
            source_id=story_id,
            source_type="story",
            target_id=requirement_id,
            target_type="requirement",
            link_type="derives_from",
            created_at=self._get_timestamp(),
        )
        await self._store.create_link(link)

        logger.info(
            "traced_story",
            story_id=story_id,
            requirement_id=requirement_id,
            name=name,
        )

        return story_id

    async def trace_design(
        self,
        design_id: str,
        story_id: str,
        name: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a design decision artifact and link it to a story.

        Args:
            design_id: Unique identifier for the design decision.
            story_id: ID of the story this design derives from.
            name: Human-readable name of the design decision.
            description: Detailed description of the design decision.
            metadata: Optional additional metadata.

        Returns:
            The design ID.
        """
        artifact = TraceableArtifact(
            id=design_id,
            artifact_type="design_decision",
            name=name,
            description=description,
            created_at=self._get_timestamp(),
            metadata=metadata or {},
        )

        await self._store.register_artifact(artifact)

        # Create link: design derives_from story
        link = TraceLink(
            id=self._generate_link_id(),
            source_id=design_id,
            source_type="design_decision",
            target_id=story_id,
            target_type="story",
            link_type="derives_from",
            created_at=self._get_timestamp(),
        )
        await self._store.create_link(link)

        logger.info(
            "traced_design",
            design_id=design_id,
            story_id=story_id,
            name=name,
        )

        return design_id

    async def trace_code(
        self,
        code_id: str,
        design_id: str,
        name: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a code artifact and link it to a design decision.

        Args:
            code_id: Unique identifier for the code (typically file path).
            design_id: ID of the design decision this code implements.
            name: Human-readable name of the code artifact.
            description: Detailed description of the code.
            metadata: Optional additional metadata.

        Returns:
            The code ID.
        """
        artifact = TraceableArtifact(
            id=code_id,
            artifact_type="code",
            name=name,
            description=description,
            created_at=self._get_timestamp(),
            metadata=metadata or {},
        )

        await self._store.register_artifact(artifact)

        # Create link: code implements design
        link = TraceLink(
            id=self._generate_link_id(),
            source_id=code_id,
            source_type="code",
            target_id=design_id,
            target_type="design_decision",
            link_type="implements",
            created_at=self._get_timestamp(),
        )
        await self._store.create_link(link)

        logger.info(
            "traced_code",
            code_id=code_id,
            design_id=design_id,
            name=name,
        )

        return code_id

    async def trace_test(
        self,
        test_id: str,
        code_id: str,
        name: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a test artifact and link it to code.

        Args:
            test_id: Unique identifier for the test (typically file path).
            code_id: ID of the code this test validates.
            name: Human-readable name of the test artifact.
            description: Detailed description of the test.
            metadata: Optional additional metadata.

        Returns:
            The test ID.
        """
        artifact = TraceableArtifact(
            id=test_id,
            artifact_type="test",
            name=name,
            description=description,
            created_at=self._get_timestamp(),
            metadata=metadata or {},
        )

        await self._store.register_artifact(artifact)

        # Create link: test tests code
        link = TraceLink(
            id=self._generate_link_id(),
            source_id=test_id,
            source_type="test",
            target_id=code_id,
            target_type="code",
            link_type="tests",
            created_at=self._get_timestamp(),
        )
        await self._store.create_link(link)

        logger.info(
            "traced_test",
            test_id=test_id,
            code_id=code_id,
            name=name,
        )

        return test_id

    async def get_requirement_for_code(self, code_id: str) -> TraceableArtifact | None:
        """Navigate upstream from code to find the originating requirement.

        Args:
            code_id: The code artifact ID to start from.

        Returns:
            The requirement artifact if found, None otherwise.
        """
        # Get the upstream chain from code
        chain = await self._store.get_trace_chain(code_id, "upstream")

        # Find the first requirement in the chain
        for artifact in chain:
            if artifact.artifact_type == "requirement":
                return artifact

        return None

    async def get_code_for_requirement(self, requirement_id: str) -> list[TraceableArtifact]:
        """Navigate downstream from requirement to find all code artifacts.

        Args:
            requirement_id: The requirement artifact ID to start from.

        Returns:
            List of code artifacts implementing the requirement.
        """
        # Get the downstream chain from requirement
        chain = await self._store.get_trace_chain(requirement_id, "downstream")

        # Filter to only code artifacts
        return [a for a in chain if a.artifact_type == "code"]

    async def get_coverage_report(self) -> dict[str, Any]:
        """Generate a coverage report showing requirement implementation status.

        Returns:
            Dictionary with coverage statistics:
                - total_requirements: Total number of requirements
                - covered_requirements: Requirements with downstream code
                - coverage_percentage: Percentage of requirements covered
                - uncovered_requirements: List of requirement IDs without code
        """
        # Get all requirements
        # Note: This requires getting all artifacts, which we can do by checking
        # for unlinked + linked requirements
        all_requirements: list[TraceableArtifact] = []
        uncovered_requirements: list[str] = []

        # First, get unlinked requirements (those with no outgoing links)
        unlinked = await self._store.get_unlinked_artifacts("requirement")
        all_requirements.extend(unlinked)
        uncovered_requirements.extend([r.id for r in unlinked])

        # For requirements that DO have outgoing links, we need to check
        # if they have downstream code
        # This is a limitation of the current protocol - we'll work with
        # what we have for now

        # Actually, requirements shouldn't have outgoing links in our model
        # (stories link TO requirements, not the other way around)
        # So unlinked requirements = all requirements

        # Check which requirements have code downstream
        covered_count = 0
        for req in list(all_requirements):  # Copy to allow modification
            code_artifacts = await self.get_code_for_requirement(req.id)
            if code_artifacts:
                covered_count += 1
                if req.id in uncovered_requirements:
                    uncovered_requirements.remove(req.id)

        total = len(all_requirements)
        coverage_pct = (covered_count / total * 100) if total > 0 else 0.0

        return {
            "total_requirements": total,
            "covered_requirements": covered_count,
            "coverage_percentage": coverage_pct,
            "uncovered_requirements": uncovered_requirements,
        }


def get_traceability_service(
    store: TraceabilityStore | None = None,
) -> TraceabilityService:
    """Factory function to create a TraceabilityService.

    Args:
        store: Optional traceability store. If not provided, creates
            an InMemoryTraceabilityStore.

    Returns:
        A configured TraceabilityService instance.
    """
    if store is None:
        store = InMemoryTraceabilityStore()

    return TraceabilityService(store)
