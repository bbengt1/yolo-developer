"""Tests for InMemoryTraceabilityStore implementation (Story 11.2).

Tests cover:
- register_artifact returns valid ID
- create_link returns valid ID
- get_artifact retrieves correct artifact
- get_links_from retrieves correct links
- get_links_to retrieves correct links
- get_trace_chain upstream navigation
- get_trace_chain downstream navigation
- get_unlinked_artifacts finds orphans
- Concurrent access safety
"""

from __future__ import annotations

import asyncio

import pytest

from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink


@pytest.fixture
def sample_requirement() -> TraceableArtifact:
    """Create a sample requirement artifact."""
    return TraceableArtifact(
        id="FR82",
        artifact_type="requirement",
        name="Requirement Traceability",
        description="System can trace code to requirements",
        created_at="2026-01-18T12:00:00Z",
    )


@pytest.fixture
def sample_story() -> TraceableArtifact:
    """Create a sample story artifact."""
    return TraceableArtifact(
        id="11-2-requirement-traceability",
        artifact_type="story",
        name="Story 11.2: Requirement Traceability",
        description="Implement requirement traceability",
        created_at="2026-01-18T12:00:00Z",
    )


@pytest.fixture
def sample_design() -> TraceableArtifact:
    """Create a sample design_decision artifact."""
    return TraceableArtifact(
        id="design-001",
        artifact_type="design_decision",
        name="DAG-based traceability",
        description="Use directed acyclic graph for trace links",
        created_at="2026-01-18T12:00:00Z",
    )


@pytest.fixture
def sample_code() -> TraceableArtifact:
    """Create a sample code artifact."""
    return TraceableArtifact(
        id="src/yolo_developer/audit/traceability.py",
        artifact_type="code",
        name="TraceabilityService",
        description="High-level traceability service",
        created_at="2026-01-18T12:00:00Z",
    )


@pytest.fixture
def sample_test() -> TraceableArtifact:
    """Create a sample test artifact."""
    return TraceableArtifact(
        id="tests/unit/audit/test_traceability.py",
        artifact_type="test",
        name="TraceabilityService tests",
        description="Unit tests for TraceabilityService",
        created_at="2026-01-18T12:00:00Z",
    )


class TestInMemoryTraceabilityStoreBasics:
    """Tests for basic InMemoryTraceabilityStore operations."""

    @pytest.mark.asyncio
    async def test_register_artifact_returns_artifact_id(
        self, sample_requirement: TraceableArtifact
    ) -> None:
        """Test that register_artifact returns the artifact ID."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        artifact_id = await store.register_artifact(sample_requirement)

        assert artifact_id == "FR82"

    @pytest.mark.asyncio
    async def test_register_artifact_stores_artifact(
        self, sample_requirement: TraceableArtifact
    ) -> None:
        """Test that register_artifact stores the artifact for later retrieval."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)

        retrieved = await store.get_artifact("FR82")
        assert retrieved is not None
        assert retrieved.id == "FR82"
        assert retrieved.name == "Requirement Traceability"

    @pytest.mark.asyncio
    async def test_get_artifact_returns_none_for_missing(self) -> None:
        """Test that get_artifact returns None for missing artifact."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        retrieved = await store.get_artifact("nonexistent")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_create_link_returns_link_id(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
    ) -> None:
        """Test that create_link returns the link ID."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)

        link = TraceLink(
            id="link-001",
            source_id="11-2-requirement-traceability",
            source_type="story",
            target_id="FR82",
            target_type="requirement",
            link_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        link_id = await store.create_link(link)
        assert link_id == "link-001"


class TestGetLinksFrom:
    """Tests for get_links_from method."""

    @pytest.mark.asyncio
    async def test_get_links_from_returns_matching_links(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
    ) -> None:
        """Test that get_links_from returns links with matching source_id."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)

        link = TraceLink(
            id="link-001",
            source_id="11-2-requirement-traceability",
            source_type="story",
            target_id="FR82",
            target_type="requirement",
            link_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )
        await store.create_link(link)

        links = await store.get_links_from("11-2-requirement-traceability")

        assert len(links) == 1
        assert links[0].id == "link-001"
        assert links[0].target_id == "FR82"

    @pytest.mark.asyncio
    async def test_get_links_from_returns_empty_for_no_matches(self) -> None:
        """Test that get_links_from returns empty list for no matches."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        links = await store.get_links_from("nonexistent")

        assert links == []


class TestGetLinksTo:
    """Tests for get_links_to method."""

    @pytest.mark.asyncio
    async def test_get_links_to_returns_matching_links(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
    ) -> None:
        """Test that get_links_to returns links with matching target_id."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)

        link = TraceLink(
            id="link-001",
            source_id="11-2-requirement-traceability",
            source_type="story",
            target_id="FR82",
            target_type="requirement",
            link_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )
        await store.create_link(link)

        links = await store.get_links_to("FR82")

        assert len(links) == 1
        assert links[0].id == "link-001"
        assert links[0].source_id == "11-2-requirement-traceability"

    @pytest.mark.asyncio
    async def test_get_links_to_returns_empty_for_no_matches(self) -> None:
        """Test that get_links_to returns empty list for no matches."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        links = await store.get_links_to("nonexistent")

        assert links == []


class TestGetTraceChain:
    """Tests for get_trace_chain method."""

    @pytest.mark.asyncio
    async def test_get_trace_chain_upstream_navigation(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
        sample_design: TraceableArtifact,
        sample_code: TraceableArtifact,
    ) -> None:
        """Test upstream navigation: code → design → story → requirement."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        # Register all artifacts
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)
        await store.register_artifact(sample_design)
        await store.register_artifact(sample_code)

        # Create links (source derives_from/implements target)
        # story → requirement
        await store.create_link(
            TraceLink(
                id="link-1",
                source_id="11-2-requirement-traceability",
                source_type="story",
                target_id="FR82",
                target_type="requirement",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )
        )
        # design → story
        await store.create_link(
            TraceLink(
                id="link-2",
                source_id="design-001",
                source_type="design_decision",
                target_id="11-2-requirement-traceability",
                target_type="story",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )
        )
        # code → design
        await store.create_link(
            TraceLink(
                id="link-3",
                source_id="src/yolo_developer/audit/traceability.py",
                source_type="code",
                target_id="design-001",
                target_type="design_decision",
                link_type="implements",
                created_at="2026-01-18T12:00:00Z",
            )
        )

        # Navigate upstream from code
        chain = await store.get_trace_chain("src/yolo_developer/audit/traceability.py", "upstream")

        # Should get: design → story → requirement
        assert len(chain) == 3
        artifact_ids = [a.id for a in chain]
        assert "design-001" in artifact_ids
        assert "11-2-requirement-traceability" in artifact_ids
        assert "FR82" in artifact_ids

    @pytest.mark.asyncio
    async def test_get_trace_chain_downstream_navigation(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
        sample_design: TraceableArtifact,
        sample_code: TraceableArtifact,
    ) -> None:
        """Test downstream navigation: requirement → story → design → code."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        # Register all artifacts
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)
        await store.register_artifact(sample_design)
        await store.register_artifact(sample_code)

        # Create links (source derives_from/implements target)
        await store.create_link(
            TraceLink(
                id="link-1",
                source_id="11-2-requirement-traceability",
                source_type="story",
                target_id="FR82",
                target_type="requirement",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )
        )
        await store.create_link(
            TraceLink(
                id="link-2",
                source_id="design-001",
                source_type="design_decision",
                target_id="11-2-requirement-traceability",
                target_type="story",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )
        )
        await store.create_link(
            TraceLink(
                id="link-3",
                source_id="src/yolo_developer/audit/traceability.py",
                source_type="code",
                target_id="design-001",
                target_type="design_decision",
                link_type="implements",
                created_at="2026-01-18T12:00:00Z",
            )
        )

        # Navigate downstream from requirement
        chain = await store.get_trace_chain("FR82", "downstream")

        # Should get: story → design → code
        assert len(chain) == 3
        artifact_ids = [a.id for a in chain]
        assert "11-2-requirement-traceability" in artifact_ids
        assert "design-001" in artifact_ids
        assert "src/yolo_developer/audit/traceability.py" in artifact_ids

    @pytest.mark.asyncio
    async def test_get_trace_chain_empty_for_unlinked(
        self, sample_requirement: TraceableArtifact
    ) -> None:
        """Test that get_trace_chain returns empty for unlinked artifact."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)

        chain = await store.get_trace_chain("FR82", "upstream")
        assert chain == []

        chain = await store.get_trace_chain("FR82", "downstream")
        assert chain == []


class TestGetUnlinkedArtifacts:
    """Tests for get_unlinked_artifacts method."""

    @pytest.mark.asyncio
    async def test_get_unlinked_artifacts_finds_orphans(
        self, sample_requirement: TraceableArtifact
    ) -> None:
        """Test that get_unlinked_artifacts finds artifacts with no outgoing links."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)

        # Create another requirement
        req2 = TraceableArtifact(
            id="FR83",
            artifact_type="requirement",
            name="Another Requirement",
            description="Something else",
            created_at="2026-01-18T12:00:00Z",
        )
        await store.register_artifact(req2)

        # Both requirements have no outgoing links
        unlinked = await store.get_unlinked_artifacts("requirement")

        assert len(unlinked) == 2
        unlinked_ids = [a.id for a in unlinked]
        assert "FR82" in unlinked_ids
        assert "FR83" in unlinked_ids

    @pytest.mark.asyncio
    async def test_get_unlinked_artifacts_excludes_linked(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
    ) -> None:
        """Test that get_unlinked_artifacts excludes artifacts with outgoing links."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)

        # Link story to requirement (story has outgoing link)
        await store.create_link(
            TraceLink(
                id="link-001",
                source_id="11-2-requirement-traceability",
                source_type="story",
                target_id="FR82",
                target_type="requirement",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )
        )

        # Story has outgoing link, should not appear
        unlinked_stories = await store.get_unlinked_artifacts("story")
        assert len(unlinked_stories) == 0

        # Requirement has no outgoing link, should appear
        unlinked_reqs = await store.get_unlinked_artifacts("requirement")
        assert len(unlinked_reqs) == 1
        assert unlinked_reqs[0].id == "FR82"

    @pytest.mark.asyncio
    async def test_get_unlinked_artifacts_returns_empty_for_no_matches(self) -> None:
        """Test that get_unlinked_artifacts returns empty for no artifacts of type."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        unlinked = await store.get_unlinked_artifacts("code")

        assert unlinked == []


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_register_artifacts(self) -> None:
        """Test that concurrent artifact registration is thread-safe."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        async def register_artifact(index: int) -> str:
            artifact = TraceableArtifact(
                id=f"req-{index}",
                artifact_type="requirement",
                name=f"Requirement {index}",
                description=f"Description {index}",
                created_at="2026-01-18T12:00:00Z",
            )
            return await store.register_artifact(artifact)

        # Register 100 artifacts concurrently
        tasks = [register_artifact(i) for i in range(100)]
        results = await asyncio.gather(*tasks)

        # All should succeed with unique IDs
        assert len(results) == 100
        assert len(set(results)) == 100  # All unique

        # All artifacts should be retrievable
        for i in range(100):
            artifact = await store.get_artifact(f"req-{i}")
            assert artifact is not None
            assert artifact.id == f"req-{i}"

    @pytest.mark.asyncio
    async def test_concurrent_create_links(
        self,
        sample_requirement: TraceableArtifact,
    ) -> None:
        """Test that concurrent link creation is thread-safe."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)

        # Register multiple story artifacts
        for i in range(50):
            story = TraceableArtifact(
                id=f"story-{i}",
                artifact_type="story",
                name=f"Story {i}",
                description=f"Description {i}",
                created_at="2026-01-18T12:00:00Z",
            )
            await store.register_artifact(story)

        async def create_link(index: int) -> str:
            link = TraceLink(
                id=f"link-{index}",
                source_id=f"story-{index}",
                source_type="story",
                target_id="FR82",
                target_type="requirement",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )
            return await store.create_link(link)

        # Create 50 links concurrently
        tasks = [create_link(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 50

        # All links should be retrievable via get_links_to
        links = await store.get_links_to("FR82")
        assert len(links) == 50


class TestGetArtifactsByType:
    """Tests for get_artifacts_by_type method (Story 11.7)."""

    @pytest.mark.asyncio
    async def test_get_artifacts_by_type_returns_matching(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
        sample_code: TraceableArtifact,
    ) -> None:
        """Test that get_artifacts_by_type returns only matching artifacts."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)
        await store.register_artifact(sample_code)

        requirements = await store.get_artifacts_by_type("requirement")

        assert len(requirements) == 1
        assert requirements[0].id == "FR82"

    @pytest.mark.asyncio
    async def test_get_artifacts_by_type_returns_multiple(self) -> None:
        """Test that get_artifacts_by_type returns all matching artifacts."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        # Register multiple requirements
        for i in range(3):
            artifact = TraceableArtifact(
                id=f"req-{i}",
                artifact_type="requirement",
                name=f"Requirement {i}",
                description=f"Description {i}",
                created_at="2026-01-18T12:00:00Z",
            )
            await store.register_artifact(artifact)

        requirements = await store.get_artifacts_by_type("requirement")

        assert len(requirements) == 3

    @pytest.mark.asyncio
    async def test_get_artifacts_by_type_returns_empty_for_no_matches(self) -> None:
        """Test that get_artifacts_by_type returns empty for no matching type."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        artifacts = await store.get_artifacts_by_type("code")

        assert artifacts == []

    @pytest.mark.asyncio
    async def test_get_artifacts_by_type_for_each_valid_type(self) -> None:
        """Test filtering for each valid artifact type."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        # Register one of each type
        valid_types = ["requirement", "story", "design_decision", "code", "test"]
        for artifact_type in valid_types:
            artifact = TraceableArtifact(
                id=f"art-{artifact_type}",
                artifact_type=artifact_type,
                name=f"Sample {artifact_type}",
                description=f"A {artifact_type} artifact",
                created_at="2026-01-18T12:00:00Z",
            )
            await store.register_artifact(artifact)

        # Verify each type can be filtered
        for artifact_type in valid_types:
            artifacts = await store.get_artifacts_by_type(artifact_type)
            assert len(artifacts) == 1
            assert artifacts[0].artifact_type == artifact_type


class TestGetArtifactsByFilters:
    """Tests for get_artifacts_by_filters method (Story 11.7)."""

    @pytest.mark.asyncio
    async def test_get_artifacts_by_filters_with_type_only(
        self,
        sample_requirement: TraceableArtifact,
        sample_story: TraceableArtifact,
    ) -> None:
        """Test filtering by artifact type only."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()
        await store.register_artifact(sample_requirement)
        await store.register_artifact(sample_story)

        results = await store.get_artifacts_by_filters(artifact_type="requirement")

        assert len(results) == 1
        assert results[0].id == "FR82"

    @pytest.mark.asyncio
    async def test_get_artifacts_by_filters_with_time_range(self) -> None:
        """Test filtering by time range."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        # Register artifacts with different timestamps
        for i, timestamp in enumerate(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-15T00:00:00Z",
                "2026-01-31T00:00:00Z",
            ]
        ):
            artifact = TraceableArtifact(
                id=f"req-{i}",
                artifact_type="requirement",
                name=f"Requirement {i}",
                description=f"Created at {timestamp}",
                created_at=timestamp,
            )
            await store.register_artifact(artifact)

        # Filter for middle of month
        results = await store.get_artifacts_by_filters(
            created_after="2026-01-10T00:00:00Z",
            created_before="2026-01-20T00:00:00Z",
        )

        assert len(results) == 1
        assert results[0].id == "req-1"

    @pytest.mark.asyncio
    async def test_get_artifacts_by_filters_combined(self) -> None:
        """Test filtering with multiple filters combined (AND logic)."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        # Register requirements and stories with various timestamps
        for i, (artifact_type, timestamp) in enumerate(
            [
                ("requirement", "2026-01-01T00:00:00Z"),
                ("requirement", "2026-01-15T00:00:00Z"),
                ("story", "2026-01-15T00:00:00Z"),
                ("requirement", "2026-01-31T00:00:00Z"),
            ]
        ):
            artifact = TraceableArtifact(
                id=f"art-{i}",
                artifact_type=artifact_type,
                name=f"Artifact {i}",
                description=f"Type {artifact_type} at {timestamp}",
                created_at=timestamp,
            )
            await store.register_artifact(artifact)

        # Filter for requirements in mid-January
        results = await store.get_artifacts_by_filters(
            artifact_type="requirement",
            created_after="2026-01-10T00:00:00Z",
            created_before="2026-01-20T00:00:00Z",
        )

        assert len(results) == 1
        assert results[0].id == "art-1"
        assert results[0].artifact_type == "requirement"

    @pytest.mark.asyncio
    async def test_get_artifacts_by_filters_no_filters_returns_all(self) -> None:
        """Test that no filters returns all artifacts."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        for i in range(3):
            artifact = TraceableArtifact(
                id=f"art-{i}",
                artifact_type="requirement",
                name=f"Artifact {i}",
                description=f"Description {i}",
                created_at="2026-01-18T12:00:00Z",
            )
            await store.register_artifact(artifact)

        results = await store.get_artifacts_by_filters()

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_artifacts_by_filters_empty_for_no_matches(self) -> None:
        """Test that non-matching filters return empty list."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        artifact = TraceableArtifact(
            id="req-1",
            artifact_type="requirement",
            name="Requirement 1",
            description="Description",
            created_at="2026-01-18T12:00:00Z",
        )
        await store.register_artifact(artifact)

        # Filter for future time range
        results = await store.get_artifacts_by_filters(
            created_after="2027-01-01T00:00:00Z",
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_get_artifacts_by_filters_created_after_inclusive(self) -> None:
        """Test that created_after is inclusive."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        artifact = TraceableArtifact(
            id="req-1",
            artifact_type="requirement",
            name="Requirement 1",
            description="Description",
            created_at="2026-01-18T12:00:00Z",
        )
        await store.register_artifact(artifact)

        # Filter with exact timestamp should include
        results = await store.get_artifacts_by_filters(
            created_after="2026-01-18T12:00:00Z",
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_artifacts_by_filters_created_before_inclusive(self) -> None:
        """Test that created_before is inclusive."""
        from yolo_developer.audit.traceability_memory_store import (
            InMemoryTraceabilityStore,
        )

        store = InMemoryTraceabilityStore()

        artifact = TraceableArtifact(
            id="req-1",
            artifact_type="requirement",
            name="Requirement 1",
            description="Description",
            created_at="2026-01-18T12:00:00Z",
        )
        await store.register_artifact(artifact)

        # Filter with exact timestamp should include
        results = await store.get_artifacts_by_filters(
            created_before="2026-01-18T12:00:00Z",
        )

        assert len(results) == 1
