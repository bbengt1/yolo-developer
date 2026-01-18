"""Tests for TraceabilityService (Story 11.2).

Tests cover:
- trace_requirement creates artifact
- trace_story creates artifact and link
- trace_design creates artifact and link
- trace_code creates artifact and link
- trace_test creates artifact and link
- get_requirement_for_code navigates upstream
- get_code_for_requirement navigates downstream
- get_coverage_report returns statistics
- get_traceability_service factory function
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.traceability_memory_store import InMemoryTraceabilityStore


class TestTraceRequirement:
    """Tests for trace_requirement method."""

    @pytest.mark.asyncio
    async def test_trace_requirement_creates_artifact(self) -> None:
        """Test that trace_requirement creates a requirement artifact."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        req_id = await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )

        assert req_id == "FR82"

        # Verify artifact was stored
        artifact = await store.get_artifact("FR82")
        assert artifact is not None
        assert artifact.artifact_type == "requirement"
        assert artifact.name == "Requirement Traceability"

    @pytest.mark.asyncio
    async def test_trace_requirement_returns_id(self) -> None:
        """Test that trace_requirement returns the requirement ID."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        req_id = await service.trace_requirement(
            requirement_id="FR81",
            name="Decision Logging",
            description="System can log decisions",
        )

        assert req_id == "FR81"


class TestTraceStory:
    """Tests for trace_story method."""

    @pytest.mark.asyncio
    async def test_trace_story_creates_artifact(self) -> None:
        """Test that trace_story creates a story artifact."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # First create requirement
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )

        # Then create story
        story_id = await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )

        assert story_id == "11-2-requirement-traceability"

        # Verify artifact was stored
        artifact = await store.get_artifact("11-2-requirement-traceability")
        assert artifact is not None
        assert artifact.artifact_type == "story"

    @pytest.mark.asyncio
    async def test_trace_story_creates_link_to_requirement(self) -> None:
        """Test that trace_story creates a link to the requirement."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )

        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )

        # Verify link was created
        links = await store.get_links_from("11-2-requirement-traceability")
        assert len(links) == 1
        assert links[0].target_id == "FR82"
        assert links[0].link_type == "derives_from"


class TestTraceDesign:
    """Tests for trace_design method."""

    @pytest.mark.asyncio
    async def test_trace_design_creates_artifact(self) -> None:
        """Test that trace_design creates a design_decision artifact."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create requirement and story first
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )

        # Create design
        design_id = await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )

        assert design_id == "design-001"

        # Verify artifact was stored
        artifact = await store.get_artifact("design-001")
        assert artifact is not None
        assert artifact.artifact_type == "design_decision"

    @pytest.mark.asyncio
    async def test_trace_design_creates_link_to_story(self) -> None:
        """Test that trace_design creates a link to the story."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )

        # Verify link was created
        links = await store.get_links_from("design-001")
        assert len(links) == 1
        assert links[0].target_id == "11-2-requirement-traceability"
        assert links[0].link_type == "derives_from"


class TestTraceCode:
    """Tests for trace_code method."""

    @pytest.mark.asyncio
    async def test_trace_code_creates_artifact(self) -> None:
        """Test that trace_code creates a code artifact."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create full chain
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )

        # Create code
        code_id = await service.trace_code(
            code_id="src/yolo_developer/audit/traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )

        assert code_id == "src/yolo_developer/audit/traceability.py"

        # Verify artifact was stored
        artifact = await store.get_artifact(code_id)
        assert artifact is not None
        assert artifact.artifact_type == "code"

    @pytest.mark.asyncio
    async def test_trace_code_creates_link_to_design(self) -> None:
        """Test that trace_code creates a link to the design."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )
        await service.trace_code(
            code_id="src/yolo_developer/audit/traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )

        # Verify link was created
        links = await store.get_links_from("src/yolo_developer/audit/traceability.py")
        assert len(links) == 1
        assert links[0].target_id == "design-001"
        assert links[0].link_type == "implements"


class TestTraceTest:
    """Tests for trace_test method."""

    @pytest.mark.asyncio
    async def test_trace_test_creates_artifact(self) -> None:
        """Test that trace_test creates a test artifact."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create full chain
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )
        await service.trace_code(
            code_id="src/yolo_developer/audit/traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )

        # Create test
        test_id = await service.trace_test(
            test_id="tests/unit/audit/test_traceability_service.py",
            code_id="src/yolo_developer/audit/traceability.py",
            name="TraceabilityService tests",
            description="Unit tests for TraceabilityService",
        )

        assert test_id == "tests/unit/audit/test_traceability_service.py"

        # Verify artifact was stored
        artifact = await store.get_artifact(test_id)
        assert artifact is not None
        assert artifact.artifact_type == "test"

    @pytest.mark.asyncio
    async def test_trace_test_creates_link_to_code(self) -> None:
        """Test that trace_test creates a link to the code."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )
        await service.trace_code(
            code_id="src/yolo_developer/audit/traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )
        await service.trace_test(
            test_id="tests/unit/audit/test_traceability_service.py",
            code_id="src/yolo_developer/audit/traceability.py",
            name="TraceabilityService tests",
            description="Unit tests for TraceabilityService",
        )

        # Verify link was created
        links = await store.get_links_from("tests/unit/audit/test_traceability_service.py")
        assert len(links) == 1
        assert links[0].target_id == "src/yolo_developer/audit/traceability.py"
        assert links[0].link_type == "tests"


class TestGetRequirementForCode:
    """Tests for get_requirement_for_code method."""

    @pytest.mark.asyncio
    async def test_get_requirement_for_code_navigates_upstream(self) -> None:
        """Test that get_requirement_for_code navigates the full chain upstream."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create full chain
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )
        await service.trace_code(
            code_id="src/yolo_developer/audit/traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )

        # Navigate upstream from code to requirement
        requirement = await service.get_requirement_for_code(
            "src/yolo_developer/audit/traceability.py"
        )

        assert requirement is not None
        assert requirement.id == "FR82"
        assert requirement.artifact_type == "requirement"

    @pytest.mark.asyncio
    async def test_get_requirement_for_code_returns_none_for_unlinked(self) -> None:
        """Test that get_requirement_for_code returns None for unlinked code."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create code without chain
        await service.trace_code(
            code_id="orphan-code.py",
            design_id="nonexistent",
            name="Orphan Code",
            description="Code without traceability",
        )

        requirement = await service.get_requirement_for_code("orphan-code.py")
        assert requirement is None

    @pytest.mark.asyncio
    async def test_get_requirement_for_code_partial_chain(self) -> None:
        """Test get_requirement_for_code with partial chain (code → design, no story).

        Verifies that partial chain navigation returns None when the chain
        does not reach a requirement, even if intermediate links exist.
        """
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create a partial chain: code → design (no story or requirement)
        await service.trace_design(
            design_id="orphan-design",
            story_id="nonexistent-story",
            name="Orphan Design",
            description="Design without upstream story",
        )
        await service.trace_code(
            code_id="partial-code.py",
            design_id="orphan-design",
            name="Partial Code",
            description="Code with partial traceability",
        )

        # Should return None since chain doesn't reach a requirement
        requirement = await service.get_requirement_for_code("partial-code.py")
        assert requirement is None


class TestGetCodeForRequirement:
    """Tests for get_code_for_requirement method."""

    @pytest.mark.asyncio
    async def test_get_code_for_requirement_navigates_downstream(self) -> None:
        """Test that get_code_for_requirement navigates the full chain downstream."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create full chain
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )
        await service.trace_code(
            code_id="src/yolo_developer/audit/traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )

        # Navigate downstream from requirement to code
        code_artifacts = await service.get_code_for_requirement("FR82")

        assert len(code_artifacts) == 1
        assert code_artifacts[0].id == "src/yolo_developer/audit/traceability.py"
        assert code_artifacts[0].artifact_type == "code"

    @pytest.mark.asyncio
    async def test_get_code_for_requirement_returns_multiple_code_artifacts(
        self,
    ) -> None:
        """Test that get_code_for_requirement returns all code artifacts."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create chain with multiple code files
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )

        # Create multiple code files
        await service.trace_code(
            code_id="traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )
        await service.trace_code(
            code_id="traceability_types.py",
            design_id="design-001",
            name="Traceability Types",
            description="Type definitions for traceability",
        )

        # Navigate downstream from requirement
        code_artifacts = await service.get_code_for_requirement("FR82")

        assert len(code_artifacts) == 2
        code_ids = [a.id for a in code_artifacts]
        assert "traceability.py" in code_ids
        assert "traceability_types.py" in code_ids

    @pytest.mark.asyncio
    async def test_get_code_for_requirement_returns_empty_for_unimplemented(
        self,
    ) -> None:
        """Test that get_code_for_requirement returns empty for unimplemented requirement."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create requirement without implementation
        await service.trace_requirement(
            requirement_id="FR83",
            name="Future Requirement",
            description="Not implemented yet",
        )

        code_artifacts = await service.get_code_for_requirement("FR83")
        assert code_artifacts == []


class TestGetCoverageReport:
    """Tests for get_coverage_report method."""

    @pytest.mark.asyncio
    async def test_get_coverage_report_returns_statistics(self) -> None:
        """Test that get_coverage_report returns coverage statistics."""
        from yolo_developer.audit.traceability import TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)

        # Create full chain for one requirement
        await service.trace_requirement(
            requirement_id="FR82",
            name="Requirement Traceability",
            description="System can trace code to requirements",
        )
        await service.trace_story(
            story_id="11-2-requirement-traceability",
            requirement_id="FR82",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
        )
        await service.trace_design(
            design_id="design-001",
            story_id="11-2-requirement-traceability",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
        )
        await service.trace_code(
            code_id="traceability.py",
            design_id="design-001",
            name="TraceabilityService",
            description="High-level traceability service",
        )

        # Create unimplemented requirement
        await service.trace_requirement(
            requirement_id="FR83",
            name="Future Requirement",
            description="Not implemented yet",
        )

        report = await service.get_coverage_report()

        assert isinstance(report, dict)
        assert report["total_requirements"] == 2
        assert report["covered_requirements"] == 1
        assert report["coverage_percentage"] == 50.0
        assert "FR83" in report["uncovered_requirements"]


class TestGetTraceabilityServiceFactory:
    """Tests for get_traceability_service factory function."""

    def test_get_traceability_service_creates_service(self) -> None:
        """Test that get_traceability_service creates a TraceabilityService."""
        from yolo_developer.audit.traceability import get_traceability_service

        store = InMemoryTraceabilityStore()
        service = get_traceability_service(store)

        assert service is not None
        assert hasattr(service, "trace_requirement")
        assert hasattr(service, "trace_story")

    def test_get_traceability_service_with_default_store(self) -> None:
        """Test that get_traceability_service creates default store when not provided."""
        from yolo_developer.audit.traceability import get_traceability_service

        service = get_traceability_service()

        assert service is not None
        assert hasattr(service, "trace_requirement")
