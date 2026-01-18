"""Tests for traceability type definitions (Story 11.2).

Tests cover:
- ArtifactType and LinkType literal type validation
- TraceableArtifact frozen dataclass functionality
- TraceLink frozen dataclass functionality
- to_dict() JSON serialization
- Validation in __post_init__ methods
- Constants exports
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


class TestArtifactType:
    """Tests for ArtifactType literal type."""

    def test_valid_artifact_types_constant_exists(self) -> None:
        """Test that VALID_ARTIFACT_TYPES constant is exported."""
        from yolo_developer.audit.traceability_types import VALID_ARTIFACT_TYPES

        assert isinstance(VALID_ARTIFACT_TYPES, frozenset)
        assert len(VALID_ARTIFACT_TYPES) == 5

    def test_valid_artifact_types_contains_expected_values(self) -> None:
        """Test that VALID_ARTIFACT_TYPES contains all expected values."""
        from yolo_developer.audit.traceability_types import VALID_ARTIFACT_TYPES

        expected = {"requirement", "story", "design_decision", "code", "test"}
        assert VALID_ARTIFACT_TYPES == expected


class TestLinkType:
    """Tests for LinkType literal type."""

    def test_valid_link_types_constant_exists(self) -> None:
        """Test that VALID_LINK_TYPES constant is exported."""
        from yolo_developer.audit.traceability_types import VALID_LINK_TYPES

        assert isinstance(VALID_LINK_TYPES, frozenset)
        assert len(VALID_LINK_TYPES) == 4

    def test_valid_link_types_contains_expected_values(self) -> None:
        """Test that VALID_LINK_TYPES contains all expected values."""
        from yolo_developer.audit.traceability_types import VALID_LINK_TYPES

        expected = {"derives_from", "implements", "tests", "documents"}
        assert VALID_LINK_TYPES == expected


class TestTraceableArtifact:
    """Tests for TraceableArtifact frozen dataclass."""

    def test_create_valid_artifact(self) -> None:
        """Test creating a valid TraceableArtifact."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="req-001",
            artifact_type="requirement",
            name="FR81 Decision Logging",
            description="System can log all agent decisions with rationale",
            created_at="2026-01-18T12:00:00Z",
        )

        assert artifact.id == "req-001"
        assert artifact.artifact_type == "requirement"
        assert artifact.name == "FR81 Decision Logging"
        assert artifact.description == "System can log all agent decisions with rationale"
        assert artifact.created_at == "2026-01-18T12:00:00Z"
        assert artifact.metadata == {}

    def test_create_artifact_with_metadata(self) -> None:
        """Test creating a TraceableArtifact with metadata."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="code-001",
            artifact_type="code",
            name="DecisionLogger",
            description="High-level decision logging",
            created_at="2026-01-18T12:00:00Z",
            metadata={"file_path": "src/yolo_developer/audit/logger.py", "lines": 150},
        )

        assert artifact.metadata == {
            "file_path": "src/yolo_developer/audit/logger.py",
            "lines": 150,
        }

    def test_artifact_is_frozen(self) -> None:
        """Test that TraceableArtifact is immutable (frozen)."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="req-001",
            artifact_type="requirement",
            name="Test",
            description="Test desc",
            created_at="2026-01-18T12:00:00Z",
        )

        with pytest.raises(AttributeError):
            artifact.id = "new-id"  # type: ignore[misc]

    def test_artifact_to_dict(self) -> None:
        """Test TraceableArtifact.to_dict() produces JSON-serializable output."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="req-001",
            artifact_type="requirement",
            name="FR81 Decision Logging",
            description="System can log all agent decisions",
            created_at="2026-01-18T12:00:00Z",
            metadata={"source": "prd.md"},
        )

        result = artifact.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == "req-001"
        assert result["artifact_type"] == "requirement"
        assert result["name"] == "FR81 Decision Logging"
        assert result["description"] == "System can log all agent decisions"
        assert result["created_at"] == "2026-01-18T12:00:00Z"
        assert result["metadata"] == {"source": "prd.md"}

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_artifact_validation_warns_on_empty_id(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that empty id logs a warning."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        with caplog.at_level(logging.WARNING):
            TraceableArtifact(
                id="",
                artifact_type="requirement",
                name="Test",
                description="Test desc",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "TraceableArtifact id is empty" in caplog.text

    def test_artifact_validation_warns_on_empty_name(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that empty name logs a warning."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        with caplog.at_level(logging.WARNING):
            TraceableArtifact(
                id="req-001",
                artifact_type="requirement",
                name="",
                description="Test desc",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "TraceableArtifact name is empty" in caplog.text

    def test_artifact_validation_warns_on_invalid_type(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that invalid artifact_type logs a warning."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        with caplog.at_level(logging.WARNING):
            TraceableArtifact(
                id="req-001",
                artifact_type="invalid_type",  # type: ignore[arg-type]
                name="Test",
                description="Test desc",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "is not a valid artifact type" in caplog.text

    def test_artifact_validation_warns_on_empty_created_at(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that empty created_at logs a warning."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        with caplog.at_level(logging.WARNING):
            TraceableArtifact(
                id="req-001",
                artifact_type="requirement",
                name="Test",
                description="Test desc",
                created_at="",
            )

        assert "TraceableArtifact created_at is empty" in caplog.text


class TestTraceLink:
    """Tests for TraceLink frozen dataclass."""

    def test_create_valid_link(self) -> None:
        """Test creating a valid TraceLink."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="req-001",
            source_type="requirement",
            target_id="story-001",
            target_type="story",
            link_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        assert link.id == "link-001"
        assert link.source_id == "req-001"
        assert link.source_type == "requirement"
        assert link.target_id == "story-001"
        assert link.target_type == "story"
        assert link.link_type == "derives_from"
        assert link.created_at == "2026-01-18T12:00:00Z"
        assert link.metadata == {}

    def test_create_link_with_metadata(self) -> None:
        """Test creating a TraceLink with metadata."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="design-001",
            source_type="design_decision",
            target_id="code-001",
            target_type="code",
            link_type="implements",
            created_at="2026-01-18T12:00:00Z",
            metadata={"confidence": 0.95, "verified": True},
        )

        assert link.metadata == {"confidence": 0.95, "verified": True}

    def test_link_is_frozen(self) -> None:
        """Test that TraceLink is immutable (frozen)."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="req-001",
            source_type="requirement",
            target_id="story-001",
            target_type="story",
            link_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        with pytest.raises(AttributeError):
            link.id = "new-id"  # type: ignore[misc]

    def test_link_to_dict(self) -> None:
        """Test TraceLink.to_dict() produces JSON-serializable output."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="req-001",
            source_type="requirement",
            target_id="story-001",
            target_type="story",
            link_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
            metadata={"agent": "analyst"},
        )

        result = link.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == "link-001"
        assert result["source_id"] == "req-001"
        assert result["source_type"] == "requirement"
        assert result["target_id"] == "story-001"
        assert result["target_type"] == "story"
        assert result["link_type"] == "derives_from"
        assert result["created_at"] == "2026-01-18T12:00:00Z"
        assert result["metadata"] == {"agent": "analyst"}

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_link_validation_warns_on_empty_id(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that empty id logs a warning."""
        from yolo_developer.audit.traceability_types import TraceLink

        with caplog.at_level(logging.WARNING):
            TraceLink(
                id="",
                source_id="req-001",
                source_type="requirement",
                target_id="story-001",
                target_type="story",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "TraceLink id is empty" in caplog.text

    def test_link_validation_warns_on_empty_source_id(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that empty source_id logs a warning."""
        from yolo_developer.audit.traceability_types import TraceLink

        with caplog.at_level(logging.WARNING):
            TraceLink(
                id="link-001",
                source_id="",
                source_type="requirement",
                target_id="story-001",
                target_type="story",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "TraceLink source_id is empty" in caplog.text

    def test_link_validation_warns_on_empty_target_id(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that empty target_id logs a warning."""
        from yolo_developer.audit.traceability_types import TraceLink

        with caplog.at_level(logging.WARNING):
            TraceLink(
                id="link-001",
                source_id="req-001",
                source_type="requirement",
                target_id="",
                target_type="story",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "TraceLink target_id is empty" in caplog.text

    def test_link_validation_warns_on_invalid_source_type(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that invalid source_type logs a warning."""
        from yolo_developer.audit.traceability_types import TraceLink

        with caplog.at_level(logging.WARNING):
            TraceLink(
                id="link-001",
                source_id="req-001",
                source_type="invalid",  # type: ignore[arg-type]
                target_id="story-001",
                target_type="story",
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "is not a valid artifact type" in caplog.text

    def test_link_validation_warns_on_invalid_target_type(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that invalid target_type logs a warning."""
        from yolo_developer.audit.traceability_types import TraceLink

        with caplog.at_level(logging.WARNING):
            TraceLink(
                id="link-001",
                source_id="req-001",
                source_type="requirement",
                target_id="story-001",
                target_type="invalid",  # type: ignore[arg-type]
                link_type="derives_from",
                created_at="2026-01-18T12:00:00Z",
            )

        assert "is not a valid artifact type" in caplog.text

    def test_link_validation_warns_on_invalid_link_type(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that invalid link_type logs a warning."""
        from yolo_developer.audit.traceability_types import TraceLink

        with caplog.at_level(logging.WARNING):
            TraceLink(
                id="link-001",
                source_id="req-001",
                source_type="requirement",
                target_id="story-001",
                target_type="story",
                link_type="invalid",  # type: ignore[arg-type]
                created_at="2026-01-18T12:00:00Z",
            )

        assert "is not a valid link type" in caplog.text

    def test_link_validation_warns_on_empty_created_at(
        self, caplog: LogCaptureFixture
    ) -> None:
        """Test that empty created_at logs a warning."""
        from yolo_developer.audit.traceability_types import TraceLink

        with caplog.at_level(logging.WARNING):
            TraceLink(
                id="link-001",
                source_id="req-001",
                source_type="requirement",
                target_id="story-001",
                target_type="story",
                link_type="derives_from",
                created_at="",
            )

        assert "TraceLink created_at is empty" in caplog.text


class TestAllArtifactTypes:
    """Tests for all artifact types in the DAG."""

    def test_create_requirement_artifact(self) -> None:
        """Test creating a requirement artifact."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="FR82",
            artifact_type="requirement",
            name="Requirement Traceability",
            description="System can trace code back to requirements",
            created_at="2026-01-18T12:00:00Z",
        )

        assert artifact.artifact_type == "requirement"

    def test_create_story_artifact(self) -> None:
        """Test creating a story artifact."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="11-2-requirement-traceability",
            artifact_type="story",
            name="Story 11.2: Requirement Traceability",
            description="Implement requirement traceability",
            created_at="2026-01-18T12:00:00Z",
        )

        assert artifact.artifact_type == "story"

    def test_create_design_decision_artifact(self) -> None:
        """Test creating a design_decision artifact."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="design-001",
            artifact_type="design_decision",
            name="DAG-based traceability",
            description="Use directed acyclic graph for trace links",
            created_at="2026-01-18T12:00:00Z",
        )

        assert artifact.artifact_type == "design_decision"

    def test_create_code_artifact(self) -> None:
        """Test creating a code artifact."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="src/yolo_developer/audit/traceability.py",
            artifact_type="code",
            name="TraceabilityService",
            description="High-level traceability service",
            created_at="2026-01-18T12:00:00Z",
        )

        assert artifact.artifact_type == "code"

    def test_create_test_artifact(self) -> None:
        """Test creating a test artifact."""
        from yolo_developer.audit.traceability_types import TraceableArtifact

        artifact = TraceableArtifact(
            id="tests/unit/audit/test_traceability.py",
            artifact_type="test",
            name="TraceabilityService tests",
            description="Unit tests for TraceabilityService",
            created_at="2026-01-18T12:00:00Z",
        )

        assert artifact.artifact_type == "test"


class TestAllLinkTypes:
    """Tests for all link types."""

    def test_derives_from_link(self) -> None:
        """Test derives_from link type."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="story-001",
            source_type="story",
            target_id="req-001",
            target_type="requirement",
            link_type="derives_from",
            created_at="2026-01-18T12:00:00Z",
        )

        assert link.link_type == "derives_from"

    def test_implements_link(self) -> None:
        """Test implements link type."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="code-001",
            source_type="code",
            target_id="design-001",
            target_type="design_decision",
            link_type="implements",
            created_at="2026-01-18T12:00:00Z",
        )

        assert link.link_type == "implements"

    def test_tests_link(self) -> None:
        """Test tests link type."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="test-001",
            source_type="test",
            target_id="code-001",
            target_type="code",
            link_type="tests",
            created_at="2026-01-18T12:00:00Z",
        )

        assert link.link_type == "tests"

    def test_documents_link(self) -> None:
        """Test documents link type."""
        from yolo_developer.audit.traceability_types import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="adr-001",
            source_type="design_decision",
            target_id="design-001",
            target_type="design_decision",
            link_type="documents",
            created_at="2026-01-18T12:00:00Z",
        )

        assert link.link_type == "documents"
