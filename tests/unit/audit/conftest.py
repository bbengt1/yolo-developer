"""Shared test fixtures for audit module tests.

Provides common test data factories for Decision, TraceableArtifact,
and TraceLink objects used across multiple test files.
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext


def create_test_decision(
    id: str = "dec-001",
    content: str = "Test decision",
    decision_type: str = "requirement_analysis",
    severity: str = "info",
) -> Decision:
    """Create a test Decision object.

    Args:
        id: Decision ID.
        content: Decision content text.
        decision_type: Type of decision.
        severity: Decision severity level.

    Returns:
        Decision object for testing.
    """
    return Decision(
        id=id,
        decision_type=decision_type,  # type: ignore[arg-type]
        content=content,
        rationale="Test rationale",
        agent=AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-123",
        ),
        context=DecisionContext(
            sprint_id="sprint-1",
            story_id="story-1",
        ),
        timestamp="2026-01-18T12:00:00Z",
        metadata={"key": "value"},
        severity=severity,  # type: ignore[arg-type]
    )


def create_test_artifact(
    id: str = "art-001",
    name: str = "Test Artifact",
    artifact_type: str = "requirement",
) -> TraceableArtifact:
    """Create a test TraceableArtifact object.

    Args:
        id: Artifact ID.
        name: Artifact name.
        artifact_type: Type of artifact.

    Returns:
        TraceableArtifact object for testing.
    """
    return TraceableArtifact(
        id=id,
        artifact_type=artifact_type,
        name=name,
        description="Test description",
        created_at="2026-01-18T12:00:00Z",
        metadata={"key": "value"},
    )


def create_test_link(
    id: str = "link-001",
    source_id: str = "art-001",
    target_id: str = "art-002",
    link_type: str = "traces_to",
) -> TraceLink:
    """Create a test TraceLink object.

    Args:
        id: Link ID.
        source_id: Source artifact ID.
        target_id: Target artifact ID.
        link_type: Type of link.

    Returns:
        TraceLink object for testing.
    """
    return TraceLink(
        id=id,
        source_id=source_id,
        source_type="requirement",
        target_id=target_id,
        target_type="story",
        link_type=link_type,
        created_at="2026-01-18T12:00:00Z",
        metadata={"key": "value"},
    )


# Pytest fixtures for dependency injection


@pytest.fixture
def test_decision() -> Decision:
    """Provide a test Decision object."""
    return create_test_decision()


@pytest.fixture
def test_artifact() -> TraceableArtifact:
    """Provide a test TraceableArtifact object."""
    return create_test_artifact()


@pytest.fixture
def test_link() -> TraceLink:
    """Provide a test TraceLink object."""
    return create_test_link()
