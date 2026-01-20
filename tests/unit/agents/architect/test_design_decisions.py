"""Tests for design decision generation (Story 7.1, Task 5).

Tests verify that the Architect agent can generate design decisions
for stories with proper ID format and structure.
"""

from __future__ import annotations

import re

import pytest

from yolo_developer.agents.architect.node import _generate_design_decisions
from yolo_developer.agents.architect.types import DesignDecision


class TestDesignDecisionGeneration:
    """Test design decision generation from stories."""

    @pytest.mark.asyncio
    async def test_generates_decision_for_story(self) -> None:
        """Test that a design decision is generated for a story."""
        stories = [
            {
                "id": "story-001",
                "title": "User Authentication",
                "role": "user",
                "action": "log in",
                "benefit": "access system",
            }
        ]

        decisions, _ = await _generate_design_decisions(stories)

        assert len(decisions) == 1
        assert isinstance(decisions[0], DesignDecision)
        assert decisions[0].story_id == "story-001"

    @pytest.mark.asyncio
    async def test_generates_multiple_decisions(self) -> None:
        """Test that decisions are generated for multiple stories."""
        stories = [{"id": f"story-{i:03d}", "title": f"Story {i}"} for i in range(3)]

        decisions, _ = await _generate_design_decisions(stories)

        assert len(decisions) == 3
        story_ids = [d.story_id for d in decisions]
        assert story_ids == ["story-000", "story-001", "story-002"]

    @pytest.mark.asyncio
    async def test_decision_id_format(self) -> None:
        """Test that decision IDs follow correct format: design-{timestamp}-{counter}."""
        stories = [{"id": "story-001", "title": "Test"}]

        decisions, _ = await _generate_design_decisions(stories)

        # Format: design-{timestamp}-{counter:03d}
        pattern = r"^design-\d+-\d{3}$"
        assert re.match(pattern, decisions[0].id)

    @pytest.mark.asyncio
    async def test_decision_has_valid_type(self) -> None:
        """Test that decision type is one of the valid types."""
        stories = [{"id": "story-001", "title": "Test"}]
        valid_types = {"pattern", "technology", "integration", "data", "security", "infrastructure"}

        decisions, _ = await _generate_design_decisions(stories)

        assert decisions[0].decision_type in valid_types

    @pytest.mark.asyncio
    async def test_decision_has_description(self) -> None:
        """Test that decision has a non-empty description."""
        stories = [{"id": "story-001", "title": "Test Story"}]

        decisions, _ = await _generate_design_decisions(stories)

        assert decisions[0].description
        assert len(decisions[0].description) > 0

    @pytest.mark.asyncio
    async def test_decision_has_rationale(self) -> None:
        """Test that decision has a non-empty rationale."""
        stories = [{"id": "story-001", "title": "Test Story"}]

        decisions, _ = await _generate_design_decisions(stories)

        assert decisions[0].rationale
        assert len(decisions[0].rationale) > 0

    @pytest.mark.asyncio
    async def test_decision_has_alternatives(self) -> None:
        """Test that decision has alternatives_considered."""
        stories = [{"id": "story-001", "title": "Test Story"}]

        decisions, _ = await _generate_design_decisions(stories)

        assert isinstance(decisions[0].alternatives_considered, tuple)

    @pytest.mark.asyncio
    async def test_decision_has_created_at(self) -> None:
        """Test that decision has ISO timestamp."""
        stories = [{"id": "story-001", "title": "Test"}]

        decisions, _ = await _generate_design_decisions(stories)

        # Should be ISO format timestamp
        assert "T" in decisions[0].created_at
        assert decisions[0].created_at.endswith("+00:00") or "Z" in decisions[0].created_at


class TestDesignDecisionEmptyInput:
    """Test handling of empty input."""

    @pytest.mark.asyncio
    async def test_empty_stories_returns_empty_list(self) -> None:
        """Test that empty stories list returns empty decisions."""
        decisions, analyses = await _generate_design_decisions([])

        assert decisions == []
        assert analyses == {}


class TestDesignDecisionIdUniqueness:
    """Test that decision IDs are unique."""

    @pytest.mark.asyncio
    async def test_sequential_decisions_have_unique_ids(self) -> None:
        """Test that multiple decisions have unique IDs."""
        stories = [{"id": f"story-{i:03d}", "title": f"Story {i}"} for i in range(5)]

        decisions, _ = await _generate_design_decisions(stories)

        ids = [d.id for d in decisions]
        assert len(ids) == len(set(ids))  # All IDs unique
