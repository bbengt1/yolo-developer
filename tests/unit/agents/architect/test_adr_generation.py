"""Tests for ADR generation (Story 7.1, Task 6).

Tests verify that the Architect agent can generate Architecture Decision
Records (ADRs) from design decisions with proper format and structure.

Updated for Story 7.3 to use generate_adrs from adr_generator.
"""

from __future__ import annotations

import re

import pytest

from yolo_developer.agents.architect.adr_generator import generate_adrs
from yolo_developer.agents.architect.types import ADR, DesignDecision


def _create_test_decision(
    decision_id: str = "design-001",
    story_id: str = "story-001",
    decision_type: str = "technology",
) -> DesignDecision:
    """Create a test design decision."""
    return DesignDecision(
        id=decision_id,
        story_id=story_id,
        decision_type=decision_type,  # type: ignore[arg-type]
        description="Test decision",
        rationale="Test rationale",
        alternatives_considered=("Alt A", "Alt B"),
    )


class TestADRGeneration:
    """Test ADR generation from design decisions."""

    @pytest.mark.asyncio
    async def test_generates_adr_for_technology_decision(self) -> None:
        """Test that ADR is generated for technology decisions."""
        decisions = [_create_test_decision(decision_type="technology")]

        adrs = await generate_adrs(decisions, {})

        assert len(adrs) >= 1
        assert isinstance(adrs[0], ADR)

    @pytest.mark.asyncio
    async def test_generates_adr_for_pattern_decision(self) -> None:
        """Test that ADR is generated for pattern decisions."""
        decisions = [_create_test_decision(decision_type="pattern")]

        adrs = await generate_adrs(decisions, {})

        assert len(adrs) >= 1

    @pytest.mark.asyncio
    async def test_adr_id_format(self) -> None:
        """Test that ADR IDs follow format: ADR-{number:03d}."""
        decisions = [_create_test_decision()]

        adrs = await generate_adrs(decisions, {})

        # Format: ADR-001, ADR-002, etc.
        pattern = r"^ADR-\d{3}$"
        assert re.match(pattern, adrs[0].id)

    @pytest.mark.asyncio
    async def test_adr_has_title(self) -> None:
        """Test that ADR has a non-empty title."""
        decisions = [_create_test_decision()]

        adrs = await generate_adrs(decisions, {})

        assert adrs[0].title
        assert len(adrs[0].title) > 0

    @pytest.mark.asyncio
    async def test_adr_has_valid_status(self) -> None:
        """Test that ADR status is valid."""
        decisions = [_create_test_decision()]
        valid_statuses = {"proposed", "accepted", "deprecated", "superseded"}

        adrs = await generate_adrs(decisions, {})

        assert adrs[0].status in valid_statuses

    @pytest.mark.asyncio
    async def test_adr_has_context(self) -> None:
        """Test that ADR has context field."""
        decisions = [_create_test_decision()]

        adrs = await generate_adrs(decisions, {})

        assert adrs[0].context
        assert len(adrs[0].context) > 0

    @pytest.mark.asyncio
    async def test_adr_has_decision(self) -> None:
        """Test that ADR has decision field."""
        decisions = [_create_test_decision()]

        adrs = await generate_adrs(decisions, {})

        assert adrs[0].decision
        assert len(adrs[0].decision) > 0

    @pytest.mark.asyncio
    async def test_adr_has_consequences(self) -> None:
        """Test that ADR has consequences field."""
        decisions = [_create_test_decision()]

        adrs = await generate_adrs(decisions, {})

        assert adrs[0].consequences
        assert len(adrs[0].consequences) > 0

    @pytest.mark.asyncio
    async def test_adr_links_to_stories(self) -> None:
        """Test that ADR is linked to story IDs."""
        decisions = [_create_test_decision(story_id="story-001")]

        adrs = await generate_adrs(decisions, {})

        assert "story-001" in adrs[0].story_ids


class TestADREmptyInput:
    """Test handling of empty input."""

    @pytest.mark.asyncio
    async def test_empty_decisions_returns_empty_list(self) -> None:
        """Test that empty decisions list returns empty ADRs."""
        adrs = await generate_adrs([], {})

        assert adrs == []


class TestADRIdUniqueness:
    """Test that ADR IDs are unique."""

    @pytest.mark.asyncio
    async def test_sequential_adrs_have_unique_ids(self) -> None:
        """Test that multiple ADRs have unique IDs."""
        decisions = [
            _create_test_decision(f"design-{i:03d}", f"story-{i:03d}", "technology")
            for i in range(3)
        ]

        adrs = await generate_adrs(decisions, {})

        ids = [a.id for a in adrs]
        assert len(ids) == len(set(ids))  # All IDs unique
