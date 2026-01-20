"""Tests for ADR story linking (Story 7.3, Task 3, 7).

Tests verify that ADRs are properly linked to stories.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.architect.types import (
    ADR,
    DesignDecision,
    TwelveFactorAnalysis,
)


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


def _create_test_analysis() -> TwelveFactorAnalysis:
    """Create a test 12-Factor analysis."""
    return TwelveFactorAnalysis(
        factor_results={},
        applicable_factors=(),
        overall_compliance=1.0,
        recommendations=(),
    )


class TestAdrStoryLinking:
    """Test ADR story linking functionality."""

    @pytest.mark.asyncio
    async def test_adr_includes_story_id(self) -> None:
        """Test that ADR includes story_ids field with related stories."""
        from yolo_developer.agents.architect.adr_generator import generate_adr

        decision = _create_test_decision(story_id="story-001")
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis)

        assert "story-001" in adr.story_ids

    @pytest.mark.asyncio
    async def test_adr_multiple_stories_linked(self) -> None:
        """Test that ADR can link to multiple stories via decision grouping."""
        from yolo_developer.agents.architect.adr_generator import generate_adr

        decision = _create_test_decision(story_id="story-001")
        analysis = _create_test_analysis()

        adr = await generate_adr(
            decision, analysis, additional_story_ids=("story-002", "story-003")
        )

        assert "story-001" in adr.story_ids
        assert "story-002" in adr.story_ids
        assert "story-003" in adr.story_ids

    @pytest.mark.asyncio
    async def test_adr_story_ids_is_tuple(self) -> None:
        """Test that story_ids is a tuple (immutable)."""
        from yolo_developer.agents.architect.adr_generator import generate_adr

        decision = _create_test_decision()
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis)

        assert isinstance(adr.story_ids, tuple)


class TestGenerateAdrsAsync:
    """Test async _generate_adrs function."""

    @pytest.mark.asyncio
    async def test_generate_adrs_is_async(self) -> None:
        """Test that generate_adrs is an async function."""
        import inspect

        from yolo_developer.agents.architect.adr_generator import generate_adrs

        assert inspect.iscoroutinefunction(generate_adrs)

    @pytest.mark.asyncio
    async def test_generate_adrs_returns_list_of_adrs(self) -> None:
        """Test that generate_adrs returns list of ADR objects."""
        from yolo_developer.agents.architect.adr_generator import generate_adrs

        decisions = [_create_test_decision()]
        analyses = {"story-001": _create_test_analysis()}

        adrs = await generate_adrs(decisions, analyses)

        assert isinstance(adrs, list)
        if adrs:
            assert isinstance(adrs[0], ADR)

    @pytest.mark.asyncio
    async def test_generate_adrs_accepts_analyses(self) -> None:
        """Test that generate_adrs accepts twelve_factor_analyses parameter."""
        from yolo_developer.agents.architect.adr_generator import generate_adrs

        decisions = [_create_test_decision()]
        analyses = {"story-001": _create_test_analysis()}

        # Should not raise
        adrs = await generate_adrs(decisions, analyses)

        assert isinstance(adrs, list)

    @pytest.mark.asyncio
    async def test_generate_adrs_empty_decisions(self) -> None:
        """Test that empty decisions returns empty list."""
        from yolo_developer.agents.architect.adr_generator import generate_adrs

        adrs = await generate_adrs([], {})

        assert adrs == []


class TestAdrIdFormat:
    """Test ADR ID formatting."""

    @pytest.mark.asyncio
    async def test_adr_id_format(self) -> None:
        """Test ADR IDs follow format ADR-{number:03d}."""
        import re

        from yolo_developer.agents.architect.adr_generator import generate_adr

        decision = _create_test_decision()
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis, adr_number=42)

        pattern = r"^ADR-\d{3}$"
        assert re.match(pattern, adr.id)
        assert adr.id == "ADR-042"


class TestAdrImmutability:
    """Test ADR immutability (frozen dataclass)."""

    @pytest.mark.asyncio
    async def test_adr_is_frozen(self) -> None:
        """Test that ADR cannot be modified after creation."""
        from yolo_developer.agents.architect.adr_generator import generate_adr

        decision = _create_test_decision()
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis)

        # Should raise FrozenInstanceError
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            adr.title = "New title"  # type: ignore[misc]


class TestAdrTimestamp:
    """Test ADR created_at timestamp."""

    @pytest.mark.asyncio
    async def test_adr_has_created_at(self) -> None:
        """Test that ADR has created_at timestamp."""
        from yolo_developer.agents.architect.adr_generator import generate_adr

        decision = _create_test_decision()
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis)

        assert adr.created_at is not None
        # Should be ISO format
        assert "T" in adr.created_at
