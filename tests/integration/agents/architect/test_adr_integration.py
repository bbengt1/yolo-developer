"""Integration tests for ADR generation (Story 7.3, Task 9).

Tests verify end-to-end ADR generation flow through the architect node,
including 12-Factor analysis integration and proper content generation.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.agents.architect import architect_node
from yolo_developer.agents.architect.adr_generator import generate_adr, generate_adrs
from yolo_developer.agents.architect.types import (
    ADR,
    DesignDecision,
    FactorResult,
    TwelveFactorAnalysis,
)
from yolo_developer.orchestrator.state import YoloState


def _create_minimal_state(stories: list[dict[str, Any]] | None = None) -> YoloState:
    """Create minimal YoloState for testing."""
    state: YoloState = {
        "messages": [],
        "current_agent": "architect",
        "handoff_context": None,
        "decisions": [],
        "gate_blocked": False,
        "gate_results": [],  # Must be list for decorator to append
        "advisory_warnings": [],
    }
    if stories:
        state["pm_output"] = {"stories": stories}
    return state


def _create_test_story(
    story_id: str = "story-001",
    title: str = "User Authentication",
) -> dict[str, Any]:
    """Create a test story."""
    return {
        "id": story_id,
        "title": title,
        "description": "Implement user authentication",
    }


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
        description="Use PostgreSQL for data persistence",
        rationale="ACID compliance required for financial data",
        alternatives_considered=("MySQL", "MongoDB"),
    )


def _create_test_analysis(
    overall_compliance: float = 0.85,
) -> TwelveFactorAnalysis:
    """Create a test 12-Factor analysis."""
    return TwelveFactorAnalysis(
        factor_results={
            "config": FactorResult(
                factor_name="config",
                applies=True,
                compliant=False,
                finding="Hardcoded configuration detected",
                recommendation="Use environment variables",
            ),
        },
        applicable_factors=("config", "backing_services"),
        overall_compliance=overall_compliance,
        recommendations=("Externalize config", "Use connection strings"),
    )


class TestArchitectNodeAdrIntegration:
    """Test architect node ADR generation integration."""

    @pytest.mark.asyncio
    async def test_architect_node_generates_adrs(self) -> None:
        """Test that architect node generates ADRs for stories."""
        stories = [
            _create_test_story("story-001", "Database Setup"),
            _create_test_story("story-002", "API Design"),
        ]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        assert "architect_output" in result
        output = result["architect_output"]
        # ADRs should be in output
        assert "adrs" in output

    @pytest.mark.asyncio
    async def test_architect_node_adrs_link_to_stories(self) -> None:
        """Test that generated ADRs link to stories."""
        stories = [_create_test_story("story-001")]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]
        if output["adrs"]:
            adr = output["adrs"][0]
            # ADRs should link to story IDs
            assert "story_ids" in adr
            assert len(adr["story_ids"]) > 0

    @pytest.mark.asyncio
    async def test_architect_node_adrs_have_full_content(self) -> None:
        """Test that ADRs have context, decision, and consequences."""
        stories = [_create_test_story("story-001")]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]
        if output["adrs"]:
            adr = output["adrs"][0]
            # ADRs should have full content (Story 7.3)
            assert "context" in adr
            assert "decision" in adr
            assert "consequences" in adr
            # Consequences should not be stub text
            assert "Stub implementation" not in adr["consequences"]


class TestGenerateAdrIntegration:
    """Test generate_adr function integration."""

    @pytest.mark.asyncio
    async def test_generate_adr_with_twelve_factor_analysis(self) -> None:
        """Test ADR generation with 12-Factor analysis."""
        decision = _create_test_decision()
        analysis = _create_test_analysis(overall_compliance=0.75)

        adr = await generate_adr(decision, analysis, adr_number=1)

        assert isinstance(adr, ADR)
        # Context should mention 12-Factor compliance
        assert "75%" in adr.context or "12-Factor" in adr.context
        # Consequences should include recommendations
        assert "recommend" in adr.consequences.lower() or "externalize" in adr.consequences.lower()

    @pytest.mark.asyncio
    async def test_generate_adr_without_analysis(self) -> None:
        """Test ADR generation without 12-Factor analysis."""
        decision = _create_test_decision()

        adr = await generate_adr(decision, None, adr_number=1)

        assert isinstance(adr, ADR)
        assert adr.context
        assert adr.decision
        assert adr.consequences

    @pytest.mark.asyncio
    async def test_generate_adr_with_additional_stories(self) -> None:
        """Test ADR generation with additional story links."""
        decision = _create_test_decision(story_id="story-001")
        analysis = _create_test_analysis()

        adr = await generate_adr(
            decision,
            analysis,
            adr_number=1,
            additional_story_ids=("story-002", "story-003"),
        )

        assert "story-001" in adr.story_ids
        assert "story-002" in adr.story_ids
        assert "story-003" in adr.story_ids


class TestGenerateAdrsIntegration:
    """Test generate_adrs batch function integration."""

    @pytest.mark.asyncio
    async def test_generate_adrs_with_twelve_factor_analyses(self) -> None:
        """Test batch ADR generation with 12-Factor analyses."""
        decisions = [
            _create_test_decision("design-001", "story-001", "technology"),
            _create_test_decision("design-002", "story-002", "pattern"),
        ]
        analyses = {
            "story-001": _create_test_analysis(0.85),
            "story-002": _create_test_analysis(0.95),
        }

        adrs = await generate_adrs(decisions, analyses)

        assert len(adrs) == 2
        for adr in adrs:
            assert isinstance(adr, ADR)
            assert adr.context
            assert adr.decision
            assert adr.consequences

    @pytest.mark.asyncio
    async def test_generate_adrs_filters_significant_decisions(self) -> None:
        """Test that only significant decisions get ADRs."""
        decisions = [
            _create_test_decision("design-001", "story-001", "technology"),
            _create_test_decision("design-002", "story-002", "security"),  # Not significant
            _create_test_decision("design-003", "story-003", "pattern"),
        ]

        adrs = await generate_adrs(decisions, {})

        # Only technology and pattern types get ADRs
        assert len(adrs) == 2
        adr_ids = [adr.id for adr in adrs]
        assert "ADR-001" in adr_ids
        assert "ADR-002" in adr_ids

    @pytest.mark.asyncio
    async def test_generate_adrs_with_dict_analyses(self) -> None:
        """Test batch ADR generation with dict-format analyses."""
        decisions = [_create_test_decision()]
        # Simulate dict from serialized state
        analyses: dict[str, Any] = {
            "story-001": {
                "applicable_factors": ["config"],
                "overall_compliance": 0.8,
                "recommendations": ["Use env vars"],
            },
        }

        adrs = await generate_adrs(decisions, analyses)

        assert len(adrs) == 1
        assert "80%" in adrs[0].context or "config" in adrs[0].context


class TestAdrContentQuality:
    """Test ADR content quality and completeness."""

    @pytest.mark.asyncio
    async def test_adr_context_mentions_story(self) -> None:
        """Test that context references the story."""
        decision = _create_test_decision(story_id="story-important-feature")
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis, adr_number=1)

        assert "story-important-feature" in adr.context

    @pytest.mark.asyncio
    async def test_adr_decision_includes_rationale(self) -> None:
        """Test that decision includes rationale."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="technology",  # type: ignore[arg-type]
            description="Use Redis for caching",
            rationale="High throughput required for real-time data",
            alternatives_considered=("Memcached", "In-memory"),
        )
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis, adr_number=1)

        assert "rationale" in adr.decision.lower() or "High throughput" in adr.decision

    @pytest.mark.asyncio
    async def test_adr_consequences_include_alternatives(self) -> None:
        """Test that consequences document alternatives."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="technology",  # type: ignore[arg-type]
            description="Use PostgreSQL",
            rationale="ACID compliance",
            alternatives_considered=("MySQL", "MongoDB"),
        )
        analysis = _create_test_analysis()

        adr = await generate_adr(decision, analysis, adr_number=1)

        assert "MySQL" in adr.consequences
        assert "MongoDB" in adr.consequences

    @pytest.mark.asyncio
    async def test_adr_title_is_descriptive(self) -> None:
        """Test that ADR title is descriptive."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="technology",  # type: ignore[arg-type]
            description="PostgreSQL for data persistence",
            rationale="ACID compliance",
            alternatives_considered=(),
        )

        adr = await generate_adr(decision, None, adr_number=1)

        assert len(adr.title) > 5
        assert "PostgreSQL" in adr.title or "persistence" in adr.title.lower()
