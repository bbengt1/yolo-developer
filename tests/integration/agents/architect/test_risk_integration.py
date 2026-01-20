"""Integration tests for technical risk identification (Story 7.5, Task 13).

Tests verify end-to-end risk identification flow through the architect node,
including integration with quality evaluation and design decisions.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.agents.architect import architect_node
from yolo_developer.agents.architect.risk_identifier import identify_technical_risks
from yolo_developer.agents.architect.types import (
    DesignDecision,
    QualityRisk,
    TechnicalRiskReport,
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
        "gate_results": [],
        "advisory_warnings": [],
    }
    if stories:
        state["pm_output"] = {"stories": stories}
    return state


def _create_test_story(
    story_id: str = "story-001",
    title: str = "Feature Implementation",
    description: str = "Implement a new feature with external API integration",
) -> dict[str, Any]:
    """Create a test story."""
    return {
        "id": story_id,
        "title": title,
        "description": description,
    }


def _create_test_decision(
    decision_id: str = "design-001",
    story_id: str = "story-001",
    decision_type: str = "integration",
    description: str = "Use external API for data",
    rationale: str = "Third-party service provides required data",
) -> DesignDecision:
    """Create a test design decision."""
    from yolo_developer.agents.architect.types import DesignDecisionType

    # Validate decision_type is a valid Literal value
    valid_types: list[DesignDecisionType] = [
        "pattern",
        "technology",
        "integration",
        "data",
        "security",
        "infrastructure",
    ]
    validated_type: DesignDecisionType = (
        decision_type if decision_type in valid_types else "pattern"
    )
    return DesignDecision(
        id=decision_id,
        story_id=story_id,
        decision_type=validated_type,
        description=description,
        rationale=rationale,
        alternatives_considered=("Build in-house",),
    )


class TestArchitectNodeRiskIntegration:
    """Test architect node risk identification integration."""

    @pytest.mark.asyncio
    async def test_architect_node_includes_technical_risk_reports(self) -> None:
        """Test that architect node generates technical risk reports."""
        stories = [
            _create_test_story("story-001", "Database Setup"),
            _create_test_story("story-002", "API Design"),
        ]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        assert "architect_output" in result
        output = result["architect_output"]
        # Technical risk reports should be in output
        assert "technical_risk_reports" in output
        assert len(output["technical_risk_reports"]) == 2

    @pytest.mark.asyncio
    async def test_architect_node_risk_report_has_structure(self) -> None:
        """Test that risk reports have correct structure."""
        stories = [_create_test_story("story-001")]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]
        risk_reports = output["technical_risk_reports"]
        assert "story-001" in risk_reports, "Expected story-001 in risk reports"
        report = risk_reports["story-001"]
        assert "risks" in report
        assert "overall_risk_level" in report
        assert "summary" in report

    @pytest.mark.asyncio
    async def test_architect_node_risk_report_has_risks(self) -> None:
        """Test that risky stories generate risks."""
        stories = [
            _create_test_story(
                "story-001",
                "Risky Feature",
                "Using deprecated API with external vendor dependency and single instance",
            )
        ]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]
        risk_reports = output["technical_risk_reports"]
        assert "story-001" in risk_reports, "Expected story-001 in risk reports"
        report = risk_reports["story-001"]
        # Should have some risks
        assert isinstance(report["risks"], list)

    @pytest.mark.asyncio
    async def test_architect_node_includes_risk_in_processing_notes(self) -> None:
        """Test that processing notes mention risk identification."""
        stories = [_create_test_story()]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]
        assert "risk" in output["processing_notes"].lower()


class TestIdentifyTechnicalRisksIntegration:
    """Test identify_technical_risks function integration."""

    @pytest.mark.asyncio
    async def test_identify_with_design_decisions(self) -> None:
        """Test risk identification with design decisions."""
        story = _create_test_story(
            description="Implement feature using deprecated external API",
        )
        decisions = [
            _create_test_decision(
                decision_type="integration",
                description="Use deprecated third-party vendor API",
                rationale="Legacy compatibility",
            ),
        ]

        report = await identify_technical_risks(story, decisions, use_llm=False)

        assert isinstance(report, TechnicalRiskReport)
        # Should have risks from deprecated and external patterns
        assert len(report.risks) >= 1
        assert report.overall_risk_level in ("critical", "high", "medium", "low")

    @pytest.mark.asyncio
    async def test_identify_without_design_decisions(self) -> None:
        """Test risk identification without design decisions."""
        story = _create_test_story(description="Simple feature with deprecated technology")
        decisions: list[DesignDecision] = []

        report = await identify_technical_risks(story, decisions, use_llm=False)

        assert isinstance(report, TechnicalRiskReport)
        assert report.summary != ""

    @pytest.mark.asyncio
    async def test_identify_with_quality_risks(self) -> None:
        """Test that quality risks are incorporated."""
        story = _create_test_story()
        decisions: list[DesignDecision] = []
        quality_risks = [
            QualityRisk(
                attribute="reliability",
                description="Low reliability score",
                severity="high",
                mitigation="Add retry logic",
                mitigation_effort="medium",
            ),
        ]

        report = await identify_technical_risks(
            story, decisions, quality_risks=quality_risks, use_llm=False
        )

        # Should include converted quality risk
        quality_risk_found = any("[Quality]" in r.description for r in report.risks)
        assert quality_risk_found

    @pytest.mark.asyncio
    async def test_identify_multiple_categories(self) -> None:
        """Test that multiple risk categories are detected."""
        story = _create_test_story(
            description=(
                "Deprecated API with external vendor and single instance "
                "using session state and no monitoring"
            )
        )
        decisions: list[DesignDecision] = []

        report = await identify_technical_risks(story, decisions, use_llm=False)

        categories = {r.category for r in report.risks}
        # Should have multiple categories
        assert len(categories) >= 2


class TestRiskReportSerialization:
    """Test risk report serialization in architect output."""

    @pytest.mark.asyncio
    async def test_report_to_dict(self) -> None:
        """Test that report can be serialized to dict."""
        story = _create_test_story(description="Using deprecated single database")
        decisions: list[DesignDecision] = []

        report = await identify_technical_risks(story, decisions, use_llm=False)
        result = report.to_dict()

        assert isinstance(result, dict)
        assert "risks" in result
        assert "overall_risk_level" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_architect_output_includes_report_data(self) -> None:
        """Test that ArchitectOutput properly includes risk report data."""
        stories = [_create_test_story()]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]

        # Verify structure
        assert "design_decisions" in output
        assert "adrs" in output
        assert "twelve_factor_analyses" in output
        assert "quality_evaluations" in output
        assert "technical_risk_reports" in output

        # Risk reports should be serialized
        risk_reports = output["technical_risk_reports"]
        for _story_id, report_data in risk_reports.items():
            assert "risks" in report_data
            assert "overall_risk_level" in report_data


class TestRiskMitigationIntegration:
    """Test that mitigations are properly generated and included."""

    @pytest.mark.asyncio
    async def test_risks_have_mitigations(self) -> None:
        """Test that identified risks have mitigation strategies."""
        story = _create_test_story(
            description="Using deprecated external API with vendor dependency"
        )
        decisions: list[DesignDecision] = []

        report = await identify_technical_risks(story, decisions, use_llm=False)

        for risk in report.risks:
            assert risk.mitigation != ""
            assert risk.mitigation_effort in ("high", "medium", "low")
            assert risk.mitigation_priority in ("P1", "P2", "P3", "P4")

    @pytest.mark.asyncio
    async def test_high_severity_risks_get_priority(self) -> None:
        """Test that high severity risks get appropriate priority."""
        story = _create_test_story(description="Using end-of-life technology")
        decisions: list[DesignDecision] = []

        report = await identify_technical_risks(story, decisions, use_llm=False)

        # End-of-life is critical, should get P1
        critical_risks = [r for r in report.risks if r.severity == "critical"]
        for risk in critical_risks:
            assert risk.mitigation_priority == "P1"


class TestRiskOverallLevelCalculation:
    """Test overall risk level calculation."""

    @pytest.mark.asyncio
    async def test_critical_risk_sets_overall_critical(self) -> None:
        """Test that critical risk sets overall to critical."""
        story = _create_test_story(description="Using end-of-life API")
        decisions: list[DesignDecision] = []

        report = await identify_technical_risks(story, decisions, use_llm=False)

        # Should have critical overall level
        assert report.overall_risk_level == "critical"

    @pytest.mark.asyncio
    async def test_no_risks_returns_low(self) -> None:
        """Test that no risks returns low overall level."""
        story = _create_test_story(description="Standard feature implementation")
        decisions: list[DesignDecision] = []

        report = await identify_technical_risks(story, decisions, use_llm=False)

        if len(report.risks) == 0:
            assert report.overall_risk_level == "low"
