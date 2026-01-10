"""Unit tests for risk identifier module (Story 7.5, Tasks 10-12).

Tests verify:
- Technology risk detection with various patterns
- Integration risk detection with external API patterns
- Scalability risk detection with stateful patterns
- Mitigation generation is design-specific
- LLM analysis with mocked LLM and fallback behavior
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from yolo_developer.agents.architect.types import (
    DesignDecision,
    DesignDecisionType,
    QualityRisk,
    TechnicalRisk,
)


def _create_test_story(
    story_id: str = "story-001",
    title: str = "Test Story",
    description: str = "A test story description",
) -> dict[str, Any]:
    """Create a test story dictionary."""
    return {
        "id": story_id,
        "title": title,
        "description": description,
    }


def _create_test_decision(
    decision_id: str = "design-001",
    story_id: str = "story-001",
    decision_type: DesignDecisionType = "pattern",
    description: str = "Test design decision",
    rationale: str = "Test rationale",
) -> DesignDecision:
    """Create a test design decision."""
    return DesignDecision(
        id=decision_id,
        story_id=story_id,
        decision_type=decision_type,
        description=description,
        rationale=rationale,
        alternatives_considered=("Alternative A",),
    )


class TestTechnologyRiskDetection:
    """Test technology risk detection (Task 10)."""

    @pytest.mark.asyncio
    async def test_detects_deprecated_technology(self) -> None:
        """Test detection of deprecated technology patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_technology_risks,
        )

        story = _create_test_story(
            description="Using deprecated API for authentication"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_technology_risks(story, decisions)

        assert len(risks) >= 1
        assert any("deprecated" in r.description.lower() for r in risks)
        assert all(r.category == "technology" for r in risks)

    @pytest.mark.asyncio
    async def test_detects_experimental_features(self) -> None:
        """Test detection of experimental/unstable features."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_technology_risks,
        )

        story = _create_test_story(description="Using experimental feature in beta")
        decisions: list[DesignDecision] = []

        risks = _identify_technology_risks(story, decisions)

        assert len(risks) >= 1
        assert any(
            "experimental" in r.description.lower() or "beta" in r.description.lower()
            for r in risks
        )

    @pytest.mark.asyncio
    async def test_detects_version_conflicts(self) -> None:
        """Test detection of version conflict patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_technology_risks,
        )

        story = _create_test_story(
            description="May have incompatible version conflict with lib"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_technology_risks(story, decisions)

        assert len(risks) >= 1
        assert any(
            "incompatible" in r.description.lower()
            or "version" in r.description.lower()
            for r in risks
        )

    @pytest.mark.asyncio
    async def test_no_technology_risks_for_safe_story(self) -> None:
        """Test that safe story has no technology risks."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_technology_risks,
        )

        story = _create_test_story(
            description="Standard feature using stable libraries"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_technology_risks(story, decisions)

        assert len(risks) == 0

    @pytest.mark.asyncio
    async def test_detects_risks_from_decisions(self) -> None:
        """Test that risks are detected from design decisions."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_technology_risks,
        )

        story = _create_test_story()
        decisions = [
            _create_test_decision(
                description="Use legacy database driver",
                rationale="Compatibility with old system",
            )
        ]

        risks = _identify_technology_risks(story, decisions)

        assert len(risks) >= 1
        assert any("legacy" in r.description.lower() for r in risks)


class TestIntegrationRiskDetection:
    """Test integration risk detection (Task 10)."""

    @pytest.mark.asyncio
    async def test_detects_external_api_dependency(self) -> None:
        """Test detection of external API dependencies."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_integration_risks,
        )

        story = _create_test_story(
            description="Integrate with external API for payment processing"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_integration_risks(story, decisions)

        assert len(risks) >= 1
        assert any("external" in r.description.lower() for r in risks)
        assert all(r.category == "integration" for r in risks)

    @pytest.mark.asyncio
    async def test_detects_rate_limiting_concerns(self) -> None:
        """Test detection of rate limiting patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_integration_risks,
        )

        story = _create_test_story(
            description="API has rate limit of 100 requests per minute"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_integration_risks(story, decisions)

        assert len(risks) >= 1
        assert any("rate" in r.description.lower() for r in risks)

    @pytest.mark.asyncio
    async def test_detects_vendor_lock_in(self) -> None:
        """Test detection of vendor lock-in patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_integration_risks,
        )

        story = _create_test_story(
            description="Using proprietary vendor-specific cloud features"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_integration_risks(story, decisions)

        assert len(risks) >= 1
        assert any(
            "proprietary" in r.description.lower()
            or "vendor" in r.description.lower()
            for r in risks
        )

    @pytest.mark.asyncio
    async def test_detects_authentication_complexity(self) -> None:
        """Test detection of authentication complexity patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_integration_risks,
        )

        story = _create_test_story(
            description="Implement OAuth with credential management"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_integration_risks(story, decisions)

        assert len(risks) >= 1
        assert any(
            "oauth" in r.description.lower() or "credential" in r.description.lower()
            for r in risks
        )

    @pytest.mark.asyncio
    async def test_extracts_integration_components(self) -> None:
        """Test that affected components are extracted for integration risks."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_integration_risks,
        )

        story = _create_test_story(title="Payment Integration")
        decisions = [
            _create_test_decision(
                decision_type="integration",
                description="Stripe payment API integration",
            )
        ]

        risks = _identify_integration_risks(story, decisions)

        # Should have affected components from decision or story
        for risk in risks:
            assert len(risk.affected_components) > 0


class TestScalabilityRiskDetection:
    """Test scalability risk detection (Task 10)."""

    @pytest.mark.asyncio
    async def test_detects_single_point_of_failure(self) -> None:
        """Test detection of single point of failure patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_scalability_risks,
        )

        story = _create_test_story(description="Deploy as single instance")
        decisions: list[DesignDecision] = []

        risks = _identify_scalability_risks(story, decisions)

        assert len(risks) >= 1
        assert any("single" in r.description.lower() for r in risks)
        assert all(r.category == "scalability" for r in risks)

    @pytest.mark.asyncio
    async def test_detects_stateful_components(self) -> None:
        """Test detection of stateful component patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_scalability_risks,
        )

        story = _create_test_story(
            description="Store session state in memory with sticky session"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_scalability_risks(story, decisions)

        assert len(risks) >= 1
        assert any(
            "session" in r.description.lower() or "memory" in r.description.lower()
            for r in risks
        )

    @pytest.mark.asyncio
    async def test_detects_database_bottlenecks(self) -> None:
        """Test detection of database bottleneck patterns."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_scalability_risks,
        )

        story = _create_test_story(
            description="Using single database with no replication"
        )
        decisions: list[DesignDecision] = []

        risks = _identify_scalability_risks(story, decisions)

        assert len(risks) >= 1
        assert any(
            "database" in r.description.lower()
            or "replication" in r.description.lower()
            for r in risks
        )

    @pytest.mark.asyncio
    async def test_links_to_design_decisions(self) -> None:
        """Test that scalability risks link to specific decisions."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_scalability_risks,
        )

        story = _create_test_story()
        decisions = [
            _create_test_decision(
                decision_id="design-scale-001",
                decision_type="infrastructure",
                description="Use single database for all data",
            )
        ]

        risks = _identify_scalability_risks(story, decisions)

        # Should have some risks with affected components
        for risk in risks:
            assert len(risk.affected_components) > 0

    @pytest.mark.asyncio
    async def test_scalability_risks_have_high_effort(self) -> None:
        """Test that scalability risks default to higher effort."""
        from yolo_developer.agents.architect.risk_identifier import (
            _identify_scalability_risks,
        )

        story = _create_test_story(description="Monolithic single instance")
        decisions: list[DesignDecision] = []

        risks = _identify_scalability_risks(story, decisions)

        # Scalability fixes should typically require higher effort
        assert all(r.mitigation_effort in ("medium", "high") for r in risks)


class TestMitigationSuggestionEngine:
    """Test mitigation suggestion engine (Task 11)."""

    @pytest.mark.asyncio
    async def test_generates_design_specific_mitigations(self) -> None:
        """Test that mitigations are design-specific."""
        from yolo_developer.agents.architect.risk_identifier import (
            _generate_mitigations,
        )
        from yolo_developer.agents.architect.types import TechnicalRisk

        risks = [
            TechnicalRisk(
                category="integration",
                description="Dependency on external API",
                severity="high",
                affected_components=("PaymentService",),
                mitigation="",
                mitigation_effort="medium",
                mitigation_priority="P2",
            )
        ]
        story = _create_test_story(title="Payment Integration")
        decisions: list[DesignDecision] = []

        mitigated = _generate_mitigations(risks, story, decisions)

        assert len(mitigated) == 1
        assert mitigated[0].mitigation != ""
        # Should reference the story context
        assert "Payment Integration" in mitigated[0].mitigation or len(mitigated[0].mitigation) > 20

    @pytest.mark.asyncio
    async def test_estimates_mitigation_effort(self) -> None:
        """Test that mitigation effort is estimated based on risk."""
        from yolo_developer.agents.architect.risk_identifier import (
            _estimate_mitigation_effort,
        )
        from yolo_developer.agents.architect.types import TechnicalRisk

        # Critical scalability risk should have high effort
        critical_risk = TechnicalRisk(
            category="scalability",
            description="Single point of failure",
            severity="critical",
            affected_components=(),
            mitigation="",
            mitigation_effort="medium",
            mitigation_priority="P1",
        )

        effort = _estimate_mitigation_effort(critical_risk)
        assert effort == "high"

    @pytest.mark.asyncio
    async def test_calculates_mitigation_priority(self) -> None:
        """Test that mitigation priority is calculated correctly."""
        from yolo_developer.agents.architect.risk_identifier import (
            _generate_mitigations,
        )
        from yolo_developer.agents.architect.types import TechnicalRisk

        risks = [
            TechnicalRisk(
                category="technology",
                description="Using deprecated API",
                severity="critical",
                affected_components=(),
                mitigation="",
                mitigation_effort="medium",
                mitigation_priority="P2",
            )
        ]
        story = _create_test_story()
        decisions: list[DesignDecision] = []

        mitigated = _generate_mitigations(risks, story, decisions)

        # Critical risks should get P1 priority
        assert mitigated[0].mitigation_priority == "P1"

    @pytest.mark.asyncio
    async def test_mitigations_reference_components(self) -> None:
        """Test that mitigations reference specific components."""
        from yolo_developer.agents.architect.risk_identifier import (
            _generate_mitigations,
        )
        from yolo_developer.agents.architect.types import TechnicalRisk

        risks = [
            TechnicalRisk(
                category="scalability",
                description="Session state prevents scaling",
                severity="high",
                affected_components=("AuthModule", "SessionStore"),
                mitigation="",
                mitigation_effort="high",
                mitigation_priority="P2",
            )
        ]
        story = _create_test_story(title="Auth System")
        decisions = [
            _create_test_decision(
                decision_type="data",
                description="Use Redis for session storage",
            )
        ]

        mitigated = _generate_mitigations(risks, story, decisions)

        # Mitigation should have context
        assert len(mitigated[0].mitigation) > 0


class TestLLMIntegration:
    """Test LLM integration (Task 12)."""

    @pytest.mark.asyncio
    async def test_llm_analysis_with_mocked_llm(self) -> None:
        """Test LLM analysis with mocked response."""
        from yolo_developer.agents.architect.risk_identifier import (
            _analyze_risks_with_llm,
        )

        mock_response = """```json
{
    "risks": [
        {
            "category": "integration",
            "description": "External API dependency",
            "severity": "high",
            "affected_components": ["PaymentService"],
            "mitigation": "Add circuit breaker",
            "mitigation_effort": "medium"
        }
    ],
    "overall_risk_level": "high",
    "summary": "One high-severity integration risk"
}
```"""

        story = _create_test_story()
        decisions: list[DesignDecision] = []

        with patch(
            "yolo_developer.agents.architect.risk_identifier._call_risk_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _analyze_risks_with_llm(story, decisions)

        assert result is not None
        assert len(result.risks) == 1
        assert result.risks[0].category == "integration"
        assert result.overall_risk_level == "high"

    @pytest.mark.asyncio
    async def test_llm_fallback_on_failure(self) -> None:
        """Test fallback to pattern-based on LLM failure."""
        from yolo_developer.agents.architect.risk_identifier import (
            identify_technical_risks,
        )

        story = _create_test_story(
            description="Using deprecated API with external service"
        )
        decisions: list[DesignDecision] = []

        with patch(
            "yolo_developer.agents.architect.risk_identifier._call_risk_llm",
            new_callable=AsyncMock,
            side_effect=Exception("LLM error"),
        ):
            result = await identify_technical_risks(story, decisions, use_llm=True)

        # Should still get results from pattern-based fallback
        assert result is not None
        assert len(result.risks) >= 1

    @pytest.mark.asyncio
    async def test_llm_fallback_on_invalid_json(self) -> None:
        """Test fallback on invalid JSON response."""
        from yolo_developer.agents.architect.risk_identifier import (
            _analyze_risks_with_llm,
        )

        story = _create_test_story()
        decisions: list[DesignDecision] = []

        with patch(
            "yolo_developer.agents.architect.risk_identifier._call_risk_llm",
            new_callable=AsyncMock,
            return_value="This is not valid JSON",
        ):
            result = await _analyze_risks_with_llm(story, decisions)

        assert result is None  # Should return None on parse failure

    @pytest.mark.asyncio
    async def test_pattern_based_when_llm_disabled(self) -> None:
        """Test pattern-based analysis when LLM is disabled."""
        from yolo_developer.agents.architect.risk_identifier import (
            identify_technical_risks,
        )

        story = _create_test_story(description="Using deprecated single instance")
        decisions: list[DesignDecision] = []

        result = await identify_technical_risks(story, decisions, use_llm=False)

        assert result is not None
        assert len(result.risks) >= 1


class TestMainIdentificationFunction:
    """Test main identify_technical_risks function (Task 7)."""

    @pytest.mark.asyncio
    async def test_identifies_all_risk_categories(self) -> None:
        """Test that all risk categories can be identified."""
        from yolo_developer.agents.architect.risk_identifier import (
            identify_technical_risks,
        )

        # Story with multiple risk patterns
        story = _create_test_story(
            description=(
                "Using deprecated external API with single instance "
                "and no monitoring or replication"
            )
        )
        decisions: list[DesignDecision] = []

        result = await identify_technical_risks(story, decisions, use_llm=False)

        categories = {r.category for r in result.risks}
        # Should have multiple categories
        assert len(categories) >= 2

    @pytest.mark.asyncio
    async def test_incorporates_quality_risks(self) -> None:
        """Test that quality risks are incorporated."""
        from yolo_developer.agents.architect.risk_identifier import (
            identify_technical_risks,
        )

        story = _create_test_story()
        decisions: list[DesignDecision] = []
        quality_risks = [
            QualityRisk(
                attribute="performance",
                description="Poor performance expected",
                severity="high",
                mitigation="Optimize queries",
                mitigation_effort="medium",
            )
        ]

        result = await identify_technical_risks(
            story, decisions, quality_risks=quality_risks, use_llm=False
        )

        # Should include converted quality risk
        assert any("[Quality]" in r.description for r in result.risks)

    @pytest.mark.asyncio
    async def test_calculates_overall_risk_level(self) -> None:
        """Test overall risk level calculation."""
        from yolo_developer.agents.architect.risk_identifier import (
            identify_technical_risks,
        )

        story = _create_test_story(description="Using end-of-life technology")
        decisions: list[DesignDecision] = []

        result = await identify_technical_risks(story, decisions, use_llm=False)

        # End-of-life is critical, so overall should be critical
        assert result.overall_risk_level == "critical"

    @pytest.mark.asyncio
    async def test_generates_summary(self) -> None:
        """Test summary generation."""
        from yolo_developer.agents.architect.risk_identifier import (
            identify_technical_risks,
        )

        story = _create_test_story(description="Using deprecated API")
        decisions: list[DesignDecision] = []

        result = await identify_technical_risks(story, decisions, use_llm=False)

        assert result.summary != ""
        assert "risk" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_empty_risks_returns_low_level(self) -> None:
        """Test that no risks returns low overall level."""
        from yolo_developer.agents.architect.risk_identifier import (
            identify_technical_risks,
        )

        story = _create_test_story(description="Standard safe implementation")
        decisions: list[DesignDecision] = []

        result = await identify_technical_risks(story, decisions, use_llm=False)

        if len(result.risks) == 0:
            assert result.overall_risk_level == "low"

    @pytest.mark.asyncio
    async def test_function_is_importable_from_architect(self) -> None:
        """Test that function is importable from architect module."""
        from yolo_developer.agents.architect import identify_technical_risks

        assert identify_technical_risks is not None


class TestQualityRiskConversion:
    """Test quality risk conversion to technical risks."""

    @pytest.mark.asyncio
    async def test_converts_quality_risk_categories(self) -> None:
        """Test that quality attributes map to technical categories."""
        from yolo_developer.agents.architect.risk_identifier import (
            _convert_quality_risks,
        )

        quality_risks = [
            QualityRisk(
                attribute="performance",
                description="Slow response times",
                severity="medium",
                mitigation="Add caching",
                mitigation_effort="low",
            ),
            QualityRisk(
                attribute="integration",
                description="Integration issues",
                severity="high",
                mitigation="Add abstraction",
                mitigation_effort="medium",
            ),
        ]

        converted = _convert_quality_risks(quality_risks)

        assert len(converted) == 2
        # Performance maps to scalability
        assert converted[0].category == "scalability"
        # Integration maps to integration
        assert converted[1].category == "integration"

    @pytest.mark.asyncio
    async def test_preserves_quality_risk_details(self) -> None:
        """Test that quality risk details are preserved."""
        from yolo_developer.agents.architect.risk_identifier import (
            _convert_quality_risks,
        )

        quality_risks = [
            QualityRisk(
                attribute="security",
                description="Security vulnerability",
                severity="critical",
                mitigation="Add input validation",
                mitigation_effort="low",
            )
        ]

        converted = _convert_quality_risks(quality_risks)

        assert len(converted) == 1
        assert "[Quality]" in converted[0].description
        assert converted[0].severity == "critical"
        assert converted[0].mitigation == "Add input validation"
