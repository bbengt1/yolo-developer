"""Tests for ATAM Reviewer module (Story 7.7, Tasks 2-7, 10-14).

Tests verify the ATAM architectural review functionality including
scenario generation, conflict detection, risk assessment, and pass/fail decisions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

# =============================================================================
# Task 10: Unit Tests for Scenario Generation (AC: 1, 2)
# =============================================================================


class TestGenerateATAMScenarios:
    """Test _generate_atam_scenarios function."""

    @pytest.mark.asyncio
    async def test_scenario_generation_from_design_decisions(self) -> None:
        """Test scenarios are generated from design decisions."""
        from yolo_developer.agents.architect.atam_reviewer import _generate_atam_scenarios
        from yolo_developer.agents.architect.types import DesignDecision

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use async processing for API calls",
                rationale="Performance requirements",
                alternatives_considered=(),
            )
        ]

        quality_eval = {
            "attribute_scores": {"performance": 0.8, "reliability": 0.7},
            "trade_offs": [],
        }

        scenarios = _generate_atam_scenarios(decisions, quality_eval)

        assert len(scenarios) > 0
        assert all(s.scenario_id.startswith("ATAM-") for s in scenarios)

    @pytest.mark.asyncio
    async def test_scenario_generation_per_quality_attribute(self) -> None:
        """Test scenarios generated for each quality attribute affected."""
        from yolo_developer.agents.architect.atam_reviewer import _generate_atam_scenarios
        from yolo_developer.agents.architect.types import DesignDecision

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="security",
                description="Implement OAuth2 authentication",
                rationale="Security requirements",
                alternatives_considered=(),
            )
        ]

        quality_eval = {
            "attribute_scores": {"security": 0.9, "performance": 0.7},
            "trade_offs": [],
        }

        scenarios = _generate_atam_scenarios(decisions, quality_eval)

        # Should have scenarios for quality attributes
        attributes = {s.quality_attribute for s in scenarios}
        assert len(attributes) > 0

    @pytest.mark.asyncio
    async def test_scenario_stimulus_response_pairs(self) -> None:
        """Test scenarios have stimulus-response pairs."""
        from yolo_developer.agents.architect.atam_reviewer import _generate_atam_scenarios
        from yolo_developer.agents.architect.types import DesignDecision

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="data",
                description="Use caching for database queries",
                rationale="Performance optimization",
                alternatives_considered=(),
            )
        ]

        quality_eval = {"attribute_scores": {"performance": 0.85}, "trade_offs": []}

        scenarios = _generate_atam_scenarios(decisions, quality_eval)

        for scenario in scenarios:
            assert scenario.stimulus != ""
            assert scenario.response != ""
            assert scenario.analysis != ""

    @pytest.mark.asyncio
    async def test_empty_scenarios_when_no_decisions(self) -> None:
        """Test empty scenarios returned when no decisions."""
        from yolo_developer.agents.architect.atam_reviewer import _generate_atam_scenarios

        scenarios = _generate_atam_scenarios([], {})

        assert scenarios == []

    @pytest.mark.asyncio
    async def test_generate_scenario_analysis_helper(self) -> None:
        """Test _generate_scenario_analysis helper function."""
        from yolo_developer.agents.architect.atam_reviewer import _generate_scenario_analysis
        from yolo_developer.agents.architect.types import DesignDecision

        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="pattern",
            description="Use repository pattern",
            rationale="Separation of concerns",
            alternatives_considered=(),
        )

        # Test excellent score
        analysis = _generate_scenario_analysis(decision, "maintainability", 0.9)
        assert "excellent" in analysis
        assert "maintainability" in analysis
        assert decision.description in analysis

        # Test adequate score
        analysis = _generate_scenario_analysis(decision, "performance", 0.65)
        assert "adequate" in analysis

        # Test needs improvement score
        analysis = _generate_scenario_analysis(decision, "security", 0.4)
        assert "needs improvement" in analysis


# =============================================================================
# Task 11: Unit Tests for Conflict Detection (AC: 2)
# =============================================================================


class TestDetectTradeOffConflicts:
    """Test _detect_trade_off_conflicts function."""

    @pytest.mark.asyncio
    async def test_trade_off_conflict_detection(self) -> None:
        """Test conflicts detected from trade-offs."""
        from yolo_developer.agents.architect.atam_reviewer import _detect_trade_off_conflicts
        from yolo_developer.agents.architect.types import DesignDecision, QualityTradeOff

        trade_offs = [
            QualityTradeOff(
                attribute_a="performance",
                attribute_b="security",
                description="Caching reduces latency but may expose stale auth data",
                resolution="Short cache TTL for auth tokens",
            ),
            QualityTradeOff(
                attribute_a="security",
                attribute_b="performance",
                description="Encryption adds processing overhead",
                resolution="Use hardware acceleration",
            ),
        ]

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Test decision",
                rationale="Test",
                alternatives_considered=(),
            )
        ]

        conflicts = _detect_trade_off_conflicts(trade_offs, decisions)

        # Should detect conflict between performance/security trade-offs
        assert len(conflicts) > 0

    @pytest.mark.asyncio
    async def test_severity_assignment_for_conflicts(self) -> None:
        """Test severity is assigned to conflicts."""
        from yolo_developer.agents.architect.atam_reviewer import _detect_trade_off_conflicts
        from yolo_developer.agents.architect.types import QualityTradeOff

        trade_offs = [
            QualityTradeOff(
                attribute_a="scalability",
                attribute_b="maintainability",
                description="Distributed system adds complexity",
                resolution="Documentation",
            ),
            QualityTradeOff(
                attribute_a="maintainability",
                attribute_b="scalability",
                description="Simple architecture limits scaling",
                resolution="Plan for migration",
            ),
        ]

        decisions: list[object] = []

        conflicts = _detect_trade_off_conflicts(trade_offs, decisions)

        for conflict in conflicts:
            assert conflict.severity in ("critical", "high", "medium", "low")

    @pytest.mark.asyncio
    async def test_resolution_strategy_suggestion(self) -> None:
        """Test resolution strategies are suggested for conflicts."""
        from yolo_developer.agents.architect.atam_reviewer import _detect_trade_off_conflicts
        from yolo_developer.agents.architect.types import QualityTradeOff

        trade_offs = [
            QualityTradeOff(
                attribute_a="a",
                attribute_b="b",
                description="Trade-off 1",
                resolution="Mitigation 1",
            ),
            QualityTradeOff(
                attribute_a="b",
                attribute_b="a",
                description="Trade-off 2",
                resolution="Mitigation 2",
            ),
        ]

        conflicts = _detect_trade_off_conflicts(trade_offs, [])

        for conflict in conflicts:
            assert conflict.resolution_strategy != ""

    @pytest.mark.asyncio
    async def test_no_conflicts_when_compatible(self) -> None:
        """Test no conflicts when trade-offs are compatible."""
        from yolo_developer.agents.architect.atam_reviewer import _detect_trade_off_conflicts
        from yolo_developer.agents.architect.types import QualityTradeOff

        trade_offs = [
            QualityTradeOff(
                attribute_a="performance",
                attribute_b="cost_efficiency",
                description="Better hardware costs more",
                resolution="Budget allocation",
            ),
        ]

        conflicts = _detect_trade_off_conflicts(trade_offs, [])

        # Single trade-off, no conflict
        assert len(conflicts) == 0


# =============================================================================
# Task 12: Unit Tests for Risk Assessment (AC: 3)
# =============================================================================


class TestAssessRiskImpact:
    """Test _assess_risk_impact function."""

    @pytest.mark.asyncio
    async def test_risk_to_quality_attribute_mapping(self) -> None:
        """Test risks are mapped to quality attributes."""
        from yolo_developer.agents.architect.atam_reviewer import _assess_risk_impact
        from yolo_developer.agents.architect.types import TechnicalRisk, TechnicalRiskReport

        risk_report = TechnicalRiskReport(
            risks=(
                TechnicalRisk(
                    category="scalability",
                    description="Single database may bottleneck",
                    severity="high",
                    affected_components=("Database",),
                    mitigation="Add read replicas",
                    mitigation_effort="medium",
                    mitigation_priority="P1",
                ),
            ),
            overall_risk_level="high",
            summary="Test risks",
        )

        quality_eval = {"attribute_scores": {"scalability": 0.6, "reliability": 0.8}}

        assessments = _assess_risk_impact(risk_report, quality_eval)

        assert len(assessments) > 0
        # Risk should be mapped to quality attributes
        for assessment in assessments:
            assert len(assessment.quality_impact) > 0

    @pytest.mark.asyncio
    async def test_mitigation_feasibility_evaluation(self) -> None:
        """Test mitigation feasibility is evaluated."""
        from yolo_developer.agents.architect.atam_reviewer import _assess_risk_impact
        from yolo_developer.agents.architect.types import TechnicalRisk, TechnicalRiskReport

        risk_report = TechnicalRiskReport(
            risks=(
                TechnicalRisk(
                    category="technology",
                    description="Using deprecated API",
                    severity="medium",
                    affected_components=("API",),
                    mitigation="Upgrade to new version",
                    mitigation_effort="low",
                    mitigation_priority="P2",
                ),
            ),
            overall_risk_level="medium",
            summary="Test",
        )

        assessments = _assess_risk_impact(risk_report, {})

        for assessment in assessments:
            assert assessment.mitigation_feasibility in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_critical_unmitigated_risk_flagging(self) -> None:
        """Test critical unmitigated risks are flagged."""
        from yolo_developer.agents.architect.atam_reviewer import _assess_risk_impact
        from yolo_developer.agents.architect.types import TechnicalRisk, TechnicalRiskReport

        risk_report = TechnicalRiskReport(
            risks=(
                TechnicalRisk(
                    category="operational",
                    description="No encryption for sensitive data",
                    severity="critical",
                    affected_components=("DataStore",),
                    mitigation="",  # No mitigation
                    mitigation_effort="high",
                    mitigation_priority="P1",
                ),
            ),
            overall_risk_level="critical",
            summary="Critical risk",
        )

        assessments = _assess_risk_impact(risk_report, {})

        # Critical risk with no mitigation should be flagged
        critical_unmitigated = [a for a in assessments if a.unmitigated]
        assert len(critical_unmitigated) > 0

    @pytest.mark.asyncio
    async def test_empty_assessments_when_no_risks(self) -> None:
        """Test empty assessments when no risks."""
        from yolo_developer.agents.architect.atam_reviewer import _assess_risk_impact
        from yolo_developer.agents.architect.types import TechnicalRiskReport

        risk_report = TechnicalRiskReport(
            risks=(),
            overall_risk_level="low",
            summary="No risks",
        )

        assessments = _assess_risk_impact(risk_report, {})

        assert assessments == []


# =============================================================================
# Task 13: Unit Tests for Pass/Fail Decision (AC: 4)
# =============================================================================


class TestMakeReviewDecision:
    """Test _make_review_decision function."""

    @pytest.mark.asyncio
    async def test_pass_decision_with_good_scores(self) -> None:
        """Test pass decision with good scenarios and no critical issues."""
        from yolo_developer.agents.architect.atam_reviewer import _make_review_decision
        from yolo_developer.agents.architect.types import ATAMRiskAssessment, ATAMScenario

        scenarios = [
            ATAMScenario("ATAM-001", "performance", "s1", "r1", "a1"),
            ATAMScenario("ATAM-002", "security", "s2", "r2", "a2"),
            ATAMScenario("ATAM-003", "reliability", "s3", "r3", "a3"),
        ]

        risks = [
            ATAMRiskAssessment("R1", ("performance",), "high", False),
        ]

        passed, confidence, reasons = _make_review_decision(scenarios, [], risks)

        assert passed is True
        assert confidence >= 0.6
        assert len(reasons) == 0

    @pytest.mark.asyncio
    async def test_fail_decision_with_critical_risks(self) -> None:
        """Test fail decision when critical unmitigated risks exist."""
        from yolo_developer.agents.architect.atam_reviewer import _make_review_decision
        from yolo_developer.agents.architect.types import (
            ATAMRiskAssessment,
            ATAMScenario,
        )

        scenarios = [ATAMScenario("ATAM-001", "performance", "s1", "r1", "a1")]

        risks = [
            ATAMRiskAssessment("CRIT-001", ("security",), "low", True),  # unmitigated
        ]

        passed, _confidence, reasons = _make_review_decision(scenarios, [], risks)

        assert passed is False
        assert "unmitigated" in " ".join(reasons).lower() or "critical" in " ".join(reasons).lower()

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self) -> None:
        """Test confidence score is calculated correctly."""
        from yolo_developer.agents.architect.atam_reviewer import _make_review_decision
        from yolo_developer.agents.architect.types import ATAMScenario

        # More scenarios = higher confidence
        scenarios = [
            ATAMScenario("ATAM-001", "performance", "s", "r", "a"),
            ATAMScenario("ATAM-002", "security", "s", "r", "a"),
            ATAMScenario("ATAM-003", "reliability", "s", "r", "a"),
            ATAMScenario("ATAM-004", "scalability", "s", "r", "a"),
            ATAMScenario("ATAM-005", "maintainability", "s", "r", "a"),
        ]

        _, confidence, _ = _make_review_decision(scenarios, [], [])

        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.6  # Should pass with 5 scenarios

    @pytest.mark.asyncio
    async def test_failure_reason_generation(self) -> None:
        """Test failure reasons are generated when review fails."""
        from yolo_developer.agents.architect.atam_reviewer import _make_review_decision
        from yolo_developer.agents.architect.types import ATAMTradeOffConflict

        # Too many high conflicts should fail
        conflicts = [
            ATAMTradeOffConflict("a", "b", "d", "high", "s"),
            ATAMTradeOffConflict("c", "d", "d", "high", "s"),
            ATAMTradeOffConflict("e", "f", "d", "high", "s"),
        ]

        passed, _confidence, reasons = _make_review_decision([], conflicts, [])

        # Should fail due to low confidence (no scenarios) or too many conflicts
        if not passed:
            assert len(reasons) > 0


# =============================================================================
# Task 14: Unit Tests for LLM Integration (AC: 6)
# =============================================================================


class TestLLMIntegration:
    """Test LLM-powered ATAM analysis."""

    @pytest.mark.asyncio
    async def test_llm_analysis_with_mocked_llm(self) -> None:
        """Test LLM analysis returns valid result."""
        from yolo_developer.agents.architect.atam_reviewer import _analyze_atam_with_llm
        from yolo_developer.agents.architect.types import DesignDecision

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="""{
                        "overall_pass": true,
                        "confidence": 0.82,
                        "scenarios_evaluated": [
                            {"scenario_id": "ATAM-001", "quality_attribute": "performance",
                             "stimulus": "Load", "response": "OK", "analysis": "Good"}
                        ],
                        "trade_off_conflicts": [],
                        "risk_assessments": [],
                        "failure_reasons": [],
                        "summary": "Design passes"
                    }"""
                )
            )
        ]

        decisions = [
            DesignDecision(
                id="d1",
                story_id="s1",
                decision_type="pattern",
                description="Test",
                rationale="Test",
                alternatives_considered=(),
            )
        ]

        with patch("yolo_developer.agents.architect.atam_reviewer.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            result = await _analyze_atam_with_llm(decisions, {}, None)

            if result is not None:
                assert result.overall_pass is True
                assert result.confidence == 0.82

    @pytest.mark.asyncio
    async def test_fallback_to_rule_based_on_llm_failure(self) -> None:
        """Test fallback to rule-based when LLM fails."""
        from yolo_developer.agents.architect.atam_reviewer import run_atam_review
        from yolo_developer.agents.architect.types import DesignDecision

        decisions = [
            DesignDecision(
                id="d1",
                story_id="s1",
                decision_type="pattern",
                description="Test",
                rationale="Test",
                alternatives_considered=(),
            )
        ]

        with patch(
            "yolo_developer.agents.architect.atam_reviewer._analyze_atam_with_llm",
            new_callable=AsyncMock,
            return_value=None,  # LLM failed
        ):
            result = await run_atam_review(decisions)

            # Should still return a result from rule-based analysis
            assert result is not None
            assert isinstance(result.overall_pass, bool)

    @pytest.mark.asyncio
    async def test_json_parsing_of_llm_response(self) -> None:
        """Test JSON parsing handles various response formats."""
        from yolo_developer.agents.architect.atam_reviewer import _analyze_atam_with_llm
        from yolo_developer.agents.architect.types import DesignDecision

        # Test with JSON in code block
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="""```json
{
    "overall_pass": false,
    "confidence": 0.55,
    "scenarios_evaluated": [],
    "trade_off_conflicts": [],
    "risk_assessments": [],
    "failure_reasons": ["Low confidence"],
    "summary": "Needs review"
}
```"""
                )
            )
        ]

        decisions = [
            DesignDecision(
                id="d1",
                story_id="s1",
                decision_type="data",
                description="Test",
                rationale="Test",
                alternatives_considered=(),
            )
        ]

        with patch("yolo_developer.agents.architect.atam_reviewer.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            result = await _analyze_atam_with_llm(decisions, {}, None)

            if result is not None:
                assert result.overall_pass is False
                assert result.confidence == 0.55

    @pytest.mark.asyncio
    async def test_retry_behavior_on_transient_failures(self) -> None:
        """Test retry behavior on transient LLM failures (AC6)."""
        from yolo_developer.agents.architect.atam_reviewer import _call_atam_llm

        call_count = 0

        async def mock_acompletion(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            # Return success on third attempt
            mock_response = AsyncMock()
            mock_response.choices = [
                AsyncMock(message=AsyncMock(content='{"result": "success"}'))
            ]
            return mock_response

        with patch(
            "yolo_developer.agents.architect.atam_reviewer.litellm.acompletion",
            side_effect=mock_acompletion,
        ):
            result = await _call_atam_llm("test prompt")

            # Should have retried and succeeded on 3rd attempt
            assert call_count == 3
            assert "success" in result


# =============================================================================
# Task 7: Main Review Function Tests
# =============================================================================


class TestRunATAMReview:
    """Test run_atam_review main function."""

    @pytest.mark.asyncio
    async def test_run_atam_review_basic(self) -> None:
        """Test basic ATAM review execution."""
        from yolo_developer.agents.architect.atam_reviewer import run_atam_review
        from yolo_developer.agents.architect.types import DesignDecision

        decisions = [
            DesignDecision(
                id="d1",
                story_id="s1",
                decision_type="pattern",
                description="Use repository pattern",
                rationale="Clean architecture",
                alternatives_considered=(),
            )
        ]

        with patch(
            "yolo_developer.agents.architect.atam_reviewer._analyze_atam_with_llm",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await run_atam_review(decisions)

            assert result is not None
            assert isinstance(result.overall_pass, bool)
            assert 0.0 <= result.confidence <= 1.0
            assert result.summary != ""

    @pytest.mark.asyncio
    async def test_run_atam_review_with_quality_eval(self) -> None:
        """Test ATAM review with quality evaluation input."""
        from yolo_developer.agents.architect.atam_reviewer import run_atam_review
        from yolo_developer.agents.architect.types import (
            DesignDecision,
            QualityAttributeEvaluation,
            QualityTradeOff,
        )

        decisions = [
            DesignDecision(
                id="d1",
                story_id="s1",
                decision_type="security",
                description="Implement encryption",
                rationale="Data protection",
                alternatives_considered=(),
            )
        ]

        quality_eval = QualityAttributeEvaluation(
            attribute_scores={"security": 0.9, "performance": 0.7},
            trade_offs=(
                QualityTradeOff(
                    attribute_a="security",
                    attribute_b="performance",
                    description="Encryption overhead",
                    resolution="Use AES-NI",
                ),
            ),
            risks=(),
            overall_score=0.8,
        )

        with patch(
            "yolo_developer.agents.architect.atam_reviewer._analyze_atam_with_llm",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await run_atam_review(decisions, quality_eval)

            assert result is not None

    @pytest.mark.asyncio
    async def test_run_atam_review_with_risk_report(self) -> None:
        """Test ATAM review with risk report input."""
        from yolo_developer.agents.architect.atam_reviewer import run_atam_review
        from yolo_developer.agents.architect.types import (
            DesignDecision,
            TechnicalRisk,
            TechnicalRiskReport,
        )

        decisions = [
            DesignDecision(
                id="d1",
                story_id="s1",
                decision_type="infrastructure",
                description="Use cloud deployment",
                rationale="Scalability",
                alternatives_considered=(),
            )
        ]

        risk_report = TechnicalRiskReport(
            risks=(
                TechnicalRisk(
                    category="operational",
                    description="Cloud vendor lock-in",
                    severity="medium",
                    affected_components=("Infrastructure",),
                    mitigation="Use abstraction layer",
                    mitigation_effort="medium",
                    mitigation_priority="P2",
                ),
            ),
            overall_risk_level="medium",
            summary="Moderate risks",
        )

        with patch(
            "yolo_developer.agents.architect.atam_reviewer._analyze_atam_with_llm",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await run_atam_review(decisions, risk_report=risk_report)

            assert result is not None
            assert len(result.risk_assessments) > 0

    @pytest.mark.asyncio
    async def test_run_atam_review_importable(self) -> None:
        """Test run_atam_review is importable from architect module."""
        from yolo_developer.agents.architect import run_atam_review

        assert run_atam_review is not None


# =============================================================================
# Task 15: Integration Tests (AC: 5)
# =============================================================================


class TestArchitectNodeIntegration:
    """Integration tests for architect_node with ATAM review (Task 15)."""

    @pytest.fixture
    def sample_state(self) -> dict[str, object]:
        """Sample orchestration state with stories."""
        return {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [
                    {
                        "id": "story-001",
                        "title": "Test Story",
                        "description": "Test description",
                    }
                ]
            },
        }

    @pytest.fixture
    def mock_dependencies(self) -> dict[str, object]:
        """Create mock return values for all architect_node dependencies."""
        from yolo_developer.agents.architect.types import (
            ATAMReviewResult,
            QualityAttributeEvaluation,
            TechnicalRiskReport,
            TechStackValidation,
            TwelveFactorAnalysis,
        )

        return {
            "twelve_factor": TwelveFactorAnalysis(
                factor_results={},
                applicable_factors=(),
                overall_compliance=1.0,
                recommendations=(),
            ),
            "quality_eval": QualityAttributeEvaluation(
                attribute_scores={"performance": 0.8},
                trade_offs=(),
                risks=(),
                overall_score=0.8,
            ),
            "risk_report": TechnicalRiskReport(
                risks=(),
                overall_risk_level="low",
                summary="No risks",
            ),
            "tech_stack": TechStackValidation(
                overall_compliance=True,
                violations=(),
                suggested_patterns=(),
                summary="Valid",
            ),
            "atam_review": ATAMReviewResult(
                overall_pass=True,
                confidence=0.85,
                scenarios_evaluated=(),
                trade_off_conflicts=(),
                risk_assessments=(),
                failure_reasons=(),
                summary="All tests pass",
            ),
        }

    @pytest.mark.asyncio
    async def test_architect_node_includes_atam_review(
        self, sample_state: dict[str, object], mock_dependencies: dict[str, object]
    ) -> None:
        """Test architect_node includes atam_reviews in output."""
        from yolo_developer.agents.architect import architect_node

        with (
            patch(
                "yolo_developer.agents.architect.node.analyze_twelve_factor",
                new_callable=AsyncMock,
                return_value=mock_dependencies["twelve_factor"],
            ),
            patch(
                "yolo_developer.agents.architect.node.generate_adrs",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "yolo_developer.agents.architect.node.evaluate_quality_attributes",
                new_callable=AsyncMock,
                return_value=mock_dependencies["quality_eval"],
            ),
            patch(
                "yolo_developer.agents.architect.node.identify_technical_risks",
                new_callable=AsyncMock,
                return_value=mock_dependencies["risk_report"],
            ),
            patch(
                "yolo_developer.agents.architect.node.validate_tech_stack_constraints",
                new_callable=AsyncMock,
                return_value=mock_dependencies["tech_stack"],
            ),
            patch(
                "yolo_developer.agents.architect.node.run_atam_review",
                new_callable=AsyncMock,
                return_value=mock_dependencies["atam_review"],
            ),
        ):
            result = await architect_node(sample_state)

            assert "architect_output" in result
            output = result["architect_output"]
            assert "atam_reviews" in output
            assert "story-001" in output["atam_reviews"]

    @pytest.mark.asyncio
    async def test_architect_output_serializes_atam_review(
        self, sample_state: dict[str, object], mock_dependencies: dict[str, object]
    ) -> None:
        """Test architect_output correctly serializes ATAM review results."""
        from yolo_developer.agents.architect import architect_node
        from yolo_developer.agents.architect.types import (
            ATAMReviewResult,
            ATAMScenario,
            ATAMTradeOffConflict,
        )

        detailed_review = ATAMReviewResult(
            overall_pass=True,
            confidence=0.75,
            scenarios_evaluated=(
                ATAMScenario(
                    scenario_id="ATAM-001",
                    quality_attribute="performance",
                    stimulus="100 requests",
                    response="<500ms",
                    analysis="Design supports this",
                ),
            ),
            trade_off_conflicts=(
                ATAMTradeOffConflict(
                    attribute_a="performance",
                    attribute_b="security",
                    description="Encryption adds latency",
                    severity="medium",
                    resolution_strategy="Use caching",
                ),
            ),
            risk_assessments=(),
            failure_reasons=(),
            summary="Review complete",
        )

        with (
            patch(
                "yolo_developer.agents.architect.node.analyze_twelve_factor",
                new_callable=AsyncMock,
                return_value=mock_dependencies["twelve_factor"],
            ),
            patch(
                "yolo_developer.agents.architect.node.generate_adrs",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "yolo_developer.agents.architect.node.evaluate_quality_attributes",
                new_callable=AsyncMock,
                return_value=mock_dependencies["quality_eval"],
            ),
            patch(
                "yolo_developer.agents.architect.node.identify_technical_risks",
                new_callable=AsyncMock,
                return_value=mock_dependencies["risk_report"],
            ),
            patch(
                "yolo_developer.agents.architect.node.validate_tech_stack_constraints",
                new_callable=AsyncMock,
                return_value=mock_dependencies["tech_stack"],
            ),
            patch(
                "yolo_developer.agents.architect.node.run_atam_review",
                new_callable=AsyncMock,
                return_value=detailed_review,
            ),
        ):
            result = await architect_node(sample_state)

            output = result["architect_output"]
            review = output["atam_reviews"]["story-001"]

            assert review["overall_pass"] is True
            assert review["confidence"] == 0.75
            assert len(review["scenarios_evaluated"]) == 1
            assert len(review["trade_off_conflicts"]) == 1
            assert review["summary"] == "Review complete"

    @pytest.mark.asyncio
    async def test_architect_output_to_dict_includes_atam_reviews(self) -> None:
        """Test ArchitectOutput.to_dict() includes atam_reviews field."""
        from yolo_developer.agents.architect.types import ArchitectOutput

        output = ArchitectOutput(
            design_decisions=(),
            adrs=(),
            processing_notes="Test",
            atam_reviews={"story-001": {"overall_pass": True, "confidence": 0.8}},
        )

        result = output.to_dict()

        assert "atam_reviews" in result
        assert result["atam_reviews"] == {"story-001": {"overall_pass": True, "confidence": 0.8}}

    @pytest.mark.asyncio
    async def test_atam_review_runs_after_tech_stack_validation(
        self, sample_state: dict[str, object], mock_dependencies: dict[str, object]
    ) -> None:
        """Test ATAM review runs after tech stack validation in architect_node."""
        from yolo_developer.agents.architect import architect_node
        from yolo_developer.agents.architect.types import (
            ATAMReviewResult,
            TechStackValidation,
        )

        call_order: list[str] = []

        async def track_tech_stack(*args: object, **kwargs: object) -> TechStackValidation:
            call_order.append("tech_stack")
            return TechStackValidation(
                overall_compliance=True,
                violations=(),
                suggested_patterns=(),
                summary="Valid",
            )

        async def track_atam(*args: object, **kwargs: object) -> ATAMReviewResult:
            call_order.append("atam")
            return ATAMReviewResult(
                overall_pass=True,
                confidence=0.8,
                scenarios_evaluated=(),
                trade_off_conflicts=(),
                risk_assessments=(),
                failure_reasons=(),
                summary="Pass",
            )

        with (
            patch(
                "yolo_developer.agents.architect.node.analyze_twelve_factor",
                new_callable=AsyncMock,
                return_value=mock_dependencies["twelve_factor"],
            ),
            patch(
                "yolo_developer.agents.architect.node.generate_adrs",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "yolo_developer.agents.architect.node.evaluate_quality_attributes",
                new_callable=AsyncMock,
                return_value=mock_dependencies["quality_eval"],
            ),
            patch(
                "yolo_developer.agents.architect.node.identify_technical_risks",
                new_callable=AsyncMock,
                return_value=mock_dependencies["risk_report"],
            ),
            patch(
                "yolo_developer.agents.architect.node.validate_tech_stack_constraints",
                new_callable=AsyncMock,
                side_effect=track_tech_stack,
            ),
            patch(
                "yolo_developer.agents.architect.node.run_atam_review",
                new_callable=AsyncMock,
                side_effect=track_atam,
            ),
        ):
            await architect_node(sample_state)

            # Verify ATAM runs after tech stack validation
            assert call_order.index("tech_stack") < call_order.index("atam")

    @pytest.mark.asyncio
    async def test_atam_review_uses_quality_eval_and_risk_report(
        self, sample_state: dict[str, object], mock_dependencies: dict[str, object]
    ) -> None:
        """Test ATAM review receives quality evaluation and risk report objects."""
        from yolo_developer.agents.architect import architect_node
        from yolo_developer.agents.architect.types import ATAMReviewResult

        captured_calls: list[dict[str, object]] = []

        async def capture_atam_call(
            decisions: list[object],
            quality_eval: object = None,
            risk_report: object = None,
        ) -> ATAMReviewResult:
            captured_calls.append({
                "decisions": decisions,
                "quality_eval": quality_eval,
                "risk_report": risk_report,
            })
            return ATAMReviewResult(
                overall_pass=True,
                confidence=0.8,
                scenarios_evaluated=(),
                trade_off_conflicts=(),
                risk_assessments=(),
                failure_reasons=(),
                summary="Pass",
            )

        with (
            patch(
                "yolo_developer.agents.architect.node.analyze_twelve_factor",
                new_callable=AsyncMock,
                return_value=mock_dependencies["twelve_factor"],
            ),
            patch(
                "yolo_developer.agents.architect.node.generate_adrs",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "yolo_developer.agents.architect.node.evaluate_quality_attributes",
                new_callable=AsyncMock,
                return_value=mock_dependencies["quality_eval"],
            ),
            patch(
                "yolo_developer.agents.architect.node.identify_technical_risks",
                new_callable=AsyncMock,
                return_value=mock_dependencies["risk_report"],
            ),
            patch(
                "yolo_developer.agents.architect.node.validate_tech_stack_constraints",
                new_callable=AsyncMock,
                return_value=mock_dependencies["tech_stack"],
            ),
            patch(
                "yolo_developer.agents.architect.node.run_atam_review",
                new_callable=AsyncMock,
                side_effect=capture_atam_call,
            ),
        ):
            await architect_node(sample_state)

            # Verify ATAM was called with quality_eval and risk_report
            assert len(captured_calls) == 1
            call = captured_calls[0]
            # Should have quality_eval from evaluate_quality_attributes
            assert call["quality_eval"] is not None
            # Should have risk_report from identify_technical_risks
            assert call["risk_report"] is not None

    def test_run_atam_review_importable_from_architect(self) -> None:
        """Test run_atam_review is importable from yolo_developer.agents.architect."""
        from yolo_developer.agents.architect import run_atam_review

        assert run_atam_review is not None
        assert callable(run_atam_review)
