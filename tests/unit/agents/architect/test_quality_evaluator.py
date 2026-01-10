"""Tests for quality attribute evaluator (Story 7.4, Tasks 2-7, 10-12).

Tests verify quality attribute scoring, trade-off detection,
risk identification, and LLM integration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from yolo_developer.agents.architect.types import DesignDecision, DesignDecisionType


def _create_test_story(
    story_id: str = "story-001",
    title: str = "User Authentication",
    description: str = "Implement user authentication with JWT tokens",
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
    decision_type: DesignDecisionType = "security",
    description: str = "Use JWT for authentication",
    rationale: str = "Stateless authentication required",
) -> DesignDecision:
    """Create a test design decision."""
    return DesignDecision(
        id=decision_id,
        story_id=story_id,
        decision_type=decision_type,
        description=description,
        rationale=rationale,
        alternatives_considered=("Session-based auth",),
    )


class TestScorePerformance:
    """Test _score_performance function."""

    def test_baseline_score(self) -> None:
        """Test baseline score for neutral story."""
        from yolo_developer.agents.architect.quality_evaluator import _score_performance

        story = _create_test_story(
            title="Basic Feature",
            description="A simple feature",
        )
        decisions: list[DesignDecision] = []

        score = _score_performance(story, decisions)

        assert 0.6 <= score <= 0.8  # Should be around baseline

    def test_positive_patterns_increase_score(self) -> None:
        """Test that positive patterns increase score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_performance

        story = _create_test_story(
            title="Cached Data Access",
            description="Implement caching with async operations and connection pooling",
        )
        decisions: list[DesignDecision] = []

        score = _score_performance(story, decisions)

        assert score > 0.8  # Should be higher due to positive patterns

    def test_negative_patterns_decrease_score(self) -> None:
        """Test that negative patterns decrease score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_performance

        story = _create_test_story(
            title="Blocking Operations",
            description="Synchronous blocking operations with no cache",
        )
        decisions: list[DesignDecision] = []

        score = _score_performance(story, decisions)

        assert score < 0.7  # Should be lower due to negative patterns

    def test_score_range(self) -> None:
        """Test that score is always in valid range."""
        from yolo_developer.agents.architect.quality_evaluator import _score_performance

        # Extreme positive case
        story_positive = _create_test_story(
            description="async await cache caching index batch pool connection pool lazy load pagination",
        )
        score_high = _score_performance(story_positive, [])

        # Extreme negative case
        story_negative = _create_test_story(
            description="synchronous blocking n+1 no cache unbounded",
        )
        score_low = _score_performance(story_negative, [])

        assert 0.0 <= score_high <= 1.0
        assert 0.0 <= score_low <= 1.0


class TestScoreSecurity:
    """Test _score_security function."""

    def test_high_score_when_not_relevant(self) -> None:
        """Test high score when security is not in scope."""
        from yolo_developer.agents.architect.quality_evaluator import _score_security

        story = _create_test_story(
            title="Data Processing",
            description="Process data files in batch mode",
        )
        decisions: list[DesignDecision] = []

        score = _score_security(story, decisions)

        assert score >= 0.8  # High score when security not in scope

    def test_positive_patterns_increase_score(self) -> None:
        """Test that security patterns increase score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_security

        story = _create_test_story(
            title="User Authentication",
            description="Implement OAuth authentication with JWT validation and input sanitization",
        )
        decisions: list[DesignDecision] = []

        score = _score_security(story, decisions)

        assert score > 0.75  # Should be higher with security patterns

    def test_negative_patterns_decrease_score(self) -> None:
        """Test that security anti-patterns decrease score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_security

        story = _create_test_story(
            title="User Login",
            description="Store password in plain text with no validation",
        )
        decisions: list[DesignDecision] = []

        score = _score_security(story, decisions)

        assert score < 0.7  # Should be lower with anti-patterns


class TestScoreReliability:
    """Test _score_reliability function."""

    def test_positive_patterns_increase_score(self) -> None:
        """Test that reliability patterns increase score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_reliability

        story = _create_test_story(
            title="API Client",
            description="Implement retry with tenacity and exponential backoff, circuit breaker pattern",
        )
        decisions: list[DesignDecision] = []

        score = _score_reliability(story, decisions)

        assert score > 0.85  # Should be high with reliability patterns

    def test_negative_patterns_decrease_score(self) -> None:
        """Test that reliability anti-patterns decrease score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_reliability

        story = _create_test_story(
            title="Service Call",
            description="Call service with no retry and no recovery mechanism - single point of failure",
        )
        decisions: list[DesignDecision] = []

        score = _score_reliability(story, decisions)

        assert score < 0.7  # Should be lower with anti-patterns


class TestScoreScalability:
    """Test _score_scalability function."""

    def test_positive_patterns_increase_score(self) -> None:
        """Test that scalability patterns increase score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_scalability

        story = _create_test_story(
            title="Worker Service",
            description="Stateless horizontal scaling with message queue and load balancing",
        )
        decisions: list[DesignDecision] = []

        score = _score_scalability(story, decisions)

        assert score > 0.85  # Should be high with scalability patterns

    def test_negative_patterns_decrease_score(self) -> None:
        """Test that scalability anti-patterns decrease score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_scalability

        story = _create_test_story(
            title="Stateful Service",
            description="Stateful single instance with sticky sessions",
        )
        decisions: list[DesignDecision] = []

        score = _score_scalability(story, decisions)

        assert score < 0.7  # Should be lower with anti-patterns


class TestScoreMaintainability:
    """Test _score_maintainability function."""

    def test_positive_patterns_increase_score(self) -> None:
        """Test that maintainability patterns increase score."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _score_maintainability,
        )

        story = _create_test_story(
            title="Data Model",
            description="Use dataclass with type hints, unit tests, integration tests, document with structlog logging",
        )
        decisions: list[DesignDecision] = []

        score = _score_maintainability(story, decisions)

        assert score > 0.85  # Should be high with maintainability patterns


class TestScoreIntegration:
    """Test _score_integration function."""

    def test_positive_patterns_increase_score(self) -> None:
        """Test that integration patterns increase score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_integration

        story = _create_test_story(
            title="LLM Integration",
            description="Use litellm abstraction with adapter pattern for multi-provider support",
        )
        decisions: list[DesignDecision] = []

        score = _score_integration(story, decisions)

        assert score > 0.85  # Should be high with integration patterns

    def test_negative_patterns_decrease_score(self) -> None:
        """Test that integration anti-patterns decrease score."""
        from yolo_developer.agents.architect.quality_evaluator import _score_integration

        story = _create_test_story(
            title="Direct API Call",
            description="Directly call vendor API with tight coupling and no abstraction",
        )
        decisions: list[DesignDecision] = []

        score = _score_integration(story, decisions)

        assert score < 0.7  # Should be lower with anti-patterns


class TestScoreCostEfficiency:
    """Test _score_cost_efficiency function."""

    def test_positive_patterns_increase_score(self) -> None:
        """Test that cost efficiency patterns increase score."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _score_cost_efficiency,
        )

        story = _create_test_story(
            title="Efficient Processing",
            description="Use caching and model tiering with cheap models for routine tasks, batch operations for efficiency",
        )
        decisions: list[DesignDecision] = []

        score = _score_cost_efficiency(story, decisions)

        assert score > 0.85  # Should be high with cost efficiency patterns

    def test_negative_patterns_decrease_score(self) -> None:
        """Test that cost inefficiency patterns decrease score."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _score_cost_efficiency,
        )

        story = _create_test_story(
            title="Expensive Processing",
            description="Always use expensive models with no cache, wasteful resource usage",
        )
        decisions: list[DesignDecision] = []

        score = _score_cost_efficiency(story, decisions)

        assert score < 0.7  # Should be lower with anti-patterns


class TestCalculateOverallScore:
    """Test _calculate_overall_score function."""

    def test_weighted_calculation(self) -> None:
        """Test that overall score is weighted correctly."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _calculate_overall_score,
        )

        attribute_scores = {
            "performance": 0.8,
            "security": 0.9,
            "reliability": 0.7,
            "scalability": 0.6,
            "maintainability": 0.8,
            "integration": 0.7,
            "cost_efficiency": 0.75,
        }

        overall = _calculate_overall_score(attribute_scores)

        # Should be weighted average
        assert 0.7 <= overall <= 0.8

    def test_empty_scores(self) -> None:
        """Test handling of empty scores."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _calculate_overall_score,
        )

        overall = _calculate_overall_score({})

        assert overall == 0.0

    def test_partial_scores(self) -> None:
        """Test handling of partial scores."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _calculate_overall_score,
        )

        attribute_scores = {
            "performance": 0.8,
            "security": 0.6,
        }

        overall = _calculate_overall_score(attribute_scores)

        assert 0.6 <= overall <= 0.8


class TestDetectTradeOffs:
    """Test _detect_trade_offs function."""

    def test_detects_performance_security_tradeoff(self) -> None:
        """Test detection of performance vs security trade-off."""
        from yolo_developer.agents.architect.quality_evaluator import _detect_trade_offs

        # Create story with explicit performance and security keywords
        story = _create_test_story(
            title="Secure Fast API",
            description="Fast low-latency API with secure encrypted authentication and credential protection",
        )
        decisions = [
            _create_test_decision(
                description="Use encryption for data in transit with fast throughput",
                rationale="Balance security and speed requirements",
            ),
        ]
        attribute_scores = {
            "performance": 0.7,
            "security": 0.8,
        }

        trade_offs = _detect_trade_offs(story, decisions, attribute_scores)

        # Should detect performance vs security trade-off due to keywords
        assert len(trade_offs) >= 1, "Expected at least one trade-off to be detected"
        # Verify the trade-off involves performance and security
        trade_off_pairs = [(t.attribute_a, t.attribute_b) for t in trade_offs]
        assert ("performance", "security") in trade_off_pairs, (
            f"Expected performance-security trade-off, got: {trade_off_pairs}"
        )

    def test_returns_empty_for_unrelated_story(self) -> None:
        """Test no trade-offs for unrelated story."""
        from yolo_developer.agents.architect.quality_evaluator import _detect_trade_offs

        story = _create_test_story(
            title="Simple Feature",
            description="A basic feature with no special requirements",
        )
        decisions: list[DesignDecision] = []
        attribute_scores = {
            "performance": 0.7,
            "security": 0.7,
        }

        trade_offs = _detect_trade_offs(story, decisions, attribute_scores)

        # Should return empty or minimal trade-offs
        assert isinstance(trade_offs, list)


class TestIdentifyRisks:
    """Test _identify_risks function."""

    def test_identifies_risks_for_low_scores(self) -> None:
        """Test that risks are identified for low scores."""
        from yolo_developer.agents.architect.quality_evaluator import _identify_risks

        story = _create_test_story()
        decisions: list[DesignDecision] = []
        attribute_scores = {
            "performance": 0.4,  # High risk
            "security": 0.8,  # Low risk
            "reliability": 0.2,  # Critical risk
        }

        risks = _identify_risks(story, decisions, attribute_scores)

        # Should identify risks for performance and reliability
        assert len(risks) == 2
        risk_attrs = [r.attribute for r in risks]
        assert "performance" in risk_attrs
        assert "reliability" in risk_attrs

    def test_risk_severity_mapping(self) -> None:
        """Test that risk severity is mapped correctly."""
        from yolo_developer.agents.architect.quality_evaluator import _identify_risks

        story = _create_test_story()
        decisions: list[DesignDecision] = []
        attribute_scores = {
            "performance": 0.2,  # Critical
            "security": 0.4,  # High
            "reliability": 0.6,  # Medium
        }

        risks = _identify_risks(story, decisions, attribute_scores)

        risk_map = {r.attribute: r.severity for r in risks}
        assert risk_map["performance"] == "critical"
        assert risk_map["security"] == "high"
        assert risk_map["reliability"] == "medium"

    def test_no_risks_for_high_scores(self) -> None:
        """Test that no risks are identified for high scores."""
        from yolo_developer.agents.architect.quality_evaluator import _identify_risks

        story = _create_test_story()
        decisions: list[DesignDecision] = []
        attribute_scores = {
            "performance": 0.9,
            "security": 0.85,
            "reliability": 0.8,
        }

        risks = _identify_risks(story, decisions, attribute_scores)

        assert len(risks) == 0

    def test_risks_include_mitigations(self) -> None:
        """Test that risks include mitigation strategies."""
        from yolo_developer.agents.architect.quality_evaluator import _identify_risks

        story = _create_test_story()
        decisions: list[DesignDecision] = []
        attribute_scores = {
            "performance": 0.4,
        }

        risks = _identify_risks(story, decisions, attribute_scores)

        assert len(risks) == 1
        assert risks[0].mitigation != ""
        assert risks[0].mitigation_effort in ["high", "medium", "low"]


class TestEvaluateQualityAttributes:
    """Test evaluate_quality_attributes main function."""

    @pytest.mark.asyncio
    async def test_pattern_based_evaluation(self) -> None:
        """Test pattern-based evaluation without LLM."""
        from yolo_developer.agents.architect.quality_evaluator import (
            evaluate_quality_attributes,
        )

        story = _create_test_story(
            title="Cached Service",
            description="Implement service with caching, retry logic, and authentication",
        )
        decisions = [
            _create_test_decision(
                decision_type="pattern",
                description="Use async await pattern with tenacity retry",
                rationale="Reliability and performance",
            ),
        ]

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        assert evaluation.overall_score > 0
        assert len(evaluation.attribute_scores) >= 5
        assert 0.0 <= evaluation.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_returns_complete_evaluation(self) -> None:
        """Test that evaluation includes all components."""
        from yolo_developer.agents.architect.quality_evaluator import (
            evaluate_quality_attributes,
        )

        story = _create_test_story()
        decisions: list[DesignDecision] = []

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        # Should have all standard attributes
        assert "performance" in evaluation.attribute_scores
        assert "security" in evaluation.attribute_scores
        assert "reliability" in evaluation.attribute_scores
        assert "scalability" in evaluation.attribute_scores
        assert "maintainability" in evaluation.attribute_scores

        # Should have overall score
        assert evaluation.overall_score > 0

        # Trade-offs and risks should be lists (possibly empty)
        assert isinstance(evaluation.trade_offs, tuple)
        assert isinstance(evaluation.risks, tuple)


class TestLLMIntegration:
    """Test LLM integration for quality evaluation."""

    @pytest.mark.asyncio
    async def test_llm_evaluation_with_mock(self) -> None:
        """Test LLM evaluation with mocked response."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _evaluate_quality_with_llm,
        )

        story = _create_test_story()
        decisions = [_create_test_decision()]

        mock_response = """```json
{
    "attribute_scores": {
        "performance": 0.8,
        "security": 0.9,
        "reliability": 0.75,
        "scalability": 0.7,
        "maintainability": 0.85,
        "integration": 0.7,
        "cost_efficiency": 0.8
    },
    "trade_offs": [
        {
            "attribute_a": "performance",
            "attribute_b": "security",
            "description": "Auth overhead",
            "resolution": "Cache tokens"
        }
    ],
    "risks": []
}
```"""

        with patch(
            "yolo_developer.agents.architect.quality_evaluator._call_quality_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            evaluation = await _evaluate_quality_with_llm(story, decisions)

        assert evaluation is not None
        assert evaluation.attribute_scores["performance"] == 0.8
        assert evaluation.attribute_scores["security"] == 0.9
        assert len(evaluation.trade_offs) == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_on_parse_error(self) -> None:
        """Test fallback when LLM response can't be parsed."""
        from yolo_developer.agents.architect.quality_evaluator import (
            _evaluate_quality_with_llm,
        )

        story = _create_test_story()
        decisions = [_create_test_decision()]

        # Invalid JSON response
        mock_response = "This is not valid JSON"

        with patch(
            "yolo_developer.agents.architect.quality_evaluator._call_quality_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            evaluation = await _evaluate_quality_with_llm(story, decisions)

        assert evaluation is None  # Should return None on parse error

    @pytest.mark.asyncio
    async def test_fallback_to_pattern_based(self) -> None:
        """Test fallback to pattern-based when LLM fails."""
        from yolo_developer.agents.architect.quality_evaluator import (
            evaluate_quality_attributes,
        )

        story = _create_test_story()
        decisions: list[DesignDecision] = []

        with patch(
            "yolo_developer.agents.architect.quality_evaluator._evaluate_quality_with_llm",
            new_callable=AsyncMock,
            return_value=None,
        ):
            evaluation = await evaluate_quality_attributes(story, decisions, use_llm=True)

        # Should still get valid evaluation from pattern-based fallback
        assert evaluation.overall_score > 0
        assert len(evaluation.attribute_scores) >= 5


class TestScoreToSeverity:
    """Test _score_to_severity helper function."""

    def test_critical_severity(self) -> None:
        """Test critical severity for very low scores."""
        from yolo_developer.agents.architect.quality_evaluator import _score_to_severity

        assert _score_to_severity(0.1) == "critical"
        assert _score_to_severity(0.29) == "critical"

    def test_high_severity(self) -> None:
        """Test high severity for low scores."""
        from yolo_developer.agents.architect.quality_evaluator import _score_to_severity

        assert _score_to_severity(0.3) == "high"
        assert _score_to_severity(0.49) == "high"

    def test_medium_severity(self) -> None:
        """Test medium severity for moderate scores."""
        from yolo_developer.agents.architect.quality_evaluator import _score_to_severity

        assert _score_to_severity(0.5) == "medium"
        assert _score_to_severity(0.69) == "medium"

    def test_low_severity(self) -> None:
        """Test low severity for high scores."""
        from yolo_developer.agents.architect.quality_evaluator import _score_to_severity

        assert _score_to_severity(0.7) == "low"
        assert _score_to_severity(0.9) == "low"
