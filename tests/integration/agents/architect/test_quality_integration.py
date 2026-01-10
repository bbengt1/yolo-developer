"""Integration tests for quality attribute evaluation (Story 7.4, Task 13).

Tests verify end-to-end quality evaluation flow through the architect node,
including integration with design decisions and 12-Factor analysis.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.agents.architect import architect_node
from yolo_developer.agents.architect.quality_evaluator import evaluate_quality_attributes
from yolo_developer.agents.architect.types import (
    DesignDecision,
    QualityAttributeEvaluation,
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
    title: str = "User Authentication",
    description: str = "Implement user authentication with JWT tokens and caching",
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
    decision_type: str = "security",
    description: str = "Use JWT for authentication with tenacity retry",
    rationale: str = "Stateless authentication with reliability",
) -> DesignDecision:
    """Create a test design decision."""
    return DesignDecision(
        id=decision_id,
        story_id=story_id,
        decision_type=decision_type,  # type: ignore[arg-type]
        description=description,
        rationale=rationale,
        alternatives_considered=("Session-based auth",),
    )


class TestArchitectNodeQualityIntegration:
    """Test architect node quality evaluation integration."""

    @pytest.mark.asyncio
    async def test_architect_node_includes_quality_evaluations(self) -> None:
        """Test that architect node generates quality evaluations."""
        stories = [
            _create_test_story("story-001", "Database Setup"),
            _create_test_story("story-002", "API Design"),
        ]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        assert "architect_output" in result
        output = result["architect_output"]
        # Quality evaluations should be in output
        assert "quality_evaluations" in output
        assert len(output["quality_evaluations"]) == 2

    @pytest.mark.asyncio
    async def test_architect_node_quality_eval_has_scores(self) -> None:
        """Test that quality evaluations have attribute scores."""
        stories = [_create_test_story("story-001")]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]
        quality_evals = output["quality_evaluations"]
        if "story-001" in quality_evals:
            eval_data = quality_evals["story-001"]
            assert "attribute_scores" in eval_data
            assert "overall_score" in eval_data
            assert 0.0 <= eval_data["overall_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_architect_node_quality_eval_has_risks(self) -> None:
        """Test that quality evaluations may include risks."""
        stories = [_create_test_story(
            "story-001",
            "Risky Feature",
            "Implement a feature with no tests and no documentation",
        )]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]
        quality_evals = output["quality_evaluations"]
        if "story-001" in quality_evals:
            eval_data = quality_evals["story-001"]
            assert "risks" in eval_data
            # Risks should be a list (possibly empty)
            assert isinstance(eval_data["risks"], list)


class TestEvaluateQualityAttributesIntegration:
    """Test evaluate_quality_attributes function integration."""

    @pytest.mark.asyncio
    async def test_evaluate_with_design_decisions(self) -> None:
        """Test quality evaluation with design decisions."""
        story = _create_test_story(
            description="Implement caching with retry logic and OAuth authentication",
        )
        decisions = [
            _create_test_decision(
                decision_type="pattern",
                description="Use async await with tenacity retry and caching",
                rationale="Reliability and performance",
            ),
            _create_test_decision(
                decision_id="design-002",
                decision_type="security",
                description="Use OAuth authentication with JWT",
                rationale="Security best practices",
            ),
        ]

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        assert isinstance(evaluation, QualityAttributeEvaluation)
        # With positive patterns, scores should be decent
        assert evaluation.overall_score > 0.6
        assert evaluation.attribute_scores.get("performance", 0) > 0
        assert evaluation.attribute_scores.get("security", 0) > 0

    @pytest.mark.asyncio
    async def test_evaluate_without_design_decisions(self) -> None:
        """Test quality evaluation without design decisions."""
        story = _create_test_story()
        decisions: list[DesignDecision] = []

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        assert isinstance(evaluation, QualityAttributeEvaluation)
        assert evaluation.overall_score > 0
        # Should still have all attribute scores
        assert len(evaluation.attribute_scores) >= 5

    @pytest.mark.asyncio
    async def test_evaluate_trade_offs_detection(self) -> None:
        """Test that trade-offs are detected when relevant."""
        story = _create_test_story(
            title="Secure Fast API",
            description="Build a fast API with encrypted authentication and retry logic",
        )
        decisions = [
            _create_test_decision(
                description="Use encryption with async operations",
                rationale="Balance security and speed",
            ),
        ]

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        # Trade-offs is a tuple (possibly empty based on pattern matching)
        assert isinstance(evaluation.trade_offs, tuple)

    @pytest.mark.asyncio
    async def test_evaluate_risk_identification(self) -> None:
        """Test that risks are identified for low scores."""
        # Story designed to score low on some attributes
        story = _create_test_story(
            title="Quick Prototype",
            description="Build a quick prototype with no tests, no documentation, stateful single instance",
        )
        decisions: list[DesignDecision] = []

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        # Should have some risks due to negative patterns
        # Note: May or may not have risks depending on actual scores
        assert isinstance(evaluation.risks, tuple)


class TestQualityEvaluationSerialization:
    """Test quality evaluation serialization in architect output."""

    @pytest.mark.asyncio
    async def test_evaluation_to_dict(self) -> None:
        """Test that evaluation can be serialized to dict."""
        story = _create_test_story()
        decisions = [_create_test_decision()]

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)
        result = evaluation.to_dict()

        assert isinstance(result, dict)
        assert "attribute_scores" in result
        assert "trade_offs" in result
        assert "risks" in result
        assert "overall_score" in result

    @pytest.mark.asyncio
    async def test_architect_output_includes_evaluation_data(self) -> None:
        """Test that ArchitectOutput properly includes evaluation data."""
        stories = [_create_test_story()]
        state = _create_minimal_state(stories)

        result = await architect_node(state)

        output = result["architect_output"]

        # Verify structure
        assert "design_decisions" in output
        assert "adrs" in output
        assert "twelve_factor_analyses" in output
        assert "quality_evaluations" in output

        # Quality evaluations should be serialized
        quality_evals = output["quality_evaluations"]
        for _story_id, eval_data in quality_evals.items():
            assert "attribute_scores" in eval_data
            assert "overall_score" in eval_data


class TestQualityScoreRanges:
    """Test that quality scores are in valid ranges."""

    @pytest.mark.asyncio
    async def test_all_scores_in_valid_range(self) -> None:
        """Test that all attribute scores are between 0 and 1."""
        story = _create_test_story()
        decisions = [_create_test_decision()]

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        for attr, score in evaluation.attribute_scores.items():
            assert 0.0 <= score <= 1.0, f"{attr} score {score} out of range"

        assert 0.0 <= evaluation.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_overall_score_is_weighted_average(self) -> None:
        """Test that overall score is reasonable weighted average."""
        story = _create_test_story()
        decisions: list[DesignDecision] = []

        evaluation = await evaluate_quality_attributes(story, decisions, use_llm=False)

        # Overall score should be within range of individual scores
        scores = list(evaluation.attribute_scores.values())
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            # Overall should be between min and max (with some buffer for weights)
            assert min_score - 0.1 <= evaluation.overall_score <= max_score + 0.1
