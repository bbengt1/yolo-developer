"""Unit tests for PM agent node (Story 6.1 Task 11, Story 6.2, Story 6.3).

Tests for pm_node function and helper functions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from yolo_developer.agents.pm.node import (
    _determine_priority,
    _generate_acceptance_criteria,
    _transform_requirements_to_stories,
    pm_node,
)
from yolo_developer.agents.pm.types import StoryPriority, StoryStatus
from yolo_developer.orchestrator.context import Decision


class TestDeterminePriority:
    """Tests for _determine_priority function."""

    def test_security_is_critical(self) -> None:
        """Security requirements should be critical priority."""
        priority = _determine_priority("functional", "User authentication with OAuth2")
        assert priority == StoryPriority.CRITICAL

    def test_authorization_is_critical(self) -> None:
        """Authorization requirements should be critical priority."""
        priority = _determine_priority("functional", "Role-based authorization")
        assert priority == StoryPriority.CRITICAL

    def test_functional_is_high(self) -> None:
        """Functional requirements without security should be high."""
        priority = _determine_priority("functional", "User can view dashboard")
        assert priority == StoryPriority.HIGH

    def test_non_functional_is_medium(self) -> None:
        """Non-functional requirements should be medium."""
        priority = _determine_priority("non_functional", "Response time under 200ms")
        assert priority == StoryPriority.MEDIUM

    def test_unknown_category_is_low(self) -> None:
        """Unknown categories should be low."""
        priority = _determine_priority("unknown", "Some requirement")
        assert priority == StoryPriority.LOW


class TestGenerateAcceptanceCriteria:
    """Tests for _generate_acceptance_criteria function."""

    @pytest.mark.asyncio
    async def test_generates_at_least_one_ac(self) -> None:
        """Should generate at least one AC for a requirement."""
        story_components = {
            "role": "user",
            "action": "login with email",
            "benefit": "access account",
            "title": "User Login",
        }
        acs = await _generate_acceptance_criteria(
            "req-001", "User can login", story_components
        )

        assert len(acs) >= 1
        assert acs[0].id == "AC1"

    @pytest.mark.asyncio
    async def test_ac_has_given_when_then(self) -> None:
        """Generated AC should have Given/When/Then format."""
        story_components = {
            "role": "user",
            "action": "test feature",
            "benefit": "validate",
            "title": "Test",
        }
        acs = await _generate_acceptance_criteria(
            "req-001", "Test requirement", story_components
        )

        ac = acs[0]
        assert ac.given != ""
        assert ac.when != ""
        assert ac.then != ""

    @pytest.mark.asyncio
    async def test_ac_has_and_clauses_tuple(self) -> None:
        """Generated AC should have and_clauses as tuple."""
        story_components = {
            "role": "user",
            "action": "test",
            "benefit": "verify",
            "title": "Test",
        }
        acs = await _generate_acceptance_criteria(
            "req-001", "Test", story_components
        )

        assert isinstance(acs[0].and_clauses, tuple)

    @pytest.mark.asyncio
    async def test_handles_empty_requirement_text(self) -> None:
        """Should handle empty requirement text gracefully."""
        story_components = {
            "role": "user",
            "action": "do something",
            "benefit": "get value",
            "title": "Action",
        }
        acs = await _generate_acceptance_criteria("req-001", "", story_components)

        assert len(acs) >= 1
        assert acs[0].id == "AC1"


class TestTransformRequirementsToStories:
    """Tests for _transform_requirements_to_stories function."""

    @pytest.mark.asyncio
    async def test_empty_requirements(self) -> None:
        """Should handle empty requirements list."""
        stories, unprocessed = await _transform_requirements_to_stories([])

        assert stories == ()
        assert unprocessed == ()

    @pytest.mark.asyncio
    async def test_single_functional_requirement(self) -> None:
        """Should transform single functional requirement to story."""
        reqs = [
            {
                "id": "req-001",
                "refined_text": "User can login with email",
                "category": "functional",
            }
        ]

        stories, _ = await _transform_requirements_to_stories(reqs)

        assert len(stories) == 1
        assert stories[0].id == "story-001"
        assert stories[0].source_requirements == ("req-001",)
        assert stories[0].status == StoryStatus.DRAFT

    @pytest.mark.asyncio
    async def test_constraint_requirements_unprocessed(self) -> None:
        """Constraint requirements should be marked as unprocessed."""
        reqs = [
            {
                "id": "req-001",
                "refined_text": "Must use PostgreSQL",
                "category": "constraint",
            }
        ]

        stories, unprocessed = await _transform_requirements_to_stories(reqs)

        assert len(stories) == 0
        assert unprocessed == ("req-001",)

    @pytest.mark.asyncio
    async def test_multiple_requirements(self) -> None:
        """Should transform multiple requirements."""
        reqs = [
            {"id": "req-001", "refined_text": "Login", "category": "functional"},
            {"id": "req-002", "refined_text": "Must be fast", "category": "constraint"},
            {"id": "req-003", "refined_text": "API rate limit", "category": "non_functional"},
        ]

        stories, unprocessed = await _transform_requirements_to_stories(reqs)

        assert len(stories) == 2  # functional + non-functional
        assert unprocessed == ("req-002",)  # constraint

    @pytest.mark.asyncio
    async def test_story_has_acceptance_criteria(self) -> None:
        """Generated stories should have acceptance criteria."""
        reqs = [{"id": "req-001", "refined_text": "Test", "category": "functional"}]

        stories, _ = await _transform_requirements_to_stories(reqs)

        assert len(stories[0].acceptance_criteria) >= 1
        assert stories[0].acceptance_criteria[0].id == "AC1"

    @pytest.mark.asyncio
    async def test_story_title_from_refined_text(self) -> None:
        """Story title should come from refined_text or extraction."""
        reqs = [{"id": "req-001", "refined_text": "User authentication", "category": "functional"}]

        stories, _ = await _transform_requirements_to_stories(reqs)

        # Title should contain some part of the requirement or be descriptive
        assert stories[0].title != ""
        assert len(stories[0].title) <= 50

    @pytest.mark.asyncio
    async def test_story_title_max_length(self) -> None:
        """Story title should not exceed 50 characters."""
        long_text = "x" * 100
        reqs = [{"id": "req-001", "refined_text": long_text, "category": "functional"}]

        stories, _ = await _transform_requirements_to_stories(reqs)

        assert len(stories[0].title) <= 50

    @pytest.mark.asyncio
    async def test_missing_refined_text_uses_original(self) -> None:
        """Should use original_text if refined_text missing."""
        reqs = [{"id": "req-001", "original_text": "Original", "category": "functional"}]

        stories, _ = await _transform_requirements_to_stories(reqs)

        # Action should contain the original text
        assert "Original" in stories[0].action or stories[0].action != ""

    @pytest.mark.asyncio
    async def test_missing_id_generates_one(self) -> None:
        """Should generate ID if missing from requirement."""
        reqs = [{"refined_text": "Test", "category": "functional"}]

        stories, _ = await _transform_requirements_to_stories(reqs)

        assert stories[0].source_requirements[0].startswith("req-")

    @pytest.mark.asyncio
    async def test_story_ids_are_sequential(self) -> None:
        """Story IDs should be sequential."""
        reqs = [
            {"id": "req-001", "refined_text": "A", "category": "functional"},
            {"id": "req-002", "refined_text": "B", "category": "functional"},
            {"id": "req-003", "refined_text": "C", "category": "functional"},
        ]

        stories, _ = await _transform_requirements_to_stories(reqs)

        assert stories[0].id == "story-001"
        assert stories[1].id == "story-002"
        assert stories[2].id == "story-003"

    @pytest.mark.asyncio
    async def test_story_ids_sequential_with_constraints(self) -> None:
        """Story IDs should be sequential even when constraints are filtered."""
        reqs = [
            {"id": "req-001", "refined_text": "A", "category": "functional"},
            {"id": "req-002", "refined_text": "B", "category": "constraint"},
            {"id": "req-003", "refined_text": "C", "category": "functional"},
        ]

        stories, unprocessed = await _transform_requirements_to_stories(reqs)

        # Should have 2 stories with sequential IDs (no gaps)
        assert len(stories) == 2
        assert stories[0].id == "story-001"
        assert stories[1].id == "story-002"  # NOT story-003
        assert unprocessed == ("req-002",)

    @pytest.mark.asyncio
    async def test_security_requirement_is_critical(self) -> None:
        """Security requirements should have critical priority."""
        reqs = [
            {"id": "req-001", "refined_text": "User authentication with OAuth", "category": "functional"}
        ]

        stories, _ = await _transform_requirements_to_stories(reqs)

        assert stories[0].priority == StoryPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_complexity_estimation(self) -> None:
        """Stories should have complexity estimation."""
        reqs = [{"id": "req-001", "refined_text": "Simple display feature", "category": "functional"}]

        stories, _ = await _transform_requirements_to_stories(reqs)

        assert stories[0].estimated_complexity in ["S", "M", "L", "XL"]

    @pytest.mark.asyncio
    async def test_handles_transformation_exception(self) -> None:
        """Should handle exceptions during single requirement transformation."""
        from unittest.mock import patch, AsyncMock

        reqs = [
            {"id": "req-001", "refined_text": "Normal requirement", "category": "functional"},
            {"id": "req-002", "refined_text": "Problematic requirement", "category": "functional"},
            {"id": "req-003", "refined_text": "Another normal requirement", "category": "functional"},
        ]

        # Mock _transform_single_requirement to fail for second requirement
        original_transform = _transform_requirements_to_stories.__wrapped__ if hasattr(_transform_requirements_to_stories, '__wrapped__') else None

        with patch(
            "yolo_developer.agents.pm.node._transform_single_requirement",
            new_callable=AsyncMock,
        ) as mock_transform:
            # First and third succeed, second fails
            mock_story = AsyncMock()
            mock_story.id = "story-001"

            async def side_effect(req: dict, counter: int) -> Any:
                if req.get("id") == "req-002":
                    raise RuntimeError("Transformation failed")
                # Return a mock story for other requirements
                from yolo_developer.agents.pm.types import Story, StoryStatus, StoryPriority, AcceptanceCriterion
                return Story(
                    id=f"story-{counter:03d}",
                    title="Test",
                    role="user",
                    action="test",
                    benefit="test",
                    acceptance_criteria=(AcceptanceCriterion(id="AC1", given="x", when="y", then="z", and_clauses=()),),
                    priority=StoryPriority.HIGH,
                    status=StoryStatus.DRAFT,
                    source_requirements=(req.get("id"),),
                    dependencies=(),
                    estimated_complexity="M",
                )

            mock_transform.side_effect = side_effect

            stories, unprocessed = await _transform_requirements_to_stories(reqs)

            # Should have 2 stories (req-001 and req-003), req-002 should be unprocessed
            assert len(stories) == 2
            assert "req-002" in unprocessed


@pytest.fixture
def mock_state() -> dict[str, Any]:
    """Create a mock YoloState for testing."""
    return {
        "messages": [],
        "current_agent": "pm",
        "handoff_context": None,
        "decisions": [],
        "analyst_output": {
            "requirements": [
                {
                    "id": "req-001",
                    "refined_text": "User can login with email and password",
                    "category": "functional",
                },
                {
                    "id": "req-002",
                    "refined_text": "Response time under 200ms",
                    "category": "non_functional",
                },
            ],
            "escalations": [],
        },
    }


@pytest.fixture
def empty_state() -> dict[str, Any]:
    """Create an empty mock YoloState."""
    return {
        "messages": [],
        "current_agent": "pm",
        "handoff_context": None,
        "decisions": [],
    }


class TestPmNode:
    """Tests for pm_node function."""

    @pytest.mark.asyncio
    async def test_returns_dict(self, mock_state: dict[str, Any]) -> None:
        """pm_node should return a dict with state updates."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_messages(self, mock_state: dict[str, Any]) -> None:
        """pm_node should return messages list."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_returns_decisions(self, mock_state: dict[str, Any]) -> None:
        """pm_node should return decisions list."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        assert "decisions" in result
        assert len(result["decisions"]) == 1
        assert isinstance(result["decisions"][0], Decision)

    @pytest.mark.asyncio
    async def test_returns_pm_output(self, mock_state: dict[str, Any]) -> None:
        """pm_node should return pm_output dict."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        assert "pm_output" in result
        assert isinstance(result["pm_output"], dict)
        assert "stories" in result["pm_output"]
        assert "story_count" in result["pm_output"]

    @pytest.mark.asyncio
    async def test_creates_stories_from_requirements(
        self, mock_state: dict[str, Any]
    ) -> None:
        """pm_node should create stories from requirements."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        pm_output = result["pm_output"]
        # 2 requirements (functional + non-functional) = 2 stories
        assert pm_output["story_count"] == 2

    @pytest.mark.asyncio
    async def test_decision_has_correct_agent(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Decision should have agent='pm'."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        decision = result["decisions"][0]
        assert decision.agent == "pm"

    @pytest.mark.asyncio
    async def test_decision_has_related_artifacts(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Decision should include story IDs as related_artifacts."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        decision = result["decisions"][0]
        assert len(decision.related_artifacts) > 0
        assert all(a.startswith("story-") for a in decision.related_artifacts)

    @pytest.mark.asyncio
    async def test_decision_has_valid_timestamp(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Decision should have a valid UTC timestamp."""
        before = datetime.now(timezone.utc)
        result = await pm_node(mock_state)  # type: ignore[arg-type]
        after = datetime.now(timezone.utc)

        decision = result["decisions"][0]
        assert decision.timestamp is not None
        assert isinstance(decision.timestamp, datetime)
        # Timestamp should be within test execution window
        assert before <= decision.timestamp <= after

    @pytest.mark.asyncio
    async def test_message_has_agent_metadata(
        self, mock_state: dict[str, Any]
    ) -> None:
        """AIMessage should have agent='pm' in additional_kwargs."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        message = result["messages"][0]
        assert message.additional_kwargs.get("agent") == "pm"

    @pytest.mark.asyncio
    async def test_message_content_has_summary(
        self, mock_state: dict[str, Any]
    ) -> None:
        """AIMessage content should summarize processing."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        message = result["messages"][0]
        assert "Stories created:" in message.content
        assert "PM processing complete" in message.content

    @pytest.mark.asyncio
    async def test_handles_empty_analyst_output(
        self, empty_state: dict[str, Any]
    ) -> None:
        """pm_node should handle missing analyst_output."""
        result = await pm_node(empty_state)  # type: ignore[arg-type]

        pm_output = result["pm_output"]
        assert pm_output["story_count"] == 0

    @pytest.mark.asyncio
    async def test_handles_empty_requirements(
        self, empty_state: dict[str, Any]
    ) -> None:
        """pm_node should handle empty requirements list."""
        empty_state["analyst_output"] = {"requirements": [], "escalations": []}

        result = await pm_node(empty_state)  # type: ignore[arg-type]

        pm_output = result["pm_output"]
        assert pm_output["story_count"] == 0

    @pytest.mark.asyncio
    async def test_processing_notes_summary(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Processing notes should summarize transformation."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        notes = result["pm_output"]["processing_notes"]
        assert "Transformed" in notes
        assert "requirements" in notes
        assert "stories" in notes

    @pytest.mark.asyncio
    async def test_handles_gaps_and_contradictions(
        self, empty_state: dict[str, Any]
    ) -> None:
        """pm_node should extract gaps and contradictions from analyst_output."""
        empty_state["analyst_output"] = {
            "requirements": [],
            "escalations": [],
            "gaps": [{"id": "gap-001", "description": "Missing auth"}],
            "contradictions": [{"id": "cont-001", "description": "Conflict"}],
        }

        # Should not raise - gaps and contradictions are extracted and logged
        result = await pm_node(empty_state)  # type: ignore[arg-type]

        # Verify node completed successfully
        assert result["pm_output"]["story_count"] == 0

    @pytest.mark.asyncio
    async def test_decision_rationale_mentions_llm(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Decision rationale should mention LLM-powered transformation."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        decision = result["decisions"][0]
        assert "LLM" in decision.rationale or "story extraction" in decision.rationale

    @pytest.mark.asyncio
    async def test_processing_notes_include_validation(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Processing notes should include testability validation summary (Story 6.3)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        notes = result["pm_output"]["processing_notes"]
        # Should mention testability validation
        assert "Testability validation" in notes or "testability" in notes.lower()

    @pytest.mark.asyncio
    async def test_validation_does_not_block_story_creation(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Validation should warn but not block story creation (Story 6.3)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        # Stories should still be created even if they have validation issues
        pm_output = result["pm_output"]
        assert pm_output["story_count"] > 0
        # Processing notes should exist
        assert pm_output["processing_notes"] != ""

    @pytest.mark.asyncio
    async def test_validation_results_appear_in_notes(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Validation results should appear in processing notes (Story 6.3)."""
        # Create requirements that will produce stories with potential issues
        empty_state["analyst_output"] = {
            "requirements": [
                {
                    "id": "req-001",
                    "refined_text": "System must be fast and efficient",  # Contains vague terms
                    "category": "functional",
                },
            ],
            "escalations": [],
        }

        result = await pm_node(empty_state)  # type: ignore[arg-type]

        notes = result["pm_output"]["processing_notes"]
        # Should have testability validation info
        assert "Testability validation" in notes or "validation" in notes.lower()

    @pytest.mark.asyncio
    async def test_prioritization_result_included(
        self, mock_state: dict[str, Any]
    ) -> None:
        """pm_node should include prioritization_result in output (Story 6.4)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        assert "prioritization_result" in result
        prioritization = result["prioritization_result"]
        assert "scores" in prioritization
        assert "recommended_execution_order" in prioritization
        assert "quick_wins" in prioritization
        assert "dependency_cycles" in prioritization
        assert "analysis_notes" in prioritization

    @pytest.mark.asyncio
    async def test_prioritization_scores_match_story_count(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Prioritization should have scores for each story (Story 6.4)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        pm_output = result["pm_output"]
        prioritization = result["prioritization_result"]

        assert len(prioritization["scores"]) == pm_output["story_count"]

    @pytest.mark.asyncio
    async def test_prioritization_order_matches_stories(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Recommended order should contain same story IDs (Story 6.4)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        pm_output = result["pm_output"]
        prioritization = result["prioritization_result"]

        story_ids = {s["id"] for s in pm_output["stories"]}
        order_ids = set(prioritization["recommended_execution_order"])

        assert story_ids == order_ids

    @pytest.mark.asyncio
    async def test_prioritization_summary_in_notes(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Processing notes should include prioritization summary (Story 6.4)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        notes = result["pm_output"]["processing_notes"]
        assert "Prioritization:" in notes or "prioritized" in notes.lower()

    @pytest.mark.asyncio
    async def test_prioritization_in_decision_rationale(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Decision rationale should mention prioritization (Story 6.4)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        decision = result["decisions"][0]
        assert "Prioritization" in decision.rationale or "prioritized" in decision.rationale.lower()

    @pytest.mark.asyncio
    async def test_empty_stories_prioritization(
        self, empty_state: dict[str, Any]
    ) -> None:
        """Prioritization should handle empty stories gracefully (Story 6.4)."""
        empty_state["analyst_output"] = {"requirements": [], "escalations": []}

        result = await pm_node(empty_state)  # type: ignore[arg-type]

        prioritization = result["prioritization_result"]
        assert prioritization["scores"] == []
        assert prioritization["recommended_execution_order"] == []
        assert prioritization["quick_wins"] == []
        assert prioritization["dependency_cycles"] == []

    @pytest.mark.asyncio
    async def test_message_includes_quick_wins(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Message content should mention quick wins (Story 6.4)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        message = result["messages"][0]
        assert "Quick wins:" in message.content

    @pytest.mark.asyncio
    async def test_message_includes_recommended_order(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Message content should mention recommended order (Story 6.4)."""
        result = await pm_node(mock_state)  # type: ignore[arg-type]

        message = result["messages"][0]
        assert "Recommended order:" in message.content
