"""Unit tests for PM agent node (Story 6.1 Task 11).

Tests for pm_node function and helper functions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from yolo_developer.agents.pm.node import (
    _generate_acceptance_criteria,
    _transform_requirements_to_stories,
    pm_node,
)
from yolo_developer.agents.pm.types import StoryPriority, StoryStatus
from yolo_developer.orchestrator.context import Decision


class TestGenerateAcceptanceCriteria:
    """Tests for _generate_acceptance_criteria function."""

    def test_generates_single_ac(self) -> None:
        """Should generate a single AC for a requirement."""
        acs = _generate_acceptance_criteria("req-001", "User can login")

        assert len(acs) == 1
        assert acs[0].id == "AC1"

    def test_ac_has_given_when_then(self) -> None:
        """Generated AC should have Given/When/Then format."""
        acs = _generate_acceptance_criteria("req-001", "Test requirement")

        ac = acs[0]
        assert "req-001" in ac.given
        assert ac.when != ""
        assert "Test requirement" in ac.then

    def test_truncates_long_requirement_text(self) -> None:
        """Should truncate requirement text in 'then' clause."""
        long_text = "x" * 100
        acs = _generate_acceptance_criteria("req-001", long_text)

        # Should include ellipsis after truncation
        assert "..." in acs[0].then

    def test_empty_and_clauses(self) -> None:
        """Generated AC should have empty and_clauses."""
        acs = _generate_acceptance_criteria("req-001", "Test")

        assert acs[0].and_clauses == ()

    def test_handles_empty_requirement_text(self) -> None:
        """Should handle empty requirement text gracefully."""
        acs = _generate_acceptance_criteria("req-001", "")

        assert len(acs) == 1
        assert acs[0].id == "AC1"
        # then clause should still be valid even with empty text
        assert "..." in acs[0].then
        assert "req-001" in acs[0].given


class TestTransformRequirementsToStories:
    """Tests for _transform_requirements_to_stories function."""

    def test_empty_requirements(self) -> None:
        """Should handle empty requirements list."""
        stories, unprocessed = _transform_requirements_to_stories([])

        assert stories == ()
        assert unprocessed == ()

    def test_single_functional_requirement(self) -> None:
        """Should transform single functional requirement to story."""
        reqs = [
            {
                "id": "req-001",
                "refined_text": "User can login with email",
                "category": "functional",
            }
        ]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert len(stories) == 1
        assert stories[0].id == "story-001"
        assert stories[0].source_requirements == ("req-001",)
        assert stories[0].priority == StoryPriority.HIGH
        assert stories[0].status == StoryStatus.DRAFT

    def test_constraint_requirements_unprocessed(self) -> None:
        """Constraint requirements should be marked as unprocessed."""
        reqs = [
            {
                "id": "req-001",
                "refined_text": "Must use PostgreSQL",
                "category": "constraint",
            }
        ]

        stories, unprocessed = _transform_requirements_to_stories(reqs)

        assert len(stories) == 0
        assert unprocessed == ("req-001",)

    def test_multiple_requirements(self) -> None:
        """Should transform multiple requirements."""
        reqs = [
            {"id": "req-001", "refined_text": "Login", "category": "functional"},
            {"id": "req-002", "refined_text": "Must be fast", "category": "constraint"},
            {"id": "req-003", "refined_text": "API rate limit", "category": "non-functional"},
        ]

        stories, unprocessed = _transform_requirements_to_stories(reqs)

        assert len(stories) == 2  # functional + non-functional
        assert unprocessed == ("req-002",)  # constraint

    def test_story_has_acceptance_criteria(self) -> None:
        """Generated stories should have acceptance criteria."""
        reqs = [{"id": "req-001", "refined_text": "Test", "category": "functional"}]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert len(stories[0].acceptance_criteria) >= 1
        assert stories[0].acceptance_criteria[0].id == "AC1"

    def test_story_title_from_refined_text(self) -> None:
        """Story title should come from refined_text."""
        reqs = [{"id": "req-001", "refined_text": "User authentication", "category": "functional"}]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert "User authentication" in stories[0].title

    def test_story_title_truncates_long_text(self) -> None:
        """Story title should truncate very long refined text."""
        long_text = "x" * 100
        reqs = [{"id": "req-001", "refined_text": long_text, "category": "functional"}]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert len(stories[0].title) <= 50

    def test_missing_refined_text_uses_original(self) -> None:
        """Should use original_text if refined_text missing."""
        reqs = [{"id": "req-001", "original_text": "Original", "category": "functional"}]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert "Original" in stories[0].action

    def test_missing_id_generates_one(self) -> None:
        """Should generate ID if missing from requirement."""
        reqs = [{"refined_text": "Test", "category": "functional"}]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert stories[0].source_requirements[0].startswith("req-")

    def test_non_functional_gets_medium_priority(self) -> None:
        """Non-functional requirements should get medium priority."""
        reqs = [{"id": "req-001", "refined_text": "Fast", "category": "non-functional"}]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert stories[0].priority == StoryPriority.MEDIUM

    def test_story_ids_are_sequential(self) -> None:
        """Story IDs should be sequential."""
        reqs = [
            {"id": "req-001", "refined_text": "A", "category": "functional"},
            {"id": "req-002", "refined_text": "B", "category": "functional"},
            {"id": "req-003", "refined_text": "C", "category": "functional"},
        ]

        stories, _ = _transform_requirements_to_stories(reqs)

        assert stories[0].id == "story-001"
        assert stories[1].id == "story-002"
        assert stories[2].id == "story-003"

    def test_story_ids_sequential_with_constraints(self) -> None:
        """Story IDs should be sequential even when constraints are filtered."""
        reqs = [
            {"id": "req-001", "refined_text": "A", "category": "functional"},
            {"id": "req-002", "refined_text": "B", "category": "constraint"},
            {"id": "req-003", "refined_text": "C", "category": "functional"},
        ]

        stories, unprocessed = _transform_requirements_to_stories(reqs)

        # Should have 2 stories with sequential IDs (no gaps)
        assert len(stories) == 2
        assert stories[0].id == "story-001"
        assert stories[1].id == "story-002"  # NOT story-003
        assert unprocessed == ("req-002",)


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
                    "category": "non-functional",
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
