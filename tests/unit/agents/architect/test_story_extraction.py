"""Tests for story extraction from state (Story 7.1, Task 4, Task 9).

Tests verify that the Architect agent can extract stories from
orchestration state in various formats.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from yolo_developer.agents.architect.node import _extract_stories_from_state
from yolo_developer.agents.pm.types import AcceptanceCriterion, Story, StoryPriority, StoryStatus


def _create_test_story(story_id: str = "story-001") -> Story:
    """Create a test story for use in tests."""
    return Story(
        id=story_id,
        title="Test Story",
        role="developer",
        action="implement feature",
        benefit="deliver value",
        acceptance_criteria=(
            AcceptanceCriterion(
                id="AC1",
                given="system is ready",
                when="action is taken",
                then="result is observed",
            ),
        ),
        priority=StoryPriority.HIGH,
        status=StoryStatus.READY,
    )


class TestStoryExtractionFromPMOutput:
    """Test story extraction from pm_output in state."""

    def test_extracts_stories_from_pm_output(self) -> None:
        """Test that stories are extracted from pm_output."""
        story = _create_test_story()
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [story.to_dict()],
                "processing_notes": "test",
            },
        }

        result = _extract_stories_from_state(state)

        assert len(result) == 1
        assert result[0]["id"] == "story-001"

    def test_extracts_multiple_stories_from_pm_output(self) -> None:
        """Test that multiple stories are extracted."""
        stories = [_create_test_story(f"story-{i:03d}") for i in range(3)]
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [s.to_dict() for s in stories],
                "processing_notes": "test",
            },
        }

        result = _extract_stories_from_state(state)

        assert len(result) == 3
        assert [s["id"] for s in result] == ["story-000", "story-001", "story-002"]


class TestStoryExtractionFromMessages:
    """Test story extraction from message metadata."""

    def test_extracts_stories_from_message_metadata(self) -> None:
        """Test that stories are extracted from message metadata when no pm_output."""
        story = _create_test_story()
        message = AIMessage(
            content="PM complete",
            additional_kwargs={
                "agent": "pm",
                "output": {
                    "stories": [story.to_dict()],
                    "processing_notes": "test",
                },
            },
        )
        state = {
            "messages": [message],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        result = _extract_stories_from_state(state)

        assert len(result) == 1
        assert result[0]["id"] == "story-001"

    def test_extracts_from_latest_pm_message(self) -> None:
        """Test that stories are extracted from the latest PM message."""
        old_story = _create_test_story("old-story")
        new_story = _create_test_story("new-story")

        old_message = AIMessage(
            content="Old PM output",
            additional_kwargs={
                "agent": "pm",
                "output": {
                    "stories": [old_story.to_dict()],
                    "processing_notes": "old",
                },
            },
        )
        new_message = AIMessage(
            content="New PM output",
            additional_kwargs={
                "agent": "pm",
                "output": {
                    "stories": [new_story.to_dict()],
                    "processing_notes": "new",
                },
            },
        )
        state = {
            "messages": [old_message, new_message],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        result = _extract_stories_from_state(state)

        assert len(result) == 1
        assert result[0]["id"] == "new-story"


class TestStoryExtractionEmptyState:
    """Test graceful handling of empty or missing data."""

    def test_empty_state_returns_empty_list(self) -> None:
        """Test that empty state returns empty list."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        result = _extract_stories_from_state(state)

        assert result == []

    def test_missing_pm_output_returns_empty_list(self) -> None:
        """Test that missing pm_output key returns empty list."""
        state = {
            "messages": [HumanMessage(content="hello")],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        result = _extract_stories_from_state(state)

        assert result == []

    def test_empty_stories_list_returns_empty_list(self) -> None:
        """Test that empty stories list returns empty list."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [],
                "processing_notes": "no stories",
            },
        }

        result = _extract_stories_from_state(state)

        assert result == []

    def test_no_pm_messages_returns_empty_list(self) -> None:
        """Test that state with only non-PM messages returns empty list."""
        analyst_message = AIMessage(
            content="Analyst output",
            additional_kwargs={
                "agent": "analyst",
                "output": {"requirements": []},
            },
        )
        state = {
            "messages": [analyst_message],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        result = _extract_stories_from_state(state)

        assert result == []
