"""Story extraction tests for Dev agent (Story 8.1, AC2).

Tests for _extract_stories_for_implementation function that extracts
stories from orchestration state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from yolo_developer.agents.dev.node import _extract_stories_for_implementation
from yolo_developer.orchestrator.state import YoloState


class TestStoryExtractionFromArchitectOutput:
    """Tests for extracting stories from architect_output."""

    def test_extracts_stories_from_architect_output(self) -> None:
        """Test extraction from architect_output with design decisions."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {"story_id": "story-001", "decision_type": "pattern"},
                    {"story_id": "story-002", "decision_type": "technology"},
                ],
            },
        }
        stories = _extract_stories_for_implementation(state)
        assert len(stories) == 2
        story_ids = {s["id"] for s in stories}
        assert "story-001" in story_ids
        assert "story-002" in story_ids

    def test_extracts_unique_story_ids(self) -> None:
        """Test extraction deduplicates story IDs."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {"story_id": "story-001", "decision_type": "pattern"},
                    {"story_id": "story-001", "decision_type": "technology"},
                ],
            },
        }
        stories = _extract_stories_for_implementation(state)
        assert len(stories) == 1
        assert stories[0]["id"] == "story-001"


class TestStoryExtractionFromPmOutput:
    """Tests for extracting stories from pm_output fallback."""

    def test_extracts_stories_from_pm_output(self) -> None:
        """Test extraction from pm_output when architect_output empty."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [
                    {"id": "story-001", "title": "Story One"},
                    {"id": "story-002", "title": "Story Two"},
                ],
            },
        }
        stories = _extract_stories_for_implementation(state)
        assert len(stories) == 2
        assert stories[0]["id"] == "story-001"
        assert stories[1]["id"] == "story-002"


class TestStoryExtractionFromMessages:
    """Tests for extracting stories from message metadata."""

    def test_extracts_stories_from_architect_message(self) -> None:
        """Test extraction from architect message metadata."""
        mock_msg = MagicMock()
        mock_msg.additional_kwargs = {
            "agent": "architect",
            "output": {
                "design_decisions": [
                    {"story_id": "story-msg-001"},
                ],
            },
        }
        state: YoloState = {
            "messages": [mock_msg],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        stories = _extract_stories_for_implementation(state)
        assert len(stories) == 1
        assert stories[0]["id"] == "story-msg-001"


class TestStoryExtractionEmptyState:
    """Tests for empty state handling."""

    def test_returns_empty_list_for_empty_state(self) -> None:
        """Test returns empty list when state has no stories."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        stories = _extract_stories_for_implementation(state)
        assert stories == []

    def test_returns_empty_list_for_empty_architect_output(self) -> None:
        """Test returns empty list when architect_output has no decisions."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {"design_decisions": []},
        }
        stories = _extract_stories_for_implementation(state)
        assert stories == []
