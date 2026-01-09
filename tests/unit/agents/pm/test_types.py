"""Unit tests for PM agent types (Story 6.1 Task 10).

Tests for StoryStatus, StoryPriority, AcceptanceCriterion, Story, and PMOutput.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    PMOutput,
    Story,
    StoryPriority,
    StoryStatus,
)


class TestStoryStatus:
    """Tests for StoryStatus enum."""

    def test_draft_value(self) -> None:
        """StoryStatus.DRAFT should have correct value."""
        assert StoryStatus.DRAFT.value == "draft"

    def test_ready_value(self) -> None:
        """StoryStatus.READY should have correct value."""
        assert StoryStatus.READY.value == "ready"

    def test_blocked_value(self) -> None:
        """StoryStatus.BLOCKED should have correct value."""
        assert StoryStatus.BLOCKED.value == "blocked"

    def test_in_progress_value(self) -> None:
        """StoryStatus.IN_PROGRESS should have correct value."""
        assert StoryStatus.IN_PROGRESS.value == "in_progress"

    def test_done_value(self) -> None:
        """StoryStatus.DONE should have correct value."""
        assert StoryStatus.DONE.value == "done"

    def test_all_values_exist(self) -> None:
        """StoryStatus should have all expected values."""
        expected = {"draft", "ready", "blocked", "in_progress", "done"}
        actual = {status.value for status in StoryStatus}
        assert actual == expected

    def test_construction_from_string(self) -> None:
        """StoryStatus should be constructable from string value."""
        status = StoryStatus("draft")
        assert status == StoryStatus.DRAFT


class TestStoryPriority:
    """Tests for StoryPriority enum."""

    def test_critical_value(self) -> None:
        """StoryPriority.CRITICAL should have correct value."""
        assert StoryPriority.CRITICAL.value == "critical"

    def test_high_value(self) -> None:
        """StoryPriority.HIGH should have correct value."""
        assert StoryPriority.HIGH.value == "high"

    def test_medium_value(self) -> None:
        """StoryPriority.MEDIUM should have correct value."""
        assert StoryPriority.MEDIUM.value == "medium"

    def test_low_value(self) -> None:
        """StoryPriority.LOW should have correct value."""
        assert StoryPriority.LOW.value == "low"

    def test_all_values_exist(self) -> None:
        """StoryPriority should have all expected values."""
        expected = {"critical", "high", "medium", "low"}
        actual = {priority.value for priority in StoryPriority}
        assert actual == expected

    def test_construction_from_string(self) -> None:
        """StoryPriority should be constructable from string value."""
        priority = StoryPriority("high")
        assert priority == StoryPriority.HIGH


class TestAcceptanceCriterion:
    """Tests for AcceptanceCriterion dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """AcceptanceCriterion should be creatable with required fields."""
        ac = AcceptanceCriterion(
            id="AC1",
            given="a user is logged in",
            when="they click logout",
            then="they are redirected to login page",
        )

        assert ac.id == "AC1"
        assert ac.given == "a user is logged in"
        assert ac.when == "they click logout"
        assert ac.then == "they are redirected to login page"
        assert ac.and_clauses == ()

    def test_creation_with_and_clauses(self) -> None:
        """AcceptanceCriterion should support and_clauses."""
        ac = AcceptanceCriterion(
            id="AC1",
            given="precondition",
            when="action",
            then="outcome",
            and_clauses=("session is invalidated", "cookies are cleared"),
        )

        assert ac.and_clauses == ("session is invalidated", "cookies are cleared")

    def test_immutability(self) -> None:
        """AcceptanceCriterion should be immutable (frozen dataclass)."""
        ac = AcceptanceCriterion(
            id="AC1",
            given="g",
            when="w",
            then="t",
        )

        with pytest.raises(AttributeError):
            ac.id = "AC2"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        ac = AcceptanceCriterion(
            id="AC1",
            given="user logged in",
            when="click logout",
            then="redirected",
            and_clauses=("session ends",),
        )

        result = ac.to_dict()

        assert result == {
            "id": "AC1",
            "given": "user logged in",
            "when": "click logout",
            "then": "redirected",
            "and_clauses": ["session ends"],
        }

    def test_to_dict_empty_and_clauses(self) -> None:
        """to_dict should handle empty and_clauses."""
        ac = AcceptanceCriterion(
            id="AC1",
            given="g",
            when="w",
            then="t",
        )

        result = ac.to_dict()
        assert result["and_clauses"] == []

    def test_equality(self) -> None:
        """AcceptanceCriterion equality should be based on field values."""
        ac1 = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        ac2 = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")

        assert ac1 == ac2

    def test_hashability(self) -> None:
        """AcceptanceCriterion should be hashable (frozen)."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")

        s = {ac}
        assert ac in s


class TestStory:
    """Tests for Story dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Story should be creatable with required fields."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="User login",
            role="visitor",
            action="log into the system",
            benefit="I can access my account",
            acceptance_criteria=(ac,),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
        )

        assert story.id == "story-001"
        assert story.title == "User login"
        assert story.role == "visitor"
        assert story.action == "log into the system"
        assert story.benefit == "I can access my account"
        assert len(story.acceptance_criteria) == 1
        assert story.priority == StoryPriority.HIGH
        assert story.status == StoryStatus.DRAFT
        assert story.source_requirements == ()
        assert story.dependencies == ()
        assert story.estimated_complexity == "M"

    def test_creation_with_all_fields(self) -> None:
        """Story should support all optional fields."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.CRITICAL,
            status=StoryStatus.IN_PROGRESS,
            source_requirements=("req-001", "req-002"),
            dependencies=("story-000",),
            estimated_complexity="XL",
        )

        assert story.source_requirements == ("req-001", "req-002")
        assert story.dependencies == ("story-000",)
        assert story.estimated_complexity == "XL"

    def test_immutability(self) -> None:
        """Story should be immutable (frozen dataclass)."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )

        with pytest.raises(AttributeError):
            story.id = "story-002"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.HIGH,
            status=StoryStatus.READY,
            source_requirements=("req-001",),
            dependencies=("story-000",),
            estimated_complexity="L",
        )

        result = story.to_dict()

        assert result["id"] == "story-001"
        assert result["title"] == "Title"
        assert result["role"] == "user"
        assert result["action"] == "action"
        assert result["benefit"] == "benefit"
        assert result["priority"] == "high"
        assert result["status"] == "ready"
        assert result["source_requirements"] == ["req-001"]
        assert result["dependencies"] == ["story-000"]
        assert result["estimated_complexity"] == "L"
        assert len(result["acceptance_criteria"]) == 1
        assert result["acceptance_criteria"][0]["id"] == "AC1"

    def test_equality(self) -> None:
        """Story equality should be based on field values."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story1 = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )
        story2 = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )

        assert story1 == story2

    def test_hashability(self) -> None:
        """Story should be hashable (frozen)."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )

        s = {story}
        assert story in s


class TestPMOutput:
    """Tests for PMOutput dataclass."""

    def test_creation_with_empty_stories(self) -> None:
        """PMOutput should be creatable with no stories."""
        output = PMOutput(stories=())

        assert output.stories == ()
        assert output.unprocessed_requirements == ()
        assert output.escalations_to_analyst == ()
        assert output.processing_notes == ""

    def test_creation_with_stories(self) -> None:
        """PMOutput should be creatable with stories."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )
        output = PMOutput(
            stories=(story,),
            unprocessed_requirements=("req-005",),
            escalations_to_analyst=("req-006",),
            processing_notes="Test notes",
        )

        assert len(output.stories) == 1
        assert output.unprocessed_requirements == ("req-005",)
        assert output.escalations_to_analyst == ("req-006",)
        assert output.processing_notes == "Test notes"

    def test_story_count_property(self) -> None:
        """story_count should return number of stories."""
        output_empty = PMOutput(stories=())
        assert output_empty.story_count == 0

        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )
        output_one = PMOutput(stories=(story,))
        assert output_one.story_count == 1

    def test_has_escalations_property_false(self) -> None:
        """has_escalations should be False when no escalations."""
        output = PMOutput(stories=())
        assert output.has_escalations is False

    def test_has_escalations_property_true(self) -> None:
        """has_escalations should be True when escalations exist."""
        output = PMOutput(
            stories=(),
            escalations_to_analyst=("req-001",),
        )
        assert output.has_escalations is True

    def test_immutability(self) -> None:
        """PMOutput should be immutable (frozen dataclass)."""
        output = PMOutput(stories=())

        with pytest.raises(AttributeError):
            output.processing_notes = "new notes"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        ac = AcceptanceCriterion(id="AC1", given="g", when="w", then="t")
        story = Story(
            id="story-001",
            title="Title",
            role="user",
            action="action",
            benefit="benefit",
            acceptance_criteria=(ac,),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )
        output = PMOutput(
            stories=(story,),
            unprocessed_requirements=("req-005",),
            escalations_to_analyst=("req-006",),
            processing_notes="Notes",
        )

        result = output.to_dict()

        assert result["story_count"] == 1
        assert result["has_escalations"] is True
        assert result["unprocessed_requirements"] == ["req-005"]
        assert result["escalations_to_analyst"] == ["req-006"]
        assert result["processing_notes"] == "Notes"
        assert len(result["stories"]) == 1
        assert result["stories"][0]["id"] == "story-001"

    def test_to_dict_empty(self) -> None:
        """to_dict should handle empty PMOutput."""
        output = PMOutput(stories=())

        result = output.to_dict()

        assert result == {
            "stories": [],
            "unprocessed_requirements": [],
            "escalations_to_analyst": [],
            "processing_notes": "",
            "story_count": 0,
            "has_escalations": False,
        }

    def test_equality(self) -> None:
        """PMOutput equality should be based on field values."""
        output1 = PMOutput(stories=(), processing_notes="test")
        output2 = PMOutput(stories=(), processing_notes="test")

        assert output1 == output2

    def test_hashability(self) -> None:
        """PMOutput should be hashable (frozen)."""
        output = PMOutput(stories=())

        s = {output}
        assert output in s
