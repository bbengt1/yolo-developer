"""Tests for sprint planning types (Story 10.3).

Tests the data types used by the sprint planning module:
- SprintStory: Story prepared for sprint planning
- SprintPlan: Complete sprint plan output
- PlanningConfig: Configuration for sprint planning
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.sm.planning_types import (
    DEFAULT_DEPENDENCY_WEIGHT,
    DEFAULT_MAX_POINTS,
    DEFAULT_MAX_STORIES,
    DEFAULT_TECH_DEBT_WEIGHT,
    DEFAULT_VALUE_WEIGHT,
    DEFAULT_VELOCITY_WEIGHT,
    PlanningConfig,
    SprintPlan,
    SprintStory,
)


class TestSprintStory:
    """Tests for SprintStory dataclass."""

    def test_create_minimal_story(self) -> None:
        """Test creating a SprintStory with required fields only."""
        story = SprintStory(
            story_id="1-1-test-story",
            title="Test Story",
        )
        assert story.story_id == "1-1-test-story"
        assert story.title == "Test Story"
        assert story.priority_score == 0.0
        assert story.dependencies == ()
        assert story.estimated_points == 1
        assert story.value_score == 0.5
        assert story.tech_debt_score == 0.0
        assert story.velocity_impact == 0.5
        assert story.dependency_score == 0.0
        assert story.metadata == {}

    def test_create_full_story(self) -> None:
        """Test creating a SprintStory with all fields."""
        story = SprintStory(
            story_id="2-3-auth-feature",
            title="Implement Authentication",
            priority_score=0.85,
            dependencies=("2-1-user-model", "2-2-db-setup"),
            estimated_points=5,
            value_score=0.9,
            tech_debt_score=0.3,
            velocity_impact=0.7,
            dependency_score=0.8,
            metadata={"epic": "auth", "sprint": 2},
        )
        assert story.story_id == "2-3-auth-feature"
        assert story.title == "Implement Authentication"
        assert story.priority_score == 0.85
        assert story.dependencies == ("2-1-user-model", "2-2-db-setup")
        assert story.estimated_points == 5
        assert story.value_score == 0.9
        assert story.tech_debt_score == 0.3
        assert story.velocity_impact == 0.7
        assert story.dependency_score == 0.8
        assert story.metadata == {"epic": "auth", "sprint": 2}

    def test_story_is_frozen(self) -> None:
        """Test that SprintStory is immutable (frozen)."""
        story = SprintStory(story_id="test", title="Test")
        with pytest.raises(AttributeError):
            story.story_id = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test SprintStory.to_dict() serialization."""
        story = SprintStory(
            story_id="1-1-test",
            title="Test Story",
            priority_score=0.75,
            dependencies=("1-0-setup",),
            estimated_points=3,
            value_score=0.8,
            tech_debt_score=0.2,
            velocity_impact=0.6,
            dependency_score=0.5,
            metadata={"key": "value"},
        )
        result = story.to_dict()

        assert result["story_id"] == "1-1-test"
        assert result["title"] == "Test Story"
        assert result["priority_score"] == 0.75
        assert result["dependencies"] == ("1-0-setup",)
        assert result["estimated_points"] == 3
        assert result["value_score"] == 0.8
        assert result["tech_debt_score"] == 0.2
        assert result["velocity_impact"] == 0.6
        assert result["dependency_score"] == 0.5
        assert result["metadata"] == {"key": "value"}


class TestSprintPlan:
    """Tests for SprintPlan dataclass."""

    def test_create_minimal_plan(self) -> None:
        """Test creating a SprintPlan with required fields only."""
        plan = SprintPlan(
            sprint_id="sprint-20260112",
            stories=(),
            total_points=0,
            capacity_used=0.0,
            planning_rationale="Empty sprint",
        )
        assert plan.sprint_id == "sprint-20260112"
        assert plan.stories == ()
        assert plan.total_points == 0
        assert plan.capacity_used == 0.0
        assert plan.planning_rationale == "Empty sprint"
        assert plan.created_at is not None  # Auto-generated

    def test_create_full_plan(self) -> None:
        """Test creating a SprintPlan with stories."""
        story1 = SprintStory(story_id="1-1", title="Story 1", estimated_points=3)
        story2 = SprintStory(story_id="1-2", title="Story 2", estimated_points=5)

        plan = SprintPlan(
            sprint_id="sprint-20260112",
            stories=(story1, story2),
            total_points=8,
            capacity_used=0.2,
            planning_rationale="Selected stories for sprint",
        )

        assert plan.sprint_id == "sprint-20260112"
        assert len(plan.stories) == 2
        assert plan.stories[0].story_id == "1-1"
        assert plan.stories[1].story_id == "1-2"
        assert plan.total_points == 8
        assert plan.capacity_used == 0.2

    def test_plan_is_frozen(self) -> None:
        """Test that SprintPlan is immutable (frozen)."""
        plan = SprintPlan(
            sprint_id="sprint-1",
            stories=(),
            total_points=0,
            capacity_used=0.0,
            planning_rationale="Test",
        )
        with pytest.raises(AttributeError):
            plan.sprint_id = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test SprintPlan.to_dict() serialization."""
        story = SprintStory(story_id="1-1", title="Story", estimated_points=2)
        plan = SprintPlan(
            sprint_id="sprint-20260112",
            stories=(story,),
            total_points=2,
            capacity_used=0.05,
            planning_rationale="Test plan",
        )
        result = plan.to_dict()

        assert result["sprint_id"] == "sprint-20260112"
        assert len(result["stories"]) == 1
        assert result["stories"][0]["story_id"] == "1-1"
        assert result["total_points"] == 2
        assert result["capacity_used"] == 0.05
        assert result["planning_rationale"] == "Test plan"
        assert "created_at" in result


class TestPlanningConfig:
    """Tests for PlanningConfig dataclass."""

    def test_default_config(self) -> None:
        """Test PlanningConfig with default values."""
        config = PlanningConfig()
        assert config.max_stories == DEFAULT_MAX_STORIES
        assert config.max_points == DEFAULT_MAX_POINTS
        assert config.value_weight == DEFAULT_VALUE_WEIGHT
        assert config.dependency_weight == DEFAULT_DEPENDENCY_WEIGHT
        assert config.velocity_weight == DEFAULT_VELOCITY_WEIGHT
        assert config.tech_debt_weight == DEFAULT_TECH_DEBT_WEIGHT

    def test_custom_config(self) -> None:
        """Test PlanningConfig with custom values."""
        config = PlanningConfig(
            max_stories=5,
            max_points=20,
            value_weight=0.5,
            dependency_weight=0.25,
            velocity_weight=0.15,
            tech_debt_weight=0.1,
        )
        assert config.max_stories == 5
        assert config.max_points == 20
        assert config.value_weight == 0.5
        assert config.dependency_weight == 0.25
        assert config.velocity_weight == 0.15
        assert config.tech_debt_weight == 0.1

    def test_config_is_frozen(self) -> None:
        """Test that PlanningConfig is immutable (frozen)."""
        config = PlanningConfig()
        with pytest.raises(AttributeError):
            config.max_stories = 5  # type: ignore[misc]

    def test_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        config = PlanningConfig()
        total = (
            config.value_weight
            + config.dependency_weight
            + config.velocity_weight
            + config.tech_debt_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_to_dict(self) -> None:
        """Test PlanningConfig.to_dict() serialization."""
        config = PlanningConfig(max_stories=8, max_points=30)
        result = config.to_dict()

        assert result["max_stories"] == 8
        assert result["max_points"] == 30
        assert "value_weight" in result
        assert "dependency_weight" in result
        assert "velocity_weight" in result
        assert "tech_debt_weight" in result


class TestConstants:
    """Tests for module constants."""

    def test_default_max_stories(self) -> None:
        """Test DEFAULT_MAX_STORIES is reasonable (5-10 per NFR-SCALE-1)."""
        assert 5 <= DEFAULT_MAX_STORIES <= 10

    def test_default_max_points(self) -> None:
        """Test DEFAULT_MAX_POINTS is reasonable for a sprint."""
        assert DEFAULT_MAX_POINTS > 0
        assert DEFAULT_MAX_POINTS <= 100

    def test_weights_are_floats(self) -> None:
        """Test all weight constants are floats between 0 and 1."""
        for weight in [
            DEFAULT_VALUE_WEIGHT,
            DEFAULT_DEPENDENCY_WEIGHT,
            DEFAULT_VELOCITY_WEIGHT,
            DEFAULT_TECH_DEBT_WEIGHT,
        ]:
            assert isinstance(weight, float)
            assert 0.0 <= weight <= 1.0
