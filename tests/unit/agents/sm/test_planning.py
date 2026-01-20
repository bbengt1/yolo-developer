"""Tests for sprint planning module (Story 10.3).

Tests the sprint planning functions:
- _calculate_priority_score: Weighted priority scoring per FR65
- _analyze_dependencies: Build dependency graph
- _topological_sort: Dependency-aware ordering
- _check_capacity: Sprint capacity management
- plan_sprint: Main planning function
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.sm.planning import (
    CircularDependencyError,
    _analyze_dependencies,
    _calculate_priority_score,
    _check_capacity,
    _dict_to_sprint_story,
    _generate_planning_rationale,
    _topological_sort,
    plan_sprint,
)
from yolo_developer.agents.sm.planning_types import (
    PlanningConfig,
    SprintStory,
)


class TestCalculatePriorityScore:
    """Tests for _calculate_priority_score function (AC #1, FR65)."""

    def test_default_weights(self) -> None:
        """Test priority calculation with default weights."""
        story = SprintStory(
            story_id="test",
            title="Test",
            value_score=1.0,
            dependency_score=1.0,
            velocity_impact=1.0,
            tech_debt_score=1.0,
        )
        config = PlanningConfig()
        score = _calculate_priority_score(story, config)
        # 1.0 * (0.4 + 0.3 + 0.2 + 0.1) = 1.0
        assert abs(score - 1.0) < 0.001

    def test_partial_scores(self) -> None:
        """Test priority calculation with partial scores."""
        story = SprintStory(
            story_id="test",
            title="Test",
            value_score=0.8,
            dependency_score=0.5,
            velocity_impact=0.6,
            tech_debt_score=0.2,
        )
        config = PlanningConfig()
        # 0.8*0.4 + 0.5*0.3 + 0.6*0.2 + 0.2*0.1 = 0.32 + 0.15 + 0.12 + 0.02 = 0.61
        score = _calculate_priority_score(story, config)
        assert abs(score - 0.61) < 0.001

    def test_custom_weights(self) -> None:
        """Test priority calculation with custom weights."""
        story = SprintStory(
            story_id="test",
            title="Test",
            value_score=1.0,
            dependency_score=0.0,
            velocity_impact=0.0,
            tech_debt_score=0.0,
        )
        config = PlanningConfig(
            value_weight=1.0,
            dependency_weight=0.0,
            velocity_weight=0.0,
            tech_debt_weight=0.0,
        )
        score = _calculate_priority_score(story, config)
        assert abs(score - 1.0) < 0.001

    def test_zero_scores(self) -> None:
        """Test priority calculation with all zero scores."""
        story = SprintStory(
            story_id="test",
            title="Test",
            value_score=0.0,
            dependency_score=0.0,
            velocity_impact=0.0,
            tech_debt_score=0.0,
        )
        config = PlanningConfig()
        score = _calculate_priority_score(story, config)
        assert score == 0.0


class TestAnalyzeDependencies:
    """Tests for _analyze_dependencies function (AC #1)."""

    def test_no_dependencies(self) -> None:
        """Test stories with no dependencies."""
        stories = [
            SprintStory(story_id="1-1", title="Story 1"),
            SprintStory(story_id="1-2", title="Story 2"),
        ]
        graph, in_degree, story_map = _analyze_dependencies(stories)

        assert len(story_map) == 2
        assert in_degree["1-1"] == 0
        assert in_degree["1-2"] == 0
        assert graph == {}

    def test_simple_dependency_chain(self) -> None:
        """Test linear dependency chain: A -> B -> C."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B", dependencies=("A",)),
            SprintStory(story_id="C", title="C", dependencies=("B",)),
        ]
        graph, in_degree, _story_map = _analyze_dependencies(stories)

        assert in_degree["A"] == 0
        assert in_degree["B"] == 1
        assert in_degree["C"] == 1
        assert "B" in graph["A"]
        assert "C" in graph["B"]

    def test_multiple_dependencies(self) -> None:
        """Test story depending on multiple others."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B"),
            SprintStory(story_id="C", title="C", dependencies=("A", "B")),
        ]
        graph, in_degree, _story_map = _analyze_dependencies(stories)

        assert in_degree["A"] == 0
        assert in_degree["B"] == 0
        assert in_degree["C"] == 2
        assert "C" in graph["A"]
        assert "C" in graph["B"]

    def test_dependency_outside_sprint(self) -> None:
        """Test dependency on story not in sprint (ignored)."""
        stories = [
            SprintStory(story_id="B", title="B", dependencies=("A",)),  # A not in sprint
        ]
        _graph, in_degree, story_map = _analyze_dependencies(stories)

        assert in_degree["B"] == 0  # External dependency ignored
        assert "A" not in story_map


class TestTopologicalSort:
    """Tests for _topological_sort function (AC #2)."""

    def test_no_dependencies(self) -> None:
        """Test sorting stories with no dependencies (by priority)."""
        stories = [
            SprintStory(story_id="low", title="Low", priority_score=0.3),
            SprintStory(story_id="high", title="High", priority_score=0.9),
            SprintStory(story_id="med", title="Med", priority_score=0.5),
        ]
        result = _topological_sort(stories)

        # Should be sorted by priority descending
        assert result[0].story_id == "high"
        assert result[1].story_id == "med"
        assert result[2].story_id == "low"

    def test_respects_dependencies(self) -> None:
        """Test that dependencies are respected over priority."""
        stories = [
            SprintStory(story_id="A", title="A", priority_score=0.3),
            SprintStory(story_id="B", title="B", priority_score=0.9, dependencies=("A",)),
        ]
        result = _topological_sort(stories)

        # A must come before B despite lower priority
        assert result[0].story_id == "A"
        assert result[1].story_id == "B"

    def test_complex_dag(self) -> None:
        """Test complex dependency graph."""
        #   A(p=0.5)  B(p=0.8)
        #      \    /  |
        #       C(p=0.9)
        #         |
        #       D(p=0.7)
        stories = [
            SprintStory(story_id="A", title="A", priority_score=0.5),
            SprintStory(story_id="B", title="B", priority_score=0.8),
            SprintStory(story_id="C", title="C", priority_score=0.9, dependencies=("A", "B")),
            SprintStory(story_id="D", title="D", priority_score=0.7, dependencies=("C",)),
        ]
        result = _topological_sort(stories)

        # B should come before A (higher priority, both have 0 deps)
        # C must come after both A and B
        # D must come after C
        a_idx = next(i for i, s in enumerate(result) if s.story_id == "A")
        b_idx = next(i for i, s in enumerate(result) if s.story_id == "B")
        c_idx = next(i for i, s in enumerate(result) if s.story_id == "C")
        d_idx = next(i for i, s in enumerate(result) if s.story_id == "D")

        assert b_idx < c_idx
        assert a_idx < c_idx
        assert c_idx < d_idx

    def test_circular_dependency_detection(self) -> None:
        """Test that circular dependencies are detected."""
        stories = [
            SprintStory(story_id="A", title="A", dependencies=("B",)),
            SprintStory(story_id="B", title="B", dependencies=("A",)),
        ]
        with pytest.raises(CircularDependencyError) as exc_info:
            _topological_sort(stories)
        assert "Circular dependency" in str(exc_info.value)

    def test_self_dependency(self) -> None:
        """Test that self-dependency is handled."""
        stories = [
            SprintStory(story_id="A", title="A", dependencies=("A",)),
        ]
        with pytest.raises(CircularDependencyError):
            _topological_sort(stories)


class TestCheckCapacity:
    """Tests for _check_capacity function (AC #3)."""

    def test_fits_capacity(self) -> None:
        """Test stories that fit within capacity."""
        stories = [
            SprintStory(story_id="1", title="1", estimated_points=3),
            SprintStory(story_id="2", title="2", estimated_points=5),
        ]
        config = PlanningConfig(max_stories=10, max_points=40)
        selected, total = _check_capacity(stories, config)

        assert len(selected) == 2
        assert total == 8

    def test_exceeds_max_stories(self) -> None:
        """Test capacity limit by story count."""
        stories = [
            SprintStory(story_id=str(i), title=str(i), estimated_points=1) for i in range(15)
        ]
        config = PlanningConfig(max_stories=10, max_points=100)
        selected, total = _check_capacity(stories, config)

        assert len(selected) == 10
        assert total == 10

    def test_exceeds_max_points(self) -> None:
        """Test capacity limit by story points."""
        stories = [
            SprintStory(story_id="1", title="1", estimated_points=20),
            SprintStory(story_id="2", title="2", estimated_points=15),
            SprintStory(story_id="3", title="3", estimated_points=10),  # Would exceed
        ]
        config = PlanningConfig(max_stories=10, max_points=35)
        selected, total = _check_capacity(stories, config)

        assert len(selected) == 2
        assert total == 35

    def test_skips_large_story(self) -> None:
        """Test skipping a large story to fit smaller ones."""
        stories = [
            SprintStory(story_id="big", title="Big", estimated_points=30),
            SprintStory(story_id="small", title="Small", estimated_points=5),
        ]
        config = PlanningConfig(max_stories=10, max_points=10)
        selected, total = _check_capacity(stories, config)

        # Big story skipped, small one fits
        assert len(selected) == 1
        assert selected[0].story_id == "small"
        assert total == 5

    def test_empty_stories(self) -> None:
        """Test with no stories."""
        config = PlanningConfig()
        selected, total = _check_capacity([], config)

        assert selected == []
        assert total == 0


class TestDictToSprintStory:
    """Tests for _dict_to_sprint_story conversion."""

    def test_minimal_dict(self) -> None:
        """Test conversion with minimal fields."""
        data = {"story_id": "1-1", "title": "Test"}
        story = _dict_to_sprint_story(data)

        assert story.story_id == "1-1"
        assert story.title == "Test"
        assert story.estimated_points == 1

    def test_missing_story_id_raises_error(self) -> None:
        """Test that missing story_id raises ValueError."""
        data = {"title": "Test Story"}
        with pytest.raises(ValueError) as exc_info:
            _dict_to_sprint_story(data)
        assert "story_id" in str(exc_info.value)

    def test_missing_title_raises_error(self) -> None:
        """Test that missing title raises ValueError."""
        data = {"story_id": "1-1"}
        with pytest.raises(ValueError) as exc_info:
            _dict_to_sprint_story(data)
        assert "title" in str(exc_info.value)

    def test_full_dict(self) -> None:
        """Test conversion with all fields."""
        data = {
            "story_id": "2-1",
            "title": "Full Story",
            "priority_score": 0.8,
            "dependencies": ["1-1", "1-2"],
            "estimated_points": 5,
            "value_score": 0.9,
            "tech_debt_score": 0.2,
            "velocity_impact": 0.7,
            "dependency_score": 0.5,
            "metadata": {"epic": "auth"},
        }
        story = _dict_to_sprint_story(data)

        assert story.story_id == "2-1"
        assert story.title == "Full Story"
        assert story.priority_score == 0.8
        assert story.dependencies == ("1-1", "1-2")
        assert story.estimated_points == 5
        assert story.value_score == 0.9
        assert story.tech_debt_score == 0.2
        assert story.velocity_impact == 0.7
        assert story.dependency_score == 0.5
        assert story.metadata == {"epic": "auth"}


class TestGeneratePlanningRationale:
    """Tests for _generate_planning_rationale function (AC #4)."""

    def test_generates_rationale(self) -> None:
        """Test rationale generation includes key information."""
        stories = [
            SprintStory(story_id="1-1", title="Story 1", estimated_points=3),
            SprintStory(story_id="1-2", title="Story 2", estimated_points=5),
        ]
        config = PlanningConfig(max_stories=10, max_points=40)
        rationale = _generate_planning_rationale(stories, 8, config)

        assert "2" in rationale  # Story count
        assert "8" in rationale  # Total points
        assert "40" in rationale  # Capacity


class TestPlanSprint:
    """Tests for plan_sprint main function (AC #2, #4)."""

    @pytest.mark.asyncio
    async def test_basic_planning(self) -> None:
        """Test basic sprint planning."""
        stories = [
            {"story_id": "1-1", "title": "Story 1", "estimated_points": 3},
            {"story_id": "1-2", "title": "Story 2", "estimated_points": 5},
        ]
        plan = await plan_sprint(stories)

        assert plan.sprint_id.startswith("sprint-")
        assert len(plan.stories) == 2
        assert plan.total_points == 8
        assert 0.0 <= plan.capacity_used <= 1.0
        assert plan.planning_rationale != ""

    @pytest.mark.asyncio
    async def test_planning_with_dependencies(self) -> None:
        """Test sprint planning respects dependencies."""
        stories = [
            {"story_id": "B", "title": "B", "dependencies": ["A"], "priority_score": 0.9},
            {"story_id": "A", "title": "A", "priority_score": 0.5},
        ]
        plan = await plan_sprint(stories)

        # A should come before B despite lower priority
        a_idx = next(i for i, s in enumerate(plan.stories) if s.story_id == "A")
        b_idx = next(i for i, s in enumerate(plan.stories) if s.story_id == "B")
        assert a_idx < b_idx

    @pytest.mark.asyncio
    async def test_planning_with_capacity_limit(self) -> None:
        """Test sprint planning respects capacity."""
        stories = [{"story_id": str(i), "title": str(i), "estimated_points": 5} for i in range(20)]
        config = PlanningConfig(max_stories=5, max_points=25)
        plan = await plan_sprint(stories, config)

        assert len(plan.stories) == 5
        assert plan.total_points == 25

    @pytest.mark.asyncio
    async def test_planning_with_circular_dependency(self) -> None:
        """Test sprint planning detects circular dependencies."""
        stories = [
            {"story_id": "A", "title": "A", "dependencies": ["B"]},
            {"story_id": "B", "title": "B", "dependencies": ["A"]},
        ]
        with pytest.raises(CircularDependencyError):
            await plan_sprint(stories)

    @pytest.mark.asyncio
    async def test_empty_stories(self) -> None:
        """Test planning with no stories."""
        plan = await plan_sprint([])

        assert len(plan.stories) == 0
        assert plan.total_points == 0
        assert plan.capacity_used == 0.0

    @pytest.mark.asyncio
    async def test_custom_config(self) -> None:
        """Test planning with custom config."""
        stories = [
            {"story_id": "1", "title": "1", "estimated_points": 5},
        ]
        config = PlanningConfig(max_points=10)
        plan = await plan_sprint(stories, config)

        assert plan.capacity_used == 0.5  # 5/10

    @pytest.mark.asyncio
    async def test_priority_recalculation(self) -> None:
        """Test that priorities are recalculated during planning."""
        stories = [
            {
                "story_id": "high-value",
                "title": "High Value",
                "value_score": 1.0,
                "dependency_score": 0.0,
                "velocity_impact": 0.0,
                "tech_debt_score": 0.0,
            },
            {
                "story_id": "low-value",
                "title": "Low Value",
                "value_score": 0.1,
                "dependency_score": 0.0,
                "velocity_impact": 0.0,
                "tech_debt_score": 0.0,
            },
        ]
        plan = await plan_sprint(stories)

        # High value should come first
        assert plan.stories[0].story_id == "high-value"
        assert plan.stories[1].story_id == "low-value"


class TestSprintPlanAuditLogging:
    """Tests for sprint plan audit logging (AC #4)."""

    @pytest.mark.asyncio
    async def test_plan_includes_created_at(self) -> None:
        """Test that sprint plan includes creation timestamp."""
        stories = [{"story_id": "1-1", "title": "Story 1"}]
        plan = await plan_sprint(stories)

        assert plan.created_at is not None
        # ISO format check
        assert "T" in plan.created_at
        assert "Z" in plan.created_at or "+" in plan.created_at or plan.created_at.endswith("00:00")

    @pytest.mark.asyncio
    async def test_plan_rationale_explains_decisions(self) -> None:
        """Test that planning rationale explains key decisions."""
        stories = [
            {"story_id": "1-1", "title": "Story 1", "estimated_points": 5},
            {"story_id": "1-2", "title": "Story 2", "estimated_points": 3, "dependencies": ["1-1"]},
        ]
        plan = await plan_sprint(stories)

        # Rationale should mention story count, points, and dependency ordering
        assert "2" in plan.planning_rationale  # Story count
        assert "8" in plan.planning_rationale  # Total points
        assert "dependencies" in plan.planning_rationale.lower()

    @pytest.mark.asyncio
    async def test_plan_to_dict_for_serialization(self) -> None:
        """Test that sprint plan can be serialized for audit."""
        stories = [{"story_id": "1-1", "title": "Story 1"}]
        plan = await plan_sprint(stories)

        plan_dict = plan.to_dict()

        assert "sprint_id" in plan_dict
        assert "stories" in plan_dict
        assert "total_points" in plan_dict
        assert "capacity_used" in plan_dict
        assert "planning_rationale" in plan_dict
        assert "created_at" in plan_dict
        assert isinstance(plan_dict["stories"], list)
