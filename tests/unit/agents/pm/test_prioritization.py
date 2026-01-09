"""Tests for PM agent prioritization module (Story 6.4).

This module tests story prioritization including:
- Value score calculation
- Dependency analysis and cycle detection
- Dependency adjustment calculation
- Quick win identification
- Full prioritization orchestration

Test Organization:
    TestValueScoring: Tests for _calculate_value_score function
    TestDependencyAnalysis: Tests for _analyze_dependencies function
    TestDependencyAdjustment: Tests for _calculate_dependency_adjustment function
    TestQuickWinDetection: Tests for _is_quick_win function
    TestCycleDetection: Tests for _detect_cycles function
    TestPrioritizeStories: Tests for prioritize_stories function
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.pm.prioritization import (
    COMPLEXITY_ADJUSTMENTS,
    PRIORITY_BASE_SCORES,
    QUICK_WIN_COMPLEXITIES,
    QUICK_WIN_MAX_AC_COUNT,
    QUICK_WIN_MIN_SCORE,
    _analyze_dependencies,
    _calculate_dependency_adjustment,
    _calculate_value_score,
    _detect_cycles,
    _is_quick_win,
    prioritize_stories,
)
from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    DependencyInfo,
    Story,
    StoryPriority,
    StoryStatus,
)


def _make_ac(ac_id: str = "AC1") -> AcceptanceCriterion:
    """Create a simple acceptance criterion for testing."""
    return AcceptanceCriterion(
        id=ac_id,
        given="precondition",
        when="action",
        then="result",
    )


def _make_story(
    story_id: str = "story-001",
    priority: StoryPriority = StoryPriority.HIGH,
    complexity: str = "M",
    ac_count: int = 2,
    source_count: int = 1,
    dependencies: tuple[str, ...] = (),
) -> Story:
    """Create a story with configurable attributes for testing."""
    acs = tuple(_make_ac(f"AC{i}") for i in range(1, ac_count + 1))
    sources = tuple(f"req-{i:03d}" for i in range(1, source_count + 1))

    return Story(
        id=story_id,
        title=f"Test Story {story_id}",
        role="user",
        action="do something",
        benefit="get value",
        acceptance_criteria=acs,
        priority=priority,
        status=StoryStatus.DRAFT,
        source_requirements=sources,
        dependencies=dependencies,
        estimated_complexity=complexity,
    )


class TestValueScoring:
    """Tests for _calculate_value_score function."""

    def test_critical_priority_highest_base_score(self) -> None:
        """CRITICAL priority should yield highest base score."""
        story = _make_story(priority=StoryPriority.CRITICAL, complexity="M")
        score, rationale = _calculate_value_score(story)

        assert score == 100
        assert "CRITICAL priority (+100)" in rationale

    def test_high_priority_base_score(self) -> None:
        """HIGH priority should yield 75 base score."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="M")
        score, rationale = _calculate_value_score(story)

        assert score == 75
        assert "HIGH priority (+75)" in rationale

    def test_medium_priority_base_score(self) -> None:
        """MEDIUM priority should yield 50 base score."""
        story = _make_story(priority=StoryPriority.MEDIUM, complexity="M")
        score, rationale = _calculate_value_score(story)

        assert score == 50
        assert "MEDIUM priority (+50)" in rationale

    def test_low_priority_base_score(self) -> None:
        """LOW priority should yield 25 base score."""
        story = _make_story(priority=StoryPriority.LOW, complexity="M")
        score, rationale = _calculate_value_score(story)

        assert score == 25
        assert "LOW priority (+25)" in rationale

    def test_small_complexity_bonus(self) -> None:
        """S complexity should add +10 to score."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="S")
        score, rationale = _calculate_value_score(story)

        assert score == 85  # 75 + 10
        assert "S complexity (+10)" in rationale

    def test_medium_complexity_neutral(self) -> None:
        """M complexity should not change score."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="M")
        score, rationale = _calculate_value_score(story)

        assert score == 75  # 75 + 0
        # M complexity doesn't appear in rationale (no adjustment)
        assert "M complexity" not in " ".join(rationale)

    def test_large_complexity_penalty(self) -> None:
        """L complexity should subtract 10 from score."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="L")
        score, rationale = _calculate_value_score(story)

        assert score == 65  # 75 - 10
        assert "L complexity (-10)" in rationale

    def test_xl_complexity_penalty(self) -> None:
        """XL complexity should subtract 20 from score."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="XL")
        score, rationale = _calculate_value_score(story)

        assert score == 55  # 75 - 20
        assert "XL complexity (-20)" in rationale

    def test_optimal_ac_count_bonus(self) -> None:
        """Stories with 3-5 ACs should get +5 bonus."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="M", ac_count=4)
        score, rationale = _calculate_value_score(story)

        assert score == 80  # 75 + 5
        assert "4 ACs (+5)" in rationale

    def test_too_few_acs_no_bonus(self) -> None:
        """Stories with <3 ACs should not get AC bonus."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="M", ac_count=2)
        score, _ = _calculate_value_score(story)

        assert score == 75  # No bonus

    def test_too_many_acs_no_bonus(self) -> None:
        """Stories with >5 ACs should not get AC bonus."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="M", ac_count=6)
        score, _ = _calculate_value_score(story)

        assert score == 75  # No bonus

    def test_multi_source_bonus(self) -> None:
        """Stories with multiple source requirements get bonus."""
        story = _make_story(
            priority=StoryPriority.HIGH, complexity="M", source_count=3
        )
        score, rationale = _calculate_value_score(story)

        assert score == 85  # 75 + 5*(3-1)
        assert "3 source reqs (+10)" in rationale

    def test_single_source_no_bonus(self) -> None:
        """Stories with single source requirement get no bonus."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="M", source_count=1)
        score, _ = _calculate_value_score(story)

        assert score == 75

    def test_score_clamped_to_100(self) -> None:
        """Score should be clamped to 100 maximum."""
        # CRITICAL (100) + S (+10) + 4 ACs (+5) + 3 sources (+10) = 125
        story = _make_story(
            priority=StoryPriority.CRITICAL,
            complexity="S",
            ac_count=4,
            source_count=3,
        )
        score, _ = _calculate_value_score(story)

        assert score == 100  # Clamped

    def test_score_clamped_to_0(self) -> None:
        """Score should be clamped to 0 minimum."""
        # LOW (25) + XL (-20) = 5, can't go negative
        story = _make_story(priority=StoryPriority.LOW, complexity="XL")
        score, _ = _calculate_value_score(story)

        assert score >= 0

    def test_priority_base_scores_constant_values(self) -> None:
        """PRIORITY_BASE_SCORES should have correct values."""
        assert PRIORITY_BASE_SCORES[StoryPriority.CRITICAL] == 100
        assert PRIORITY_BASE_SCORES[StoryPriority.HIGH] == 75
        assert PRIORITY_BASE_SCORES[StoryPriority.MEDIUM] == 50
        assert PRIORITY_BASE_SCORES[StoryPriority.LOW] == 25

    def test_complexity_adjustments_constant_values(self) -> None:
        """COMPLEXITY_ADJUSTMENTS should have correct values."""
        assert COMPLEXITY_ADJUSTMENTS["S"] == 10
        assert COMPLEXITY_ADJUSTMENTS["M"] == 0
        assert COMPLEXITY_ADJUSTMENTS["L"] == -10
        assert COMPLEXITY_ADJUSTMENTS["XL"] == -20


class TestDependencyAnalysis:
    """Tests for _analyze_dependencies function."""

    def test_no_dependencies_returns_zero_counts(self) -> None:
        """Stories with no dependencies should have zero counts."""
        stories = (
            _make_story("story-001"),
            _make_story("story-002"),
        )

        dep_info, cycles = _analyze_dependencies(stories)

        assert dep_info["story-001"]["blocking_count"] == 0
        assert dep_info["story-001"]["blocked_by_count"] == 0
        assert dep_info["story-002"]["blocking_count"] == 0
        assert dep_info["story-002"]["blocked_by_count"] == 0
        assert cycles == []

    def test_linear_dependency_chain(self) -> None:
        """A -> B -> C: A blocks B, B blocks C."""
        stories = (
            _make_story("A"),
            _make_story("B", dependencies=("A",)),
            _make_story("C", dependencies=("B",)),
        )

        dep_info, cycles = _analyze_dependencies(stories)

        # A blocks B
        assert dep_info["A"]["blocking_count"] == 1
        assert dep_info["A"]["blocked_by_count"] == 0
        assert "B" in dep_info["A"]["blocking_story_ids"]

        # B blocked by A, blocks C
        assert dep_info["B"]["blocking_count"] == 1
        assert dep_info["B"]["blocked_by_count"] == 1
        assert "C" in dep_info["B"]["blocking_story_ids"]
        assert "A" in dep_info["B"]["blocked_by_story_ids"]

        # C blocked by B
        assert dep_info["C"]["blocking_count"] == 0
        assert dep_info["C"]["blocked_by_count"] == 1
        assert "B" in dep_info["C"]["blocked_by_story_ids"]

        assert cycles == []

    def test_story_blocks_multiple_stories(self) -> None:
        """Story that unblocks multiple stories."""
        stories = (
            _make_story("A"),
            _make_story("B", dependencies=("A",)),
            _make_story("C", dependencies=("A",)),
            _make_story("D", dependencies=("A",)),
        )

        dep_info, _ = _analyze_dependencies(stories)

        # A blocks 3 stories
        assert dep_info["A"]["blocking_count"] == 3
        assert set(dep_info["A"]["blocking_story_ids"]) == {"B", "C", "D"}

    def test_story_blocked_by_multiple_stories(self) -> None:
        """Story blocked by multiple dependencies."""
        stories = (
            _make_story("A"),
            _make_story("B"),
            _make_story("C"),
            _make_story("D", dependencies=("A", "B", "C")),
        )

        dep_info, _ = _analyze_dependencies(stories)

        # D blocked by 3 stories
        assert dep_info["D"]["blocked_by_count"] == 3
        assert set(dep_info["D"]["blocked_by_story_ids"]) == {"A", "B", "C"}

    def test_dependency_to_external_story_ignored(self) -> None:
        """Dependencies to stories not in the collection are ignored."""
        stories = (
            _make_story("A", dependencies=("external-story",)),
        )

        dep_info, _ = _analyze_dependencies(stories)

        # External dependency not counted
        assert dep_info["A"]["blocked_by_count"] == 0


class TestCycleDetection:
    """Tests for _detect_cycles function."""

    def test_no_cycles_in_dag(self) -> None:
        """No cycles in directed acyclic graph."""
        story_ids = {"A", "B", "C"}
        dependencies = {
            "A": [],
            "B": ["A"],
            "C": ["B"],
        }

        cycles = _detect_cycles(story_ids, dependencies)

        assert cycles == []

    def test_simple_cycle_detected(self) -> None:
        """A -> B -> A cycle is detected."""
        story_ids = {"A", "B"}
        dependencies = {
            "A": ["B"],
            "B": ["A"],
        }

        cycles = _detect_cycles(story_ids, dependencies)

        assert len(cycles) >= 1  # At least one cycle detected

    def test_three_node_cycle_detected(self) -> None:
        """A -> B -> C -> A cycle is detected."""
        story_ids = {"A", "B", "C"}
        dependencies = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],
        }

        cycles = _detect_cycles(story_ids, dependencies)

        assert len(cycles) >= 1

    def test_cycle_in_larger_graph(self) -> None:
        """Cycle detected in larger graph with non-cycle nodes."""
        story_ids = {"A", "B", "C", "D", "E"}
        dependencies = {
            "A": [],
            "B": ["A"],
            "C": ["B", "D"],  # C depends on B and D
            "D": ["E"],
            "E": ["D"],  # D <-> E cycle
        }

        cycles = _detect_cycles(story_ids, dependencies)

        assert len(cycles) >= 1

    def test_in_cycle_flag_set(self) -> None:
        """Stories in cycle should have in_cycle=True."""
        stories = (
            _make_story("A", dependencies=("B",)),
            _make_story("B", dependencies=("A",)),
            _make_story("C"),  # Not in cycle
        )

        dep_info, cycles = _analyze_dependencies(stories)

        assert dep_info["A"]["in_cycle"] is True
        assert dep_info["B"]["in_cycle"] is True
        assert dep_info["C"]["in_cycle"] is False
        assert len(cycles) >= 1


class TestDependencyAdjustment:
    """Tests for _calculate_dependency_adjustment function."""

    def test_no_dependencies_zero_adjustment(self) -> None:
        """No dependencies means zero adjustment."""
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        adjustment, rationale = _calculate_dependency_adjustment(dep_info)

        assert adjustment == 0
        assert rationale == []

    def test_blocks_many_stories_plus_20(self) -> None:
        """Blocking 3+ stories gives +20 adjustment."""
        dep_info: DependencyInfo = {
            "blocking_count": 4,
            "blocked_by_count": 0,
            "blocking_story_ids": ["B", "C", "D", "E"],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        adjustment, rationale = _calculate_dependency_adjustment(dep_info)

        assert adjustment == 20
        assert "unblocks 4 stories (+20)" in rationale

    def test_blocks_few_stories_plus_10(self) -> None:
        """Blocking 1-2 stories gives +10 adjustment."""
        dep_info: DependencyInfo = {
            "blocking_count": 2,
            "blocked_by_count": 0,
            "blocking_story_ids": ["B", "C"],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        adjustment, rationale = _calculate_dependency_adjustment(dep_info)

        assert adjustment == 10
        assert "unblocks 2 stories (+10)" in rationale

    def test_blocked_by_many_minus_20(self) -> None:
        """Blocked by 3+ stories gives -20 adjustment."""
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 4,
            "blocking_story_ids": [],
            "blocked_by_story_ids": ["A", "B", "C", "D"],
            "in_cycle": False,
        }

        adjustment, rationale = _calculate_dependency_adjustment(dep_info)

        assert adjustment == -20
        assert "blocked by 4 stories (-20)" in rationale

    def test_blocked_by_few_minus_10(self) -> None:
        """Blocked by 1-2 stories gives -10 adjustment."""
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 2,
            "blocking_story_ids": [],
            "blocked_by_story_ids": ["A", "B"],
            "in_cycle": False,
        }

        adjustment, rationale = _calculate_dependency_adjustment(dep_info)

        assert adjustment == -10
        assert "blocked by 2 stories (-10)" in rationale

    def test_in_cycle_minus_5(self) -> None:
        """Being in a cycle gives -5 adjustment."""
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": True,
        }

        adjustment, rationale = _calculate_dependency_adjustment(dep_info)

        assert adjustment == -5
        assert "in dependency cycle (-5)" in rationale

    def test_combined_adjustments_clamped_positive(self) -> None:
        """Combined positive adjustments are clamped to +20."""
        # Theoretically could exceed +20, but we don't have that case
        dep_info: DependencyInfo = {
            "blocking_count": 5,  # +20
            "blocked_by_count": 0,
            "blocking_story_ids": ["B", "C", "D", "E", "F"],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        adjustment, _ = _calculate_dependency_adjustment(dep_info)

        assert adjustment <= 20

    def test_combined_adjustments_clamped_negative(self) -> None:
        """Combined negative adjustments are clamped to -20."""
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 5,  # -20
            "blocking_story_ids": [],
            "blocked_by_story_ids": ["A", "B", "C", "D", "E"],
            "in_cycle": True,  # -5
        }

        adjustment, _ = _calculate_dependency_adjustment(dep_info)

        assert adjustment >= -20


class TestQuickWinDetection:
    """Tests for _is_quick_win function."""

    def test_high_priority_small_complexity_is_quick_win(self) -> None:
        """HIGH priority + S complexity + no blockers = quick win."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="S", ac_count=3)
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        raw_score = 85  # HIGH (75) + S (+10)
        is_quick = _is_quick_win(story, raw_score, dep_info)

        assert is_quick is True

    def test_critical_priority_medium_complexity_is_quick_win(self) -> None:
        """CRITICAL priority + M complexity = quick win."""
        story = _make_story(priority=StoryPriority.CRITICAL, complexity="M", ac_count=3)
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        raw_score = 100
        is_quick = _is_quick_win(story, raw_score, dep_info)

        assert is_quick is True

    def test_xl_complexity_not_quick_win(self) -> None:
        """XL complexity is not a quick win regardless of priority."""
        story = _make_story(priority=StoryPriority.CRITICAL, complexity="XL", ac_count=3)
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        raw_score = 80
        is_quick = _is_quick_win(story, raw_score, dep_info)

        assert is_quick is False

    def test_large_complexity_not_quick_win(self) -> None:
        """L complexity is not a quick win."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="L", ac_count=3)
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        raw_score = 65
        is_quick = _is_quick_win(story, raw_score, dep_info)

        assert is_quick is False

    def test_low_score_not_quick_win(self) -> None:
        """Score below threshold is not a quick win."""
        story = _make_story(priority=StoryPriority.LOW, complexity="S", ac_count=3)
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        raw_score = 35  # LOW (25) + S (+10)
        is_quick = _is_quick_win(story, raw_score, dep_info)

        assert is_quick is False

    def test_blocked_story_not_quick_win(self) -> None:
        """Blocked story is not a quick win."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="S", ac_count=3)
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 1,
            "blocking_story_ids": [],
            "blocked_by_story_ids": ["other-story"],
            "in_cycle": False,
        }

        raw_score = 85
        is_quick = _is_quick_win(story, raw_score, dep_info)

        assert is_quick is False

    def test_too_many_acs_not_quick_win(self) -> None:
        """More than 4 ACs is not a quick win."""
        story = _make_story(priority=StoryPriority.HIGH, complexity="S", ac_count=5)
        dep_info: DependencyInfo = {
            "blocking_count": 0,
            "blocked_by_count": 0,
            "blocking_story_ids": [],
            "blocked_by_story_ids": [],
            "in_cycle": False,
        }

        raw_score = 85
        is_quick = _is_quick_win(story, raw_score, dep_info)

        assert is_quick is False

    def test_quick_win_constants(self) -> None:
        """Quick win threshold constants should have correct values."""
        assert QUICK_WIN_MIN_SCORE == 60
        assert QUICK_WIN_MAX_AC_COUNT == 4
        assert QUICK_WIN_COMPLEXITIES == frozenset({"S", "M"})


class TestPrioritizeStories:
    """Tests for prioritize_stories function."""

    def test_empty_stories_returns_empty_result(self) -> None:
        """Empty story list returns empty result."""
        result = prioritize_stories(())

        assert result["scores"] == []
        assert result["recommended_execution_order"] == []
        assert result["quick_wins"] == []
        assert result["dependency_cycles"] == []
        assert result["analysis_notes"] == "No stories to prioritize"

    def test_single_story_returns_single_result(self) -> None:
        """Single story returns single result."""
        story = _make_story("story-001", priority=StoryPriority.HIGH)
        result = prioritize_stories((story,))

        assert len(result["scores"]) == 1
        assert result["recommended_execution_order"] == ["story-001"]
        assert result["scores"][0]["story_id"] == "story-001"

    def test_stories_sorted_by_final_score_descending(self) -> None:
        """Stories should be sorted by final score descending."""
        stories = (
            _make_story("low", priority=StoryPriority.LOW),
            _make_story("high", priority=StoryPriority.HIGH),
            _make_story("critical", priority=StoryPriority.CRITICAL),
        )

        result = prioritize_stories(stories)

        assert result["recommended_execution_order"] == ["critical", "high", "low"]

    def test_priority_score_fields_populated(self) -> None:
        """PriorityScore should have all required fields."""
        story = _make_story("story-001", priority=StoryPriority.HIGH, complexity="S")
        result = prioritize_stories((story,))

        score = result["scores"][0]
        assert score["story_id"] == "story-001"
        assert isinstance(score["raw_score"], int)
        assert isinstance(score["dependency_adjustment"], int)
        assert isinstance(score["final_score"], int)
        assert isinstance(score["is_quick_win"], bool)
        assert isinstance(score["scoring_rationale"], str)

    def test_final_score_is_clamped(self) -> None:
        """Final score should be clamped to 0-100."""
        # Very high scoring story
        story = _make_story(
            "story-001",
            priority=StoryPriority.CRITICAL,
            complexity="S",
            ac_count=4,
            source_count=3,
        )
        result = prioritize_stories((story,))

        assert result["scores"][0]["final_score"] <= 100
        assert result["scores"][0]["final_score"] >= 0

    def test_quick_wins_identified(self) -> None:
        """Quick wins should be identified and listed."""
        stories = (
            _make_story("quick", priority=StoryPriority.HIGH, complexity="S", ac_count=3),
            _make_story("slow", priority=StoryPriority.LOW, complexity="XL", ac_count=6),
        )

        result = prioritize_stories(stories)

        assert "quick" in result["quick_wins"]
        assert "slow" not in result["quick_wins"]

    def test_dependency_cycles_detected(self) -> None:
        """Dependency cycles should be detected and reported."""
        stories = (
            _make_story("A", dependencies=("B",)),
            _make_story("B", dependencies=("A",)),
        )

        result = prioritize_stories(stories)

        assert len(result["dependency_cycles"]) >= 1

    def test_scoring_rationale_includes_factors(self) -> None:
        """Scoring rationale should include relevant factors."""
        story = _make_story(
            "story-001",
            priority=StoryPriority.HIGH,
            complexity="S",
            ac_count=4,
        )
        result = prioritize_stories((story,))

        rationale = result["scores"][0]["scoring_rationale"]
        assert "HIGH priority" in rationale
        assert "S complexity" in rationale
        assert "4 ACs" in rationale

    def test_analysis_notes_summary(self) -> None:
        """Analysis notes should summarize the prioritization."""
        stories = (
            _make_story("a", priority=StoryPriority.HIGH, complexity="S", ac_count=3),
            _make_story("b", priority=StoryPriority.MEDIUM, complexity="M"),
        )

        result = prioritize_stories(stories)

        assert "Prioritized 2 stories" in result["analysis_notes"]
        assert "quick win" in result["analysis_notes"].lower()

    def test_dependency_adjustment_affects_order(self) -> None:
        """Stories with positive dependency adjustment should rank higher."""
        stories = (
            _make_story("unblocks-many", priority=StoryPriority.MEDIUM),
            _make_story("blocked", priority=StoryPriority.MEDIUM, dependencies=("unblocks-many",)),
            _make_story("also-blocked", priority=StoryPriority.MEDIUM, dependencies=("unblocks-many",)),
            _make_story("also-blocked-2", priority=StoryPriority.MEDIUM, dependencies=("unblocks-many",)),
        )

        result = prioritize_stories(stories)

        # "unblocks-many" blocks 3 stories, gets +20 adjustment
        unblocks_score = next(
            s for s in result["scores"] if s["story_id"] == "unblocks-many"
        )
        assert unblocks_score["dependency_adjustment"] == 20
        assert result["recommended_execution_order"][0] == "unblocks-many"

    def test_stories_unchanged_by_prioritization(self) -> None:
        """Original stories should not be modified by prioritization."""
        original_story = _make_story("story-001")
        original_id = original_story.id
        original_priority = original_story.priority

        prioritize_stories((original_story,))

        assert original_story.id == original_id
        assert original_story.priority == original_priority
