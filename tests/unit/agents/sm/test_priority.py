"""Tests for priority scoring module (Story 10.11).

Tests the priority scoring functions:
- calculate_dependency_score: How many stories depend on this one
- calculate_dependency_scores: Batch dependency calculation
- calculate_priority_score: Weighted priority scoring per FR65
- normalize_scores: Min-max normalization
- normalize_results: Normalize PriorityResult objects
- score_stories: End-to-end scoring and ordering
- update_stories_with_scores: Update SprintStory with scores
- _generate_score_explanation: Audit trail generation

References:
    - FR65: SM Agent can calculate weighted priority scores for story selection
    - Story 10.3: Sprint Planning (original _calculate_priority_score)
"""

from __future__ import annotations

from yolo_developer.agents.sm.planning_types import SprintStory
from yolo_developer.agents.sm.priority import (
    _generate_score_explanation,
    calculate_dependency_score,
    calculate_dependency_scores,
    calculate_priority_score,
    normalize_results,
    normalize_scores,
    score_stories,
    update_stories_with_scores,
)
from yolo_developer.agents.sm.priority_types import (
    DEFAULT_DEPENDENCY_WEIGHT,
    DEFAULT_TECH_DEBT_WEIGHT,
    DEFAULT_VALUE_WEIGHT,
    DEFAULT_VELOCITY_WEIGHT,
    MAX_SCORE,
    MIN_SCORE,
    PriorityFactors,
    PriorityResult,
    PriorityScoringConfig,
)


class TestPriorityTypes:
    """Tests for priority scoring types (Task 1)."""

    def test_priority_factors_defaults(self) -> None:
        """Test PriorityFactors with default values."""
        factors = PriorityFactors(story_id="test-1")
        assert factors.story_id == "test-1"
        assert factors.value_score == 0.5
        assert factors.dependency_score == 0.0
        assert factors.velocity_impact == 0.5
        assert factors.tech_debt_score == 0.0

    def test_priority_factors_custom(self) -> None:
        """Test PriorityFactors with custom values."""
        factors = PriorityFactors(
            story_id="test-2",
            value_score=0.9,
            dependency_score=0.5,
            velocity_impact=0.7,
            tech_debt_score=0.3,
        )
        assert factors.value_score == 0.9
        assert factors.dependency_score == 0.5

    def test_priority_factors_to_dict(self) -> None:
        """Test PriorityFactors serialization."""
        factors = PriorityFactors(story_id="test-3", value_score=0.8)
        d = factors.to_dict()
        assert d["story_id"] == "test-3"
        assert d["value_score"] == 0.8
        assert "dependency_score" in d

    def test_priority_result_defaults(self) -> None:
        """Test PriorityResult with defaults."""
        result = PriorityResult(
            story_id="test-1",
            priority_score=0.75,
        )
        assert result.story_id == "test-1"
        assert result.priority_score == 0.75
        assert result.normalized_score is None
        assert result.explanation == ""
        assert result.factor_contributions == {}

    def test_priority_result_full(self) -> None:
        """Test PriorityResult with all fields."""
        result = PriorityResult(
            story_id="test-2",
            priority_score=0.65,
            normalized_score=0.8,
            explanation="High value story",
            factor_contributions={"value": 0.32, "dependency": 0.18},
        )
        assert result.normalized_score == 0.8
        assert "value" in result.factor_contributions

    def test_priority_result_to_dict(self) -> None:
        """Test PriorityResult serialization."""
        result = PriorityResult(
            story_id="test-3",
            priority_score=0.5,
            factor_contributions={"value": 0.2},
        )
        d = result.to_dict()
        assert d["story_id"] == "test-3"
        assert d["priority_score"] == 0.5
        assert d["factor_contributions"] == {"value": 0.2}

    def test_priority_scoring_config_defaults(self) -> None:
        """Test PriorityScoringConfig defaults match planning defaults."""
        config = PriorityScoringConfig()
        assert config.value_weight == DEFAULT_VALUE_WEIGHT
        assert config.dependency_weight == DEFAULT_DEPENDENCY_WEIGHT
        assert config.velocity_weight == DEFAULT_VELOCITY_WEIGHT
        assert config.tech_debt_weight == DEFAULT_TECH_DEBT_WEIGHT
        assert config.normalize_scores is True
        assert config.include_explanation is True
        assert config.min_score_threshold == 0.0

    def test_priority_scoring_config_custom(self) -> None:
        """Test PriorityScoringConfig with custom values."""
        config = PriorityScoringConfig(
            value_weight=0.5,
            normalize_scores=False,
            min_score_threshold=0.3,
        )
        assert config.value_weight == 0.5
        assert config.normalize_scores is False
        assert config.min_score_threshold == 0.3

    def test_priority_scoring_config_to_dict(self) -> None:
        """Test PriorityScoringConfig serialization."""
        config = PriorityScoringConfig(value_weight=0.6)
        d = config.to_dict()
        assert d["value_weight"] == 0.6
        assert "normalize_scores" in d

    def test_constants_valid_range(self) -> None:
        """Test scoring constants are valid."""
        assert MIN_SCORE == 0.0
        assert MAX_SCORE == 1.0
        assert MIN_SCORE < MAX_SCORE


class TestCalculateDependencyScore:
    """Tests for calculate_dependency_score function (Task 2.1)."""

    def test_no_dependents(self) -> None:
        """Test story with no dependents returns 0."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B"),
            SprintStory(story_id="C", title="C"),
        ]
        score = calculate_dependency_score("A", stories)
        assert score == 0.0

    def test_single_dependent(self) -> None:
        """Test story with one dependent."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B", dependencies=("A",)),
            SprintStory(story_id="C", title="C"),
        ]
        score = calculate_dependency_score("A", stories)
        # 1 dependent out of 2 other stories = 0.5
        assert abs(score - 0.5) < 0.001

    def test_multiple_dependents(self) -> None:
        """Test story that many depend on."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B", dependencies=("A",)),
            SprintStory(story_id="C", title="C", dependencies=("A",)),
        ]
        score = calculate_dependency_score("A", stories)
        # 2 dependents out of 2 other stories = 1.0
        assert score == 1.0

    def test_all_depend_on_one(self) -> None:
        """Test story that all others depend on."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B", dependencies=("A",)),
            SprintStory(story_id="C", title="C", dependencies=("A",)),
            SprintStory(story_id="D", title="D", dependencies=("A",)),
        ]
        score = calculate_dependency_score("A", stories)
        assert score == 1.0

    def test_empty_stories(self) -> None:
        """Test with empty story list."""
        score = calculate_dependency_score("A", [])
        assert score == 0.0

    def test_single_story(self) -> None:
        """Test with only one story."""
        stories = [SprintStory(story_id="A", title="A")]
        score = calculate_dependency_score("A", stories)
        assert score == 0.0

    def test_story_not_in_list(self) -> None:
        """Test with story_id not in the list."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B"),
        ]
        score = calculate_dependency_score("X", stories)
        assert score == 0.0


class TestCalculateDependencyScores:
    """Tests for calculate_dependency_scores function (Task 2.1)."""

    def test_multiple_stories(self) -> None:
        """Test calculating dependency scores for all stories."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B", dependencies=("A",)),
            SprintStory(story_id="C", title="C", dependencies=("A", "B")),
        ]
        scores = calculate_dependency_scores(stories)
        assert "A" in scores
        assert "B" in scores
        assert "C" in scores
        # A has B and C depending on it = 1.0
        assert scores["A"] == 1.0
        # B has C depending on it = 0.5
        assert abs(scores["B"] - 0.5) < 0.001
        # C has no dependents = 0.0
        assert scores["C"] == 0.0

    def test_empty_stories(self) -> None:
        """Test with empty list."""
        scores = calculate_dependency_scores([])
        assert scores == {}


class TestCalculatePriorityScore:
    """Tests for calculate_priority_score function (Task 2.2, FR65)."""

    def test_default_weights(self) -> None:
        """Test priority calculation with default weights."""
        factors = PriorityFactors(
            story_id="test",
            value_score=1.0,
            dependency_score=1.0,
            velocity_impact=1.0,
            tech_debt_score=1.0,
        )
        config = PriorityScoringConfig()
        result = calculate_priority_score(factors, config)
        # 1.0 * (0.4 + 0.3 + 0.2 + 0.1) = 1.0
        assert abs(result.priority_score - 1.0) < 0.001

    def test_partial_scores(self) -> None:
        """Test priority calculation with partial scores."""
        factors = PriorityFactors(
            story_id="test",
            value_score=0.8,
            dependency_score=0.5,
            velocity_impact=0.6,
            tech_debt_score=0.2,
        )
        config = PriorityScoringConfig()
        # 0.8*0.4 + 0.5*0.3 + 0.6*0.2 + 0.2*0.1 = 0.32 + 0.15 + 0.12 + 0.02 = 0.61
        result = calculate_priority_score(factors, config)
        assert abs(result.priority_score - 0.61) < 0.001

    def test_custom_weights(self) -> None:
        """Test priority calculation with custom weights."""
        factors = PriorityFactors(
            story_id="test",
            value_score=1.0,
            dependency_score=0.0,
            velocity_impact=0.0,
            tech_debt_score=0.0,
        )
        config = PriorityScoringConfig(
            value_weight=1.0,
            dependency_weight=0.0,
            velocity_weight=0.0,
            tech_debt_weight=0.0,
        )
        result = calculate_priority_score(factors, config)
        assert abs(result.priority_score - 1.0) < 0.001

    def test_zero_scores(self) -> None:
        """Test priority calculation with all zero scores."""
        factors = PriorityFactors(
            story_id="test",
            value_score=0.0,
            dependency_score=0.0,
            velocity_impact=0.0,
            tech_debt_score=0.0,
        )
        config = PriorityScoringConfig()
        result = calculate_priority_score(factors, config)
        assert result.priority_score == 0.0

    def test_factor_contributions(self) -> None:
        """Test that factor contributions are calculated correctly."""
        factors = PriorityFactors(
            story_id="test",
            value_score=0.8,
            dependency_score=0.5,
            velocity_impact=0.6,
            tech_debt_score=0.2,
        )
        config = PriorityScoringConfig()
        result = calculate_priority_score(factors, config)
        assert "value" in result.factor_contributions
        assert "dependency" in result.factor_contributions
        assert abs(result.factor_contributions["value"] - 0.32) < 0.001
        assert abs(result.factor_contributions["dependency"] - 0.15) < 0.001

    def test_explanation_included(self) -> None:
        """Test that explanation is generated when configured."""
        factors = PriorityFactors(story_id="test", value_score=0.8)
        config = PriorityScoringConfig(include_explanation=True)
        result = calculate_priority_score(factors, config)
        assert result.explanation != ""
        assert "test" in result.explanation
        assert "Value" in result.explanation

    def test_explanation_excluded(self) -> None:
        """Test that explanation is empty when disabled."""
        factors = PriorityFactors(story_id="test", value_score=0.8)
        config = PriorityScoringConfig(include_explanation=False)
        result = calculate_priority_score(factors, config)
        assert result.explanation == ""


class TestNormalizeScores:
    """Tests for normalize_scores function (Task 2.3)."""

    def test_normal_range(self) -> None:
        """Test normalization of scores in normal range."""
        scores = [0.2, 0.5, 0.8]
        normalized = normalize_scores(scores)
        assert normalized[0] == 0.0  # min
        assert normalized[2] == 1.0  # max
        assert abs(normalized[1] - 0.5) < 0.001  # middle

    def test_all_same(self) -> None:
        """Test normalization when all scores are equal."""
        scores = [0.5, 0.5, 0.5]
        normalized = normalize_scores(scores)
        assert all(abs(s - 0.5) < 0.001 for s in normalized)

    def test_empty_list(self) -> None:
        """Test normalization of empty list."""
        normalized = normalize_scores([])
        assert normalized == []

    def test_single_value(self) -> None:
        """Test normalization of single value."""
        normalized = normalize_scores([0.7])
        assert normalized == [0.5]

    def test_two_values(self) -> None:
        """Test normalization of two values."""
        normalized = normalize_scores([0.3, 0.7])
        assert normalized[0] == 0.0
        assert normalized[1] == 1.0

    def test_preserves_order(self) -> None:
        """Test that normalization preserves relative order."""
        scores = [0.1, 0.5, 0.3, 0.9, 0.7]
        normalized = normalize_scores(scores)
        # Check order is preserved
        assert normalized[0] < normalized[2] < normalized[1] < normalized[4] < normalized[3]


class TestNormalizeResults:
    """Tests for normalize_results function (Task 2.3)."""

    def test_updates_normalized_score(self) -> None:
        """Test that normalized_score is set on results."""
        results = [
            PriorityResult(story_id="A", priority_score=0.2),
            PriorityResult(story_id="B", priority_score=0.8),
        ]
        normalized = normalize_results(results)
        assert normalized[0].normalized_score == 0.0
        assert normalized[1].normalized_score == 1.0

    def test_preserves_other_fields(self) -> None:
        """Test that other fields are preserved."""
        results = [
            PriorityResult(
                story_id="A",
                priority_score=0.5,
                explanation="Test",
                factor_contributions={"value": 0.2},
            ),
        ]
        normalized = normalize_results(results)
        assert normalized[0].story_id == "A"
        assert normalized[0].explanation == "Test"
        assert normalized[0].factor_contributions == {"value": 0.2}

    def test_empty_list(self) -> None:
        """Test normalization of empty results list."""
        normalized = normalize_results([])
        assert normalized == []


class TestScoreStories:
    """Tests for score_stories function (Task 2.4)."""

    def test_basic_scoring(self) -> None:
        """Test basic story scoring."""
        stories = [
            SprintStory(story_id="A", title="A", value_score=0.3),
            SprintStory(story_id="B", title="B", value_score=0.9),
        ]
        results = score_stories(stories)
        # B should be first (higher value)
        assert results[0].story_id == "B"
        assert results[1].story_id == "A"

    def test_dependency_score_calculation(self) -> None:
        """Test that dependency scores are calculated."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B", dependencies=("A",)),
        ]
        results = score_stories(stories)
        # A should have higher dependency score (B depends on it)
        result_a = next(r for r in results if r.story_id == "A")
        assert result_a.factor_contributions["dependency"] > 0

    def test_custom_config(self) -> None:
        """Test scoring with custom config."""
        stories = [
            SprintStory(story_id="A", title="A", value_score=0.5),
        ]
        config = PriorityScoringConfig(
            value_weight=1.0,
            dependency_weight=0.0,
            velocity_weight=0.0,
            tech_debt_weight=0.0,
        )
        results = score_stories(stories, config)
        assert len(results) == 1
        # With only value weight, score should be 0.5
        assert abs(results[0].priority_score - 0.5) < 0.001

    def test_threshold_filtering(self) -> None:
        """Test that low-score stories are filtered."""
        # B needs value_score=1.0 to reach score=0.5 (1.0*0.4 + 0.5*0.2 = 0.5)
        # A with value_score=0.1 gets score=0.14 (0.1*0.4 + 0.5*0.2 = 0.14)
        stories = [
            SprintStory(story_id="A", title="A", value_score=0.1),
            SprintStory(story_id="B", title="B", value_score=1.0),
        ]
        config = PriorityScoringConfig(min_score_threshold=0.5, normalize_scores=False)
        results = score_stories(stories, config)
        # Only B should pass threshold
        assert len(results) == 1
        assert results[0].story_id == "B"

    def test_empty_stories(self) -> None:
        """Test scoring empty story list."""
        results = score_stories([])
        assert results == []

    def test_normalization_enabled(self) -> None:
        """Test that normalization is applied when enabled."""
        stories = [
            SprintStory(story_id="A", title="A", value_score=0.3),
            SprintStory(story_id="B", title="B", value_score=0.9),
        ]
        config = PriorityScoringConfig(normalize_scores=True)
        results = score_stories(stories, config)
        # Check normalized scores are set
        assert results[0].normalized_score is not None
        assert results[1].normalized_score is not None

    def test_normalization_disabled(self) -> None:
        """Test that normalization is skipped when disabled."""
        stories = [
            SprintStory(story_id="A", title="A", value_score=0.3),
        ]
        config = PriorityScoringConfig(normalize_scores=False)
        results = score_stories(stories, config)
        assert results[0].normalized_score is None


class TestUpdateStoriesWithScores:
    """Tests for update_stories_with_scores function (Task 2.4)."""

    def test_updates_priority_score(self) -> None:
        """Test that priority_score is updated on stories."""
        stories = [SprintStory(story_id="A", title="A")]
        results = [PriorityResult(story_id="A", priority_score=0.75)]
        updated = update_stories_with_scores(stories, results)
        assert updated[0].priority_score == 0.75

    def test_uses_normalized_score(self) -> None:
        """Test that normalized_score is used when available."""
        stories = [SprintStory(story_id="A", title="A")]
        results = [
            PriorityResult(story_id="A", priority_score=0.5, normalized_score=0.8)
        ]
        updated = update_stories_with_scores(stories, results)
        assert updated[0].priority_score == 0.8

    def test_preserves_unmatched_stories(self) -> None:
        """Test that unmatched stories are preserved unchanged."""
        stories = [
            SprintStory(story_id="A", title="A"),
            SprintStory(story_id="B", title="B"),
        ]
        results = [PriorityResult(story_id="A", priority_score=0.75)]
        updated = update_stories_with_scores(stories, results)
        assert len(updated) == 2
        # B should be unchanged
        b_story = next(s for s in updated if s.story_id == "B")
        assert b_story.priority_score == 0.0


class TestGenerateScoreExplanation:
    """Tests for _generate_score_explanation function (Task 2.5)."""

    def test_includes_story_id(self) -> None:
        """Test that explanation includes story ID."""
        factors = PriorityFactors(story_id="test-story", value_score=0.8)
        config = PriorityScoringConfig()
        contributions = {"value": 0.32, "dependency": 0.0, "velocity": 0.1, "tech_debt": 0.0}
        explanation = _generate_score_explanation(factors, config, contributions)
        assert "test-story" in explanation

    def test_includes_all_factors(self) -> None:
        """Test that explanation includes all factors."""
        factors = PriorityFactors(
            story_id="test",
            value_score=0.8,
            dependency_score=0.5,
            velocity_impact=0.6,
            tech_debt_score=0.2,
        )
        config = PriorityScoringConfig()
        contributions = {"value": 0.32, "dependency": 0.15, "velocity": 0.12, "tech_debt": 0.02}
        explanation = _generate_score_explanation(factors, config, contributions)
        assert "Value" in explanation
        assert "Dependency" in explanation
        assert "Velocity" in explanation
        assert "Tech Debt" in explanation

    def test_includes_total(self) -> None:
        """Test that explanation includes total score."""
        factors = PriorityFactors(story_id="test", value_score=0.5)
        config = PriorityScoringConfig()
        contributions = {"value": 0.2, "dependency": 0.0, "velocity": 0.1, "tech_debt": 0.0}
        explanation = _generate_score_explanation(factors, config, contributions)
        assert "Total" in explanation


class TestBackwardCompatibility:
    """Tests for backward compatibility with planning.py (Task 3.4)."""

    def test_planning_config_compatible(self) -> None:
        """Test that PriorityScoringConfig can be created from PlanningConfig."""
        from yolo_developer.agents.sm.planning_types import PlanningConfig

        planning_config = PlanningConfig(
            value_weight=0.5,
            dependency_weight=0.25,
            velocity_weight=0.15,
            tech_debt_weight=0.1,
        )
        scoring_config = PriorityScoringConfig.from_planning_config(planning_config)
        assert scoring_config.value_weight == 0.5
        assert scoring_config.dependency_weight == 0.25
        assert scoring_config.velocity_weight == 0.15
        assert scoring_config.tech_debt_weight == 0.1

    def test_planning_score_matches_priority_score(self) -> None:
        """Test that planning._calculate_priority_score matches priority module."""
        from yolo_developer.agents.sm.planning import _calculate_priority_score
        from yolo_developer.agents.sm.planning_types import PlanningConfig

        story = SprintStory(
            story_id="test",
            title="Test",
            value_score=0.8,
            dependency_score=0.5,
            velocity_impact=0.6,
            tech_debt_score=0.2,
        )
        config = PlanningConfig()

        # Get score from planning module
        planning_score = _calculate_priority_score(story, config)

        # Get score from priority module
        factors = PriorityFactors(
            story_id=story.story_id,
            value_score=story.value_score,
            dependency_score=story.dependency_score,
            velocity_impact=story.velocity_impact,
            tech_debt_score=story.tech_debt_score,
        )
        scoring_config = PriorityScoringConfig.from_planning_config(config)
        priority_result = calculate_priority_score(factors, scoring_config)

        assert abs(planning_score - priority_result.priority_score) < 0.001
