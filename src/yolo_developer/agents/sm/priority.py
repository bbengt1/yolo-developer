"""Priority scoring module for SM agent (Story 10.11).

This module provides priority scoring functionality for the SM agent:

- Dependency score calculation: How many stories depend on this one
- Priority score calculation: Weighted composite score per FR65
- Score normalization: Scale scores across a set to 0.0-1.0
- Batch scoring: Score and order a list of stories
- Explanation generation: Audit trail for scoring decisions

Key Concepts:
- **Priority Score**: Composite weighted score based on value, dependencies,
  velocity impact, and tech debt reduction (per FR65)
- **Dependency Score**: Stories that others depend on are prioritized higher
- **Normalization**: Scores are scaled to 0.0-1.0 for consistent ordering

Example:
    >>> from yolo_developer.agents.sm.priority import (
    ...     calculate_priority_score,
    ...     calculate_dependency_score,
    ...     score_stories,
    ... )
    >>> from yolo_developer.agents.sm.priority_types import (
    ...     PriorityFactors,
    ...     PriorityScoringConfig,
    ... )
    >>>
    >>> factors = PriorityFactors(
    ...     story_id="1-1",
    ...     value_score=0.8,
    ...     dependency_score=0.5,
    ... )
    >>> config = PriorityScoringConfig()
    >>> result = calculate_priority_score(factors, config)
    >>> result.priority_score
    0.57

Architecture Note:
    Per ADR-005 and ADR-007, this module follows async-first patterns and
    integrates with the audit logging system via structured logging.

References:
    - FR65: SM Agent can calculate weighted priority scores for story selection
    - Story 10.3: Sprint Planning (original _calculate_priority_score)
    - ADR-001: Frozen dataclasses for internal state
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

from yolo_developer.agents.sm.planning_types import SprintStory
from yolo_developer.agents.sm.priority_types import (
    MAX_SCORE,
    MIN_SCORE,
    PriorityFactors,
    PriorityResult,
    PriorityScoringConfig,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Dependency Score Calculation (Task 2.1)
# =============================================================================


def calculate_dependency_score(
    story_id: str,
    all_stories: Sequence[SprintStory],
) -> float:
    """Calculate how many stories depend on this one (normalized).

    Stories that others depend on should be prioritized higher because
    blocking them blocks downstream work. The score is normalized to
    0.0-1.0 based on the proportion of other stories that depend on this one.

    Args:
        story_id: The story ID to calculate dependency score for.
        all_stories: All stories in the sprint (to count dependents).

    Returns:
        Dependency score between 0.0 and 1.0.
        - 0.0 means no other stories depend on this
        - 1.0 means all other stories depend on this

    Example:
        >>> stories = [
        ...     SprintStory(story_id="A", title="A"),
        ...     SprintStory(story_id="B", title="B", dependencies=("A",)),
        ...     SprintStory(story_id="C", title="C", dependencies=("A",)),
        ... ]
        >>> calculate_dependency_score("A", stories)
        1.0
        >>> calculate_dependency_score("B", stories)
        0.0
    """
    if not all_stories:
        return MIN_SCORE

    # Count how many stories list this story_id in their dependencies
    dependents = sum(
        1 for s in all_stories if story_id in s.dependencies and s.story_id != story_id
    )

    max_possible = len(all_stories) - 1
    if max_possible <= 0:
        return MIN_SCORE

    score = min(dependents / max_possible, MAX_SCORE)

    logger.debug(
        "dependency_score_calculated",
        story_id=story_id,
        dependents=dependents,
        max_possible=max_possible,
        score=score,
    )

    return score


def calculate_dependency_scores(
    stories: Sequence[SprintStory],
) -> dict[str, float]:
    """Calculate dependency scores for all stories in a set.

    Convenience function to calculate dependency scores for all stories
    at once, useful when preparing stories for priority scoring.

    Args:
        stories: List of stories to calculate dependency scores for.

    Returns:
        Dictionary mapping story_id to dependency score.

    Example:
        >>> stories = [
        ...     SprintStory(story_id="A", title="A"),
        ...     SprintStory(story_id="B", title="B", dependencies=("A",)),
        ... ]
        >>> scores = calculate_dependency_scores(stories)
        >>> scores["A"]
        1.0
        >>> scores["B"]
        0.0
    """
    return {s.story_id: calculate_dependency_score(s.story_id, stories) for s in stories}


# =============================================================================
# Priority Score Calculation (Task 2.2)
# =============================================================================


def calculate_priority_score(
    factors: PriorityFactors,
    config: PriorityScoringConfig,
) -> PriorityResult:
    """Calculate weighted priority score for a story (FR65).

    The priority score is a composite of four factors:
    - Value score (default 40%): Business value of the story
    - Dependency score (default 30%): How many stories depend on this
    - Velocity impact (default 20%): Expected effect on team velocity
    - Tech debt score (default 10%): Tech debt reduction

    Args:
        factors: The input scoring factors for the story.
        config: Configuration with weights and options.

    Returns:
        PriorityResult with calculated score and optional explanation.

    Example:
        >>> factors = PriorityFactors(
        ...     story_id="1-1",
        ...     value_score=0.8,
        ...     dependency_score=0.5,
        ...     velocity_impact=0.6,
        ...     tech_debt_score=0.2,
        ... )
        >>> config = PriorityScoringConfig()
        >>> result = calculate_priority_score(factors, config)
        >>> 0.5 < result.priority_score < 0.7
        True
    """
    # Calculate weighted contributions
    value_contribution = factors.value_score * config.value_weight
    dependency_contribution = factors.dependency_score * config.dependency_weight
    velocity_contribution = factors.velocity_impact * config.velocity_weight
    tech_debt_contribution = factors.tech_debt_score * config.tech_debt_weight

    # Sum for composite score
    priority_score = (
        value_contribution
        + dependency_contribution
        + velocity_contribution
        + tech_debt_contribution
    )

    # Clamp to valid range
    priority_score = max(MIN_SCORE, min(MAX_SCORE, priority_score))

    factor_contributions = {
        "value": value_contribution,
        "dependency": dependency_contribution,
        "velocity": velocity_contribution,
        "tech_debt": tech_debt_contribution,
    }

    # Generate explanation if requested
    explanation = ""
    if config.include_explanation:
        explanation = _generate_score_explanation(factors, config, factor_contributions)

    logger.debug(
        "priority_score_calculated",
        story_id=factors.story_id,
        priority_score=priority_score,
        factor_contributions=factor_contributions,
    )

    return PriorityResult(
        story_id=factors.story_id,
        priority_score=priority_score,
        normalized_score=None,  # Set by normalize_scores if called
        explanation=explanation,
        factor_contributions=factor_contributions,
    )


# =============================================================================
# Score Normalization (Task 2.3)
# =============================================================================


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Normalize scores to 0.0-1.0 range using min-max normalization.

    Uses the formula: (x - min) / (max - min)

    Args:
        scores: List of raw scores to normalize.

    Returns:
        List of normalized scores in same order.
        - Empty input returns empty list
        - Single value returns [0.5]
        - All equal values return [0.5, 0.5, ...]

    Example:
        >>> normalize_scores([0.2, 0.5, 0.8])
        [0.0, 0.5, 1.0]
        >>> normalize_scores([0.5, 0.5, 0.5])
        [0.5, 0.5, 0.5]
        >>> normalize_scores([])
        []
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    # Handle edge case where all scores are equal
    if max_score == min_score:
        return [0.5 for _ in scores]

    normalized = [(s - min_score) / (max_score - min_score) for s in scores]

    logger.debug(
        "scores_normalized",
        count=len(scores),
        min_raw=min_score,
        max_raw=max_score,
    )

    return normalized


def normalize_results(results: Sequence[PriorityResult]) -> list[PriorityResult]:
    """Normalize priority results and update their normalized_score field.

    Args:
        results: List of PriorityResult objects to normalize.

    Returns:
        List of new PriorityResult objects with normalized_score set.

    Example:
        >>> results = [
        ...     PriorityResult(story_id="A", priority_score=0.2),
        ...     PriorityResult(story_id="B", priority_score=0.8),
        ... ]
        >>> normalized = normalize_results(results)
        >>> normalized[0].normalized_score
        0.0
        >>> normalized[1].normalized_score
        1.0
    """
    if not results:
        return []

    raw_scores = [r.priority_score for r in results]
    normalized_scores = normalize_scores(raw_scores)

    return [
        replace(result, normalized_score=norm_score)
        for result, norm_score in zip(results, normalized_scores, strict=True)
    ]


# =============================================================================
# Batch Scoring (Task 2.4)
# =============================================================================


def score_stories(
    stories: Sequence[SprintStory],
    config: PriorityScoringConfig | None = None,
) -> list[PriorityResult]:
    """Score and order a list of stories by priority.

    This is the main entry point for batch priority scoring. It:
    1. Calculates dependency scores for all stories
    2. Calculates priority scores for each story
    3. Optionally normalizes scores across the set
    4. Filters stories below the minimum threshold
    5. Returns results sorted by score (highest first)

    Args:
        stories: List of stories to score.
        config: Scoring configuration (uses defaults if None).

    Returns:
        List of PriorityResult sorted by priority_score descending.
        Stories below min_score_threshold are filtered out.

    Example:
        >>> stories = [
        ...     SprintStory(story_id="A", title="A", value_score=0.3),
        ...     SprintStory(story_id="B", title="B", value_score=0.9),
        ... ]
        >>> results = score_stories(stories)
        >>> results[0].story_id
        'B'
    """
    if config is None:
        config = PriorityScoringConfig()

    if not stories:
        logger.debug("score_stories_empty_input")
        return []

    # Calculate dependency scores for all stories first
    dep_scores = calculate_dependency_scores(stories)

    # Build factors and calculate scores
    results: list[PriorityResult] = []
    for story in stories:
        factors = PriorityFactors(
            story_id=story.story_id,
            value_score=story.value_score,
            dependency_score=dep_scores.get(story.story_id, story.dependency_score),
            velocity_impact=story.velocity_impact,
            tech_debt_score=story.tech_debt_score,
        )
        result = calculate_priority_score(factors, config)
        results.append(result)

    # Normalize if configured
    if config.normalize_scores:
        results = normalize_results(results)

    # Filter by threshold
    if config.min_score_threshold > MIN_SCORE:
        score_field = "normalized_score" if config.normalize_scores else "priority_score"
        results = [
            r
            for r in results
            if (getattr(r, score_field) or r.priority_score) >= config.min_score_threshold
        ]

    # Sort by score descending (use normalized if available, else priority)
    def sort_key(r: PriorityResult) -> float:
        if config.normalize_scores and r.normalized_score is not None:
            return r.normalized_score
        return r.priority_score

    results.sort(key=sort_key, reverse=True)

    logger.info(
        "stories_scored",
        total=len(stories),
        scored=len(results),
        normalized=config.normalize_scores,
        threshold=config.min_score_threshold,
    )

    return results


def update_stories_with_scores(
    stories: Sequence[SprintStory],
    results: Sequence[PriorityResult],
    config: PriorityScoringConfig | None = None,
) -> list[SprintStory]:
    """Update SprintStory objects with calculated priority scores.

    Creates new SprintStory objects with updated priority_score and
    dependency_score fields based on scoring results.

    Args:
        stories: Original stories to update.
        results: Scoring results from score_stories().
        config: Scoring configuration used (needed to reverse-calculate
            raw dependency score from weighted contribution). If None,
            uses default config.

    Returns:
        List of new SprintStory objects with updated scores.

    Example:
        >>> stories = [SprintStory(story_id="A", title="A")]
        >>> results = [PriorityResult(story_id="A", priority_score=0.7)]
        >>> updated = update_stories_with_scores(stories, results)
        >>> updated[0].priority_score
        0.7
    """
    if config is None:
        config = PriorityScoringConfig()

    result_map = {r.story_id: r for r in results}

    updated: list[SprintStory] = []
    for story in stories:
        if story.story_id in result_map:
            result = result_map[story.story_id]
            # Use normalized score if available, else priority score
            score = (
                result.normalized_score
                if result.normalized_score is not None
                else result.priority_score
            )
            # Reverse the weighted contribution to get raw dependency score
            # dep_contribution = raw_score * weight, so raw_score = dep_contribution / weight
            dep_contribution = result.factor_contributions.get("dependency", 0.0)
            raw_dep_score = (
                dep_contribution / config.dependency_weight
                if dep_contribution > 0 and config.dependency_weight > 0
                else story.dependency_score
            )
            updated_story = replace(
                story,
                priority_score=score,
                dependency_score=raw_dep_score,
            )
            updated.append(updated_story)
        else:
            updated.append(story)

    return updated


# =============================================================================
# Explanation Generation (Task 2.5)
# =============================================================================


def _generate_score_explanation(
    factors: PriorityFactors,
    config: PriorityScoringConfig,
    contributions: dict[str, float],
) -> str:
    """Generate human-readable explanation of score calculation.

    Creates an audit-friendly explanation showing how each factor
    contributed to the final priority score.

    Args:
        factors: The input scoring factors.
        config: The scoring configuration with weights.
        contributions: The calculated weighted contributions.

    Returns:
        Human-readable explanation string.

    Example:
        >>> factors = PriorityFactors(story_id="A", value_score=0.8)
        >>> config = PriorityScoringConfig()
        >>> contributions = {"value": 0.32, "dependency": 0.0, ...}
        >>> explanation = _generate_score_explanation(factors, config, contributions)
        >>> "value" in explanation
        True
    """
    total = sum(contributions.values())

    parts = [
        f"Priority score breakdown for {factors.story_id}:",
        f"  Value: {factors.value_score:.2f} x {config.value_weight:.1f} = {contributions['value']:.3f}",
        f"  Dependency: {factors.dependency_score:.2f} x {config.dependency_weight:.1f} = {contributions['dependency']:.3f}",
        f"  Velocity: {factors.velocity_impact:.2f} x {config.velocity_weight:.1f} = {contributions['velocity']:.3f}",
        f"  Tech Debt: {factors.tech_debt_score:.2f} x {config.tech_debt_weight:.1f} = {contributions['tech_debt']:.3f}",
        f"  Total: {total:.3f}",
    ]

    return "\n".join(parts)
