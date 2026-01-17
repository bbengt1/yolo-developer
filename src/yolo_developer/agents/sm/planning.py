"""Sprint planning module for SM agent (Story 10.3).

This module provides sprint planning functionality for the SM agent:

- Priority scoring: Calculate weighted priority scores per FR65
- Dependency analysis: Build dependency graph for story ordering
- Topological sort: Order stories respecting dependencies
- Capacity management: Select stories within sprint capacity
- Plan generation: Create complete sprint plans with audit logging

Key Concepts:
- **Priority Score**: Composite weighted score based on value, dependencies,
  velocity impact, and tech debt reduction (per FR65)
- **Dependency-Aware Ordering**: Stories are ordered using topological sort
  to respect dependencies while maximizing priority
- **Capacity Management**: Stories are selected to fit within sprint limits
  (max stories and max points per NFR-SCALE-1)

Example:
    >>> from yolo_developer.agents.sm.planning import plan_sprint
    >>> from yolo_developer.agents.sm.planning_types import PlanningConfig
    >>>
    >>> stories = [
    ...     {"story_id": "1-1", "title": "Setup DB", "estimated_points": 3},
    ...     {"story_id": "1-2", "title": "Add Auth", "dependencies": ["1-1"]},
    ... ]
    >>> plan = await plan_sprint(stories)
    >>> plan.sprint_id
    'sprint-20260112'

Architecture Note:
    Per ADR-005 and ADR-007, this module follows async-first patterns and
    integrates with the audit logging system via Decision records.

References:
    - FR9: SM Agent can plan sprints by prioritizing and sequencing stories
    - FR65: SM Agent can calculate weighted priority scores for story selection
    - NFR-SCALE-1: MVP supports 5-10 stories per sprint
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

from yolo_developer.agents.sm.planning_types import (
    PlanningConfig,
    SprintPlan,
    SprintStory,
)
from yolo_developer.agents.sm.priority import (
    calculate_dependency_scores,
)
from yolo_developer.agents.sm.priority_types import (
    PriorityFactors,
    PriorityScoringConfig,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class CircularDependencyError(ValueError):
    """Raised when circular dependencies are detected in stories."""


# =============================================================================
# Priority Scoring (Task 2.1, FR65)
# =============================================================================


def _calculate_priority_score(story: SprintStory, config: PlanningConfig) -> float:
    """Calculate weighted priority score for a story (FR65).

    The priority score is a composite of four factors:
    - Value score (default 40%): Business value of the story
    - Dependency score (default 30%): How many stories depend on this
    - Velocity impact (default 20%): Expected effect on team velocity
    - Tech debt score (default 10%): Tech debt reduction

    Note:
        This function delegates to the priority module (Story 10.11) for
        consistency. For more advanced features like normalization and
        explanations, use the priority module directly.

    Args:
        story: The story to calculate priority for.
        config: Planning configuration with weights.

    Returns:
        Weighted priority score between 0.0 and 1.0.

    Example:
        >>> story = SprintStory(story_id="1", title="T", value_score=0.8)
        >>> config = PlanningConfig()
        >>> _calculate_priority_score(story, config)
        0.61
    """
    # Delegate to priority module for consistent scoring (Story 10.11)
    from yolo_developer.agents.sm.priority import calculate_priority_score

    factors = PriorityFactors(
        story_id=story.story_id,
        value_score=story.value_score,
        dependency_score=story.dependency_score,
        velocity_impact=story.velocity_impact,
        tech_debt_score=story.tech_debt_score,
    )
    scoring_config = PriorityScoringConfig.from_planning_config(config, include_explanation=False)
    result = calculate_priority_score(factors, scoring_config)
    return result.priority_score


# =============================================================================
# Dependency Analysis (Task 2.2)
# =============================================================================


def _analyze_dependencies(
    stories: Sequence[SprintStory],
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, SprintStory]]:
    """Build dependency graph from stories.

    Creates an adjacency list graph and in-degree count for topological sorting.
    Dependencies on stories not in the sprint are ignored.

    Args:
        stories: List of stories to analyze.

    Returns:
        Tuple of (graph, in_degree, story_map):
        - graph: Adjacency list mapping story_id to list of dependent story_ids
        - in_degree: Map of story_id to count of dependencies
        - story_map: Map of story_id to SprintStory object

    Example:
        >>> stories = [SprintStory(story_id="A", title="A"),
        ...            SprintStory(story_id="B", title="B", dependencies=("A",))]
        >>> graph, in_degree, story_map = _analyze_dependencies(stories)
        >>> in_degree["B"]
        1
    """
    graph: dict[str, list[str]] = defaultdict(list)
    story_map: dict[str, SprintStory] = {s.story_id: s for s in stories}
    in_degree: dict[str, int] = {s.story_id: 0 for s in stories}

    for story in stories:
        for dep in story.dependencies:
            # Only count dependencies that are in this sprint
            if dep in story_map:
                graph[dep].append(story.story_id)
                in_degree[story.story_id] += 1

    logger.debug(
        "dependency_analysis_complete",
        story_count=len(stories),
        edges=sum(len(v) for v in graph.values()),
    )

    return dict(graph), in_degree, story_map


# =============================================================================
# Topological Sort (Task 2.3)
# =============================================================================


def _topological_sort(stories: Sequence[SprintStory]) -> list[SprintStory]:
    """Order stories respecting dependencies using Kahn's algorithm.

    Stories are sorted topologically (dependencies first), with ties broken
    by priority score (higher priority first). This ensures that:
    1. All dependencies are satisfied before dependent stories
    2. Among available stories, higher priority ones come first

    Args:
        stories: List of stories to sort.

    Returns:
        List of stories in execution order.

    Raises:
        CircularDependencyError: If circular dependencies are detected.

    Example:
        >>> stories = [SprintStory(story_id="B", title="B", dependencies=("A",)),
        ...            SprintStory(story_id="A", title="A")]
        >>> result = _topological_sort(stories)
        >>> [s.story_id for s in result]
        ['A', 'B']
    """
    if not stories:
        return []

    graph, in_degree, story_map = _analyze_dependencies(stories)

    # Start with stories that have no dependencies
    available: list[str] = [sid for sid, deg in in_degree.items() if deg == 0]
    result: list[SprintStory] = []

    while available:
        # Sort available by priority descending
        available.sort(key=lambda sid: -story_map[sid].priority_score)
        current = available.pop(0)
        result.append(story_map[current])

        # Reduce in-degree for dependents
        for dependent in graph.get(current, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                available.append(dependent)

    # Check for circular dependencies
    if len(result) != len(stories):
        remaining = [sid for sid, deg in in_degree.items() if deg > 0]
        logger.warning(
            "circular_dependency_detected",
            remaining_stories=remaining,
        )
        raise CircularDependencyError(f"Circular dependency detected in stories: {remaining}")

    logger.debug(
        "topological_sort_complete",
        story_count=len(result),
        order=[s.story_id for s in result],
    )

    return result


# =============================================================================
# Capacity Management (Task 3)
# =============================================================================


def _check_capacity(
    stories: Sequence[SprintStory],
    config: PlanningConfig,
) -> tuple[list[SprintStory], int]:
    """Select stories that fit within sprint capacity.

    Stories are selected in order until either:
    - Maximum story count is reached
    - Maximum story points capacity is reached

    Large stories that don't fit are skipped to allow smaller ones.

    Args:
        stories: Ordered list of stories to select from.
        config: Planning configuration with capacity limits.

    Returns:
        Tuple of (selected_stories, total_points).

    Example:
        >>> stories = [SprintStory(story_id="1", title="1", estimated_points=5)]
        >>> config = PlanningConfig(max_points=10)
        >>> selected, total = _check_capacity(stories, config)
        >>> total
        5
    """
    selected: list[SprintStory] = []
    total_points = 0

    for story in stories:
        # Check story count limit
        if len(selected) >= config.max_stories:
            logger.debug(
                "capacity_limit_story_count",
                max_stories=config.max_stories,
            )
            break

        # Check points capacity (skip if doesn't fit)
        if total_points + story.estimated_points > config.max_points:
            logger.debug(
                "capacity_skip_story",
                story_id=story.story_id,
                story_points=story.estimated_points,
                remaining_capacity=config.max_points - total_points,
            )
            continue

        selected.append(story)
        total_points += story.estimated_points

    logger.info(
        "capacity_check_complete",
        selected_count=len(selected),
        total_points=total_points,
        max_stories=config.max_stories,
        max_points=config.max_points,
    )

    return selected, total_points


# =============================================================================
# Conversion Utilities
# =============================================================================


def _dict_to_sprint_story(data: dict[str, Any]) -> SprintStory:
    """Convert a dictionary to SprintStory.

    Handles flexible input formats with sensible defaults.

    Args:
        data: Dictionary with story data. Must contain 'story_id' and 'title'.

    Returns:
        SprintStory instance.

    Raises:
        ValueError: If required fields 'story_id' or 'title' are missing.

    Example:
        >>> data = {"story_id": "1-1", "title": "Test"}
        >>> story = _dict_to_sprint_story(data)
        >>> story.story_id
        '1-1'
    """
    # Validate required fields
    if "story_id" not in data:
        raise ValueError("Story data missing required field: 'story_id'")
    if "title" not in data:
        raise ValueError("Story data missing required field: 'title'")

    # Handle dependencies as list or tuple
    deps = data.get("dependencies", ())
    if isinstance(deps, list):
        deps = tuple(deps)

    return SprintStory(
        story_id=data["story_id"],
        title=data["title"],
        priority_score=data.get("priority_score", 0.0),
        dependencies=deps,
        estimated_points=data.get("estimated_points", 1),
        value_score=data.get("value_score", 0.5),
        tech_debt_score=data.get("tech_debt_score", 0.0),
        velocity_impact=data.get("velocity_impact", 0.5),
        dependency_score=data.get("dependency_score", 0.0),
        metadata=data.get("metadata", {}),
    )


def _generate_planning_rationale(
    stories: Sequence[SprintStory],
    total_points: int,
    config: PlanningConfig,
) -> str:
    """Generate planning rationale for audit logging.

    Creates a human-readable explanation of the sprint planning decisions.

    Args:
        stories: Selected stories for the sprint.
        total_points: Total story points selected.
        config: Planning configuration used.

    Returns:
        Rationale string for audit.

    Example:
        >>> stories = [SprintStory(story_id="1", title="1", estimated_points=5)]
        >>> rationale = _generate_planning_rationale(stories, 5, PlanningConfig())
        >>> "1 stories" in rationale
        True
    """
    story_count = len(stories)
    capacity_pct = (total_points / config.max_points * 100) if config.max_points else 0

    rationale_parts = [
        f"Selected {story_count} stories with {total_points} total points",
        f"({capacity_pct:.1f}% of {config.max_points} point capacity).",
    ]

    if story_count > 0:
        # List top priority stories
        top_stories = stories[:3]
        story_list = ", ".join(s.story_id for s in top_stories)
        rationale_parts.append(f"Top priorities: {story_list}.")

        # Note any dependency ordering
        dep_ordered = [s for s in stories if s.dependencies]
        if dep_ordered:
            rationale_parts.append(f"{len(dep_ordered)} stories ordered by dependencies.")

    return " ".join(rationale_parts)


# =============================================================================
# Main Planning Function (Task 4)
# =============================================================================


async def plan_sprint(
    stories: Sequence[dict[str, Any]],
    config: PlanningConfig | None = None,
) -> SprintPlan:
    """Generate a sprint plan from available stories (FR9).

    This is the main entry point for sprint planning. It:
    1. Converts story dictionaries to SprintStory objects
    2. Calculates priority scores for each story
    3. Orders stories using topological sort (dependencies + priority)
    4. Selects stories within capacity constraints
    5. Creates a SprintPlan with audit rationale

    Args:
        stories: List of story dictionaries with at minimum story_id and title.
        config: Optional planning configuration. Uses defaults if not provided.

    Returns:
        SprintPlan with ordered stories and planning rationale.

    Raises:
        CircularDependencyError: If circular dependencies are detected.

    Example:
        >>> stories = [
        ...     {"story_id": "1-1", "title": "Setup", "estimated_points": 3},
        ...     {"story_id": "1-2", "title": "Build", "dependencies": ["1-1"]},
        ... ]
        >>> plan = await plan_sprint(stories)
        >>> plan.stories[0].story_id
        '1-1'
    """
    config = config or PlanningConfig()

    logger.info(
        "sprint_planning_started",
        story_count=len(stories),
        max_stories=config.max_stories,
        max_points=config.max_points,
    )

    # Handle empty stories
    if not stories:
        logger.info("sprint_planning_empty", reason="no_stories_provided")
        return SprintPlan(
            sprint_id=f"sprint-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            stories=(),
            total_points=0,
            capacity_used=0.0,
            planning_rationale="No stories available for sprint planning.",
        )

    # Step 1: Convert to SprintStory objects
    sprint_stories = [_dict_to_sprint_story(s) for s in stories]

    # Step 2a: Calculate dependency scores for all stories (Story 10.11)
    # This determines how many stories depend on each story
    dep_scores = calculate_dependency_scores(sprint_stories)

    # Step 2b: Update stories with dependency scores, then calculate priority
    stories_with_deps = [
        replace(story, dependency_score=dep_scores.get(story.story_id, 0.0))
        for story in sprint_stories
    ]

    # Step 2c: Calculate priority scores using updated dependency scores
    scored_stories = [
        replace(story, priority_score=_calculate_priority_score(story, config))
        for story in stories_with_deps
    ]

    logger.debug(
        "priority_scores_calculated",
        scores={s.story_id: s.priority_score for s in scored_stories},
        dependency_scores=dep_scores,
    )

    # Step 3: Sort by dependencies then priority (topological sort)
    ordered_stories = _topological_sort(scored_stories)

    # Step 4: Apply capacity constraints
    selected_stories, total_points = _check_capacity(ordered_stories, config)

    # Step 5: Calculate capacity used
    capacity_used = total_points / config.max_points if config.max_points else 0.0

    # Step 6: Generate planning rationale for audit
    rationale = _generate_planning_rationale(selected_stories, total_points, config)

    # Create sprint plan
    plan = SprintPlan(
        sprint_id=f"sprint-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        stories=tuple(selected_stories),
        total_points=total_points,
        capacity_used=capacity_used,
        planning_rationale=rationale,
    )

    logger.info(
        "sprint_planning_complete",
        sprint_id=plan.sprint_id,
        selected_stories=len(plan.stories),
        total_points=plan.total_points,
        capacity_used=plan.capacity_used,
    )

    return plan
