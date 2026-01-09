"""Story prioritization module for PM agent (Story 6.4).

This module provides functions for prioritizing user stories based on
value, dependencies, and quick win identification.

Key Functions:
    prioritize_stories: Main function that prioritizes a collection of stories.
    _calculate_value_score: Calculate base priority score from story attributes.
    _analyze_dependencies: Build dependency graph and detect cycles.
    _calculate_dependency_adjustment: Calculate score adjustment from dependencies.
    _is_quick_win: Determine if a story qualifies as a quick win.

Algorithm Overview:
    1. Calculate raw value score (0-100) for each story based on:
       - Priority enum base score (CRITICAL=100, HIGH=75, MEDIUM=50, LOW=25)
       - Complexity adjustment (S=+10, M=0, L=-10, XL=-20)
       - AC count bonus (+5 for 3-5 ACs)
       - Multi-source bonus (+5 per extra source requirement)

    2. Analyze dependencies to build graph with:
       - Blocking count (how many stories depend on this one)
       - Blocked-by count (how many dependencies this story has)
       - Cycle detection via DFS

    3. Calculate dependency adjustment (-20 to +20):
       - +20 if blocking 3+ stories, +10 if blocking 1-2
       - -20 if blocked by 3+, -10 if blocked by 1-2
       - -5 if in a cycle

    4. Identify quick wins:
       - Raw score >= 60
       - Complexity S or M
       - AC count <= 4
       - Not blocked

Example:
    >>> from yolo_developer.agents.pm.prioritization import prioritize_stories
    >>> from yolo_developer.agents.pm.types import Story, StoryPriority, StoryStatus
    >>>
    >>> stories = (story1, story2, story3)
    >>> result = prioritize_stories(stories)
    >>> result["recommended_execution_order"]
    ['story-001', 'story-002', 'story-003']

References:
    - ADR-001: TypedDict for internal state
    - FR44: Prioritize stories based on value and dependencies
"""

from __future__ import annotations

import structlog

from yolo_developer.agents.pm.types import (
    DependencyInfo,
    PrioritizationResult,
    PriorityScore,
    Story,
    StoryPriority,
)

logger = structlog.get_logger(__name__)

# Base scores from StoryPriority enum
PRIORITY_BASE_SCORES: dict[StoryPriority, int] = {
    StoryPriority.CRITICAL: 100,
    StoryPriority.HIGH: 75,
    StoryPriority.MEDIUM: 50,
    StoryPriority.LOW: 25,
}

# Complexity adjustments (value vs effort consideration)
COMPLEXITY_ADJUSTMENTS: dict[str, int] = {
    "S": 10,  # Low effort, high relative value
    "M": 0,  # Neutral
    "L": -10,  # Higher effort reduces priority
    "XL": -20,  # Much higher effort, may need breakdown
}

# Quick win thresholds
QUICK_WIN_MIN_SCORE: int = 60
QUICK_WIN_MAX_AC_COUNT: int = 4
QUICK_WIN_COMPLEXITIES: frozenset[str] = frozenset({"S", "M"})


def _calculate_value_score(story: Story) -> tuple[int, list[str]]:
    """Calculate base priority score from story attributes.

    Args:
        story: The story to score.

    Returns:
        Tuple of (score, rationale_parts) where score is 0-100 and
        rationale_parts is a list of scoring factor descriptions.

    Example:
        >>> score, rationale = _calculate_value_score(high_priority_story)
        >>> score
        85
        >>> rationale
        ['HIGH priority (+75)', 'S complexity (+10)']
    """
    rationale_parts: list[str] = []

    # Base score from priority
    base_score = PRIORITY_BASE_SCORES.get(story.priority, 50)
    rationale_parts.append(f"{story.priority.name} priority (+{base_score})")

    score = base_score

    # Complexity adjustment
    complexity_adj = COMPLEXITY_ADJUSTMENTS.get(story.estimated_complexity, 0)
    if complexity_adj != 0:
        score += complexity_adj
        sign = "+" if complexity_adj > 0 else ""
        rationale_parts.append(f"{story.estimated_complexity} complexity ({sign}{complexity_adj})")

    # Well-specified stories get bonus (3-5 ACs is optimal)
    ac_count = len(story.acceptance_criteria)
    if 3 <= ac_count <= 5:
        score += 5
        rationale_parts.append(f"{ac_count} ACs (+5)")

    # Multi-source stories (covers more requirements)
    source_count = len(story.source_requirements)
    if source_count > 1:
        bonus = 5 * (source_count - 1)
        score += bonus
        rationale_parts.append(f"{source_count} source reqs (+{bonus})")

    # Clamp to 0-100
    final_score = max(0, min(100, score))

    logger.debug(
        "prioritization_value_score",
        story_id=story.id,
        raw_score=final_score,
        priority=story.priority.value,
        complexity=story.estimated_complexity,
        ac_count=ac_count,
    )

    return final_score, rationale_parts


def _detect_cycles(
    story_ids: set[str],
    dependencies: dict[str, list[str]],
) -> list[list[str]]:
    """Detect dependency cycles using DFS with path tracking.

    Args:
        story_ids: Set of all story IDs in the collection.
        dependencies: Dict mapping story_id to list of dependency story_ids.

    Returns:
        List of cycles, where each cycle is a list of story IDs.
        Empty list if no cycles found.

    Example:
        >>> cycles = _detect_cycles({"A", "B", "C"}, {"A": ["B"], "B": ["C"], "C": ["A"]})
        >>> len(cycles)
        1
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def dfs(node: str, path: list[str]) -> None:
        if node not in story_ids:
            # Dependency points to story outside collection, skip
            return

        if node in rec_stack:
            # Found a cycle - extract cycle portion from path
            cycle_start = path.index(node)
            cycle = [*path[cycle_start:], node]
            cycles.append(cycle)
            return

        if node in visited:
            return

        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for dep in dependencies.get(node, []):
            dfs(dep, path)

        path.pop()
        rec_stack.remove(node)

    for story_id in story_ids:
        if story_id not in visited:
            dfs(story_id, [])

    return cycles


def _analyze_dependencies(
    stories: tuple[Story, ...],
) -> tuple[dict[str, DependencyInfo], list[list[str]]]:
    """Build dependency graph and detect cycles.

    Args:
        stories: Tuple of stories to analyze.

    Returns:
        Tuple of (dependency_info_map, cycles) where:
        - dependency_info_map maps story_id to DependencyInfo
        - cycles is list of detected cycles

    Example:
        >>> dep_info, cycles = _analyze_dependencies(stories)
        >>> dep_info["story-001"]["blocking_count"]
        2
    """
    story_ids = {s.id for s in stories}

    # Build reverse dependency map (who blocks whom)
    # dependencies[A] = [B, C] means A depends on B and C (B and C block A)
    dependencies: dict[str, list[str]] = {}
    # blocked_by[A] = [X, Y] means X and Y depend on A (A blocks X and Y)
    blocked_by: dict[str, list[str]] = {s.id: [] for s in stories}

    for story in stories:
        # Filter to only include dependencies within the story collection
        story_deps = [dep for dep in story.dependencies if dep in story_ids]
        dependencies[story.id] = story_deps

        # Update reverse map
        for dep in story_deps:
            if dep in blocked_by:
                blocked_by[dep].append(story.id)

    # Detect cycles
    cycles = _detect_cycles(story_ids, dependencies)

    # Find stories in cycles
    stories_in_cycles: set[str] = set()
    for cycle in cycles:
        stories_in_cycles.update(cycle)

    # Build DependencyInfo for each story
    dep_info_map: dict[str, DependencyInfo] = {}

    for story in stories:
        story_deps = dependencies.get(story.id, [])
        blocking_stories = blocked_by.get(story.id, [])

        dep_info_map[story.id] = DependencyInfo(
            blocking_count=len(blocking_stories),
            blocked_by_count=len(story_deps),
            blocking_story_ids=blocking_stories,
            blocked_by_story_ids=story_deps,
            in_cycle=story.id in stories_in_cycles,
        )

    logger.debug(
        "prioritization_dependency_analysis",
        story_count=len(stories),
        cycle_count=len(cycles),
        stories_in_cycles=len(stories_in_cycles),
    )

    return dep_info_map, cycles


def _calculate_dependency_adjustment(dep_info: DependencyInfo) -> tuple[int, list[str]]:
    """Calculate score adjustment based on dependencies.

    Args:
        dep_info: DependencyInfo for the story.

    Returns:
        Tuple of (adjustment, rationale_parts) where adjustment is -20 to +20.

    Example:
        >>> adj, rationale = _calculate_dependency_adjustment(info)
        >>> adj
        10
        >>> rationale
        ['unblocks 2 stories (+10)']
    """
    rationale_parts: list[str] = []
    adjustment = 0

    # Boost for blocking many stories (critical path)
    if dep_info["blocking_count"] >= 3:
        adjustment += 20
        rationale_parts.append(f"unblocks {dep_info['blocking_count']} stories (+20)")
    elif dep_info["blocking_count"] >= 1:
        adjustment += 10
        rationale_parts.append(f"unblocks {dep_info['blocking_count']} stories (+10)")

    # Penalty for being blocked
    if dep_info["blocked_by_count"] >= 3:
        adjustment -= 20
        rationale_parts.append(f"blocked by {dep_info['blocked_by_count']} stories (-20)")
    elif dep_info["blocked_by_count"] >= 1:
        adjustment -= 10
        rationale_parts.append(f"blocked by {dep_info['blocked_by_count']} stories (-10)")

    # Penalty for cycle involvement
    if dep_info["in_cycle"]:
        adjustment -= 5
        rationale_parts.append("in dependency cycle (-5)")

    # Clamp to -20 to +20
    final_adjustment = max(-20, min(20, adjustment))

    return final_adjustment, rationale_parts


def _is_quick_win(
    story: Story,
    raw_score: int,
    dep_info: DependencyInfo,
) -> bool:
    """Determine if a story qualifies as a quick win.

    Quick wins are stories with high value and low effort that aren't blocked.

    Args:
        story: The story to evaluate.
        raw_score: The story's raw value score.
        dep_info: The story's dependency info.

    Returns:
        True if story qualifies as a quick win.

    Example:
        >>> is_quick = _is_quick_win(simple_high_priority_story, 85, no_deps_info)
        >>> is_quick
        True
    """
    # Must have reasonable value
    if raw_score < QUICK_WIN_MIN_SCORE:
        return False

    # Must be low effort
    if story.estimated_complexity not in QUICK_WIN_COMPLEXITIES:
        return False

    # Must have manageable scope
    if len(story.acceptance_criteria) > QUICK_WIN_MAX_AC_COUNT:
        return False

    # Must not be blocked
    if dep_info["blocked_by_count"] > 0:
        return False

    return True


def prioritize_stories(stories: tuple[Story, ...]) -> PrioritizationResult:
    """Prioritize stories by value and dependencies.

    Main function that orchestrates the prioritization process:
    1. Calculate value scores for all stories
    2. Analyze dependencies and detect cycles
    3. Calculate dependency adjustments
    4. Identify quick wins
    5. Build priority scores and sort

    Args:
        stories: Tuple of stories to prioritize.

    Returns:
        PrioritizationResult with scores, order, quick wins, and cycles.

    Example:
        >>> result = prioritize_stories((story1, story2, story3))
        >>> result["recommended_execution_order"]
        ['story-001', 'story-003', 'story-002']
        >>> result["quick_wins"]
        ['story-001']
    """
    logger.info(
        "prioritization_started",
        story_count=len(stories),
    )

    if not stories:
        logger.info("prioritization_empty", story_count=0)
        return PrioritizationResult(
            scores=[],
            recommended_execution_order=[],
            quick_wins=[],
            dependency_cycles=[],
            analysis_notes="No stories to prioritize",
        )

    # Step 1: Calculate value scores
    value_scores: dict[str, tuple[int, list[str]]] = {}
    for story in stories:
        value_scores[story.id] = _calculate_value_score(story)

    # Step 2: Analyze dependencies
    dep_info_map, cycles = _analyze_dependencies(stories)

    # Step 3-5: Build priority scores
    priority_scores: list[PriorityScore] = []
    quick_wins: list[str] = []

    for story in stories:
        raw_score, value_rationale = value_scores[story.id]
        dep_info = dep_info_map[story.id]

        # Calculate dependency adjustment
        dep_adjustment, dep_rationale = _calculate_dependency_adjustment(dep_info)

        # Calculate final score (clamped)
        final_score = max(0, min(100, raw_score + dep_adjustment))

        # Check quick win
        is_quick = _is_quick_win(story, raw_score, dep_info)
        if is_quick:
            quick_wins.append(story.id)

        # Build rationale string
        all_rationale = value_rationale + dep_rationale
        if is_quick:
            all_rationale.append("quick win")
        rationale_str = ", ".join(all_rationale)

        priority_scores.append(
            PriorityScore(
                story_id=story.id,
                raw_score=raw_score,
                dependency_adjustment=dep_adjustment,
                final_score=final_score,
                is_quick_win=is_quick,
                scoring_rationale=rationale_str,
            )
        )

        logger.debug(
            "prioritization_story_scored",
            story_id=story.id,
            raw_score=raw_score,
            dep_adjustment=dep_adjustment,
            final_score=final_score,
            is_quick_win=is_quick,
        )

    # Step 6: Sort by final score descending
    priority_scores.sort(key=lambda s: s["final_score"], reverse=True)
    recommended_order = [s["story_id"] for s in priority_scores]

    # Build analysis notes
    notes_parts = [
        f"Prioritized {len(stories)} stories",
    ]
    if quick_wins:
        notes_parts.append(f"{len(quick_wins)} quick wins identified")
    if cycles:
        notes_parts.append(f"{len(cycles)} dependency cycles detected")

    analysis_notes = "; ".join(notes_parts)

    logger.info(
        "prioritization_completed",
        story_count=len(stories),
        quick_win_count=len(quick_wins),
        cycle_count=len(cycles),
        top_story=recommended_order[0] if recommended_order else None,
    )

    return PrioritizationResult(
        scores=priority_scores,
        recommended_execution_order=recommended_order,
        quick_wins=quick_wins,
        dependency_cycles=cycles,
        analysis_notes=analysis_notes,
    )
