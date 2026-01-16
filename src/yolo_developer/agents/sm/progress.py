"""Sprint progress tracking for SM agent (Story 10.9).

This module provides visibility into sprint execution status:
- Completed, in-progress, remaining, and blocked stories
- Current story and agent executing
- Estimated completion time based on historical performance

Key functions:
- track_progress(): Main entry point for progress tracking
- get_progress_summary(): Human-readable summary string
- get_progress_for_display(): Formatted dict for Rich rendering

Example:
    >>> from yolo_developer.agents.sm.progress import track_progress
    >>> progress = await track_progress(state, sprint_plan)
    >>> progress.snapshot.progress_percentage
    50.0

References:
    - FR16: System can track sprint progress and completion status
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - NFR-PERF-3: Real-time status updates <1 second refresh
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

import structlog

from yolo_developer.agents.sm.progress_types import (
    CompletionEstimate,
    ProgressConfig,
    SprintProgress,
    SprintProgressSnapshot,
    StoryProgress,
    StoryStatus,
)

if TYPE_CHECKING:
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)


# =============================================================================
# Story Tracking Functions (Task 2)
# =============================================================================


def _get_story_status(story: dict[str, Any]) -> StoryStatus:
    """Determine story status from story dict.

    Args:
        story: Story dictionary from sprint plan

    Returns:
        StoryStatus literal value
    """
    status = story.get("status", "backlog")
    if status == "backlog":
        return "backlog"
    if status == "in_progress":
        return "in_progress"
    if status == "completed":
        return "completed"
    if status == "blocked":
        return "blocked"
    if status == "failed":
        return "failed"
    return "backlog"


def _categorize_stories_by_status(
    sprint_plan: dict[str, Any] | None,
) -> dict[str, list[StoryProgress]]:
    """Categorize stories by their status.

    Groups stories into completed, in_progress, remaining (backlog),
    and blocked (blocked or failed) categories.

    Args:
        sprint_plan: Sprint plan dict containing stories list

    Returns:
        Dictionary with lists of StoryProgress by category
    """
    result: dict[str, list[StoryProgress]] = {
        "completed": [],
        "in_progress": [],
        "remaining": [],
        "blocked": [],
    }

    if not sprint_plan:
        return result

    stories = sprint_plan.get("stories", [])
    for story in stories:
        status = _get_story_status(story)
        # Note: duration_ms can be provided directly in the story dict (pre-calculated),
        # or callers can use _calculate_story_duration() to compute it from timestamps.
        # The story dict value takes precedence for flexibility in data sources.
        progress = StoryProgress(
            story_id=story.get("story_id", "unknown"),
            title=story.get("title", "Unknown"),
            status=status,
            started_at=story.get("started_at"),
            completed_at=story.get("completed_at"),
            agent_history=tuple(story.get("agent_history", [])),
            duration_ms=story.get("duration_ms"),
            blocked_reason=story.get("blocked_reason"),
        )

        if status == "completed":
            result["completed"].append(progress)
        elif status == "in_progress":
            result["in_progress"].append(progress)
        elif status == "backlog":
            result["remaining"].append(progress)
        elif status in ("blocked", "failed"):
            result["blocked"].append(progress)

    return result


def _get_completed_stories(
    state: YoloState,
    sprint_plan: dict[str, Any] | None,
) -> list[StoryProgress]:
    """Get list of completed stories with metadata.

    Args:
        state: Current orchestration state (reserved for future expansion)
        sprint_plan: Sprint plan dict

    Returns:
        List of completed StoryProgress objects
    """
    _ = state  # Reserved for future expansion
    categories = _categorize_stories_by_status(sprint_plan)
    return categories["completed"]


def _get_in_progress_stories(
    state: YoloState,
    sprint_plan: dict[str, Any] | None,
) -> list[StoryProgress]:
    """Get list of in-progress stories.

    Args:
        state: Current orchestration state (reserved for future expansion)
        sprint_plan: Sprint plan dict

    Returns:
        List of in-progress StoryProgress objects
    """
    _ = state  # Reserved for future expansion
    categories = _categorize_stories_by_status(sprint_plan)
    return categories["in_progress"]


def _get_remaining_stories(
    state: YoloState,
    sprint_plan: dict[str, Any] | None,
) -> list[StoryProgress]:
    """Get list of remaining (backlog) stories.

    Args:
        state: Current orchestration state (reserved for future expansion)
        sprint_plan: Sprint plan dict

    Returns:
        List of remaining StoryProgress objects
    """
    _ = state  # Reserved for future expansion
    categories = _categorize_stories_by_status(sprint_plan)
    return categories["remaining"]


def _get_blocked_stories(
    state: YoloState,
    sprint_plan: dict[str, Any] | None,
) -> list[StoryProgress]:
    """Get list of blocked or failed stories.

    Args:
        state: Current orchestration state (reserved for future expansion)
        sprint_plan: Sprint plan dict

    Returns:
        List of blocked/failed StoryProgress objects
    """
    _ = state  # Reserved for future expansion
    categories = _categorize_stories_by_status(sprint_plan)
    return categories["blocked"]


def _get_current_story(sprint_plan: dict[str, Any] | None) -> str | None:
    """Get the currently executing story ID.

    Args:
        sprint_plan: Sprint plan dict

    Returns:
        Story ID of in-progress story, or None
    """
    if not sprint_plan:
        return None

    stories = sprint_plan.get("stories", [])
    for story in stories:
        if _get_story_status(story) == "in_progress":
            story_id = story.get("story_id")
            if isinstance(story_id, str):
                return story_id
            return None

    return None


def _get_current_agent(state: YoloState) -> str | None:
    """Get the currently executing agent.

    Args:
        state: Current orchestration state

    Returns:
        Agent name, or None if not available
    """
    return state.get("current_agent")


# =============================================================================
# Progress Calculation Functions (Task 3)
# =============================================================================


def _calculate_progress_percentage(completed: int, total: int) -> float:
    """Calculate progress percentage.

    Args:
        completed: Number of completed stories
        total: Total number of stories

    Returns:
        Percentage (0.0-100.0)
    """
    if total == 0:
        return 0.0
    return (completed / total) * 100.0


def _calculate_story_duration(
    started_at: str | None,
    completed_at: str | None,
) -> float | None:
    """Calculate story duration in milliseconds.

    Args:
        started_at: ISO timestamp when story started
        completed_at: ISO timestamp when story completed

    Returns:
        Duration in milliseconds, or None if timestamps unavailable
    """
    if not started_at or not completed_at:
        return None

    try:
        # Handle both Z suffix and +00:00 timezone format
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        delta = end - start
        return delta.total_seconds() * 1000.0
    except (ValueError, TypeError):
        return None


def _build_progress_snapshot(
    sprint_id: str,
    completed: list[StoryProgress],
    in_progress: list[StoryProgress],
    remaining: list[StoryProgress],
    blocked: list[StoryProgress],
    current_agent: str | None,
) -> SprintProgressSnapshot:
    """Build a progress snapshot from categorized stories.

    Args:
        sprint_id: Sprint identifier
        completed: List of completed stories
        in_progress: List of in-progress stories
        remaining: List of remaining stories
        blocked: List of blocked stories
        current_agent: Currently executing agent

    Returns:
        SprintProgressSnapshot with computed metrics
    """
    total = len(completed) + len(in_progress) + len(remaining) + len(blocked)
    current_story: str | None = None
    if in_progress:
        current_story = in_progress[0].story_id

    return SprintProgressSnapshot(
        sprint_id=sprint_id,
        total_stories=total,
        stories_completed=len(completed),
        stories_in_progress=len(in_progress),
        stories_remaining=len(remaining),
        stories_blocked=len(blocked),
        current_story=current_story,
        current_agent=current_agent,
        progress_percentage=_calculate_progress_percentage(len(completed), total),
    )


# =============================================================================
# Completion Estimation Functions (Task 4)
# =============================================================================


def _calculate_average_story_duration(completed: list[StoryProgress]) -> float | None:
    """Calculate average duration of completed stories.

    Args:
        completed: List of completed stories

    Returns:
        Average duration in milliseconds, or None if insufficient data
    """
    durations = [s.duration_ms for s in completed if s.duration_ms is not None]
    if not durations:
        return None
    return mean(durations)


def _calculate_estimation_confidence(completed: list[StoryProgress]) -> float:
    """Calculate confidence level for completion estimates.

    Confidence is based on:
    - Sample size: More completed stories = higher confidence
    - Variance: Lower variance in durations = higher confidence

    Args:
        completed: List of completed stories with duration data

    Returns:
        Confidence level (0.0-1.0)
    """
    durations = [s.duration_ms for s in completed if s.duration_ms is not None]

    if len(durations) < 2:
        return 0.3  # Low confidence with minimal data

    # Factor 1: Sample size (max contribution 0.5)
    sample_factor = min(len(durations) / 10, 0.5)

    # Factor 2: Low variance (max contribution 0.5)
    avg = mean(durations)
    if avg > 0:
        variance_ratio = stdev(durations) / avg
        # Lower variance = higher confidence
        variance_factor = max(0, 0.5 - variance_ratio * 0.25)
    else:
        variance_factor = 0.25

    return min(sample_factor + variance_factor, 1.0)


def _get_estimation_factors(
    completed_count: int,
    avg_duration: float,
    confidence: float,
    confidence_threshold: float,
) -> tuple[str, ...]:
    """Get factors that influenced the completion estimate.

    Args:
        completed_count: Number of completed stories used
        avg_duration: Average story duration in ms
        confidence: Calculated confidence level
        confidence_threshold: Threshold for flagging low confidence

    Returns:
        Tuple of factor descriptions
    """
    factors: list[str] = [
        f"based_on_{completed_count}_completed_stories",
        f"avg_duration_{avg_duration:.0f}ms",
    ]

    if confidence < confidence_threshold:
        factors.append("low_confidence_estimate")

    return tuple(factors)


def _build_completion_estimate(
    completed: list[StoryProgress],
    remaining_count: int,
    confidence_threshold: float,
) -> CompletionEstimate:
    """Build completion estimate from historical data.

    Uses average duration of completed stories to estimate
    remaining time for outstanding work.

    Args:
        completed: List of completed stories with duration data
        remaining_count: Number of stories still to complete
        confidence_threshold: Threshold for flagging low confidence

    Returns:
        CompletionEstimate with timing and confidence info
    """
    avg_duration = _calculate_average_story_duration(completed)
    confidence = _calculate_estimation_confidence(completed)

    if avg_duration is None or remaining_count == 0:
        return CompletionEstimate(
            estimated_completion_time=None,
            estimated_remaining_ms=None,
            confidence=0.0,
            factors=("insufficient_data",),
        )

    estimated_remaining_ms = avg_duration * remaining_count
    factors = _get_estimation_factors(
        len(completed),
        avg_duration,
        confidence,
        confidence_threshold,
    )

    estimated_completion_time = (
        datetime.now(timezone.utc) + timedelta(milliseconds=estimated_remaining_ms)
    ).isoformat()

    return CompletionEstimate(
        estimated_completion_time=estimated_completion_time,
        estimated_remaining_ms=estimated_remaining_ms,
        confidence=confidence,
        factors=factors,
    )


# =============================================================================
# Main Progress Tracking Function (Task 5)
# =============================================================================


async def track_progress(
    state: YoloState,
    sprint_plan: dict[str, Any] | None = None,
    config: ProgressConfig | None = None,
) -> SprintProgress:
    """Track sprint progress and provide completion estimates (FR16, FR66).

    Analyzes the current sprint state to provide visibility into:
    - Completed, in-progress, remaining, and blocked stories
    - Current story and agent executing
    - Estimated completion time based on historical performance

    This is the main entry point for sprint progress tracking.

    Args:
        state: Current orchestration state
        sprint_plan: Sprint plan dict containing stories list
        config: Progress tracking configuration

    Returns:
        SprintProgress with full progress information

    Example:
        >>> progress = await track_progress(state, sprint_plan)
        >>> progress.snapshot.progress_percentage
        50.0
    """
    config = config or ProgressConfig()

    logger.info(
        "progress_tracking_started",
        sprint_id=sprint_plan.get("sprint_id") if sprint_plan else None,
    )

    # Step 1: Categorize stories by status (single call for efficiency)
    # Note: We call _categorize_stories_by_status once here instead of calling
    # _get_completed_stories/_get_in_progress_stories/etc. individually,
    # which would each trigger separate categorization calls.
    _ = state  # Reserved for future expansion (e.g., state-based status overrides)
    categories = _categorize_stories_by_status(sprint_plan)
    completed = categories["completed"]
    in_progress = categories["in_progress"]
    remaining = categories["remaining"]
    blocked = categories["blocked"]

    # Step 2: Build progress snapshot
    sprint_id = sprint_plan.get("sprint_id", "unknown") if sprint_plan else "unknown"
    current_agent = _get_current_agent(state)

    snapshot = _build_progress_snapshot(
        sprint_id=sprint_id,
        completed=completed,
        in_progress=in_progress,
        remaining=remaining,
        blocked=blocked,
        current_agent=current_agent,
    )

    # Step 3: Calculate completion estimate if configured
    completion_estimate: CompletionEstimate | None = None
    if config.include_estimates and completed:
        completion_estimate = _build_completion_estimate(
            completed=completed,
            remaining_count=len(remaining) + len(in_progress),
            confidence_threshold=config.estimate_confidence_threshold,
        )

    logger.info(
        "progress_tracking_complete",
        completed_count=len(completed),
        in_progress_count=len(in_progress),
        remaining_count=len(remaining),
        blocked_count=len(blocked),
        progress_percentage=snapshot.progress_percentage,
    )

    return SprintProgress(
        snapshot=snapshot,
        completed_stories=tuple(completed),
        in_progress_stories=tuple(in_progress),
        remaining_stories=tuple(remaining),
        blocked_stories=tuple(blocked),
        completion_estimate=completion_estimate,
    )


# =============================================================================
# Query Helpers for CLI/SDK (Task 6)
# =============================================================================


def get_progress_summary(progress: SprintProgress) -> str:
    """Generate a human-readable progress summary.

    Creates a brief summary suitable for CLI output or logging.

    Args:
        progress: SprintProgress from track_progress()

    Returns:
        Human-readable summary string

    Example:
        >>> summary = get_progress_summary(progress)
        >>> print(summary)
        Sprint sprint-test: 50.0% complete (5/10 stories)
        Current: 5-3 (dev agent)
    """
    s = progress.snapshot
    lines = [
        f"Sprint {s.sprint_id}: {s.progress_percentage:.1f}% complete "
        f"({s.stories_completed}/{s.total_stories} stories)",
    ]

    if s.current_story:
        agent_info = f" ({s.current_agent} agent)" if s.current_agent else ""
        lines.append(f"Current: {s.current_story}{agent_info}")

    if s.stories_blocked > 0:
        lines.append(f"Blocked: {s.stories_blocked} stories")

    if progress.completion_estimate and progress.completion_estimate.estimated_remaining_ms:
        hours = progress.completion_estimate.estimated_remaining_ms / 3600000.0
        conf = progress.completion_estimate.confidence
        lines.append(f"Estimated remaining: {hours:.1f} hours (confidence: {conf:.0%})")

    return "\n".join(lines)


def get_progress_for_display(progress: SprintProgress) -> dict[str, Any]:
    """Format progress data for Rich rendering.

    Creates a dictionary optimized for display with Rich tables/panels.

    Args:
        progress: SprintProgress from track_progress()

    Returns:
        Dictionary formatted for Rich rendering

    Example:
        >>> data = get_progress_for_display(progress)
        >>> table = Table()
        >>> for key, value in data.items():
        ...     table.add_row(key, str(value))
    """
    s = progress.snapshot
    result: dict[str, Any] = {
        "sprint_id": s.sprint_id,
        "progress_percentage": s.progress_percentage,
        "stories_completed": s.stories_completed,
        "stories_in_progress": s.stories_in_progress,
        "stories_remaining": s.stories_remaining,
        "stories_blocked": s.stories_blocked,
        "total_stories": s.total_stories,
        "current_story": s.current_story,
        "current_agent": s.current_agent,
    }

    if progress.completion_estimate:
        e = progress.completion_estimate
        result["estimated_completion_time"] = e.estimated_completion_time
        result["estimated_remaining_ms"] = e.estimated_remaining_ms
        result["confidence"] = e.confidence
        result["estimation_factors"] = list(e.factors)

    return result


def get_stories_by_status(
    progress: SprintProgress,
    status: str,
) -> list[StoryProgress]:
    """Get stories filtered by status.

    Helper function to retrieve stories by their status category.

    Args:
        progress: SprintProgress from track_progress()
        status: Status to filter by ("completed", "in_progress", "backlog", "blocked")

    Returns:
        List of matching StoryProgress objects

    Example:
        >>> completed = get_stories_by_status(progress, "completed")
        >>> len(completed)
        5
    """
    if status == "completed":
        return list(progress.completed_stories)
    if status == "in_progress":
        return list(progress.in_progress_stories)
    if status in ("backlog", "remaining"):
        return list(progress.remaining_stories)
    if status in ("blocked", "failed"):
        return list(progress.blocked_stories)
    return []
