"""Type definitions for sprint progress tracking (Story 10.9).

This module provides the data types used by the sprint progress tracking module:

- StoryStatus: Literal type for story execution status
- StoryProgress: Progress information for a single story
- SprintProgressSnapshot: Point-in-time snapshot of sprint progress
- CompletionEstimate: Estimated completion information
- SprintProgress: Complete sprint progress information
- ProgressConfig: Configuration for progress tracking

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.progress_types import (
    ...     StoryProgress,
    ...     SprintProgress,
    ...     SprintProgressSnapshot,
    ...     ProgressConfig,
    ... )
    >>>
    >>> story = StoryProgress(
    ...     story_id="1-1-auth",
    ...     title="User Authentication",
    ...     status="completed",
    ...     duration_ms=3600000.0,
    ... )
    >>> story.to_dict()
    {'story_id': '1-1-auth', 'title': 'User Authentication', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR16: System can track sprint progress and completion status
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - NFR-PERF-3: Real-time status updates <1 second refresh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

StoryStatus = Literal["backlog", "in_progress", "completed", "blocked", "failed"]
"""Status of a story in the sprint.

Values:
    backlog: Story not yet started
    in_progress: Story currently being worked on
    completed: Story finished successfully
    blocked: Story cannot proceed (gate blocked, dependency issue)
    failed: Story failed and cannot be recovered
"""


# =============================================================================
# Constants
# =============================================================================

DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD: float = 0.7
"""Minimum confidence level for completion estimates to be considered reliable.

Estimates with confidence below this threshold are flagged with
'low_confidence_estimate' in their factors.
"""

VALID_STORY_STATUSES: frozenset[str] = frozenset(
    {
        "backlog",
        "in_progress",
        "completed",
        "blocked",
        "failed",
    }
)
"""Set of valid story status values.

Used for runtime validation of status values.
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class StoryProgress:
    """Progress information for a single story.

    Tracks the current status and timing of a story within the sprint,
    including its execution history and any blocking issues.

    Attributes:
        story_id: Unique story identifier (e.g., "1-2-user-auth")
        title: Human-readable story title
        status: Current execution status
        started_at: ISO timestamp when story execution began (None if not started)
        completed_at: ISO timestamp when story completed (None if not completed)
        agent_history: Tuple of agent names that have worked on this story
        duration_ms: Total duration in milliseconds (None if not completed)
        blocked_reason: Reason for blocking (None if not blocked)

    Example:
        >>> story = StoryProgress(
        ...     story_id="1-1-setup",
        ...     title="Project Setup",
        ...     status="completed",
        ...     started_at="2026-01-16T10:00:00+00:00",
        ...     completed_at="2026-01-16T11:00:00+00:00",
        ...     agent_history=("analyst", "pm", "dev"),
        ...     duration_ms=3600000.0,
        ... )
    """

    story_id: str
    title: str
    status: StoryStatus
    started_at: str | None = None
    completed_at: str | None = None
    agent_history: tuple[str, ...] = field(default_factory=tuple)
    duration_ms: float | None = None
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the story progress.
            Note: agent_history tuple is converted to list for JSON compatibility.
        """
        return {
            "story_id": self.story_id,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "agent_history": list(self.agent_history),
            "duration_ms": self.duration_ms,
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class SprintProgressSnapshot:
    """Point-in-time snapshot of sprint progress.

    Provides counts and identifiers for current sprint state, giving
    a high-level overview of progress without detailed story information.

    Attributes:
        sprint_id: Unique sprint identifier (e.g., "sprint-20260116")
        total_stories: Total number of stories in the sprint
        stories_completed: Number of completed stories
        stories_in_progress: Number of stories currently being worked on
        stories_remaining: Number of stories not yet started
        stories_blocked: Number of blocked stories
        current_story: Story ID currently being worked on (None if none active)
        current_agent: Agent currently executing (None if none active)
        progress_percentage: Completion percentage (0.0-100.0)

    Example:
        >>> snapshot = SprintProgressSnapshot(
        ...     sprint_id="sprint-20260116",
        ...     total_stories=10,
        ...     stories_completed=5,
        ...     stories_in_progress=2,
        ...     stories_remaining=2,
        ...     stories_blocked=1,
        ...     current_story="5-3-feature",
        ...     current_agent="dev",
        ...     progress_percentage=50.0,
        ... )
    """

    sprint_id: str
    total_stories: int
    stories_completed: int
    stories_in_progress: int
    stories_remaining: int
    stories_blocked: int
    current_story: str | None
    current_agent: str | None
    progress_percentage: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the snapshot.
        """
        return {
            "sprint_id": self.sprint_id,
            "total_stories": self.total_stories,
            "stories_completed": self.stories_completed,
            "stories_in_progress": self.stories_in_progress,
            "stories_remaining": self.stories_remaining,
            "stories_blocked": self.stories_blocked,
            "current_story": self.current_story,
            "current_agent": self.current_agent,
            "progress_percentage": self.progress_percentage,
        }


@dataclass(frozen=True)
class CompletionEstimate:
    """Estimated completion information for the sprint.

    Based on historical data from completed stories, providing
    time estimates and confidence levels.

    Attributes:
        estimated_completion_time: ISO timestamp of estimated completion
            (None if insufficient data)
        estimated_remaining_ms: Estimated remaining time in milliseconds
            (None if insufficient data)
        confidence: Confidence level of the estimate (0.0-1.0)
        factors: Tuple of factors that influenced the estimate

    Example:
        >>> estimate = CompletionEstimate(
        ...     estimated_completion_time="2026-01-16T15:00:00+00:00",
        ...     estimated_remaining_ms=18000000.0,
        ...     confidence=0.85,
        ...     factors=("based_on_5_completed_stories", "avg_duration_3600000ms"),
        ... )
    """

    estimated_completion_time: str | None
    estimated_remaining_ms: float | None
    confidence: float
    factors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the estimate.
            Note: factors tuple is converted to list for JSON compatibility.
        """
        return {
            "estimated_completion_time": self.estimated_completion_time,
            "estimated_remaining_ms": self.estimated_remaining_ms,
            "confidence": self.confidence,
            "factors": list(self.factors),
        }


@dataclass(frozen=True)
class SprintProgress:
    """Complete sprint progress information.

    Combines snapshot, detailed story lists, and completion estimates
    to provide full visibility into sprint execution state.

    Attributes:
        snapshot: High-level progress snapshot
        completed_stories: Tuple of completed story progress records
        in_progress_stories: Tuple of in-progress story progress records
        remaining_stories: Tuple of not-yet-started story progress records
        blocked_stories: Tuple of blocked story progress records
        completion_estimate: Estimated completion information (None if unavailable)
        created_at: ISO timestamp when this progress was captured

    Example:
        >>> progress = SprintProgress(
        ...     snapshot=snapshot,
        ...     completed_stories=(story1, story2),
        ...     in_progress_stories=(story3,),
        ...     remaining_stories=(story4,),
        ...     blocked_stories=(),
        ...     completion_estimate=estimate,
        ... )
    """

    snapshot: SprintProgressSnapshot
    completed_stories: tuple[StoryProgress, ...]
    in_progress_stories: tuple[StoryProgress, ...]
    remaining_stories: tuple[StoryProgress, ...]
    blocked_stories: tuple[StoryProgress, ...]
    completion_estimate: CompletionEstimate | None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with all nested structures serialized.
        """
        return {
            "snapshot": self.snapshot.to_dict(),
            "completed_stories": [s.to_dict() for s in self.completed_stories],
            "in_progress_stories": [s.to_dict() for s in self.in_progress_stories],
            "remaining_stories": [s.to_dict() for s in self.remaining_stories],
            "blocked_stories": [s.to_dict() for s in self.blocked_stories],
            "completion_estimate": (
                self.completion_estimate.to_dict() if self.completion_estimate else None
            ),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ProgressConfig:
    """Configuration for progress tracking.

    Controls what information is included in progress reports and
    how estimates are calculated.

    Attributes:
        include_estimates: Whether to include completion estimates
        estimate_confidence_threshold: Minimum confidence for reliable estimates
        track_agent_history: Whether to track which agents worked on stories
        include_blocked_details: Whether to include blocking reason details

    Example:
        >>> config = ProgressConfig(
        ...     include_estimates=True,
        ...     estimate_confidence_threshold=0.8,
        ... )
    """

    include_estimates: bool = True
    estimate_confidence_threshold: float = DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD
    track_agent_history: bool = True
    include_blocked_details: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "include_estimates": self.include_estimates,
            "estimate_confidence_threshold": self.estimate_confidence_threshold,
            "track_agent_history": self.track_agent_history,
            "include_blocked_details": self.include_blocked_details,
        }
