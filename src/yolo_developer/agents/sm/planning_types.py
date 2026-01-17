"""Type definitions for sprint planning (Story 10.3).

This module provides the data types used by the sprint planning module:

- SprintStory: A story prepared for sprint planning with priority scores
- SprintPlan: Complete sprint plan output with ordered stories
- PlanningConfig: Configuration for sprint planning algorithm

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.planning_types import (
    ...     SprintStory,
    ...     SprintPlan,
    ...     PlanningConfig,
    ... )
    >>>
    >>> story = SprintStory(
    ...     story_id="1-1-user-auth",
    ...     title="Implement user authentication",
    ...     estimated_points=5,
    ...     value_score=0.9,
    ... )
    >>> story.to_dict()
    {'story_id': '1-1-user-auth', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR9: SM Agent can plan sprints by prioritizing and sequencing stories
    - FR65: SM Agent can calculate weighted priority scores for story selection
    - NFR-SCALE-1: MVP supports 5-10 stories per sprint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_STORIES: int = 10
"""Default maximum stories per sprint (per NFR-SCALE-1: MVP supports 5-10)."""

DEFAULT_MAX_POINTS: int = 40
"""Default maximum story points per sprint capacity."""

DEFAULT_VALUE_WEIGHT: float = 0.4
"""Default weight for value score in priority calculation (per FR65)."""

DEFAULT_DEPENDENCY_WEIGHT: float = 0.3
"""Default weight for dependency score in priority calculation (per FR65)."""

DEFAULT_VELOCITY_WEIGHT: float = 0.2
"""Default weight for velocity impact in priority calculation (per FR65)."""

DEFAULT_TECH_DEBT_WEIGHT: float = 0.1
"""Default weight for tech debt score in priority calculation (per FR65)."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class SprintStory:
    """A story prepared for sprint planning.

    Contains all the information needed to prioritize and sequence
    a story within a sprint, including various scoring factors.

    Attributes:
        story_id: Unique story identifier (e.g., "1-2-user-auth")
        title: Human-readable story title
        priority_score: Composite weighted score (calculated by planner)
        dependencies: Tuple of story IDs this depends on
        estimated_points: Story points estimate for capacity planning
        story_points: Points value for velocity tracking (default 1.0)
        value_score: Business value component (0.0-1.0)
        tech_debt_score: Tech debt reduction component (0.0-1.0)
        velocity_impact: Expected velocity impact (0.0-1.0)
        dependency_score: Score based on how many stories depend on this (0.0-1.0)
        metadata: Additional metadata for tracking

    Example:
        >>> story = SprintStory(
        ...     story_id="2-1-db-setup",
        ...     title="Setup database schema",
        ...     estimated_points=3,
        ...     value_score=0.8,
        ...     dependencies=(),
        ... )
    """

    story_id: str
    title: str
    priority_score: float = 0.0
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    estimated_points: int = 1
    story_points: float = 1.0
    value_score: float = 0.5
    tech_debt_score: float = 0.0
    velocity_impact: float = 0.5
    dependency_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the story.
        """
        return {
            "story_id": self.story_id,
            "title": self.title,
            "priority_score": self.priority_score,
            "dependencies": self.dependencies,
            "estimated_points": self.estimated_points,
            "story_points": self.story_points,
            "value_score": self.value_score,
            "tech_debt_score": self.tech_debt_score,
            "velocity_impact": self.velocity_impact,
            "dependency_score": self.dependency_score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SprintPlan:
    """Complete sprint plan output.

    Contains the ordered list of stories selected for a sprint,
    along with capacity metrics and planning rationale for audit.

    Attributes:
        sprint_id: Unique sprint identifier (e.g., "sprint-20260112")
        stories: Ordered tuple of stories for execution sequence
        total_points: Sum of estimated points for selected stories
        capacity_used: Percentage of capacity used (0.0-1.0)
        planning_rationale: Explanation of planning decisions for audit
        created_at: ISO timestamp when plan was created

    Example:
        >>> plan = SprintPlan(
        ...     sprint_id="sprint-20260112",
        ...     stories=(story1, story2),
        ...     total_points=8,
        ...     capacity_used=0.2,
        ...     planning_rationale="Selected high-value stories",
        ... )
    """

    sprint_id: str
    stories: tuple[SprintStory, ...]
    total_points: int
    capacity_used: float
    planning_rationale: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested stories.
        """
        return {
            "sprint_id": self.sprint_id,
            "stories": [s.to_dict() for s in self.stories],
            "total_points": self.total_points,
            "capacity_used": self.capacity_used,
            "planning_rationale": self.planning_rationale,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class PlanningConfig:
    """Configuration for sprint planning algorithm.

    Controls the sprint planning behavior including capacity limits
    and priority scoring weights per FR65.

    Attributes:
        max_stories: Maximum stories per sprint (default 10, per NFR-SCALE-1)
        max_points: Maximum story points capacity (default 40)
        value_weight: Weight for business value in priority (default 0.4)
        dependency_weight: Weight for dependency score (default 0.3)
        velocity_weight: Weight for velocity impact (default 0.2)
        tech_debt_weight: Weight for tech debt reduction (default 0.1)

    Note:
        Weights should sum to 1.0 for normalized priority scores.

    Example:
        >>> config = PlanningConfig(max_stories=5, max_points=20)
        >>> config.value_weight
        0.4
    """

    max_stories: int = DEFAULT_MAX_STORIES
    max_points: int = DEFAULT_MAX_POINTS
    value_weight: float = DEFAULT_VALUE_WEIGHT
    dependency_weight: float = DEFAULT_DEPENDENCY_WEIGHT
    velocity_weight: float = DEFAULT_VELOCITY_WEIGHT
    tech_debt_weight: float = DEFAULT_TECH_DEBT_WEIGHT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "max_stories": self.max_stories,
            "max_points": self.max_points,
            "value_weight": self.value_weight,
            "dependency_weight": self.dependency_weight,
            "velocity_weight": self.velocity_weight,
            "tech_debt_weight": self.tech_debt_weight,
        }
