"""PM agent module for story transformation (Story 6.1, 6.3, 6.4, 6.5).

The PM agent is responsible for:
- Transforming crystallized requirements into user stories
- Generating acceptance criteria in Given/When/Then format
- Story prioritization and dependency identification
- Ensuring AC are testable and measurable (via quality gate)
- Escalating unclear requirements back to Analyst

Example:
    >>> from yolo_developer.agents.pm import (
    ...     pm_node,
    ...     PMOutput,
    ...     Story,
    ...     StoryStatus,
    ...     StoryPriority,
    ...     AcceptanceCriterion,
    ...     prioritize_stories,
    ... )
    >>>
    >>> # Create an acceptance criterion
    >>> ac = AcceptanceCriterion(
    ...     id="AC1",
    ...     given="user logged in",
    ...     when="click logout",
    ...     then="session ends",
    ... )
    >>>
    >>> # Create a story
    >>> story = Story(
    ...     id="story-001",
    ...     title="User logout",
    ...     role="user",
    ...     action="logout",
    ...     benefit="secure session",
    ...     acceptance_criteria=(ac,),
    ...     priority=StoryPriority.HIGH,
    ...     status=StoryStatus.DRAFT,
    ... )
    >>>
    >>> # Run the PM node
    >>> result = await pm_node(state)
    >>>
    >>> # Prioritize stories
    >>> prioritization = prioritize_stories((story,))

Architecture:
    The pm_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O
    - Is decorated with @quality_gate("ac_measurability")

References:
    - ADR-001: TypedDict for internal state
    - ADR-005: LangGraph node patterns
    - ADR-006: Quality gate patterns
    - FR42-48: PM Agent capabilities
"""

from __future__ import annotations

from yolo_developer.agents.pm.dependencies import analyze_dependencies
from yolo_developer.agents.pm.node import pm_node
from yolo_developer.agents.pm.prioritization import prioritize_stories
from yolo_developer.agents.pm.testability import validate_story_testability
from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    DependencyAnalysisResult,
    DependencyEdge,
    DependencyGraph,
    DependencyInfo,
    PMOutput,
    PrioritizationResult,
    PriorityScore,
    Story,
    StoryPriority,
    StoryStatus,
    TestabilityResult,
)

__all__ = [
    "AcceptanceCriterion",
    "DependencyAnalysisResult",
    "DependencyEdge",
    "DependencyGraph",
    "DependencyInfo",
    "PMOutput",
    "PrioritizationResult",
    "PriorityScore",
    "Story",
    "StoryPriority",
    "StoryStatus",
    "TestabilityResult",
    "analyze_dependencies",
    "pm_node",
    "prioritize_stories",
    "validate_story_testability",
]
