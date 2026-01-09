"""Type definitions for PM agent (Story 6.1).

This module provides the data types used by the PM agent:

- StoryStatus: Status of a user story in the workflow
- StoryPriority: Priority level for story ordering
- AcceptanceCriterion: A single acceptance criterion in Given/When/Then format
- Story: A user story with acceptance criteria and metadata
- PMOutput: Complete output from PM processing

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.pm.types import (
    ...     Story,
    ...     StoryStatus,
    ...     StoryPriority,
    ...     AcceptanceCriterion,
    ...     PMOutput,
    ... )
    >>>
    >>> ac = AcceptanceCriterion(
    ...     id="AC1",
    ...     given="a user is logged in",
    ...     when="they click the logout button",
    ...     then="they are redirected to the login page",
    ... )
    >>> story = Story(
    ...     id="story-001",
    ...     title="User logout",
    ...     role="authenticated user",
    ...     action="logout from the system",
    ...     benefit="my session is terminated securely",
    ...     acceptance_criteria=(ac,),
    ...     priority=StoryPriority.HIGH,
    ...     status=StoryStatus.DRAFT,
    ...     source_requirements=("req-001",),
    ... )

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StoryStatus(str, Enum):
    """Status of a user story in the workflow.

    Used to track story progression through the development lifecycle.

    Values:
        DRAFT: Story is being written, not yet ready for development.
        READY: Story is complete and ready for development.
        BLOCKED: Story is blocked by dependencies or external factors.
        IN_PROGRESS: Story is currently being implemented.
        DONE: Story implementation is complete.
    """

    DRAFT = "draft"
    READY = "ready"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class StoryPriority(str, Enum):
    """Priority level for story ordering.

    Used to determine which stories should be implemented first.

    Values:
        CRITICAL: Must be done immediately, blocks other work.
        HIGH: Important, should be done soon.
        MEDIUM: Normal priority, do after higher priority items.
        LOW: Nice to have, do when time permits.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class AcceptanceCriterion:
    """A single acceptance criterion in Given/When/Then format.

    Immutable dataclass representing a testable condition that must be
    satisfied for a story to be considered complete. Follows the BDD
    Given/When/Then pattern for clarity and testability.

    Attributes:
        id: Unique identifier within the story (e.g., "AC1", "AC2").
        given: Precondition describing the initial context.
        when: Action or event that triggers the behavior.
        then: Expected outcome that should occur.
        and_clauses: Additional conditions or outcomes (optional).

    Example:
        >>> ac = AcceptanceCriterion(
        ...     id="AC1",
        ...     given="a user is logged in",
        ...     when="they click the logout button",
        ...     then="they are redirected to the login page",
        ...     and_clauses=("their session is invalidated",),
        ... )
        >>> ac.id
        'AC1'
    """

    id: str
    given: str
    when: str
    then: str
    and_clauses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, and_clauses as list.

        Example:
            >>> ac = AcceptanceCriterion(
            ...     id="AC1",
            ...     given="user logged in",
            ...     when="click logout",
            ...     then="redirected",
            ... )
            >>> ac.to_dict()["id"]
            'AC1'
        """
        return {
            "id": self.id,
            "given": self.given,
            "when": self.when,
            "then": self.then,
            "and_clauses": list(self.and_clauses),
        }


@dataclass(frozen=True)
class Story:
    """A user story with acceptance criteria and metadata.

    Immutable dataclass representing a complete user story that follows
    the "As a / I want / So that" format with structured acceptance criteria.

    Attributes:
        id: Unique story identifier (e.g., "story-001").
        title: Short descriptive title of the story.
        role: The user role from "As a {role}" part.
        action: What the user wants from "I want {action}" part.
        benefit: The benefit from "So that {benefit}" part.
        acceptance_criteria: Tuple of AcceptanceCriterion objects.
        priority: Story priority for ordering.
        status: Current status in the workflow.
        source_requirements: Requirement IDs this story addresses.
        dependencies: Story IDs this story depends on.
        estimated_complexity: Complexity estimate (S, M, L, XL).

    Example:
        >>> ac = AcceptanceCriterion(
        ...     id="AC1",
        ...     given="precondition",
        ...     when="action",
        ...     then="outcome",
        ... )
        >>> story = Story(
        ...     id="story-001",
        ...     title="User login",
        ...     role="visitor",
        ...     action="log into the system",
        ...     benefit="I can access my account",
        ...     acceptance_criteria=(ac,),
        ...     priority=StoryPriority.HIGH,
        ...     status=StoryStatus.DRAFT,
        ...     source_requirements=("req-001",),
        ... )
        >>> story.id
        'story-001'
    """

    id: str
    title: str
    role: str
    action: str
    benefit: str
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    priority: StoryPriority
    status: StoryStatus
    source_requirements: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    estimated_complexity: str = "M"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, enums as string values,
            nested AcceptanceCriterion objects as dicts.

        Example:
            >>> ac = AcceptanceCriterion(
            ...     id="AC1", given="g", when="w", then="t"
            ... )
            >>> story = Story(
            ...     id="story-001",
            ...     title="Test",
            ...     role="user",
            ...     action="do",
            ...     benefit="get",
            ...     acceptance_criteria=(ac,),
            ...     priority=StoryPriority.MEDIUM,
            ...     status=StoryStatus.DRAFT,
            ... )
            >>> story.to_dict()["priority"]
            'medium'
        """
        return {
            "id": self.id,
            "title": self.title,
            "role": self.role,
            "action": self.action,
            "benefit": self.benefit,
            "acceptance_criteria": [ac.to_dict() for ac in self.acceptance_criteria],
            "priority": self.priority.value,
            "status": self.status.value,
            "source_requirements": list(self.source_requirements),
            "dependencies": list(self.dependencies),
            "estimated_complexity": self.estimated_complexity,
        }


@dataclass(frozen=True)
class PMOutput:
    """Complete output from PM agent processing.

    Immutable dataclass containing all results from transforming
    crystallized requirements into user stories with acceptance criteria.

    Attributes:
        stories: Tuple of Story objects created from requirements.
        unprocessed_requirements: Requirement IDs that couldn't be transformed.
        escalations_to_analyst: Requirement IDs needing analyst clarification.
        processing_notes: Notes about the processing for audit trail.

    Properties:
        story_count: Number of stories generated.
        has_escalations: True if there are escalations to analyst.

    Example:
        >>> output = PMOutput(
        ...     stories=(),
        ...     unprocessed_requirements=("req-005",),
        ...     escalations_to_analyst=("req-006",),
        ...     processing_notes="Processed 10 requirements",
        ... )
        >>> output.story_count
        0
        >>> output.has_escalations
        True
    """

    stories: tuple[Story, ...]
    unprocessed_requirements: tuple[str, ...] = ()
    escalations_to_analyst: tuple[str, ...] = ()
    processing_notes: str = ""

    @property
    def story_count(self) -> int:
        """Get the number of stories generated.

        Returns:
            Count of stories in the output.

        Example:
            >>> output = PMOutput(stories=())
            >>> output.story_count
            0
        """
        return len(self.stories)

    @property
    def has_escalations(self) -> bool:
        """Check if there are escalations to analyst.

        Returns:
            True if there are requirements needing clarification.

        Example:
            >>> output = PMOutput(stories=(), escalations_to_analyst=("req-001",))
            >>> output.has_escalations
            True
        """
        return len(self.escalations_to_analyst) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields including computed properties,
            nested Story objects as dicts.

        Example:
            >>> output = PMOutput(
            ...     stories=(),
            ...     processing_notes="Done",
            ... )
            >>> output.to_dict()["story_count"]
            0
        """
        return {
            "stories": [s.to_dict() for s in self.stories],
            "unprocessed_requirements": list(self.unprocessed_requirements),
            "escalations_to_analyst": list(self.escalations_to_analyst),
            "processing_notes": self.processing_notes,
            "story_count": self.story_count,
            "has_escalations": self.has_escalations,
        }
