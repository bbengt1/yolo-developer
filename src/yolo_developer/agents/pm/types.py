"""Type definitions for PM agent (Story 6.1, 6.3, 6.4, 6.5, 6.6).

This module provides the data types used by the PM agent:

- StoryStatus: Status of a user story in the workflow
- StoryPriority: Priority level for story ordering
- AcceptanceCriterion: A single acceptance criterion in Given/When/Then format
- Story: A user story with acceptance criteria and metadata
- PMOutput: Complete output from PM processing
- TestabilityResult: Result of AC testability validation (Story 6.3)
- DependencyInfo: Dependency analysis for a single story (Story 6.4)
- PriorityScore: Priority score breakdown for a single story (Story 6.4)
- PrioritizationResult: Complete prioritization analysis result (Story 6.4)
- DependencyEdge: A single edge in the dependency graph (Story 6.5)
- DependencyGraph: Graph representation of story dependencies (Story 6.5)
- DependencyAnalysisResult: Complete dependency analysis result (Story 6.5)
- CoverageMapping: Mapping of original AC to covering sub-stories (Story 6.6)
- EpicBreakdownResult: Result of breaking down a large story (Story 6.6)

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
from typing import Any, TypedDict


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


class TestabilityResult(TypedDict):
    """Result of AC testability validation (Story 6.3).

    TypedDict containing the complete results from validating a story's
    acceptance criteria for testability issues.

    Fields:
        is_valid: True if all critical checks pass (no vague terms, no structural issues).
        vague_terms_found: List of (ac_id, term) tuples for each vague term found.
        missing_edge_cases: List of missing edge case category names
            (e.g., "error_handling", "empty_input", "boundary").
        ac_count_warning: Warning message if AC count is unusual, or None.
        validation_notes: List of detailed validation findings.

    Example:
        >>> result: TestabilityResult = {
        ...     "is_valid": False,
        ...     "vague_terms_found": [("AC1", "fast"), ("AC2", "easy")],
        ...     "missing_edge_cases": ["boundary"],
        ...     "ac_count_warning": None,
        ...     "validation_notes": ["Found vague terms in 2 ACs"],
        ... }
    """

    is_valid: bool
    vague_terms_found: list[tuple[str, str]]
    missing_edge_cases: list[str]
    ac_count_warning: str | None
    validation_notes: list[str]


class DependencyInfo(TypedDict):
    """Dependency analysis for a single story (Story 6.4).

    TypedDict containing dependency graph information for a story,
    used in prioritization to adjust scores based on dependencies.

    Fields:
        blocking_count: Number of stories that depend on this one.
        blocked_by_count: Number of unfinished dependencies this story has.
        blocking_story_ids: IDs of stories that depend on this story.
        blocked_by_story_ids: IDs of stories this story depends on.
        in_cycle: True if story is part of a dependency cycle.

    Example:
        >>> info: DependencyInfo = {
        ...     "blocking_count": 2,
        ...     "blocked_by_count": 1,
        ...     "blocking_story_ids": ["story-002", "story-003"],
        ...     "blocked_by_story_ids": ["story-000"],
        ...     "in_cycle": False,
        ... }
    """

    blocking_count: int
    blocked_by_count: int
    blocking_story_ids: list[str]
    blocked_by_story_ids: list[str]
    in_cycle: bool


class PriorityScore(TypedDict):
    """Priority score breakdown for a single story (Story 6.4).

    TypedDict containing the complete priority score breakdown for a story,
    including raw value score, dependency adjustments, and quick win status.

    Fields:
        story_id: The story's unique identifier.
        raw_score: Base priority score (0-100) based on value factors.
        dependency_adjustment: Score adjustment (-20 to +20) based on dependencies.
        final_score: Clamped sum of raw_score + dependency_adjustment (0-100).
        is_quick_win: True if story qualifies as a quick win.
        scoring_rationale: Human-readable explanation of score factors.

    Example:
        >>> score: PriorityScore = {
        ...     "story_id": "story-001",
        ...     "raw_score": 85,
        ...     "dependency_adjustment": 10,
        ...     "final_score": 95,
        ...     "is_quick_win": True,
        ...     "scoring_rationale": "HIGH priority (+75), S complexity (+10), unblocks 2 stories (+10)",
        ... }
    """

    story_id: str
    raw_score: int
    dependency_adjustment: int
    final_score: int
    is_quick_win: bool
    scoring_rationale: str


class PrioritizationResult(TypedDict):
    """Complete prioritization analysis result (Story 6.4).

    TypedDict containing the full results of story prioritization,
    including all scores, recommended order, and dependency analysis.

    Fields:
        scores: List of PriorityScore for each story.
        recommended_execution_order: Story IDs sorted by final_score descending.
        quick_wins: Story IDs flagged as quick wins.
        dependency_cycles: Detected cycles as lists of story ID chains.
        analysis_notes: Summary of prioritization analysis.

    Example:
        >>> result: PrioritizationResult = {
        ...     "scores": [{"story_id": "story-001", ...}],
        ...     "recommended_execution_order": ["story-001", "story-002"],
        ...     "quick_wins": ["story-001"],
        ...     "dependency_cycles": [],
        ...     "analysis_notes": "Prioritized 2 stories, 1 quick win identified",
        ... }
    """

    scores: list[PriorityScore]
    recommended_execution_order: list[str]
    quick_wins: list[str]
    dependency_cycles: list[list[str]]
    analysis_notes: str


class DependencyEdge(TypedDict):
    """A single edge in the dependency graph (Story 6.5).

    TypedDict representing a directed dependency relationship between two stories.
    The edge direction is from the dependent story to the story it depends on.

    Fields:
        from_story_id: The ID of the dependent story (the one that needs another).
        to_story_id: The ID of the dependency (the story that must be done first).
        reason: Human-readable explanation of why this dependency exists.

    Example:
        >>> edge: DependencyEdge = {
        ...     "from_story_id": "story-002",
        ...     "to_story_id": "story-001",
        ...     "reason": "Requires user authentication from story-001",
        ... }
    """

    from_story_id: str
    to_story_id: str
    reason: str


class DependencyGraph(TypedDict):
    """Graph representation of story dependencies (Story 6.5).

    TypedDict containing the complete dependency graph structure for a set of stories.
    Supports querying "what does X depend on" via adjacency_list and "what depends on X"
    via reverse_adjacency_list.

    Fields:
        nodes: List of all story IDs in the graph.
        edges: List of DependencyEdge objects representing all dependencies.
        adjacency_list: Maps story_id -> list of story IDs it depends on.
        reverse_adjacency_list: Maps story_id -> list of story IDs that depend on it.

    Example:
        >>> graph: DependencyGraph = {
        ...     "nodes": ["story-001", "story-002", "story-003"],
        ...     "edges": [
        ...         {"from_story_id": "story-002", "to_story_id": "story-001", "reason": "auth"},
        ...         {"from_story_id": "story-003", "to_story_id": "story-001", "reason": "db"},
        ...     ],
        ...     "adjacency_list": {"story-002": ["story-001"], "story-003": ["story-001"]},
        ...     "reverse_adjacency_list": {"story-001": ["story-002", "story-003"]},
        ... }
    """

    nodes: list[str]
    edges: list[DependencyEdge]
    adjacency_list: dict[str, list[str]]
    reverse_adjacency_list: dict[str, list[str]]


class DependencyAnalysisResult(TypedDict):
    """Complete dependency analysis result (Story 6.5).

    TypedDict containing the full results of story dependency analysis,
    including the dependency graph, detected cycles, and critical path information.

    Fields:
        graph: The complete DependencyGraph structure.
        cycles: List of detected cycles, each as a list of story IDs forming the cycle.
        critical_paths: List of critical paths (all equal-length longest paths per AC4).
        critical_path_length: Length of the critical path (0 if cycles exist).
        has_cycles: True if any dependency cycles were detected.
        analysis_notes: Summary of the dependency analysis.

    Example:
        >>> result: DependencyAnalysisResult = {
        ...     "graph": {"nodes": [...], "edges": [...], ...},
        ...     "cycles": [],
        ...     "critical_paths": [["story-001", "story-002", "story-003"]],
        ...     "critical_path_length": 3,
        ...     "has_cycles": False,
        ...     "analysis_notes": "Analyzed 5 stories, found 3 dependencies, critical path length: 3",
        ... }
    """

    graph: DependencyGraph
    cycles: list[list[str]]
    critical_paths: list[list[str]]
    critical_path_length: int
    has_cycles: bool
    analysis_notes: str


class CoverageMapping(TypedDict):
    """Mapping of original requirement aspects to covering sub-stories (Story 6.6).

    TypedDict tracking how sub-stories cover the original story's acceptance criteria
    during epic breakdown. Used to ensure full requirement coverage.

    Fields:
        original_ac_id: The ID of the original acceptance criterion being tracked.
        covering_story_ids: List of sub-story IDs that cover this AC.
        is_covered: True if at least one sub-story addresses this AC.

    Example:
        >>> mapping: CoverageMapping = {
        ...     "original_ac_id": "AC1",
        ...     "covering_story_ids": ["story-001.1", "story-001.2"],
        ...     "is_covered": True,
        ... }
    """

    original_ac_id: str
    covering_story_ids: list[str]
    is_covered: bool


class EpicBreakdownResult(TypedDict):
    """Result of breaking down a large story into sub-stories (Story 6.6).

    TypedDict containing the complete results of epic breakdown analysis,
    including generated sub-stories, coverage validation, and rationale.

    Fields:
        original_story_id: ID of the story that was broken down.
        sub_stories: Tuple of generated sub-Story objects.
        coverage_mappings: List of CoverageMapping showing how ACs are covered.
        breakdown_rationale: Human-readable explanation of why/how breakdown occurred.
        is_valid: True if all original ACs are covered by sub-stories.

    Example:
        >>> result: EpicBreakdownResult = {
        ...     "original_story_id": "story-001",
        ...     "sub_stories": (sub_story_1, sub_story_2),
        ...     "coverage_mappings": [{"original_ac_id": "AC1", ...}],
        ...     "breakdown_rationale": "Story had 6 ACs and XL complexity",
        ...     "is_valid": True,
        ... }
    """

    original_story_id: str
    sub_stories: tuple[Story, ...]
    coverage_mappings: list[CoverageMapping]
    breakdown_rationale: str
    is_valid: bool
