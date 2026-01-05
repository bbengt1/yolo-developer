"""Decision data structures for historical decision queries.

This module defines dataclasses and enums for representing agent decisions
made during project execution. Decisions are stored with rationale and
metadata to enable semantic similarity search for learning from past decisions.

Example:
    >>> from yolo_developer.memory.decisions import Decision, DecisionType
    >>>
    >>> decision = Decision(
    ...     id="dec-001",
    ...     agent_type="Architect",
    ...     context="Choosing between REST and GraphQL",
    ...     rationale="REST aligns with team expertise and project timeline",
    ...     decision_type=DecisionType.ARCHITECTURE_CHOICE,
    ... )
    >>> decision.to_embedding_text()
    'Architect decision (architecture_choice): Choosing between REST and GraphQL. Rationale: REST aligns with team expertise and project timeline.'

Security Note:
    Decisions are stored per-project to ensure isolation. Decision data
    should not contain sensitive information beyond project decision metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Valid agent types per AC4 specification
VALID_AGENT_TYPES: frozenset[str] = frozenset({"Analyst", "PM", "Architect", "Dev", "SM", "TEA"})


def validate_agent_type(agent_type: str) -> str:
    """Validate that agent_type is one of the allowed values.

    Args:
        agent_type: The agent type string to validate.

    Returns:
        The validated agent type string.

    Raises:
        ValueError: If agent_type is not in VALID_AGENT_TYPES.

    Example:
        >>> validate_agent_type("Architect")
        'Architect'
        >>> validate_agent_type("Invalid")
        Raises ValueError
    """
    if agent_type not in VALID_AGENT_TYPES:
        raise ValueError(
            f"Invalid agent_type '{agent_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_AGENT_TYPES))}"
        )
    return agent_type


class DecisionType(Enum):
    """Types of decisions that agents can make.

    Decision types categorize the nature of choices made during project
    execution. Each type maps to a specific decision domain that agents
    can query for similar historical situations.

    Attributes:
        REQUIREMENT_CLARIFICATION: Clarifying ambiguous requirements.
        STORY_PRIORITIZATION: Ordering stories by priority.
        ARCHITECTURE_CHOICE: Technical architecture decisions.
        IMPLEMENTATION_APPROACH: How to implement a feature or fix.
        TEST_STRATEGY: Testing approach decisions.
        CONFLICT_RESOLUTION: Resolving conflicts between requirements or agents.
    """

    REQUIREMENT_CLARIFICATION = "requirement_clarification"
    STORY_PRIORITIZATION = "story_prioritization"
    ARCHITECTURE_CHOICE = "architecture_choice"
    IMPLEMENTATION_APPROACH = "implementation_approach"
    TEST_STRATEGY = "test_strategy"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass(frozen=True)
class Decision:
    """A decision made by an agent during project execution.

    Represents a single decision with its context, rationale, and outcome.
    Decisions are immutable (frozen) to ensure consistency once recorded.
    Each decision has metadata for filtering and semantic search.

    Attributes:
        id: Unique identifier for the decision.
        agent_type: Type of agent that made the decision (Analyst, PM, Architect, Dev, SM, TEA).
        context: The situation or problem that required a decision.
        rationale: The reasoning behind the decision.
        outcome: The result of the decision (if known).
        decision_type: Category of the decision.
        artifact_type: Type of artifact involved (requirement, story, design, code).
        artifact_ids: IDs of related artifacts.
        timestamp: When the decision was made.

    Raises:
        ValueError: If agent_type is not a valid agent type.

    Example:
        >>> decision = Decision(
        ...     id="dec-001",
        ...     agent_type="Architect",
        ...     context="Database selection for user service",
        ...     rationale="PostgreSQL for ACID compliance",
        ... )
        >>> decision.agent_type
        'Architect'
    """

    id: str
    agent_type: str
    context: str
    rationale: str
    outcome: str | None = None
    decision_type: DecisionType | None = None
    artifact_type: str | None = None
    artifact_ids: tuple[str, ...] = field(default_factory=tuple)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate agent_type after initialization."""
        validate_agent_type(self.agent_type)

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding storage.

        Creates a text string suitable for vector embedding that captures
        the decision's agent type, context, rationale, and outcome. This
        text is used for semantic similarity search in ChromaDB.

        Returns:
            Text representation of the decision for embedding.

        Example:
            >>> decision = Decision(
            ...     id="dec-001",
            ...     agent_type="Architect",
            ...     context="Choosing database",
            ...     rationale="PostgreSQL for reliability",
            ...     decision_type=DecisionType.ARCHITECTURE_CHOICE,
            ... )
            >>> decision.to_embedding_text()
            'Architect decision (architecture_choice): Choosing database. Rationale: PostgreSQL for reliability.'
        """
        type_str = f" ({self.decision_type.value})" if self.decision_type else ""
        text = f"{self.agent_type} decision{type_str}: {self.context}. Rationale: {self.rationale}."

        if self.outcome:
            text += f" Outcome: {self.outcome}."

        return text


@dataclass(frozen=True)
class DecisionResult:
    """Result from a decision similarity search.

    Wraps a Decision with its similarity score from a vector search.
    Used when querying decisions by semantic similarity.

    Attributes:
        decision: The matched Decision.
        similarity: Similarity score 0.0-1.0 (higher is more similar).

    Example:
        >>> result = DecisionResult(decision=decision, similarity=0.87)
        >>> result.similarity
        0.87
    """

    decision: Decision
    similarity: float


@dataclass
class DecisionFilter:
    """Filter criteria for decision queries.

    Supports filtering decisions by agent type, time range, artifact type,
    and decision type. Used with ChromaDB where clauses for efficient
    metadata filtering combined with semantic search.

    Attributes:
        agent_type: Filter by agent type (Analyst, PM, Architect, Dev, SM, TEA).
        time_range_start: Start of time range filter.
        time_range_end: End of time range filter.
        artifact_type: Filter by artifact type (requirement, story, design, code).
        decision_type: Filter by decision category.

    Example:
        >>> filter_obj = DecisionFilter(agent_type="Architect", artifact_type="design")
        >>> where_clause = filter_obj.to_chromadb_where()
    """

    agent_type: str | None = None
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    artifact_type: str | None = None
    decision_type: DecisionType | None = None

    def to_chromadb_where(self) -> dict[str, Any] | None:
        """Convert filter to ChromaDB where clause.

        Generates a ChromaDB-compatible where clause dictionary for
        metadata filtering. Returns None if no filters are set.

        Returns:
            ChromaDB where clause dictionary, or None if no filters.

        Example:
            >>> filter_obj = DecisionFilter(agent_type="PM")
            >>> filter_obj.to_chromadb_where()
            {'agent_type': {'$eq': 'PM'}}
        """
        conditions: list[dict[str, Any]] = []

        if self.agent_type:
            conditions.append({"agent_type": {"$eq": self.agent_type}})

        if self.artifact_type:
            conditions.append({"artifact_type": {"$eq": self.artifact_type}})

        if self.decision_type:
            conditions.append({"decision_type": {"$eq": self.decision_type.value}})

        if self.time_range_start:
            # Store timestamp as ISO 8601 string for ChromaDB
            conditions.append({"timestamp": {"$gte": self.time_range_start.isoformat()}})

        if self.time_range_end:
            conditions.append({"timestamp": {"$lte": self.time_range_end.isoformat()}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}
