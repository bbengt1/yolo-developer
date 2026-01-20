"""SOP constraint validation for seed documents (Story 4.5).

This module provides data models and functions for validating seed documents
against established Standard Operating Procedure (SOP) constraints:

- SOPCategory: Enum for constraint categories (ARCHITECTURE, SECURITY, etc.)
- ConflictSeverity: Enum for conflict severity (HARD, SOFT)
- SOPConstraint: Dataclass representing a learned constraint
- SOPConflict: Dataclass representing a detected conflict
- SOPValidationResult: Complete validation result
- SOPStore: Protocol for constraint storage
- InMemorySOPStore: Simple in-memory implementation
- validate_against_sop: Main validation function

Example:
    >>> from yolo_developer.seed.sop import (
    ...     SOPConstraint,
    ...     SOPCategory,
    ...     ConflictSeverity,
    ...     InMemorySOPStore,
    ...     validate_against_sop,
    ... )
    >>>
    >>> # Create a store with constraints
    >>> store = InMemorySOPStore()
    >>> await store.add_constraint(SOPConstraint(
    ...     id="arch-001",
    ...     rule_text="All API endpoints must use REST conventions",
    ...     category=SOPCategory.ARCHITECTURE,
    ...     source="architecture.md",
    ...     severity=ConflictSeverity.HARD,
    ... ))
    >>>
    >>> # Validate seed against constraints
    >>> result = await validate_against_sop("Build a GraphQL API", store)
    >>> if not result.passed:
    ...     for conflict in result.conflicts:
    ...         print(f"{conflict.severity.value}: {conflict.description}")
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


class SOPCategory(str, Enum):
    """Category of SOP constraint.

    Categorizes constraints by their domain to help with
    severity assignment and conflict resolution.

    Values:
        ARCHITECTURE: System design patterns, module boundaries
        SECURITY: Authentication, authorization, data protection
        PERFORMANCE: Response times, resource limits, caching
        NAMING: Conventions for files, functions, variables
        TESTING: Coverage requirements, test patterns
        DEPENDENCY: Library choices, version constraints
    """

    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"
    NAMING = "naming"
    TESTING = "testing"
    DEPENDENCY = "dependency"


class ConflictSeverity(str, Enum):
    """Severity level of detected conflict.

    Indicates whether the conflict blocks processing or is advisory.

    Values:
        HARD: Blocks processing, must be resolved before proceeding
        SOFT: Advisory, can be overridden with acknowledgment
    """

    HARD = "hard"
    SOFT = "soft"


# Category to default severity mapping
CATEGORY_SEVERITY_MAP: dict[SOPCategory, ConflictSeverity] = {
    SOPCategory.ARCHITECTURE: ConflictSeverity.HARD,
    SOPCategory.SECURITY: ConflictSeverity.HARD,
    SOPCategory.PERFORMANCE: ConflictSeverity.SOFT,
    SOPCategory.NAMING: ConflictSeverity.SOFT,
    SOPCategory.TESTING: ConflictSeverity.SOFT,
    SOPCategory.DEPENDENCY: ConflictSeverity.HARD,
}


@dataclass(frozen=True)
class SOPConstraint:
    """A learned SOP constraint/rule.

    Represents an established pattern or rule that seed documents
    should be validated against.

    Attributes:
        id: Unique identifier for the constraint
        rule_text: The constraint rule in natural language
        category: The category of constraint
        source: Where this constraint was learned from (e.g., "architecture.md")
        severity: Default severity when violated
        created_at: When this constraint was added (ISO format)

    Example:
        >>> constraint = SOPConstraint(
        ...     id="sec-001",
        ...     rule_text="All API endpoints must require authentication",
        ...     category=SOPCategory.SECURITY,
        ...     source="security-policy.md",
        ...     severity=ConflictSeverity.HARD,
        ... )
    """

    id: str
    rule_text: str
    category: SOPCategory
    source: str
    severity: ConflictSeverity
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, enums as string values.
        """
        return {
            "id": self.id,
            "rule_text": self.rule_text,
            "category": self.category.value,
            "source": self.source,
            "severity": self.severity.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SOPConstraint:
        """Create from dictionary.

        Args:
            data: Dictionary with constraint data

        Returns:
            SOPConstraint instance
        """
        return cls(
            id=data["id"],
            rule_text=data["rule_text"],
            category=SOPCategory(data["category"]),
            source=data["source"],
            severity=ConflictSeverity(data["severity"]),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass(frozen=True)
class SOPConflict:
    """A detected conflict between seed and SOP constraint.

    Represents a specific contradiction between the seed document
    and an established constraint.

    Attributes:
        constraint: The violated constraint
        seed_text: The exact conflicting text from the seed
        severity: Severity of this specific conflict
        description: Why this is a conflict
        resolution_options: Suggested ways to resolve the conflict

    Example:
        >>> conflict = SOPConflict(
        ...     constraint=constraint,
        ...     seed_text="Use GraphQL for all endpoints",
        ...     severity=ConflictSeverity.HARD,
        ...     description="Conflicts with REST API requirement",
        ...     resolution_options=("Use REST instead", "Request exception"),
        ... )
    """

    constraint: SOPConstraint
    seed_text: str
    severity: ConflictSeverity
    description: str
    resolution_options: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields including nested constraint.
        """
        return {
            "constraint": self.constraint.to_dict(),
            "seed_text": self.seed_text,
            "severity": self.severity.value,
            "description": self.description,
            "resolution_options": list(self.resolution_options),
        }


@dataclass
class SOPValidationResult:
    """Result of SOP validation against seed content.

    Contains all detected conflicts and validation status.

    Attributes:
        conflicts: List of detected conflicts
        passed: Whether validation passed (no HARD conflicts)
        override_applied: Whether user overrode SOFT conflicts

    Example:
        >>> result = SOPValidationResult(
        ...     conflicts=[conflict1, conflict2],
        ...     passed=False,
        ...     override_applied=False,
        ... )
        >>> if not result.passed:
        ...     print(f"Found {result.hard_conflict_count} blocking conflicts")
    """

    conflicts: list[SOPConflict] = field(default_factory=list)
    passed: bool = True
    override_applied: bool = False

    @property
    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.conflicts) > 0

    @property
    def hard_conflict_count(self) -> int:
        """Count of HARD severity conflicts."""
        return sum(1 for c in self.conflicts if c.severity == ConflictSeverity.HARD)

    @property
    def soft_conflict_count(self) -> int:
        """Count of SOFT severity conflicts."""
        return sum(1 for c in self.conflicts if c.severity == ConflictSeverity.SOFT)

    @property
    def hard_conflicts(self) -> list[SOPConflict]:
        """Get only HARD conflicts."""
        return [c for c in self.conflicts if c.severity == ConflictSeverity.HARD]

    @property
    def soft_conflicts(self) -> list[SOPConflict]:
        """Get only SOFT conflicts."""
        return [c for c in self.conflicts if c.severity == ConflictSeverity.SOFT]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields and conflict details.
        """
        return {
            "conflicts": [c.to_dict() for c in self.conflicts],
            "passed": self.passed,
            "override_applied": self.override_applied,
            "hard_conflict_count": self.hard_conflict_count,
            "soft_conflict_count": self.soft_conflict_count,
        }


@runtime_checkable
class SOPStore(Protocol):
    """Protocol for SOP constraint storage.

    Defines the interface for storing and retrieving SOP constraints.
    Implementations can use in-memory storage, databases, or vector stores.
    """

    async def add_constraint(self, constraint: SOPConstraint) -> None:
        """Add a constraint to the store.

        Args:
            constraint: The constraint to add
        """
        ...

    async def get_constraints(
        self,
        category: SOPCategory | None = None,
    ) -> list[SOPConstraint]:
        """Get constraints, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of matching constraints
        """
        ...

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SOPConstraint]:
        """Search for semantically similar constraints.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of similar constraints
        """
        ...

    async def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint by ID.

        Args:
            constraint_id: ID of constraint to remove

        Returns:
            True if removed, False if not found
        """
        ...

    async def clear(self) -> None:
        """Remove all constraints from the store."""
        ...


class InMemorySOPStore:
    """Simple in-memory SOP store for testing and simple use cases.

    Stores constraints in a dictionary. Does not persist across restarts.
    For production, use ChromaDBSOPStore.

    Example:
        >>> store = InMemorySOPStore()
        >>> await store.add_constraint(constraint)
        >>> constraints = await store.get_constraints()
    """

    def __init__(self) -> None:
        """Initialize empty store."""
        self._constraints: dict[str, SOPConstraint] = {}

    async def add_constraint(self, constraint: SOPConstraint) -> None:
        """Add a constraint to the store.

        Args:
            constraint: The constraint to add
        """
        self._constraints[constraint.id] = constraint
        logger.debug("constraint_added", constraint_id=constraint.id)

    async def get_constraints(
        self,
        category: SOPCategory | None = None,
    ) -> list[SOPConstraint]:
        """Get constraints, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of matching constraints
        """
        constraints = list(self._constraints.values())
        if category is not None:
            constraints = [c for c in constraints if c.category == category]
        return constraints

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SOPConstraint]:
        """Search for similar constraints using simple text matching.

        Note: For semantic search, use ChromaDBSOPStore instead.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of constraints containing query terms
        """
        query_lower = query.lower()
        results: list[SOPConstraint] = []
        for constraint in self._constraints.values():
            if query_lower in constraint.rule_text.lower():
                results.append(constraint)
                if len(results) >= limit:
                    break
        return results

    async def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint by ID.

        Args:
            constraint_id: ID of constraint to remove

        Returns:
            True if removed, False if not found
        """
        if constraint_id in self._constraints:
            del self._constraints[constraint_id]
            logger.debug("constraint_removed", constraint_id=constraint_id)
            return True
        return False

    async def clear(self) -> None:
        """Remove all constraints from the store."""
        self._constraints.clear()
        logger.debug("store_cleared")


# LLM prompt for SOP validation
SOP_VALIDATION_PROMPT = """You are a technical validator checking requirements against established constraints.

Your task is to identify any conflicts between the seed requirements and the established SOP constraints.

CONSTRAINTS DATABASE:
{constraints_json}

SEED REQUIREMENTS:
{seed_content}

For each potential conflict, analyze:
1. Which constraint (by ID) is potentially violated
2. What specific text in the seed conflicts with it
3. Is this a HARD conflict (architectural/security/dependency - must be resolved) or SOFT (preference/convention - can be overridden)
4. Why this is a conflict
5. What resolution options exist

IMPORTANT:
- Only report actual conflicts, not potential future issues
- Be precise about what text in the seed conflicts
- Consider the constraint's category when assigning severity
- Provide actionable resolution options

Return your analysis as JSON:
{{
  "conflicts": [
    {{
      "constraint_id": "string - the ID from the constraint",
      "seed_text": "exact conflicting text from the seed",
      "severity": "HARD" or "SOFT",
      "description": "clear explanation of why this conflicts",
      "resolution_options": ["option1", "option2"]
    }}
  ]
}}

If no conflicts are found, return: {{"conflicts": []}}
"""


def _parse_json_response(content: str | None) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw response content from LLM

    Returns:
        Parsed JSON as dictionary
    """
    if not content:
        return {"conflicts": []}

    # Strip markdown code blocks if present
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed: dict[str, Any] = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("json_parse_error", error=str(e), content=content[:100])
        return {"conflicts": []}


@retry(
    retry=retry_if_exception_type((litellm.exceptions.RateLimitError,)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
)
async def _validate_with_llm(
    seed_content: str,
    constraints: list[SOPConstraint],
    model: str,
) -> list[dict[str, Any]]:
    """Validate seed against constraints using LLM.

    Args:
        seed_content: The seed document content
        constraints: List of constraints to check against
        model: LLM model to use

    Returns:
        List of raw conflict dictionaries from LLM
    """
    if not constraints:
        return []

    constraints_json = json.dumps(
        [c.to_dict() for c in constraints],
        indent=2,
    )

    prompt = SOP_VALIDATION_PROMPT.format(
        constraints_json=constraints_json,
        seed_content=seed_content,
    )

    logger.debug(
        "validating_against_sop",
        constraint_count=len(constraints),
        seed_length=len(seed_content),
    )

    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze the seed against these constraints."},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    result = _parse_json_response(response.choices[0].message.content)
    conflicts: list[dict[str, Any]] = result.get("conflicts", [])
    return conflicts


def _build_conflict(
    raw_conflict: dict[str, Any],
    constraints_by_id: dict[str, SOPConstraint],
) -> SOPConflict | None:
    """Build SOPConflict from raw LLM response data.

    Args:
        raw_conflict: Raw conflict dictionary from LLM
        constraints_by_id: Mapping of constraint IDs to constraints

    Returns:
        SOPConflict or None if constraint not found
    """
    constraint_id = raw_conflict.get("constraint_id", "")
    constraint = constraints_by_id.get(constraint_id)

    if constraint is None:
        logger.warning(
            "unknown_constraint_id",
            constraint_id=constraint_id,
        )
        return None

    # Parse severity
    raw_severity = str(raw_conflict.get("severity", "SOFT")).upper()
    try:
        severity = ConflictSeverity(raw_severity.lower())
    except ValueError:
        # Default based on category
        severity = CATEGORY_SEVERITY_MAP.get(constraint.category, ConflictSeverity.SOFT)

    # Build resolution options tuple
    resolution_options = raw_conflict.get("resolution_options", [])
    if isinstance(resolution_options, list):
        resolution_options = tuple(str(opt) for opt in resolution_options)
    else:
        resolution_options = ()

    return SOPConflict(
        constraint=constraint,
        seed_text=str(raw_conflict.get("seed_text", "")),
        severity=severity,
        description=str(raw_conflict.get("description", "")),
        resolution_options=resolution_options,
    )


async def validate_against_sop(
    seed_content: str,
    sop_store: SOPStore,
    model: str = "gpt-4o-mini",
) -> SOPValidationResult:
    """Validate seed content against SOP constraints.

    Uses LLM to analyze seed document against stored constraints
    and detect any conflicts.

    Args:
        seed_content: The seed document content to validate
        sop_store: Store containing SOP constraints
        model: LLM model to use for analysis (default: gpt-4o-mini)

    Returns:
        SOPValidationResult with detected conflicts and status

    Example:
        >>> store = InMemorySOPStore()
        >>> await store.add_constraint(constraint)
        >>> result = await validate_against_sop("Build a thing", store)
        >>> if not result.passed:
        ...     print(f"Found {len(result.conflicts)} conflicts")
    """
    logger.info("starting_sop_validation", seed_length=len(seed_content))

    # Get all constraints
    constraints = await sop_store.get_constraints()

    if not constraints:
        logger.info("no_constraints_to_validate")
        return SOPValidationResult(conflicts=[], passed=True)

    # Create lookup map
    constraints_by_id = {c.id: c for c in constraints}

    # Validate with LLM
    raw_conflicts = await _validate_with_llm(seed_content, constraints, model)

    # Build conflict objects
    conflicts: list[SOPConflict] = []
    for raw_conflict in raw_conflicts:
        conflict = _build_conflict(raw_conflict, constraints_by_id)
        if conflict is not None:
            conflicts.append(conflict)

    # Determine if passed (no HARD conflicts)
    has_hard_conflicts = any(c.severity == ConflictSeverity.HARD for c in conflicts)

    result = SOPValidationResult(
        conflicts=conflicts,
        passed=not has_hard_conflicts,
        override_applied=False,
    )

    logger.info(
        "sop_validation_complete",
        conflict_count=len(conflicts),
        hard_count=result.hard_conflict_count,
        soft_count=result.soft_conflict_count,
        passed=result.passed,
    )

    return result


def generate_constraint_id(category: SOPCategory) -> str:
    """Generate a unique constraint ID.

    Args:
        category: Constraint category for prefix

    Returns:
        Unique ID like "arch-abc123"
    """
    prefix = category.value[:4]
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}-{suffix}"
