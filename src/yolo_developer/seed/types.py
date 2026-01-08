"""Seed parsing data types for natural language document processing (Story 4.1).

This module provides data models for parsing natural language seed documents
into structured components:

- SeedSource: Enum for input source types (file, text, URL)
- ComponentType: Enum for component categories (goal, feature, constraint, etc.)
- ConstraintCategory: Enum for constraint types (technical, business, etc.)
- SeedComponent: Generic parsed component with type and confidence
- SeedGoal: High-level project objective
- SeedFeature: Discrete functional capability
- SeedConstraint: Technical, business, or other limitation
- SeedParseResult: Complete parse result with all extracted components

Example:
    >>> from yolo_developer.seed.types import (
    ...     SeedSource,
    ...     SeedGoal,
    ...     SeedFeature,
    ...     SeedConstraint,
    ...     SeedParseResult,
    ...     ConstraintCategory,
    ... )
    >>>
    >>> # Create a goal
    >>> goal = SeedGoal(
    ...     title="Build E-commerce Platform",
    ...     description="Create an online store for selling products",
    ...     priority=1,
    ...     rationale="Expand business reach to online customers",
    ... )
    >>>
    >>> # Create a feature
    >>> feature = SeedFeature(
    ...     name="Shopping Cart",
    ...     description="Users can add products and manage quantities",
    ...     user_value="Convenient pre-checkout collection",
    ... )
    >>>
    >>> # Create a constraint
    >>> constraint = SeedConstraint(
    ...     category=ConstraintCategory.TECHNICAL,
    ...     description="Must use Python 3.10+",
    ...     impact="Limits deployment options",
    ... )
    >>>
    >>> # Create a parse result
    >>> result = SeedParseResult(
    ...     goals=(goal,),
    ...     features=(feature,),
    ...     constraints=(constraint,),
    ...     raw_content="Original seed document content",
    ...     source=SeedSource.TEXT,
    ... )
    >>> print(result.goal_count)
    1

Security Note:
    All dataclasses are frozen (immutable) to prevent accidental mutation.
    This ensures parse results maintain data integrity throughout processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yolo_developer.seed.ambiguity import Ambiguity
    from yolo_developer.seed.sop import SOPValidationResult


class SeedSource(Enum):
    """Source type for seed input.

    Indicates where the seed content originated from, which may
    affect how it is preprocessed and validated.

    Values:
        FILE: Content read from a file on disk
        TEXT: Content provided directly as text
        URL: Content fetched from a URL
    """

    FILE = "file"
    TEXT = "text"
    URL = "url"


class ComponentType(Enum):
    """Type of extracted component from seed document.

    Categorizes the semantic type of each component extracted
    during seed parsing.

    Values:
        GOAL: High-level project objective
        FEATURE: Discrete functional capability
        CONSTRAINT: Technical, business, or other limitation
        CONTEXT: Background information or context
        UNKNOWN: Component type could not be determined
    """

    GOAL = "goal"
    FEATURE = "feature"
    CONSTRAINT = "constraint"
    CONTEXT = "context"
    UNKNOWN = "unknown"


class ConstraintCategory(Enum):
    """Category of constraint extracted from seed document.

    Classifies constraints by their nature to help with
    architectural decision-making.

    Values:
        TECHNICAL: Technology, language, framework constraints
        BUSINESS: Business rules, policies, requirements
        TIMELINE: Deadlines, milestones, schedule constraints
        RESOURCE: Team size, budget, infrastructure limitations
        COMPLIANCE: Legal, regulatory, security requirements
    """

    TECHNICAL = "technical"
    BUSINESS = "business"
    TIMELINE = "timeline"
    RESOURCE = "resource"
    COMPLIANCE = "compliance"


@dataclass(frozen=True)
class SeedComponent:
    """Generic component extracted from seed document.

    Represents a single piece of content extracted from a seed
    document before it is fully classified as a goal, feature,
    or constraint.

    Attributes:
        component_type: The semantic type of this component
        content: The extracted text content
        confidence: Confidence score for extraction (0.0-1.0)
        source_line: Line number in source document (optional)
        metadata: Additional key-value pairs as tuple of tuples

    Example:
        >>> component = SeedComponent(
        ...     component_type=ComponentType.GOAL,
        ...     content="Build a REST API",
        ...     confidence=0.95,
        ...     source_line=10,
        ... )
    """

    component_type: ComponentType
    content: str
    confidence: float
    source_line: int | None = None
    metadata: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, metadata as dict.
        """
        return {
            "component_type": self.component_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "source_line": self.source_line,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SeedGoal:
    """High-level project objective extracted from seed document.

    Represents a goal that captures what the project should achieve
    and why it matters. Goals inform feature prioritization and
    architectural decisions.

    Attributes:
        title: Short title for the goal
        description: Detailed description of what needs to be achieved
        priority: Priority score (1-5, 1 being highest)
        rationale: Why this goal matters (optional)

    Example:
        >>> goal = SeedGoal(
        ...     title="Build E-commerce Platform",
        ...     description="Create an online store for product sales",
        ...     priority=1,
        ...     rationale="Expand market reach to online customers",
        ... )
    """

    title: str
    description: str
    priority: int
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields.
        """
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SeedFeature:
    """Discrete functional capability extracted from seed document.

    Represents a feature that the system should implement.
    Features are self-contained, actionable capabilities that
    deliver user value.

    Attributes:
        name: Short name for the feature
        description: Detailed description of the capability
        user_value: The value this feature provides to users (optional)
        related_goals: Tuple of goal titles this feature supports

    Example:
        >>> feature = SeedFeature(
        ...     name="User Authentication",
        ...     description="Allow users to register and log in",
        ...     user_value="Secure access to personalized content",
        ...     related_goals=("Build E-commerce Platform",),
        ... )
    """

    name: str
    description: str
    user_value: str | None = None
    related_goals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, related_goals as list.
        """
        return {
            "name": self.name,
            "description": self.description,
            "user_value": self.user_value,
            "related_goals": list(self.related_goals),
        }


@dataclass(frozen=True)
class SeedConstraint:
    """Constraint extracted from seed document.

    Represents a limitation or requirement that affects how
    the system should be built. Constraints inform architectural
    decisions and scope boundaries.

    Attributes:
        category: The type of constraint
        description: What the constraint is
        impact: How this constraint affects the project (optional)
        related_items: Tuple of goal/feature names affected by this constraint

    Example:
        >>> constraint = SeedConstraint(
        ...     category=ConstraintCategory.TECHNICAL,
        ...     description="Must use Python 3.10 or higher",
        ...     impact="Limits deployment to modern environments",
        ...     related_items=("API Service",),
        ... )
    """

    category: ConstraintCategory
    description: str
    impact: str | None = None
    related_items: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, category as string value.
        """
        return {
            "category": self.category.value,
            "description": self.description,
            "impact": self.impact,
            "related_items": list(self.related_items),
        }


@dataclass(frozen=True)
class SeedParseResult:
    """Complete result from parsing a seed document.

    Contains all extracted goals, features, and constraints,
    along with the original content and metadata about the parse.
    Optionally includes ambiguity detection and SOP validation results.

    Attributes:
        goals: Tuple of extracted goals
        features: Tuple of extracted features
        constraints: Tuple of extracted constraints
        raw_content: Original seed document content
        source: Where the seed content came from
        metadata: Additional key-value pairs as tuple of tuples
        ambiguities: Tuple of detected ambiguities (optional)
        ambiguity_confidence: Confidence score reflecting ambiguity impact (0.0-1.0)
        sop_validation: SOP constraint validation result (optional)

    Example:
        >>> result = SeedParseResult(
        ...     goals=(SeedGoal(title="Goal", description="Desc", priority=1),),
        ...     features=(SeedFeature(name="Feature", description="Desc"),),
        ...     constraints=(),
        ...     raw_content="Original document",
        ...     source=SeedSource.TEXT,
        ... )
        >>> print(result.goal_count)
        1
    """

    goals: tuple[SeedGoal, ...]
    features: tuple[SeedFeature, ...]
    constraints: tuple[SeedConstraint, ...]
    raw_content: str
    source: SeedSource
    metadata: tuple[tuple[str, Any], ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    ambiguity_confidence: float = 1.0
    sop_validation: SOPValidationResult | None = None

    @property
    def goal_count(self) -> int:
        """Return the number of goals extracted."""
        return len(self.goals)

    @property
    def feature_count(self) -> int:
        """Return the number of features extracted."""
        return len(self.features)

    @property
    def constraint_count(self) -> int:
        """Return the number of constraints extracted."""
        return len(self.constraints)

    @property
    def has_ambiguities(self) -> bool:
        """Return True if any ambiguities were detected."""
        return len(self.ambiguities) > 0

    @property
    def ambiguity_count(self) -> int:
        """Return the number of ambiguities detected."""
        return len(self.ambiguities)

    @property
    def has_sop_conflicts(self) -> bool:
        """Return True if any SOP conflicts were detected."""
        return (
            self.sop_validation is not None
            and self.sop_validation.has_conflicts
        )

    @property
    def sop_passed(self) -> bool:
        """Return True if SOP validation passed (no HARD conflicts)."""
        return self.sop_validation is None or self.sop_validation.passed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, nested objects serialized.
        """
        return {
            "goals": [goal.to_dict() for goal in self.goals],
            "features": [feature.to_dict() for feature in self.features],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "raw_content": self.raw_content,
            "source": self.source.value,
            "metadata": dict(self.metadata),
            "ambiguities": [amb.to_dict() for amb in self.ambiguities],
            "ambiguity_confidence": self.ambiguity_confidence,
            "sop_validation": (
                self.sop_validation.to_dict()
                if self.sop_validation is not None
                else None
            ),
        }
