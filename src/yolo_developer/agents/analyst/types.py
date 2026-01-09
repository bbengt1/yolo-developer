"""Type definitions for Analyst agent (Story 5.1, 5.2, 5.3, 5.4).

This module provides the data types used by the Analyst agent:

- CrystallizedRequirement: A refined, categorized requirement with testability
- AnalystOutput: Complete output from analyst processing
- GapType: Enum for types of identified gaps
- Severity: Enum for gap severity levels
- IdentifiedGap: A gap or missing requirement identified during analysis
- RequirementCategory: Primary requirement category (functional, non_functional, constraint)
- FunctionalSubCategory: Sub-categories for functional requirements
- NonFunctionalSubCategory: Sub-categories for non-functional requirements (ISO 25010)
- ConstraintSubCategory: Sub-categories for constraints

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.analyst.types import (
    ...     CrystallizedRequirement,
    ...     AnalystOutput,
    ...     IdentifiedGap,
    ...     GapType,
    ...     Severity,
    ... )
    >>>
    >>> req = CrystallizedRequirement(
    ...     id="req-001",
    ...     original_text="System should be fast",
    ...     refined_text="Response time < 200ms at 95th percentile",
    ...     category="non-functional",
    ...     testable=True,
    ... )
    >>> req.to_dict()
    {'id': 'req-001', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RequirementCategory(str, Enum):
    """Primary requirement category per IEEE 830/ISO 29148 standards.

    Used to classify requirements into fundamental categories for
    appropriate handling by downstream agents.

    Values:
        FUNCTIONAL: What the system should DO - features, behaviors, capabilities.
        NON_FUNCTIONAL: How well the system should DO it - quality attributes.
        CONSTRAINT: Limitations on HOW it can be built - restrictions, mandates.
    """

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"


class FunctionalSubCategory(str, Enum):
    """Sub-categories for functional requirements.

    Used to further classify functional requirements for routing
    and specialized handling.

    Values:
        USER_MANAGEMENT: Authentication, profiles, roles, permissions.
        DATA_OPERATIONS: CRUD, validation, storage, retrieval.
        INTEGRATION: APIs, external services, webhooks.
        REPORTING: Reports, analytics, exports, dashboards.
        WORKFLOW: Business processes, state machines, approvals.
        COMMUNICATION: Notifications, messaging, alerts.
    """

    USER_MANAGEMENT = "user_management"
    DATA_OPERATIONS = "data_operations"
    INTEGRATION = "integration"
    REPORTING = "reporting"
    WORKFLOW = "workflow"
    COMMUNICATION = "communication"


class NonFunctionalSubCategory(str, Enum):
    """Sub-categories for non-functional requirements per ISO 25010.

    Used to classify quality attributes according to ISO 25010
    software quality model.

    Values:
        PERFORMANCE: Response time, throughput, resource efficiency.
        SECURITY: Authentication, encryption, audit trails, access control.
        USABILITY: User experience, learnability, accessibility.
        RELIABILITY: Availability, fault tolerance, recoverability.
        SCALABILITY: Load handling, growth capacity, elasticity.
        MAINTAINABILITY: Code quality, modularity, testability.
        ACCESSIBILITY: WCAG compliance, inclusive design, assistive tech.
    """

    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    ACCESSIBILITY = "accessibility"


class ConstraintSubCategory(str, Enum):
    """Sub-categories for constraint requirements.

    Used to classify constraints by their nature and origin.

    Values:
        TECHNICAL: Tech stack, platforms, protocols, frameworks.
        BUSINESS: Budget, stakeholder requirements, market needs.
        REGULATORY: Compliance, legal requirements, standards, certifications.
        RESOURCE: Team capacity, skills, tools availability.
        TIMELINE: Deadlines, milestones, release schedules.
    """

    TECHNICAL = "technical"
    BUSINESS = "business"
    REGULATORY = "regulatory"
    RESOURCE = "resource"
    TIMELINE = "timeline"


@dataclass(frozen=True)
class CategorizationResult:
    """Result of categorizing a requirement (Story 5.4).

    Immutable dataclass containing the categorization outcome for a
    requirement, including primary category, sub-category, confidence
    score, and rationale for audit trail.

    Note:
        This type is exported for external API consumers who may want to
        represent categorization results separately from requirements.
        Internally, categorization is embedded directly into
        CrystallizedRequirement fields (sub_category, category_confidence,
        category_rationale) for convenience.

    Attributes:
        category: Primary category (RequirementCategory enum).
        sub_category: Optional sub-category string (e.g., "user_management").
            None if sub-category could not be determined.
        confidence: Confidence score 0.0-1.0 for the categorization.
            1.0 = high confidence, 0.0 = low confidence.
        rationale: Explanation of why this category was assigned,
            including keywords or patterns that drove the decision.

    Example:
        >>> result = CategorizationResult(
        ...     category=RequirementCategory.FUNCTIONAL,
        ...     sub_category="user_management",
        ...     confidence=0.95,
        ...     rationale="Contains 'login', 'user' - clear functional",
        ... )
        >>> result.to_dict()
        {'category': 'functional', 'sub_category': 'user_management', ...}
    """

    category: RequirementCategory
    sub_category: str | None
    confidence: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, category enum as string value.

        Example:
            >>> result = CategorizationResult(
            ...     category=RequirementCategory.NON_FUNCTIONAL,
            ...     sub_category="performance",
            ...     confidence=0.9,
            ...     rationale="Response time mentioned",
            ... )
            >>> result.to_dict()["category"]
            'non_functional'
        """
        return {
            "category": self.category.value,
            "sub_category": self.sub_category,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


class GapType(str, Enum):
    """Type of identified gap in requirements.

    Used to categorize gaps found during requirement analysis for
    appropriate handling and prioritization.

    Values:
        EDGE_CASE: Missing edge case handling (boundary conditions, errors).
        IMPLIED_REQUIREMENT: Unstated but logically implied requirement.
        PATTERN_SUGGESTION: Feature suggested by domain patterns.
    """

    EDGE_CASE = "edge_case"
    IMPLIED_REQUIREMENT = "implied_requirement"
    PATTERN_SUGGESTION = "pattern_suggestion"


class Severity(str, Enum):
    """Severity level for identified gaps.

    Based on impact to implementation success and system quality.

    Values:
        CRITICAL: Security, data integrity, or core functionality at risk.
        HIGH: Major feature gaps or integration issues.
        MEDIUM: User experience gaps or minor edge cases.
        LOW: Nice-to-have features or optimization opportunities.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class IdentifiedGap:
    """A gap or missing requirement identified during analysis.

    Immutable dataclass representing a gap found in requirements that
    needs attention. Each gap is categorized, assigned severity, and
    traced to source requirements.

    Attributes:
        id: Unique identifier for this gap (e.g., "gap-001").
        description: Human-readable description of the gap.
        gap_type: Category of gap (edge_case, implied_requirement, pattern_suggestion).
        severity: Impact severity (critical, high, medium, low).
        source_requirements: Tuple of requirement IDs this gap relates to.
        rationale: Explanation of why this gap was identified.

    Example:
        >>> gap = IdentifiedGap(
        ...     id="gap-001",
        ...     description="Missing error handling for invalid input",
        ...     gap_type=GapType.EDGE_CASE,
        ...     severity=Severity.HIGH,
        ...     source_requirements=("req-001", "req-002"),
        ...     rationale="Input validation requires error response",
        ... )
        >>> gap.severity
        <Severity.HIGH: 'high'>
    """

    id: str
    description: str
    gap_type: GapType
    severity: Severity
    source_requirements: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all gap fields, enums as string values.

        Example:
            >>> gap = IdentifiedGap(
            ...     id="gap-001",
            ...     description="Missing logout",
            ...     gap_type=GapType.IMPLIED_REQUIREMENT,
            ...     severity=Severity.MEDIUM,
            ...     source_requirements=("req-001",),
            ...     rationale="Login implies logout needed",
            ... )
            >>> gap.to_dict()["severity"]
            'medium'
        """
        return {
            "id": self.id,
            "description": self.description,
            "gap_type": self.gap_type.value,
            "severity": self.severity.value,
            "source_requirements": list(self.source_requirements),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class CrystallizedRequirement:
    """A refined, categorized requirement extracted from seed content.

    Immutable dataclass representing a single requirement that has been
    processed by the Analyst agent. Each requirement is categorized,
    assessed for testability, and enhanced with scope and implementation hints.

    Attributes:
        id: Unique identifier for this requirement (e.g., "req-001").
        original_text: The original text from the seed document.
        refined_text: The clarified, refined requirement text.
        category: Category of the requirement:
            - "functional": Feature or behavior
            - "non-functional": Quality attribute (performance, security, etc.)
            - "constraint": Technical or business constraint
        testable: Whether this requirement can be objectively tested.
        scope_notes: Optional clarification of scope boundaries (in-scope vs
            out-of-scope items, edge cases). None if no scope clarification needed.
        implementation_hints: Tuple of implementation suggestions referencing
            project patterns, libraries, or architectural conventions.
        confidence: Confidence score (0.0-1.0) for the crystallization quality.
            1.0 = high confidence the refinement is accurate, 0.0 = low confidence.
        sub_category: Optional sub-category string (e.g., "user_management",
            "performance"). None if sub-category could not be determined.
            (Added in Story 5.4)
        category_confidence: Confidence score (0.0-1.0) for the category assignment.
            Distinct from `confidence` which is for crystallization quality.
            (Added in Story 5.4)
        category_rationale: Explanation of why this category/sub-category was
            assigned, including keywords or patterns that drove the decision.
            Used for audit trail. (Added in Story 5.4)

    Example:
        >>> req = CrystallizedRequirement(
        ...     id="req-001",
        ...     original_text="The system should be fast",
        ...     refined_text="API response time < 200ms for 95th percentile",
        ...     category="non-functional",
        ...     testable=True,
        ...     scope_notes="Applies to GET endpoints; POST excluded",
        ...     implementation_hints=("Use async handlers", "Add caching"),
        ...     confidence=0.9,
        ... )
        >>> req.category
        'non-functional'
    """

    id: str
    original_text: str
    refined_text: str
    category: str
    testable: bool
    scope_notes: str | None = None
    implementation_hints: tuple[str, ...] = ()
    confidence: float = 1.0
    # Story 5.4 categorization fields
    sub_category: str | None = None
    category_confidence: float = 1.0
    category_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all requirement fields including enhanced fields
            and categorization fields (Story 5.4).

        Example:
            >>> req = CrystallizedRequirement(
            ...     id="req-001",
            ...     original_text="orig",
            ...     refined_text="ref",
            ...     category="functional",
            ...     testable=True,
            ... )
            >>> req.to_dict()
            {'id': 'req-001', 'original_text': 'orig', ...}
        """
        return {
            "id": self.id,
            "original_text": self.original_text,
            "refined_text": self.refined_text,
            "category": self.category,
            "testable": self.testable,
            "scope_notes": self.scope_notes,
            "implementation_hints": list(self.implementation_hints),
            "confidence": self.confidence,
            # Story 5.4 categorization fields
            "sub_category": self.sub_category,
            "category_confidence": self.category_confidence,
            "category_rationale": self.category_rationale,
        }


@dataclass(frozen=True)
class AnalystOutput:
    """Complete output from Analyst agent processing.

    Immutable dataclass containing all results from analyzing seed content:
    crystallized requirements, identified gaps, and contradictions.

    Attributes:
        requirements: Tuple of CrystallizedRequirement objects.
        identified_gaps: Tuple of strings describing missing information (legacy).
        contradictions: Tuple of strings describing conflicting requirements.
        structured_gaps: Tuple of IdentifiedGap objects with full gap details
            (Story 5.3). Defaults to empty for backward compatibility.

    Example:
        >>> req = CrystallizedRequirement(
        ...     id="req-001",
        ...     original_text="orig",
        ...     refined_text="ref",
        ...     category="functional",
        ...     testable=True,
        ... )
        >>> output = AnalystOutput(
        ...     requirements=(req,),
        ...     identified_gaps=("Missing auth details",),
        ...     contradictions=(),
        ... )
        >>> len(output.requirements)
        1
    """

    requirements: tuple[CrystallizedRequirement, ...]
    identified_gaps: tuple[str, ...]
    contradictions: tuple[str, ...]
    structured_gaps: tuple[IdentifiedGap, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Serializes nested CrystallizedRequirement and IdentifiedGap objects.

        Returns:
            Dictionary with all output fields including structured_gaps.

        Example:
            >>> output = AnalystOutput(
            ...     requirements=(),
            ...     identified_gaps=("gap1",),
            ...     contradictions=(),
            ... )
            >>> d = output.to_dict()
            >>> d["identified_gaps"]
            ['gap1']
        """
        return {
            "requirements": [r.to_dict() for r in self.requirements],
            "identified_gaps": list(self.identified_gaps),
            "contradictions": list(self.contradictions),
            "structured_gaps": [g.to_dict() for g in self.structured_gaps],
        }
