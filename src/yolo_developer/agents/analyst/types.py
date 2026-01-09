"""Type definitions for Analyst agent (Story 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7).

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
- ImplementabilityStatus: Status of implementability validation (Story 5.5)
- ComplexityLevel: Complexity level estimate for implementation (Story 5.5)
- DependencyType: Type of external dependency (Story 5.5)
- ExternalDependency: An external dependency required for implementation (Story 5.5)
- ImplementabilityResult: Result of implementability validation (Story 5.5)
- ContradictionType: Type of contradiction between requirements (Story 5.6)
- Contradiction: A contradiction identified between requirements (Story 5.6)
- EscalationReason: Reason for escalating to PM agent (Story 5.7)
- EscalationPriority: Priority level for escalation (Story 5.7)
- Escalation: An escalation to PM with full context (Story 5.7)

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

from dataclasses import dataclass, field
from datetime import datetime, timezone
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


# =============================================================================
# Story 5.5: Implementability Validation Types
# =============================================================================


class ImplementabilityStatus(str, Enum):
    """Status of implementability validation (Story 5.5).

    Used to indicate whether a requirement can be implemented as stated.

    Values:
        IMPLEMENTABLE: Requirement can be implemented with available resources.
        NEEDS_CLARIFICATION: Requirement is ambiguous or needs more details.
        NOT_IMPLEMENTABLE: Requirement is technically impossible or infeasible.
    """

    IMPLEMENTABLE = "implementable"
    NEEDS_CLARIFICATION = "needs_clarification"
    NOT_IMPLEMENTABLE = "not_implementable"


class ComplexityLevel(str, Enum):
    """Complexity level estimate for implementation (Story 5.5).

    Used to assess how difficult a requirement will be to implement.

    Values:
        LOW: Simple, well-understood patterns (CRUD, basic validation).
        MEDIUM: Multi-component with standard integrations.
        HIGH: Complex patterns requiring careful design (real-time, distributed).
        VERY_HIGH: Cutting-edge or highly specialized (ML/AI, consensus).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DependencyType(str, Enum):
    """Type of external dependency (Story 5.5).

    Used to categorize what kind of external resource a requirement needs.

    Values:
        API: External API integration (REST, GraphQL, webhooks).
        LIBRARY: Third-party library or SDK.
        SERVICE: Cloud service (AWS, GCP, Azure, SaaS).
        INFRASTRUCTURE: Database, cache, queue, storage.
        DATA_SOURCE: External data feed or data provider.
    """

    API = "api"
    LIBRARY = "library"
    SERVICE = "service"
    INFRASTRUCTURE = "infrastructure"
    DATA_SOURCE = "data_source"


@dataclass(frozen=True)
class ExternalDependency:
    """An external dependency required for implementation (Story 5.5).

    Immutable dataclass representing an external resource that must be
    available to implement a requirement. Each dependency is categorized
    by type with availability and criticality information.

    Attributes:
        name: Name of the dependency (e.g., "PostgreSQL", "Stripe API").
        dependency_type: Category of dependency (api, library, service, etc.).
        description: Human-readable description of what the dependency provides.
        availability_notes: Notes about availability or accessibility.
        criticality: How critical this dependency is ("required", "optional",
            "recommended").

    Example:
        >>> dep = ExternalDependency(
        ...     name="PostgreSQL",
        ...     dependency_type=DependencyType.INFRASTRUCTURE,
        ...     description="Relational database for persistent storage",
        ...     availability_notes="Widely available, managed services exist",
        ...     criticality="required",
        ... )
        >>> dep.to_dict()["dependency_type"]
        'infrastructure'
    """

    name: str
    dependency_type: DependencyType
    description: str
    availability_notes: str
    criticality: str  # "required", "optional", "recommended"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, dependency_type enum as string value.

        Example:
            >>> dep = ExternalDependency(
            ...     name="Redis",
            ...     dependency_type=DependencyType.INFRASTRUCTURE,
            ...     description="In-memory cache",
            ...     availability_notes="Easy to provision",
            ...     criticality="optional",
            ... )
            >>> dep.to_dict()["criticality"]
            'optional'
        """
        return {
            "name": self.name,
            "dependency_type": self.dependency_type.value,
            "description": self.description,
            "availability_notes": self.availability_notes,
            "criticality": self.criticality,
        }


@dataclass(frozen=True)
class ImplementabilityResult:
    """Result of implementability validation for a requirement (Story 5.5).

    Immutable dataclass containing the complete validation outcome for a
    requirement, including status, complexity assessment, dependencies,
    issues found, and remediation suggestions.

    Attributes:
        status: Overall implementability status.
        complexity: Estimated complexity level for implementation.
        dependencies: Tuple of external dependencies identified.
        issues: Tuple of issue descriptions found during validation.
        remediation_suggestions: Tuple of suggested fixes for issues.
        rationale: Explanation of the validation decision.

    Example:
        >>> result = ImplementabilityResult(
        ...     status=ImplementabilityStatus.NOT_IMPLEMENTABLE,
        ...     complexity=ComplexityLevel.HIGH,
        ...     dependencies=(),
        ...     issues=("100% uptime guarantee is impossible",),
        ...     remediation_suggestions=("Use 99.9% SLA instead",),
        ...     rationale="Absolute guarantees violate physics",
        ... )
        >>> result.to_dict()["status"]
        'not_implementable'
    """

    status: ImplementabilityStatus
    complexity: ComplexityLevel
    dependencies: tuple[ExternalDependency, ...]
    issues: tuple[str, ...]
    remediation_suggestions: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, enums as string values,
            dependencies as list of dicts.

        Example:
            >>> result = ImplementabilityResult(
            ...     status=ImplementabilityStatus.IMPLEMENTABLE,
            ...     complexity=ComplexityLevel.LOW,
            ...     dependencies=(),
            ...     issues=(),
            ...     remediation_suggestions=(),
            ...     rationale="Simple CRUD requirement",
            ... )
            >>> result.to_dict()["complexity"]
            'low'
        """
        return {
            "status": self.status.value,
            "complexity": self.complexity.value,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "issues": list(self.issues),
            "remediation_suggestions": list(self.remediation_suggestions),
            "rationale": self.rationale,
        }


# =============================================================================
# Story 5.6: Contradiction Flagging Types
# =============================================================================


class ContradictionType(str, Enum):
    """Type of contradiction between requirements (Story 5.6).

    Used to categorize how requirements conflict with each other.

    Values:
        DIRECT: Explicit opposing statements (must vs must not, encrypted vs plaintext).
        IMPLICIT_RESOURCE: Competing resource needs (memory-intensive vs low-memory).
        IMPLICIT_BEHAVIOR: Contradictory behaviors (stateless vs session-required).
        SEMANTIC: Logically incompatible meanings (real-time vs eventual consistency).

    Example:
        >>> ContradictionType.DIRECT.value
        'direct'
        >>> ContradictionType("semantic") == ContradictionType.SEMANTIC
        True
    """

    DIRECT = "direct"
    IMPLICIT_RESOURCE = "implicit_resource"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class Contradiction:
    """A contradiction identified between requirements (Story 5.6).

    Immutable dataclass representing a conflict between two or more
    requirements that needs resolution before implementation.

    Attributes:
        id: Unique identifier for this contradiction (e.g., "conflict-001").
        contradiction_type: Category of conflict (direct, implicit_resource, etc.).
        requirement_ids: Tuple of requirement IDs involved in the conflict.
        description: Human-readable description of the contradiction.
        explanation: Detailed explanation of why these requirements conflict.
        severity: Impact severity (critical, high, medium, low).
        resolution_suggestions: Tuple of suggested ways to resolve the conflict.

    Example:
        >>> conflict = Contradiction(
        ...     id="conflict-001",
        ...     contradiction_type=ContradictionType.SEMANTIC,
        ...     requirement_ids=("req-001", "req-002"),
        ...     description="Real-time vs eventual consistency conflict",
        ...     explanation="req-001 requires real-time updates, req-002 specifies eventual consistency",
        ...     severity=Severity.HIGH,
        ...     resolution_suggestions=("Clarify which operations need real-time", "Split into two features"),
        ... )
        >>> conflict.severity
        <Severity.HIGH: 'high'>
    """

    id: str
    contradiction_type: ContradictionType
    requirement_ids: tuple[str, ...]
    description: str
    explanation: str
    severity: Severity
    resolution_suggestions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, enums as string values,
            requirement_ids and resolution_suggestions as lists.

        Example:
            >>> conflict = Contradiction(
            ...     id="conflict-001",
            ...     contradiction_type=ContradictionType.DIRECT,
            ...     requirement_ids=("req-001", "req-002"),
            ...     description="Encryption conflict",
            ...     explanation="One requires encryption, other forbids it",
            ...     severity=Severity.CRITICAL,
            ...     resolution_suggestions=("Clarify security requirements",),
            ... )
            >>> conflict.to_dict()["severity"]
            'critical'
        """
        return {
            "id": self.id,
            "contradiction_type": self.contradiction_type.value,
            "requirement_ids": list(self.requirement_ids),
            "description": self.description,
            "explanation": self.explanation,
            "severity": self.severity.value,
            "resolution_suggestions": list(self.resolution_suggestions),
        }


# =============================================================================
# Story 5.7: Escalation to PM Types
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with timezone info.

    Using timezone-aware datetime for consistency and to avoid
    deprecation warnings from naive datetime comparisons.
    """
    return datetime.now(timezone.utc)


class EscalationReason(str, Enum):
    """Reason for escalating an issue to the PM agent (Story 5.7).

    Used to categorize why a requirement issue cannot be resolved by the
    Analyst and needs PM decision-making.

    Values:
        UNRESOLVABLE_AMBIGUITY: Requirement is too vague to crystallize
            even after analysis; needs stakeholder clarification.
        CONFLICTING_REQUIREMENTS: Critical contradiction between requirements
            that cannot be resolved without business prioritization.
        MISSING_DOMAIN_KNOWLEDGE: Requirement references domain concepts
            or business rules that the Analyst cannot infer.
        STAKEHOLDER_DECISION_NEEDED: Multiple valid interpretations exist;
            needs product decision on which direction to take.
        SCOPE_CLARIFICATION: Unclear whether feature/behavior is in scope
            for current iteration or project.

    Example:
        >>> EscalationReason.CONFLICTING_REQUIREMENTS.value
        'conflicting_requirements'
        >>> EscalationReason("missing_domain_knowledge")
        <EscalationReason.MISSING_DOMAIN_KNOWLEDGE: 'missing_domain_knowledge'>
    """

    UNRESOLVABLE_AMBIGUITY = "unresolvable_ambiguity"
    CONFLICTING_REQUIREMENTS = "conflicting_requirements"
    MISSING_DOMAIN_KNOWLEDGE = "missing_domain_knowledge"
    STAKEHOLDER_DECISION_NEEDED = "stakeholder_decision_needed"
    SCOPE_CLARIFICATION = "scope_clarification"


class EscalationPriority(str, Enum):
    """Priority level for escalation to PM (Story 5.7).

    Used to indicate how urgently the PM needs to address the escalation.

    Values:
        URGENT: Blocks all further progress; must be resolved immediately.
        HIGH: Blocks significant work; should be resolved soon.
        NORMAL: Does not block progress; can be resolved in normal workflow.

    Example:
        >>> EscalationPriority.URGENT.value
        'urgent'
        >>> EscalationPriority("high") == EscalationPriority.HIGH
        True
    """

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"


@dataclass(frozen=True)
class Escalation:
    """An escalation to PM with full context for decision-making (Story 5.7).

    Immutable dataclass representing an issue that the Analyst cannot resolve
    and needs to escalate to the PM agent. Contains all context needed for
    PM to make a decision without re-analyzing.

    Attributes:
        id: Unique identifier for this escalation (e.g., "esc-001").
        reason: Category of why escalation is needed.
        priority: How urgently PM needs to address this.
        summary: Brief one-line description of the issue.
        context: Full context for PM including analysis attempts.
        original_requirements: Tuple of requirement IDs involved.
        analysis_attempts: Tuple of strings describing what was tried.
        decision_requested: Clear, actionable question for PM.
        related_gaps: Tuple of gap IDs related to this escalation.
        related_contradictions: Tuple of contradiction IDs related to this.
        timestamp: When the escalation was created. Defaults to current UTC time.

    Example:
        >>> esc = Escalation(
        ...     id="esc-001",
        ...     reason=EscalationReason.CONFLICTING_REQUIREMENTS,
        ...     priority=EscalationPriority.HIGH,
        ...     summary="Real-time vs batch processing conflict",
        ...     context="req-001 requires real-time updates, req-002 specifies batch only",
        ...     original_requirements=("req-001", "req-002"),
        ...     analysis_attempts=("Tried scope separation", "Checked for priority hints"),
        ...     decision_requested="Should we prioritize real-time or batch processing?",
        ...     related_gaps=(),
        ...     related_contradictions=("conflict-001",),
        ... )
        >>> esc.priority
        <EscalationPriority.HIGH: 'high'>
    """

    id: str
    reason: EscalationReason
    priority: EscalationPriority
    summary: str
    context: str
    original_requirements: tuple[str, ...]
    analysis_attempts: tuple[str, ...]
    decision_requested: str
    related_gaps: tuple[str, ...] = ()
    related_contradictions: tuple[str, ...] = ()
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, enums as string values,
            tuples as lists, timestamp as ISO format string.

        Example:
            >>> esc = Escalation(
            ...     id="esc-001",
            ...     reason=EscalationReason.SCOPE_CLARIFICATION,
            ...     priority=EscalationPriority.NORMAL,
            ...     summary="Unclear scope for feature X",
            ...     context="Feature X mentioned but not detailed",
            ...     original_requirements=("req-005",),
            ...     analysis_attempts=("Searched for scope hints",),
            ...     decision_requested="Is feature X in scope?",
            ... )
            >>> esc.to_dict()["reason"]
            'scope_clarification'
        """
        return {
            "id": self.id,
            "reason": self.reason.value,
            "priority": self.priority.value,
            "summary": self.summary,
            "context": self.context,
            "original_requirements": list(self.original_requirements),
            "analysis_attempts": list(self.analysis_attempts),
            "decision_requested": self.decision_requested,
            "related_gaps": list(self.related_gaps),
            "related_contradictions": list(self.related_contradictions),
            "timestamp": self.timestamp.isoformat(),
        }


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
        implementability_status: Status of implementability validation
            ("implementable", "needs_clarification", "not_implementable").
            None if not yet validated. (Added in Story 5.5)
        complexity: Estimated complexity level for implementation
            ("low", "medium", "high", "very_high"). None if not assessed.
            (Added in Story 5.5)
        external_dependencies: Tuple of external dependency dicts identified
            for this requirement. Each dict contains name, dependency_type,
            description, availability_notes, and criticality.
            Note: Uses dict instead of ExternalDependency objects to enable
            direct JSON serialization and compatibility with to_dict() output.
            For typed access, use ExternalDependency.to_dict() format.
            (Added in Story 5.5)
        implementability_issues: Tuple of issue descriptions found during
            implementability validation. Empty if no issues. (Added in Story 5.5)
        implementability_rationale: Explanation of implementability validation
            decision, including why the requirement passed or failed.
            (Added in Story 5.5)

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
    # Story 5.5 implementability fields
    implementability_status: str | None = None
    complexity: str | None = None
    external_dependencies: tuple[dict[str, Any], ...] = ()
    implementability_issues: tuple[str, ...] = ()
    implementability_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all requirement fields including enhanced fields,
            categorization fields (Story 5.4), and implementability fields
            (Story 5.5).

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
            # Story 5.5 implementability fields
            "implementability_status": self.implementability_status,
            "complexity": self.complexity,
            "external_dependencies": list(self.external_dependencies),
            "implementability_issues": list(self.implementability_issues),
            "implementability_rationale": self.implementability_rationale,
        }


@dataclass(frozen=True)
class AnalystOutput:
    """Complete output from Analyst agent processing.

    Immutable dataclass containing all results from analyzing seed content:
    crystallized requirements, identified gaps, contradictions, and escalations.

    Attributes:
        requirements: Tuple of CrystallizedRequirement objects.
        identified_gaps: Tuple of strings describing missing information (legacy).
        contradictions: Tuple of strings describing conflicting requirements (legacy).
        structured_gaps: Tuple of IdentifiedGap objects with full gap details
            (Story 5.3). Defaults to empty for backward compatibility.
        structured_contradictions: Tuple of Contradiction objects with full conflict
            details including severity and resolution suggestions (Story 5.6).
            Defaults to empty for backward compatibility.
        escalations: Tuple of Escalation objects representing issues that need
            PM decision-making (Story 5.7). Defaults to empty for backward compatibility.

    Properties:
        escalation_needed: True if there are any escalations pending PM review.

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
    structured_contradictions: tuple[Contradiction, ...] = ()
    escalations: tuple[Escalation, ...] = ()

    @property
    def escalation_needed(self) -> bool:
        """Check if any escalations are pending PM review.

        Returns:
            True if there are any escalations, False otherwise.

        Example:
            >>> output = AnalystOutput(
            ...     requirements=(),
            ...     identified_gaps=(),
            ...     contradictions=(),
            ... )
            >>> output.escalation_needed
            False
        """
        return len(self.escalations) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Serializes nested CrystallizedRequirement, IdentifiedGap,
        Contradiction, and Escalation objects.

        Returns:
            Dictionary with all output fields including structured_gaps,
            structured_contradictions, escalations, and escalation_needed flag.

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
            "structured_contradictions": [c.to_dict() for c in self.structured_contradictions],
            "escalations": [e.to_dict() for e in self.escalations],
            "escalation_needed": self.escalation_needed,
        }
