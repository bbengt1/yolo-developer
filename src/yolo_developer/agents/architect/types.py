"""Type definitions for Architect agent (Story 7.1, 7.2, 7.4, 7.6, 7.7).

This module provides the data types used by the Architect agent:

- DesignDecisionType: Literal type for design decision categories
- DesignDecision: A design decision made for a story
- ADRStatus: Literal type for ADR lifecycle status
- ADR: An Architecture Decision Record
- ArchitectOutput: Complete output from architect processing
- FactorResult: Result of a single 12-Factor compliance check (Story 7.2)
- TwelveFactorAnalysis: Complete 12-Factor compliance analysis (Story 7.2)
- TWELVE_FACTORS: Tuple of all 12 factor names (Story 7.2)
- QUALITY_ATTRIBUTES: Tuple of quality attribute names (Story 7.4)
- RiskSeverity: Literal type for risk severity levels (Story 7.4)
- MitigationEffort: Literal type for mitigation effort levels (Story 7.4)
- QualityRisk: A risk to meeting a quality attribute (Story 7.4)
- QualityTradeOff: A trade-off between quality attributes (Story 7.4)
- QualityAttributeEvaluation: Complete quality attribute evaluation (Story 7.4)
- TechStackCategory: Category of technology in the stack (Story 7.6)
- ConstraintViolation: A violation of tech stack constraints (Story 7.6)
- StackPattern: A stack-specific pattern suggestion (Story 7.6)
- TechStackValidation: Complete tech stack validation result (Story 7.6)
- ATAMScenario: A quality attribute scenario for ATAM review (Story 7.7)
- ATAMTradeOffConflict: A conflict between trade-offs (Story 7.7)
- ATAMRiskAssessment: Assessment of risk impact (Story 7.7)
- ATAMReviewResult: Complete ATAM review result (Story 7.7)
- MitigationFeasibility: Feasibility level for mitigations (Story 7.7)

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.architect.types import (
    ...     DesignDecision,
    ...     ADR,
    ...     ArchitectOutput,
    ... )
    >>>
    >>> decision = DesignDecision(
    ...     id="design-001",
    ...     story_id="story-001",
    ...     decision_type="pattern",
    ...     description="Use Repository pattern",
    ...     rationale="Decouples data access from business logic",
    ...     alternatives_considered=("Active Record", "DAO"),
    ... )
    >>> decision.to_dict()
    {'id': 'design-001', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

DesignDecisionType = Literal[
    "pattern",
    "technology",
    "integration",
    "data",
    "security",
    "infrastructure",
]
"""Type of architectural design decision.

Values:
    pattern: Architectural pattern selection (e.g., Repository, Factory)
    technology: Technology/framework choice (e.g., PostgreSQL, Redis)
    integration: Integration approach (e.g., REST API, message queue)
    data: Data model/storage decisions (e.g., event sourcing, CQRS)
    security: Security architecture (e.g., OAuth2, JWT)
    infrastructure: Infrastructure choice (e.g., Docker, Kubernetes)
"""

ADRStatus = Literal[
    "proposed",
    "accepted",
    "deprecated",
    "superseded",
]
"""Lifecycle status of an Architecture Decision Record.

Values:
    proposed: Decision is being considered
    accepted: Decision has been accepted and is in effect
    deprecated: Decision is no longer recommended
    superseded: Decision has been replaced by another ADR
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class DesignDecision:
    """A design decision made for a story.

    Represents an architectural decision with full context including
    the rationale and alternatives that were considered.

    Attributes:
        id: Unique identifier for the decision (format: design-{timestamp}-{counter})
        story_id: ID of the story this decision applies to
        decision_type: Category of the decision
        description: Clear description of what was decided
        rationale: Why this decision was made
        alternatives_considered: Other options that were evaluated
        created_at: ISO timestamp when decision was created

    Example:
        >>> decision = DesignDecision(
        ...     id="design-1704412345-001",
        ...     story_id="story-001",
        ...     decision_type="pattern",
        ...     description="Use Repository pattern for data access",
        ...     rationale="Provides clean separation of concerns",
        ...     alternatives_considered=("Active Record", "DAO"),
        ... )
    """

    id: str
    story_id: str
    decision_type: DesignDecisionType
    description: str
    rationale: str
    alternatives_considered: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the design decision.
        """
        return {
            "id": self.id,
            "story_id": self.story_id,
            "decision_type": self.decision_type,
            "description": self.description,
            "rationale": self.rationale,
            "alternatives_considered": list(self.alternatives_considered),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ADR:
    """An Architecture Decision Record.

    Follows the standard ADR format with Title, Status, Context,
    Decision, and Consequences sections.

    Attributes:
        id: Unique identifier (format: ADR-{number:03d})
        title: Descriptive title of the decision
        status: Current lifecycle status
        context: Why this decision was needed
        decision: What was decided
        consequences: Positive and negative effects
        story_ids: Stories this ADR relates to
        created_at: ISO timestamp when ADR was created

    Example:
        >>> adr = ADR(
        ...     id="ADR-001",
        ...     title="Use PostgreSQL for persistence",
        ...     status="accepted",
        ...     context="Need reliable ACID-compliant database",
        ...     decision="PostgreSQL for all persistent data",
        ...     consequences="Good: Reliability, Bad: Operational complexity",
        ...     story_ids=("story-001", "story-002"),
        ... )
    """

    id: str
    title: str
    status: ADRStatus
    context: str
    decision: str
    consequences: str
    story_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the ADR.
        """
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "story_ids": list(self.story_ids),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ArchitectOutput:
    """Complete output from Architect agent processing.

    Contains all design decisions and ADRs generated during
    architect_node execution, plus processing notes.

    Attributes:
        design_decisions: Tuple of design decisions made
        adrs: Tuple of ADRs generated
        processing_notes: Notes about the processing (stats, issues, etc.)
        twelve_factor_analyses: Dict mapping story IDs to 12-Factor analyses (Story 7.2)
        quality_evaluations: Dict mapping story IDs to quality evaluations (Story 7.4)
        technical_risk_reports: Dict mapping story IDs to technical risk reports (Story 7.5)
        tech_stack_validations: Dict mapping story IDs to tech stack validations (Story 7.6)
        atam_reviews: Dict mapping story IDs to ATAM review results (Story 7.7)

    Example:
        >>> output = ArchitectOutput(
        ...     design_decisions=(decision1, decision2),
        ...     adrs=(adr1,),
        ...     processing_notes="Processed 2 stories, generated 2 decisions",
        ... )
        >>> output.to_dict()
        {'design_decisions': [...], 'adrs': [...], ...}
    """

    design_decisions: tuple[DesignDecision, ...] = field(default_factory=tuple)
    adrs: tuple[ADR, ...] = field(default_factory=tuple)
    processing_notes: str = ""
    twelve_factor_analyses: dict[str, Any] = field(default_factory=dict)
    quality_evaluations: dict[str, Any] = field(default_factory=dict)
    technical_risk_reports: dict[str, Any] = field(default_factory=dict)
    tech_stack_validations: dict[str, Any] = field(default_factory=dict)
    atam_reviews: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested design decisions and ADRs.
        """
        return {
            "design_decisions": [d.to_dict() for d in self.design_decisions],
            "adrs": [a.to_dict() for a in self.adrs],
            "processing_notes": self.processing_notes,
            "twelve_factor_analyses": self.twelve_factor_analyses,
            "quality_evaluations": self.quality_evaluations,
            "technical_risk_reports": self.technical_risk_reports,
            "tech_stack_validations": self.tech_stack_validations,
            "atam_reviews": self.atam_reviews,
        }


# =============================================================================
# Twelve-Factor Types (Story 7.2)
# =============================================================================

TWELVE_FACTORS: tuple[str, ...] = (
    "codebase",
    "dependencies",
    "config",
    "backing_services",
    "build_release_run",
    "processes",
    "port_binding",
    "concurrency",
    "disposability",
    "dev_prod_parity",
    "logs",
    "admin_processes",
)
"""Tuple of all 12 factor names from the 12-Factor App methodology.

Values:
    codebase: One codebase tracked in revision control, many deploys
    dependencies: Explicitly declare and isolate dependencies
    config: Store config in the environment
    backing_services: Treat backing services as attached resources
    build_release_run: Strictly separate build and run stages
    processes: Execute the app as one or more stateless processes
    port_binding: Export services via port binding
    concurrency: Scale out via the process model
    disposability: Maximize robustness with fast startup and graceful shutdown
    dev_prod_parity: Keep development, staging, and production similar
    logs: Treat logs as event streams
    admin_processes: Run admin/management tasks as one-off processes
"""


@dataclass(frozen=True)
class FactorResult:
    """Result of a single 12-Factor compliance check.

    Represents the analysis of a story against one of the 12 factors,
    indicating whether it applies and if compliant.

    Attributes:
        factor_name: Name of the factor (from TWELVE_FACTORS)
        applies: Whether this factor is relevant to the story
        compliant: Whether the story complies (None if not applicable)
        finding: Description of what was found during analysis
        recommendation: Suggested improvement (empty if compliant)

    Example:
        >>> result = FactorResult(
        ...     factor_name="config",
        ...     applies=True,
        ...     compliant=False,
        ...     finding="Hardcoded database URL detected",
        ...     recommendation="Use environment variable DATABASE_URL",
        ... )
    """

    factor_name: str
    applies: bool
    compliant: bool | None
    finding: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the factor result.
        """
        return {
            "factor_name": self.factor_name,
            "applies": self.applies,
            "compliant": self.compliant,
            "finding": self.finding,
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class TwelveFactorAnalysis:
    """Complete 12-Factor compliance analysis for a story.

    Contains the results of analyzing a story against all 12 factors,
    with an overall compliance score and aggregated recommendations.

    Attributes:
        factor_results: Dict mapping factor names to their results
        applicable_factors: Tuple of factor names that apply to this story
        overall_compliance: Score from 0.0 to 1.0 (ratio of compliant factors)
        recommendations: Tuple of aggregated recommendations

    Example:
        >>> analysis = TwelveFactorAnalysis(
        ...     factor_results={"config": result1, "processes": result2},
        ...     applicable_factors=("config", "processes"),
        ...     overall_compliance=0.5,
        ...     recommendations=("Use env vars for config",),
        ... )
        >>> analysis.to_dict()
        {'factor_results': {...}, 'overall_compliance': 0.5, ...}
    """

    factor_results: dict[str, FactorResult]
    applicable_factors: tuple[str, ...]
    overall_compliance: float
    recommendations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested factor results.
        """
        return {
            "factor_results": {
                name: result.to_dict()
                for name, result in self.factor_results.items()
            },
            "applicable_factors": list(self.applicable_factors),
            "overall_compliance": self.overall_compliance,
            "recommendations": list(self.recommendations),
        }


# =============================================================================
# Quality Attribute Types (Story 7.4)
# =============================================================================

QUALITY_ATTRIBUTES: tuple[str, ...] = (
    "performance",
    "security",
    "reliability",
    "scalability",
    "maintainability",
    "integration",
    "cost_efficiency",
)
"""Tuple of quality attribute names for NFR evaluation.

Values:
    performance: Response time, throughput, resource efficiency
    security: Authentication, authorization, data protection
    reliability: Fault tolerance, recovery, consistency
    scalability: Horizontal scaling, load handling
    maintainability: Code clarity, testability, documentation
    integration: Multi-provider support, protocol compliance
    cost_efficiency: Model tiering, caching, token optimization
"""

RiskSeverity = Literal["critical", "high", "medium", "low"]
"""Severity level for quality risks.

Values:
    critical: Score 0.0-0.3, requires immediate attention
    high: Score 0.3-0.5, significant risk to NFRs
    medium: Score 0.5-0.7, moderate risk with workarounds
    low: Score 0.7-1.0, acceptable risk level
"""

MitigationEffort = Literal["high", "medium", "low"]
"""Effort level for implementing risk mitigations.

Values:
    high: Significant architectural changes required
    medium: Moderate code changes or configuration
    low: Simple fixes or minor adjustments
"""


@dataclass(frozen=True)
class QualityRisk:
    """A risk to meeting a quality attribute requirement.

    Represents an identified risk to a specific NFR, including severity
    categorization and suggested mitigation strategy.

    Attributes:
        attribute: Quality attribute at risk (from QUALITY_ATTRIBUTES)
        description: Description of the risk
        severity: Risk severity level
        mitigation: Suggested mitigation strategy
        mitigation_effort: Estimated effort to implement mitigation

    Example:
        >>> risk = QualityRisk(
        ...     attribute="performance",
        ...     description="High latency in database queries",
        ...     severity="high",
        ...     mitigation="Add database indexing and connection pooling",
        ...     mitigation_effort="medium",
        ... )
    """

    attribute: str
    description: str
    severity: RiskSeverity
    mitigation: str
    mitigation_effort: MitigationEffort

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the quality risk.
        """
        return {
            "attribute": self.attribute,
            "description": self.description,
            "severity": self.severity,
            "mitigation": self.mitigation,
            "mitigation_effort": self.mitigation_effort,
        }


@dataclass(frozen=True)
class QualityTradeOff:
    """A trade-off between two quality attributes.

    Documents conflicts between quality attributes and the chosen
    resolution approach.

    Attributes:
        attribute_a: First quality attribute in the trade-off
        attribute_b: Second quality attribute in the trade-off
        description: Description of the conflict
        resolution: Chosen approach to balance both attributes

    Example:
        >>> tradeoff = QualityTradeOff(
        ...     attribute_a="performance",
        ...     attribute_b="security",
        ...     description="Encryption adds 50ms latency per request",
        ...     resolution="Use async encryption and cache encrypted results",
        ... )
    """

    attribute_a: str
    attribute_b: str
    description: str
    resolution: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the quality trade-off.
        """
        return {
            "attribute_a": self.attribute_a,
            "attribute_b": self.attribute_b,
            "description": self.description,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class QualityAttributeEvaluation:
    """Complete quality attribute evaluation for a design.

    Contains scores for each quality attribute, identified trade-offs,
    risks with mitigations, and an overall weighted score.

    Attributes:
        attribute_scores: Dict mapping attribute names to scores (0.0-1.0)
        trade_offs: Tuple of identified trade-offs between attributes
        risks: Tuple of identified risks with mitigations
        overall_score: Weighted overall quality score (0.0-1.0)

    Example:
        >>> evaluation = QualityAttributeEvaluation(
        ...     attribute_scores={"performance": 0.8, "security": 0.7},
        ...     trade_offs=(tradeoff,),
        ...     risks=(risk,),
        ...     overall_score=0.75,
        ... )
        >>> evaluation.to_dict()
        {'attribute_scores': {...}, 'trade_offs': [...], ...}
    """

    attribute_scores: dict[str, float]
    trade_offs: tuple[QualityTradeOff, ...] = field(default_factory=tuple)
    risks: tuple[QualityRisk, ...] = field(default_factory=tuple)
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested trade-offs and risks.
        """
        return {
            "attribute_scores": dict(self.attribute_scores),
            "trade_offs": [t.to_dict() for t in self.trade_offs],
            "risks": [r.to_dict() for r in self.risks],
            "overall_score": self.overall_score,
        }


# =============================================================================
# Technical Risk Types (Story 7.5)
# =============================================================================

TechnicalRiskCategory = Literal[
    "technology",
    "integration",
    "scalability",
    "compatibility",
    "operational",
]
"""Category of technical risk.

Values:
    technology: Library/framework concerns (deprecated, experimental, version conflicts)
    integration: External service concerns (rate limiting, API instability, vendor lock-in)
    scalability: Growth/load concerns (single point of failure, stateful, bottlenecks)
    compatibility: Cross-system concerns (version mismatch, protocol incompatibility)
    operational: Runtime concerns (monitoring gaps, deployment complexity)
"""

MitigationPriority = Literal["P1", "P2", "P3", "P4"]
"""Priority level for addressing a risk mitigation.

Values:
    P1: Urgent - must address immediately (critical severity or easy high-impact fix)
    P2: High - address in current sprint
    P3: Medium - address when convenient
    P4: Low - nice to have, defer if needed
"""


def calculate_mitigation_priority(
    severity: RiskSeverity, effort: MitigationEffort
) -> MitigationPriority:
    """Calculate mitigation priority from severity and effort.

    Priority matrix:
    | Severity | High Effort | Medium Effort | Low Effort |
    |----------|-------------|---------------|------------|
    | Critical | P1          | P1            | P1         |
    | High     | P2          | P1            | P1         |
    | Medium   | P3          | P2            | P2         |
    | Low      | P4          | P3            | P3         |

    Args:
        severity: Risk severity level.
        effort: Mitigation effort level.

    Returns:
        Calculated mitigation priority.
    """
    priority_matrix: dict[tuple[RiskSeverity, MitigationEffort], MitigationPriority] = {
        ("critical", "high"): "P1",
        ("critical", "medium"): "P1",
        ("critical", "low"): "P1",
        ("high", "high"): "P2",
        ("high", "medium"): "P1",
        ("high", "low"): "P1",
        ("medium", "high"): "P3",
        ("medium", "medium"): "P2",
        ("medium", "low"): "P2",
        ("low", "high"): "P4",
        ("low", "medium"): "P3",
        ("low", "low"): "P3",
    }
    return priority_matrix.get((severity, effort), "P3")


@dataclass(frozen=True)
class TechnicalRisk:
    """A technical risk identified in a design.

    Represents a risk to successful implementation including category,
    severity, affected components, and mitigation strategy.

    Attributes:
        category: Type of technical risk
        description: What the risk is
        severity: Risk severity level (critical/high/medium/low)
        affected_components: Components affected by this risk
        mitigation: Suggested mitigation strategy
        mitigation_effort: Effort to implement mitigation
        mitigation_priority: Priority for addressing (P1-P4)

    Example:
        >>> risk = TechnicalRisk(
        ...     category="integration",
        ...     description="External API has no SLA",
        ...     severity="high",
        ...     affected_components=("AuthService",),
        ...     mitigation="Implement circuit breaker",
        ...     mitigation_effort="medium",
        ...     mitigation_priority="P1",
        ... )
    """

    category: TechnicalRiskCategory
    description: str
    severity: RiskSeverity
    mitigation: str
    mitigation_effort: MitigationEffort
    mitigation_priority: MitigationPriority
    affected_components: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the technical risk.
        """
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "affected_components": list(self.affected_components),
            "mitigation": self.mitigation,
            "mitigation_effort": self.mitigation_effort,
            "mitigation_priority": self.mitigation_priority,
        }


def calculate_overall_risk_level(risks: list[TechnicalRisk]) -> RiskSeverity:
    """Calculate overall risk level from a list of risks.

    Returns the highest severity among all risks, or "low" if no risks.

    Args:
        risks: List of technical risks.

    Returns:
        Overall risk level (highest severity found).
    """
    if not risks:
        return "low"

    severity_order: dict[RiskSeverity, int] = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    max_severity: RiskSeverity = "low"
    max_order = 0

    for risk in risks:
        order = severity_order.get(risk.severity, 0)
        if order > max_order:
            max_order = order
            max_severity = risk.severity

    return max_severity


@dataclass(frozen=True)
class TechnicalRiskReport:
    """Complete technical risk analysis report.

    Contains all identified risks with overall assessment and summary.

    Attributes:
        risks: Tuple of identified technical risks
        overall_risk_level: Highest severity among all risks
        summary: Brief description of key risks

    Example:
        >>> report = TechnicalRiskReport(
        ...     risks=(risk1, risk2),
        ...     overall_risk_level="high",
        ...     summary="Two integration risks identified",
        ... )
    """

    risks: tuple[TechnicalRisk, ...]
    overall_risk_level: RiskSeverity
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested risks.
        """
        return {
            "risks": [r.to_dict() for r in self.risks],
            "overall_risk_level": self.overall_risk_level,
            "summary": self.summary,
        }


# =============================================================================
# Tech Stack Types (Story 7.6)
# =============================================================================

TechStackCategory = Literal[
    "runtime",
    "framework",
    "database",
    "testing",
    "tooling",
]
"""Category of technology in the configured stack.

Values:
    runtime: Language runtime (e.g., Python 3.10+, Node.js 20+)
    framework: Application framework (e.g., LangGraph, FastAPI, Django)
    database: Data storage (e.g., ChromaDB, PostgreSQL, Neo4j)
    testing: Test framework (e.g., pytest, pytest-asyncio)
    tooling: Build/dev tools (e.g., uv, ruff, mypy)
"""


@dataclass(frozen=True)
class ConstraintViolation:
    """A violation of tech stack constraints.

    Represents a design decision that conflicts with the configured
    technology stack, including version mismatches and unconfigured
    technologies.

    Attributes:
        technology: The technology that violates constraints
        expected_version: Configured version requirement (None if not configured)
        actual_version: Version used in the design decision
        severity: Severity of the violation (critical, high, medium, low)
        suggested_alternative: How to fix the violation

    Example:
        >>> violation = ConstraintViolation(
        ...     technology="SQLite",
        ...     expected_version=None,
        ...     actual_version="3.x",
        ...     severity="critical",
        ...     suggested_alternative="Use ChromaDB as configured for vector storage",
        ... )
    """

    technology: str
    expected_version: str | None
    actual_version: str
    severity: RiskSeverity
    suggested_alternative: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the constraint violation.
        """
        return {
            "technology": self.technology,
            "expected_version": self.expected_version,
            "actual_version": self.actual_version,
            "severity": self.severity,
            "suggested_alternative": self.suggested_alternative,
        }


@dataclass(frozen=True)
class StackPattern:
    """A stack-specific pattern suggestion.

    Represents a recommended pattern based on the configured technology
    stack, with rationale explaining why it applies.

    Attributes:
        pattern_name: Short identifier for the pattern
        description: What the pattern is about
        rationale: Why this pattern applies to the configured stack
        applicable_technologies: Technologies this pattern relates to

    Example:
        >>> pattern = StackPattern(
        ...     pattern_name="pytest-fixtures",
        ...     description="Use pytest fixtures for test setup/teardown",
        ...     rationale="pytest is configured as test framework; fixtures provide clean test isolation",
        ...     applicable_technologies=("pytest", "pytest-asyncio"),
        ... )
    """

    pattern_name: str
    description: str
    rationale: str
    applicable_technologies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the stack pattern.
        """
        return {
            "pattern_name": self.pattern_name,
            "description": self.description,
            "rationale": self.rationale,
            "applicable_technologies": list(self.applicable_technologies),
        }


@dataclass(frozen=True)
class TechStackValidation:
    """Complete tech stack validation result.

    Contains the overall compliance status, any constraint violations,
    and suggested patterns for the configured technology stack.

    Attributes:
        overall_compliance: Overall compliance (False if any critical violations)
        violations: Tuple of constraint violations found
        suggested_patterns: Tuple of stack-specific pattern suggestions
        summary: Brief description of validation results

    Example:
        >>> validation = TechStackValidation(
        ...     overall_compliance=False,
        ...     violations=(violation,),
        ...     suggested_patterns=(pattern,),
        ...     summary="1 constraint violation found",
        ... )
        >>> validation.to_dict()
        {'overall_compliance': False, 'violations': [...], ...}
    """

    overall_compliance: bool
    violations: tuple[ConstraintViolation, ...]
    suggested_patterns: tuple[StackPattern, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested violations and patterns.
        """
        return {
            "overall_compliance": self.overall_compliance,
            "violations": [v.to_dict() for v in self.violations],
            "suggested_patterns": [p.to_dict() for p in self.suggested_patterns],
            "summary": self.summary,
        }


# =============================================================================
# ATAM Review Types (Story 7.7)
# =============================================================================


@dataclass(frozen=True)
class ATAMScenario:
    """A quality attribute scenario for ATAM architectural review.

    Represents a concrete expression of how the system should respond
    to a stimulus for a specific quality attribute.

    Attributes:
        scenario_id: Unique identifier (format: ATAM-{number})
        quality_attribute: Quality attribute being evaluated
        stimulus: What triggers the scenario (e.g., "100 concurrent requests")
        response: Expected system response (e.g., "95th percentile < 500ms")
        analysis: Assessment of how design addresses this scenario

    Example:
        >>> scenario = ATAMScenario(
        ...     scenario_id="ATAM-001",
        ...     quality_attribute="performance",
        ...     stimulus="100 concurrent API requests",
        ...     response="95th percentile < 500ms",
        ...     analysis="Design supports async processing and caching",
        ... )
    """

    scenario_id: str
    quality_attribute: str
    stimulus: str
    response: str
    analysis: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the ATAM scenario.
        """
        return {
            "scenario_id": self.scenario_id,
            "quality_attribute": self.quality_attribute,
            "stimulus": self.stimulus,
            "response": self.response,
            "analysis": self.analysis,
        }


@dataclass(frozen=True)
class ATAMTradeOffConflict:
    """A conflict between quality attribute trade-offs.

    Represents a situation where trade-off resolutions between quality
    attributes are in conflict and need reconciliation.

    Attributes:
        attribute_a: First quality attribute in the conflict
        attribute_b: Second quality attribute in the conflict
        description: Description of the conflict
        severity: Conflict severity (critical, high, medium, low)
        resolution_strategy: Suggested approach to resolve the conflict

    Example:
        >>> conflict = ATAMTradeOffConflict(
        ...     attribute_a="performance",
        ...     attribute_b="security",
        ...     description="Encryption adds latency to all requests",
        ...     severity="medium",
        ...     resolution_strategy="Use async encryption with caching",
        ... )
    """

    attribute_a: str
    attribute_b: str
    description: str
    severity: RiskSeverity
    resolution_strategy: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the trade-off conflict.
        """
        return {
            "attribute_a": self.attribute_a,
            "attribute_b": self.attribute_b,
            "description": self.description,
            "severity": self.severity,
            "resolution_strategy": self.resolution_strategy,
        }


MitigationFeasibility = Literal["high", "medium", "low"]
"""Feasibility level for implementing risk mitigations.

Values:
    high: Mitigation is straightforward to implement
    medium: Mitigation requires moderate effort
    low: Mitigation is difficult or costly to implement
"""


@dataclass(frozen=True)
class ATAMRiskAssessment:
    """Assessment of a technical risk's impact on quality attributes.

    Represents the ATAM-style evaluation of how a technical risk
    affects quality attributes and whether mitigation is feasible.

    Attributes:
        risk_id: Reference to the original TechnicalRisk
        quality_impact: Quality attributes affected by this risk
        mitigation_feasibility: How feasible is the mitigation (high/medium/low)
        unmitigated: Whether the risk remains unmitigated

    Example:
        >>> assessment = ATAMRiskAssessment(
        ...     risk_id="RISK-001",
        ...     quality_impact=("reliability", "performance"),
        ...     mitigation_feasibility="high",
        ...     unmitigated=False,
        ... )
    """

    risk_id: str
    quality_impact: tuple[str, ...]
    mitigation_feasibility: MitigationFeasibility
    unmitigated: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the risk assessment.
        """
        return {
            "risk_id": self.risk_id,
            "quality_impact": list(self.quality_impact),
            "mitigation_feasibility": self.mitigation_feasibility,
            "unmitigated": self.unmitigated,
        }


@dataclass(frozen=True)
class ATAMReviewResult:
    """Complete result of an ATAM architectural review.

    Contains the overall pass/fail decision, confidence score,
    evaluated scenarios, detected conflicts, risk assessments,
    and any failure reasons.

    Attributes:
        overall_pass: Whether the design passes ATAM review
        confidence: Confidence score from 0.0 to 1.0
        scenarios_evaluated: Quality attribute scenarios evaluated
        trade_off_conflicts: Conflicts between trade-offs
        risk_assessments: Assessments of technical risks
        failure_reasons: Specific reasons if review failed
        summary: Brief summary of the review outcome

    Example:
        >>> result = ATAMReviewResult(
        ...     overall_pass=True,
        ...     confidence=0.85,
        ...     scenarios_evaluated=(scenario,),
        ...     trade_off_conflicts=(),
        ...     risk_assessments=(),
        ...     failure_reasons=(),
        ...     summary="Design passes ATAM review with 85% confidence",
        ... )
        >>> result.to_dict()
        {'overall_pass': True, 'confidence': 0.85, ...}
    """

    overall_pass: bool
    confidence: float
    scenarios_evaluated: tuple[ATAMScenario, ...]
    trade_off_conflicts: tuple[ATAMTradeOffConflict, ...]
    risk_assessments: tuple[ATAMRiskAssessment, ...]
    failure_reasons: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested components.
        """
        return {
            "overall_pass": self.overall_pass,
            "confidence": self.confidence,
            "scenarios_evaluated": [s.to_dict() for s in self.scenarios_evaluated],
            "trade_off_conflicts": [c.to_dict() for c in self.trade_off_conflicts],
            "risk_assessments": [r.to_dict() for r in self.risk_assessments],
            "failure_reasons": list(self.failure_reasons),
            "summary": self.summary,
        }
