"""Type definitions for Architect agent (Story 7.1, 7.2, 7.4).

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
