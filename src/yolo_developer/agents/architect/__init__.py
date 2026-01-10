"""Architect agent module for design decisions and ADR generation (Stories 7.1-7.8).

The Architect agent is responsible for:
- Generating design decisions for stories
- Producing Architecture Decision Records (ADRs) with 12-Factor analysis
- Evaluating designs against quality attributes
- Identifying technical risks
- Ensuring designs follow 12-Factor principles

Example:
    >>> from yolo_developer.agents.architect import (
    ...     architect_node,
    ...     ArchitectOutput,
    ...     DesignDecision,
    ...     ADR,
    ...     DesignDecisionType,
    ...     ADRStatus,
    ... )
    >>>
    >>> # Create a design decision
    >>> decision = DesignDecision(
    ...     id="design-001",
    ...     story_id="story-001",
    ...     decision_type="pattern",
    ...     description="Use Repository pattern",
    ...     rationale="Clean separation of concerns",
    ...     alternatives_considered=("Active Record",),
    ... )
    >>>
    >>> # Run the architect node
    >>> result = await architect_node(state)

Architecture:
    The architect_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O
    - Integrates with architecture_validation gate (Story 7.1)

References:
    - ADR-001: TypedDict for internal state
    - ADR-005: LangGraph node patterns
    - ADR-006: Quality gate patterns
    - FR49-56: Architect Agent capabilities
"""

from __future__ import annotations

from yolo_developer.agents.architect import types
from yolo_developer.agents.architect.adr_generator import generate_adr, generate_adrs
from yolo_developer.agents.architect.atam_reviewer import run_atam_review
from yolo_developer.agents.architect.node import architect_node
from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching
from yolo_developer.agents.architect.quality_evaluator import evaluate_quality_attributes
from yolo_developer.agents.architect.risk_identifier import identify_technical_risks
from yolo_developer.agents.architect.tech_stack_validator import (
    validate_tech_stack_constraints,
)
from yolo_developer.agents.architect.twelve_factor import (
    analyze_twelve_factor,
    analyze_twelve_factor_with_llm,
)
from yolo_developer.agents.architect.types import (
    ADR,
    QUALITY_ATTRIBUTES,
    TWELVE_FACTORS,
    ADRStatus,
    ArchitectOutput,
    ATAMReviewResult,
    ATAMRiskAssessment,
    ATAMScenario,
    ATAMTradeOffConflict,
    ConstraintViolation,
    DesignDecision,
    DesignDecisionType,
    FactorResult,
    MitigationEffort,
    MitigationFeasibility,
    MitigationPriority,
    PatternCheckSeverity,
    PatternDeviation,
    PatternMatchingResult,
    PatternViolation,
    QualityAttributeEvaluation,
    QualityRisk,
    QualityTradeOff,
    RiskSeverity,
    StackPattern,
    TechnicalRisk,
    TechnicalRiskCategory,
    TechnicalRiskReport,
    TechStackCategory,
    TechStackValidation,
    TwelveFactorAnalysis,
    calculate_mitigation_priority,
    calculate_overall_risk_level,
)

__all__ = [
    "ADR",
    "QUALITY_ATTRIBUTES",
    "TWELVE_FACTORS",
    "ADRStatus",
    "ATAMReviewResult",
    "ATAMRiskAssessment",
    "ATAMScenario",
    "ATAMTradeOffConflict",
    "ArchitectOutput",
    "ConstraintViolation",
    "DesignDecision",
    "DesignDecisionType",
    "FactorResult",
    "MitigationEffort",
    "MitigationFeasibility",
    "MitigationPriority",
    "PatternCheckSeverity",
    "PatternDeviation",
    "PatternMatchingResult",
    "PatternViolation",
    "QualityAttributeEvaluation",
    "QualityRisk",
    "QualityTradeOff",
    "RiskSeverity",
    "StackPattern",
    "TechStackCategory",
    "TechStackValidation",
    "TechnicalRisk",
    "TechnicalRiskCategory",
    "TechnicalRiskReport",
    "TwelveFactorAnalysis",
    "analyze_twelve_factor",
    "analyze_twelve_factor_with_llm",
    "architect_node",
    "calculate_mitigation_priority",
    "calculate_overall_risk_level",
    "evaluate_quality_attributes",
    "generate_adr",
    "generate_adrs",
    "identify_technical_risks",
    "run_atam_review",
    "run_pattern_matching",
    "types",
    "validate_tech_stack_constraints",
]
