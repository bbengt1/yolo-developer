"""Deployment blocking types and functions (Story 9.7).

This module provides the data types and functions used for deployment blocking:

Types:
- BlockingReasonType: Literal type for blocking reason types
- BlockingReason: A reason why deployment was blocked
- RemediationStep: A step to remediate a blocking reason
- DeploymentDecision: The deployment decision result
- DeploymentOverride: Acknowledgment for overriding a deployment block
- DeploymentDecisionReport: Complete deployment decision report

Functions:
- _generate_reason_id: Generate unique reason ID
- _generate_step_id: Generate unique step ID
- generate_blocking_reasons: Generate all blocking reasons
- generate_remediation_steps: Generate all remediation steps
- evaluate_deployment_decision: Evaluate deployment decision
- create_override: Create override with acknowledgment
- validate_override: Validate override completeness
- generate_deployment_decision_report: Generate complete report
- get_deployment_threshold: Get threshold from config
- is_deployment_blocking_enabled: Check if blocking is enabled

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.tea.blocking import (
    ...     BlockingReason,
    ...     RemediationStep,
    ...     DeploymentDecision,
    ...     DeploymentOverride,
    ...     DeploymentDecisionReport,
    ... )
    >>>
    >>> reason = BlockingReason(
    ...     reason_id="BR-LOW-001",
    ...     reason_type="low_confidence",
    ...     description="Confidence score 75 is below threshold 90",
    ...     threshold_value=90,
    ...     actual_value=75,
    ...     related_findings=("F-CONFIDENCE-75",),
    ... )
    >>> reason.to_dict()
    {'reason_id': 'BR-LOW-001', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from yolo_developer.agents.tea.risk import RiskReport
    from yolo_developer.agents.tea.scoring import ConfidenceResult
    from yolo_developer.agents.tea.types import DeploymentRecommendation, ValidationResult

logger = structlog.get_logger(__name__)


# =============================================================================
# Literal Types
# =============================================================================

BlockingReasonType = Literal[
    "low_confidence",
    "critical_risk",
    "validation_failed",
    "high_risk_count",
]
"""Type of blocking reason.

Values:
    low_confidence: Confidence score is below deployment threshold
    critical_risk: Critical risk finding(s) present
    validation_failed: Validation failed for one or more artifacts
    high_risk_count: High risk count exceeds safe deployment threshold
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class BlockingReason:
    """A reason why deployment was blocked.

    Represents a single reason for blocking deployment, with details
    about the threshold violation and related findings.

    Attributes:
        reason_id: Unique identifier (e.g., "BR-LOW-001")
        reason_type: Type of blocking reason
        description: Human-readable description
        threshold_value: The threshold that was not met (if applicable)
        actual_value: The actual value that triggered blocking (if applicable)
        related_findings: IDs of findings that contributed to this reason

    Example:
        >>> reason = BlockingReason(
        ...     reason_id="BR-LOW-001",
        ...     reason_type="low_confidence",
        ...     description="Confidence score 75 is below threshold 90",
        ...     threshold_value=90,
        ...     actual_value=75,
        ...     related_findings=("F-CONFIDENCE-75",),
        ... )
        >>> reason.to_dict()
        {'reason_id': 'BR-LOW-001', ...}
    """

    reason_id: str
    reason_type: BlockingReasonType
    description: str
    threshold_value: int | None = None
    actual_value: int | None = None
    related_findings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the blocking reason.
        """
        return {
            "reason_id": self.reason_id,
            "reason_type": self.reason_type,
            "description": self.description,
            "threshold_value": self.threshold_value,
            "actual_value": self.actual_value,
            "related_findings": list(self.related_findings),
        }


@dataclass(frozen=True)
class RemediationStep:
    """A step to remediate a blocking reason.

    Represents a single actionable step to address a blocking reason
    and improve deployment readiness.

    Attributes:
        step_id: Unique identifier (e.g., "RS-001")
        priority: Priority order (1=highest)
        action: Specific action to take
        expected_impact: How this will improve the situation
        related_reason_id: The blocking reason this addresses

    Example:
        >>> step = RemediationStep(
        ...     step_id="RS-001",
        ...     priority=1,
        ...     action="Address critical finding in src/auth/handler.py",
        ...     expected_impact="Resolving critical findings removes deployment blocker",
        ...     related_reason_id="BR-CRI-001",
        ... )
        >>> step.to_dict()
        {'step_id': 'RS-001', ...}
    """

    step_id: str
    priority: int
    action: str
    expected_impact: str
    related_reason_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the remediation step.
        """
        return {
            "step_id": self.step_id,
            "priority": self.priority,
            "action": self.action,
            "expected_impact": self.expected_impact,
            "related_reason_id": self.related_reason_id,
        }


@dataclass(frozen=True)
class DeploymentDecision:
    """The deployment decision result.

    Captures the final deployment decision, including whether deployment
    is blocked and the recommendation.

    Attributes:
        is_blocked: Whether deployment is blocked
        recommendation: The deployment recommendation
        evaluated_at: ISO timestamp of evaluation

    Example:
        >>> decision = DeploymentDecision(
        ...     is_blocked=True,
        ...     recommendation="block",
        ... )
        >>> decision.to_dict()
        {'is_blocked': True, ...}
    """

    is_blocked: bool
    recommendation: DeploymentRecommendation
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the deployment decision.
        """
        return {
            "is_blocked": self.is_blocked,
            "recommendation": self.recommendation,
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True)
class DeploymentOverride:
    """Acknowledgment for overriding a deployment block.

    Captures the explicit acknowledgment required to override a deployment
    block and proceed despite known risks.

    Attributes:
        acknowledged_by: Who acknowledged the override
        acknowledged_at: ISO timestamp
        acknowledgment_reason: Why override is being used
        acknowledged_risks: Risks that were acknowledged

    Example:
        >>> override = DeploymentOverride(
        ...     acknowledged_by="user@example.com",
        ...     acknowledged_at="2026-01-12T10:30:00Z",
        ...     acknowledgment_reason="Critical hotfix for production outage",
        ...     acknowledged_risks=("Confidence score 75 is below threshold 90",),
        ... )
        >>> override.to_dict()
        {'acknowledged_by': 'user@example.com', ...}
    """

    acknowledged_by: str
    acknowledged_at: str
    acknowledgment_reason: str
    acknowledged_risks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the deployment override.
        """
        return {
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "acknowledgment_reason": self.acknowledgment_reason,
            "acknowledged_risks": list(self.acknowledged_risks),
        }


@dataclass(frozen=True)
class DeploymentDecisionReport:
    """Complete deployment decision report.

    Contains the full deployment decision with blocking reasons,
    remediation steps, optional override, and supporting data.

    Attributes:
        decision: The deployment decision
        blocking_reasons: Reasons for blocking (empty if not blocked)
        remediation_steps: Steps to fix blocking (empty if not blocked)
        override: Override acknowledgment if provided
        confidence_result: Full confidence scoring result
        risk_report: Full risk categorization report
        created_at: ISO timestamp

    Example:
        >>> from yolo_developer.agents.tea.blocking import DeploymentDecision
        >>> decision = DeploymentDecision(is_blocked=False, recommendation="deploy")
        >>> report = DeploymentDecisionReport(decision=decision)
        >>> report.to_dict()
        {'decision': {...}, 'blocking_reasons': [], ...}
    """

    decision: DeploymentDecision
    blocking_reasons: tuple[BlockingReason, ...] = field(default_factory=tuple)
    remediation_steps: tuple[RemediationStep, ...] = field(default_factory=tuple)
    override: DeploymentOverride | None = None
    confidence_result: ConfidenceResult | None = None
    risk_report: RiskReport | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested components.
        """
        return {
            "decision": self.decision.to_dict(),
            "blocking_reasons": [r.to_dict() for r in self.blocking_reasons],
            "remediation_steps": [s.to_dict() for s in self.remediation_steps],
            "override": self.override.to_dict() if self.override else None,
            "confidence_result": self.confidence_result.to_dict()
            if self.confidence_result
            else None,
            "risk_report": self.risk_report.to_dict() if self.risk_report else None,
            "created_at": self.created_at,
        }


# =============================================================================
# Reason Templates
# =============================================================================

REASON_TEMPLATES: dict[BlockingReasonType, str] = {
    "low_confidence": "Confidence score {actual} is below deployment threshold {threshold}",
    "critical_risk": "Critical risk finding(s) present: {count} critical issue(s) must be resolved",
    "validation_failed": "Validation failed for {count} artifact(s): {artifacts}",
    "high_risk_count": "High risk count ({actual}) exceeds safe deployment threshold ({threshold})",
}


# =============================================================================
# Remediation Templates
# =============================================================================

REMEDIATION_TEMPLATES: dict[BlockingReasonType, list[tuple[str, str]]] = {
    "low_confidence": [
        (
            "Increase test coverage to improve confidence score",
            "Higher coverage increases confidence score by ~8-12 points",
        ),
        (
            "Fix failing tests to improve test execution score",
            "Passing tests contribute to test execution score",
        ),
        (
            "Address validation findings to reduce penalties",
            "Resolving findings removes score penalties",
        ),
    ],
    "critical_risk": [
        (
            "Address critical findings immediately",
            "Resolving critical findings removes deployment blocker",
        ),
        ("Review security-related findings", "Security fixes are essential before deployment"),
        ("Verify fixes with additional tests", "Tests confirm fixes are effective"),
    ],
    "validation_failed": [
        ("Fix validation errors in affected artifacts", "Validation must pass for deployment"),
        ("Re-run validation after fixes", "Confirm fixes resolve validation issues"),
        (
            "Review validation rules for false positives",
            "Adjust rules if validation is overly strict",
        ),
    ],
    "high_risk_count": [
        (
            "Prioritize and address high-severity findings",
            "Reducing high-risk count below threshold allows deployment",
        ),
        (
            "Consider breaking changes into smaller deployments",
            "Smaller changes reduce accumulated risk",
        ),
        ("Add tests for high-risk areas", "Tests mitigate risk for critical code"),
    ],
}


# =============================================================================
# ID Generation Functions
# =============================================================================


def _generate_reason_id(reason_type: BlockingReasonType, sequence: int) -> str:
    """Generate unique reason ID.

    Format: "BR-{reason_type[:3].upper()}-{seq:03d}" (e.g., "BR-LOW-001")

    Args:
        reason_type: Type of blocking reason.
        sequence: Sequence number for this reason type.

    Returns:
        Unique reason ID string.

    Example:
        >>> _generate_reason_id("low_confidence", 1)
        'BR-LOW-001'
        >>> _generate_reason_id("critical_risk", 2)
        'BR-CRI-002'
    """
    prefix = reason_type[:3].upper()
    return f"BR-{prefix}-{sequence:03d}"


def _generate_step_id(sequence: int) -> str:
    """Generate unique step ID.

    Format: "RS-{seq:03d}" (e.g., "RS-001")

    Args:
        sequence: Sequence number for this step.

    Returns:
        Unique step ID string.

    Example:
        >>> _generate_step_id(1)
        'RS-001'
        >>> _generate_step_id(10)
        'RS-010'
    """
    return f"RS-{sequence:03d}"


# =============================================================================
# Blocking Reason Generation Functions
# =============================================================================


def _generate_low_confidence_reason(score: int, threshold: int) -> BlockingReason:
    """Generate blocking reason for low confidence score.

    Args:
        score: Actual confidence score.
        threshold: Required threshold.

    Returns:
        BlockingReason for low confidence.
    """
    return BlockingReason(
        reason_id=_generate_reason_id("low_confidence", 1),
        reason_type="low_confidence",
        description=REASON_TEMPLATES["low_confidence"].format(actual=score, threshold=threshold),
        threshold_value=threshold,
        actual_value=score,
        related_findings=(f"F-CONFIDENCE-{score}",),
    )


def _generate_critical_risk_reason(risk_report: RiskReport) -> BlockingReason:
    """Generate blocking reason for critical risks.

    Args:
        risk_report: Risk report with critical findings.

    Returns:
        BlockingReason for critical risks.
    """
    # Collect finding IDs from critical risks
    related_findings = tuple(
        risk.risk_id for risk in risk_report.risks if risk.risk_level == "critical"
    )

    return BlockingReason(
        reason_id=_generate_reason_id("critical_risk", 1),
        reason_type="critical_risk",
        description=REASON_TEMPLATES["critical_risk"].format(count=risk_report.critical_count),
        threshold_value=None,
        actual_value=risk_report.critical_count,
        related_findings=related_findings,
    )


def _generate_validation_failed_reason(
    validation_results: tuple[ValidationResult, ...],
) -> BlockingReason:
    """Generate blocking reason for validation failures.

    Args:
        validation_results: Validation results to analyze.

    Returns:
        BlockingReason for validation failures.
    """
    failed_artifacts = [
        r.artifact_id for r in validation_results if r.validation_status == "failed"
    ]
    failed_count = len(failed_artifacts)
    artifacts_str = ", ".join(failed_artifacts[:3])  # Show first 3
    if failed_count > 3:
        artifacts_str += f" (+{failed_count - 3} more)"

    # Collect finding IDs from failed validations
    related_findings: list[str] = []
    for result in validation_results:
        if result.validation_status == "failed":
            for finding in result.findings:
                related_findings.append(finding.finding_id)

    return BlockingReason(
        reason_id=_generate_reason_id("validation_failed", 1),
        reason_type="validation_failed",
        description=REASON_TEMPLATES["validation_failed"].format(
            count=failed_count, artifacts=artifacts_str
        ),
        threshold_value=None,
        actual_value=failed_count,
        related_findings=tuple(related_findings),
    )


def get_high_risk_count_threshold() -> int:
    """Get threshold for blocking on accumulated high risks.

    Reads from YoloConfig if available, returns 5 as default.

    Returns:
        High risk count threshold (default 5).

    Example:
        >>> threshold = get_high_risk_count_threshold()
        >>> threshold
        5
    """
    # Future: Read from config if high_risk_threshold is added
    # For now, return sensible default
    return 5


def _generate_high_risk_count_reason(risk_report: RiskReport) -> BlockingReason | None:
    """Generate blocking reason for high risk count.

    Returns reason if high risk count exceeds configured threshold.

    Args:
        risk_report: Risk report with high-level risks.

    Returns:
        BlockingReason if high risk count exceeds threshold, None otherwise.
    """
    threshold = get_high_risk_count_threshold()

    if risk_report.high_count <= threshold:
        return None

    # Collect finding IDs from high risks
    related_findings = tuple(
        risk.risk_id for risk in risk_report.risks if risk.risk_level == "high"
    )

    return BlockingReason(
        reason_id=_generate_reason_id("high_risk_count", 1),
        reason_type="high_risk_count",
        description=REASON_TEMPLATES["high_risk_count"].format(
            actual=risk_report.high_count, threshold=threshold
        ),
        threshold_value=threshold,
        actual_value=risk_report.high_count,
        related_findings=related_findings,
    )


def generate_blocking_reasons(
    confidence_result: ConfidenceResult,
    risk_report: RiskReport,
    validation_results: tuple[ValidationResult, ...],
) -> tuple[BlockingReason, ...]:
    """Generate all blocking reasons from validation data.

    Analyzes confidence score, risk report, and validation results
    to generate comprehensive blocking reasons.

    Args:
        confidence_result: Confidence scoring result.
        risk_report: Risk categorization report.
        validation_results: Validation results from TEA.

    Returns:
        Tuple of BlockingReason instances (empty if no blocking conditions).

    Example:
        >>> reasons = generate_blocking_reasons(
        ...     confidence_result=confidence_result,
        ...     risk_report=risk_report,
        ...     validation_results=validation_results,
        ... )
        >>> len(reasons)
        2
    """
    reasons: list[BlockingReason] = []

    # Check confidence score
    if not confidence_result.passed_threshold:
        reasons.append(
            _generate_low_confidence_reason(
                score=confidence_result.score,
                threshold=confidence_result.threshold_value,
            )
        )

    # Check critical risks
    if risk_report.critical_count > 0:
        reasons.append(_generate_critical_risk_reason(risk_report))

    # Check validation failures
    failed_count = sum(1 for r in validation_results if r.validation_status == "failed")
    if failed_count > 0:
        reasons.append(_generate_validation_failed_reason(validation_results))

    # Check high risk accumulation
    high_risk_reason = _generate_high_risk_count_reason(risk_report)
    if high_risk_reason is not None:
        reasons.append(high_risk_reason)

    logger.debug(
        "blocking_reasons_generated",
        reason_count=len(reasons),
        reason_types=[r.reason_type for r in reasons],
    )

    return tuple(reasons)


# =============================================================================
# Remediation Step Generation Functions
# =============================================================================


def _generate_confidence_remediation(reason: BlockingReason) -> list[RemediationStep]:
    """Generate remediation steps for low confidence.

    Args:
        reason: The blocking reason for low confidence.

    Returns:
        List of RemediationStep instances.
    """
    steps: list[RemediationStep] = []
    templates = REMEDIATION_TEMPLATES["low_confidence"]

    for i, (action, impact) in enumerate(templates, start=1):
        steps.append(
            RemediationStep(
                step_id=_generate_step_id(len(steps) + 100 + i),  # Offset for confidence
                priority=100 + i,  # Lower priority than critical
                action=action,
                expected_impact=impact,
                related_reason_id=reason.reason_id,
            )
        )

    return steps


def _generate_risk_remediation(reason: BlockingReason) -> list[RemediationStep]:
    """Generate remediation steps for critical or high risk.

    Args:
        reason: The blocking reason for risks.

    Returns:
        List of RemediationStep instances.
    """
    steps: list[RemediationStep] = []

    if reason.reason_type == "critical_risk":
        templates = REMEDIATION_TEMPLATES["critical_risk"]
        priority_base = 1  # Highest priority
    else:
        templates = REMEDIATION_TEMPLATES["high_risk_count"]
        priority_base = 50  # Medium priority

    for i, (action, impact) in enumerate(templates, start=1):
        steps.append(
            RemediationStep(
                step_id=_generate_step_id(len(steps) + priority_base + i),
                priority=priority_base + i,
                action=action,
                expected_impact=impact,
                related_reason_id=reason.reason_id,
            )
        )

    return steps


def _generate_validation_remediation(
    reason: BlockingReason,
    validation_results: tuple[ValidationResult, ...],
) -> list[RemediationStep]:
    """Generate remediation steps for validation failures.

    Args:
        reason: The blocking reason for validation failure.
        validation_results: Validation results to analyze.

    Returns:
        List of RemediationStep instances.
    """
    steps: list[RemediationStep] = []
    templates = REMEDIATION_TEMPLATES["validation_failed"]

    # Add specific steps for failed artifacts
    failed_artifacts = [r for r in validation_results if r.validation_status == "failed"]

    for i, artifact in enumerate(failed_artifacts[:5], start=1):  # Max 5 specific steps
        steps.append(
            RemediationStep(
                step_id=_generate_step_id(200 + i),
                priority=20 + i,  # After critical, before general confidence
                action=f"Fix validation errors in {artifact.artifact_id}",
                expected_impact="Resolving validation errors allows deployment",
                related_reason_id=reason.reason_id,
            )
        )

    # Add general templates
    for i, (action, impact) in enumerate(templates, start=1):
        steps.append(
            RemediationStep(
                step_id=_generate_step_id(220 + i),
                priority=30 + i,
                action=action,
                expected_impact=impact,
                related_reason_id=reason.reason_id,
            )
        )

    return steps


def generate_remediation_steps(
    blocking_reasons: tuple[BlockingReason, ...],
    validation_results: tuple[ValidationResult, ...],
) -> tuple[RemediationStep, ...]:
    """Generate all remediation steps for blocking reasons.

    Combines remediation steps for all blocking reasons, sorted by priority.

    Args:
        blocking_reasons: Tuple of blocking reasons to remediate.
        validation_results: Validation results for context.

    Returns:
        Tuple of RemediationStep instances sorted by priority.

    Example:
        >>> steps = generate_remediation_steps(
        ...     blocking_reasons=(reason1, reason2),
        ...     validation_results=validation_results,
        ... )
        >>> steps[0].priority < steps[-1].priority
        True
    """
    all_steps: list[RemediationStep] = []

    for reason in blocking_reasons:
        if reason.reason_type == "low_confidence":
            all_steps.extend(_generate_confidence_remediation(reason))
        elif reason.reason_type in ("critical_risk", "high_risk_count"):
            all_steps.extend(_generate_risk_remediation(reason))
        elif reason.reason_type == "validation_failed":
            all_steps.extend(_generate_validation_remediation(reason, validation_results))

    # Sort by priority and reassign IDs for clean sequence
    all_steps.sort(key=lambda s: s.priority)

    # Reassign step IDs with clean sequence
    final_steps: list[RemediationStep] = []
    for i, step in enumerate(all_steps, start=1):
        final_steps.append(
            RemediationStep(
                step_id=_generate_step_id(i),
                priority=i,
                action=step.action,
                expected_impact=step.expected_impact,
                related_reason_id=step.related_reason_id,
            )
        )

    logger.debug(
        "remediation_steps_generated",
        step_count=len(final_steps),
    )

    return tuple(final_steps)


# =============================================================================
# Deployment Decision Evaluation
# =============================================================================


def get_deployment_threshold() -> int:
    """Get deployment confidence threshold from config.

    Reads from YoloConfig.quality.confidence_threshold if available.
    Returns 90 as default if not configured.

    Returns:
        Confidence threshold (0-100).

    Example:
        >>> threshold = get_deployment_threshold()
        >>> threshold
        90
    """
    try:
        from yolo_developer.config import load_config
        from yolo_developer.config.loader import ConfigurationError

        config = load_config()
        # Use test_coverage_threshold as a proxy for confidence threshold
        # The quality config might not have a dedicated confidence_threshold field
        threshold = int(config.quality.test_coverage_threshold * 100)
        if 0 < threshold <= 100:
            logger.debug("deployment_threshold_from_config", threshold=threshold)
            return threshold

    except FileNotFoundError:
        logger.debug("deployment_threshold_config_not_found_using_default")
    except ConfigurationError as e:
        logger.warning("deployment_threshold_config_error_using_default", error=str(e))
    except ImportError:
        logger.debug("deployment_threshold_config_module_unavailable_using_default")
    except Exception as e:
        logger.warning(
            "deployment_threshold_unexpected_error_using_default",
            error=str(e),
            error_type=type(e).__name__,
        )

    return 90


def is_deployment_blocking_enabled() -> bool:
    """Check if deployment blocking is enabled.

    For future config support, returns True by default.

    Returns:
        True if deployment blocking is enabled.

    Example:
        >>> is_deployment_blocking_enabled()
        True
    """
    # Future: Read from config if deployment blocking can be disabled
    return True


def evaluate_deployment_decision(
    confidence_result: ConfidenceResult,
    risk_report: RiskReport,
    validation_results: tuple[ValidationResult, ...],
) -> DeploymentDecision:
    """Evaluate deployment decision based on confidence, risks, and validation.

    Checks:
    1. Confidence score against threshold
    2. Critical risks that block deployment
    3. Failed validation results

    Args:
        confidence_result: Confidence scoring result.
        risk_report: Risk categorization report.
        validation_results: Validation results from TEA.

    Returns:
        DeploymentDecision with blocking status and recommendation.

    Example:
        >>> decision = evaluate_deployment_decision(
        ...     confidence_result=confidence_result,
        ...     risk_report=risk_report,
        ...     validation_results=validation_results,
        ... )
        >>> decision.is_blocked
        True
    """
    is_blocked = False
    recommendation: DeploymentRecommendation = "deploy"

    # Check confidence threshold
    if not confidence_result.passed_threshold:
        is_blocked = True
        recommendation = "block"
        logger.debug(
            "deployment_blocked_low_confidence",
            score=confidence_result.score,
            threshold=confidence_result.threshold_value,
        )

    # Check critical risks
    if risk_report.critical_count > 0:
        is_blocked = True
        recommendation = "block"
        logger.debug(
            "deployment_blocked_critical_risks",
            critical_count=risk_report.critical_count,
        )

    # Check validation failures
    failed_count = sum(1 for r in validation_results if r.validation_status == "failed")
    if failed_count > 0:
        is_blocked = True
        recommendation = "block"
        logger.debug(
            "deployment_blocked_validation_failed",
            failed_count=failed_count,
        )

    # Check high risk accumulation (uses configured threshold)
    high_risk_threshold = get_high_risk_count_threshold()
    if risk_report.high_count > high_risk_threshold:
        is_blocked = True
        recommendation = "block"
        logger.debug(
            "deployment_blocked_high_risk_count",
            high_count=risk_report.high_count,
        )

    # If not blocked but has warnings, set deploy_with_warnings
    if not is_blocked:
        warning_count = sum(1 for r in validation_results if r.validation_status == "warning")
        if warning_count > 0 or risk_report.high_count > 0:
            recommendation = "deploy_with_warnings"
            logger.debug(
                "deployment_with_warnings",
                warning_count=warning_count,
                high_risk_count=risk_report.high_count,
            )

    logger.info(
        "deployment_decision_evaluated",
        is_blocked=is_blocked,
        recommendation=recommendation,
    )

    return DeploymentDecision(
        is_blocked=is_blocked,
        recommendation=recommendation,
    )


# =============================================================================
# Override Handling
# =============================================================================


def create_override(
    acknowledged_by: str,
    acknowledgment_reason: str,
    blocking_reasons: tuple[BlockingReason, ...],
) -> DeploymentOverride:
    """Create override with acknowledgment data.

    Extracts acknowledged risks from blocking reasons and creates
    an override record with full audit context.

    Args:
        acknowledged_by: Who is acknowledging the override.
        acknowledgment_reason: Why override is being used.
        blocking_reasons: The blocking reasons being overridden.

    Returns:
        DeploymentOverride with full acknowledgment data.

    Example:
        >>> override = create_override(
        ...     acknowledged_by="user@example.com",
        ...     acknowledgment_reason="Critical hotfix",
        ...     blocking_reasons=(reason1, reason2),
        ... )
        >>> override.acknowledged_by
        'user@example.com'
    """
    # Extract risk descriptions from blocking reasons
    acknowledged_risks = tuple(reason.description for reason in blocking_reasons)

    override = DeploymentOverride(
        acknowledged_by=acknowledged_by,
        acknowledged_at=datetime.now(timezone.utc).isoformat(),
        acknowledgment_reason=acknowledgment_reason,
        acknowledged_risks=acknowledged_risks,
    )

    logger.warning(
        "deployment_override_created",
        acknowledged_by=acknowledged_by,
        acknowledged_risk_count=len(acknowledged_risks),
        acknowledgment_reason=acknowledgment_reason,
    )

    return override


def validate_override(override: DeploymentOverride) -> bool:
    """Validate override completeness.

    Checks that all required fields have non-empty values.

    Args:
        override: The override to validate.

    Returns:
        True if override is complete, False otherwise.

    Example:
        >>> override = DeploymentOverride(
        ...     acknowledged_by="user@example.com",
        ...     acknowledged_at="2026-01-12T10:30:00Z",
        ...     acknowledgment_reason="Test",
        ...     acknowledged_risks=("Risk 1",),
        ... )
        >>> validate_override(override)
        True
    """
    if not override.acknowledged_by or not override.acknowledged_by.strip():
        logger.warning("override_validation_failed", reason="acknowledged_by is empty")
        return False

    if not override.acknowledged_at or not override.acknowledged_at.strip():
        logger.warning("override_validation_failed", reason="acknowledged_at is empty")
        return False

    if not override.acknowledgment_reason or not override.acknowledgment_reason.strip():
        logger.warning("override_validation_failed", reason="acknowledgment_reason is empty")
        return False

    if not override.acknowledged_risks:
        logger.warning("override_validation_failed", reason="acknowledged_risks is empty")
        return False

    logger.debug("override_validation_passed")
    return True


# =============================================================================
# Report Generation
# =============================================================================


def generate_deployment_decision_report(
    confidence_result: ConfidenceResult,
    risk_report: RiskReport,
    validation_results: tuple[ValidationResult, ...],
    override: DeploymentOverride | None = None,
) -> DeploymentDecisionReport:
    """Generate complete deployment decision report.

    Evaluates deployment decision, generates blocking reasons and
    remediation steps if blocked, and builds complete report.

    Args:
        confidence_result: Confidence scoring result from Story 9.4.
        risk_report: Risk categorization report from Story 9.5.
        validation_results: Validation results from TEA validation.
        override: Optional override acknowledgment.

    Returns:
        Complete deployment decision report.

    Example:
        >>> report = generate_deployment_decision_report(
        ...     confidence_result=confidence_result,
        ...     risk_report=risk_report,
        ...     validation_results=validation_results,
        ... )
        >>> report.decision.is_blocked
        True
    """
    # Evaluate deployment decision
    decision = evaluate_deployment_decision(
        confidence_result=confidence_result,
        risk_report=risk_report,
        validation_results=validation_results,
    )

    # Generate blocking reasons if blocked
    blocking_reasons: tuple[BlockingReason, ...] = ()
    remediation_steps: tuple[RemediationStep, ...] = ()

    if decision.is_blocked:
        blocking_reasons = generate_blocking_reasons(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=validation_results,
        )
        remediation_steps = generate_remediation_steps(
            blocking_reasons=blocking_reasons,
            validation_results=validation_results,
        )

    logger.info(
        "deployment_decision_report_generated",
        is_blocked=decision.is_blocked,
        blocking_reason_count=len(blocking_reasons),
        remediation_step_count=len(remediation_steps),
        has_override=override is not None,
    )

    return DeploymentDecisionReport(
        decision=decision,
        blocking_reasons=blocking_reasons,
        remediation_steps=remediation_steps,
        override=override,
        confidence_result=confidence_result,
        risk_report=risk_report,
    )
