"""Quality threshold rejection types and validation (Story 4.7).

This module provides data models and functions for validating seed quality
against configurable thresholds and generating rejection feedback:

- QualityThreshold: Configurable threshold values for quality checks
- RejectionReason: Details about why a specific threshold failed
- RejectionResult: Complete rejection result with pass/fail and reasons
- validate_quality_thresholds: Validate metrics against thresholds
- generate_remediation_steps: Generate actionable remediation suggestions

Example:
    >>> from yolo_developer.seed.rejection import (
    ...     QualityThreshold,
    ...     validate_quality_thresholds,
    ... )
    >>> from yolo_developer.seed.report import QualityMetrics
    >>>
    >>> # Create quality metrics (normally from generate_validation_report)
    >>> metrics = QualityMetrics(
    ...     ambiguity_score=0.85,
    ...     sop_score=0.90,
    ...     extraction_score=0.80,
    ...     overall_score=0.85,
    ... )
    >>> # Check if metrics pass thresholds
    >>> thresholds = QualityThreshold()  # uses defaults
    >>> result = validate_quality_thresholds(metrics, thresholds)
    >>> result.passed
    True

Security Note:
    Rejection messages may contain information about seed quality.
    Override actions should be logged for audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from yolo_developer.seed.report import QualityMetrics, ValidationReport

logger = structlog.get_logger(__name__)


# =============================================================================
# Default Threshold Constants
# =============================================================================


# Default minimum scores for quality thresholds
DEFAULT_OVERALL_THRESHOLD = 0.70
DEFAULT_AMBIGUITY_THRESHOLD = 0.60
DEFAULT_SOP_THRESHOLD = 0.80


# =============================================================================
# Data Models
# =============================================================================


@dataclass(frozen=True)
class QualityThreshold:
    """Configurable quality threshold values.

    Immutable dataclass defining minimum acceptable scores for each
    quality dimension. Seeds with scores below these thresholds are rejected.

    Attributes:
        overall: Minimum overall quality score (default: 0.70)
        ambiguity: Minimum ambiguity score (default: 0.60)
        sop: Minimum SOP compliance score (default: 0.80)

    Example:
        >>> # Use default thresholds
        >>> thresholds = QualityThreshold()
        >>> thresholds.overall
        0.7

        >>> # Custom thresholds
        >>> strict = QualityThreshold(overall=0.85, ambiguity=0.75, sop=0.90)
    """

    overall: float = DEFAULT_OVERALL_THRESHOLD
    ambiguity: float = DEFAULT_AMBIGUITY_THRESHOLD
    sop: float = DEFAULT_SOP_THRESHOLD

    def __post_init__(self) -> None:
        """Validate threshold values are in valid range."""
        for name, value in [
            ("overall", self.overall),
            ("ambiguity", self.ambiguity),
            ("sop", self.sop),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Threshold '{name}' must be between 0.0 and 1.0, got {value}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all threshold fields.
        """
        return {
            "overall": self.overall,
            "ambiguity": self.ambiguity,
            "sop": self.sop,
        }


@dataclass(frozen=True)
class RejectionReason:
    """Details about why a specific threshold failed.

    Immutable dataclass capturing the specifics of a threshold failure,
    including the threshold name, actual score, and required score.

    Attributes:
        threshold_name: Name of the threshold that failed (overall, ambiguity, sop)
        actual_score: The actual score achieved
        required_score: The minimum required score
        description: Human-readable description of the failure

    Example:
        >>> reason = RejectionReason(
        ...     threshold_name="overall",
        ...     actual_score=0.52,
        ...     required_score=0.70,
        ...     description="Overall quality score is below minimum threshold",
        ... )
    """

    threshold_name: str
    actual_score: float
    required_score: float
    description: str = ""

    def __post_init__(self) -> None:
        """Generate description if not provided."""
        if not self.description:
            # Use object.__setattr__ since dataclass is frozen
            desc = (
                f"{self.threshold_name.title()} score {self.actual_score:.2f} "
                f"is below required {self.required_score:.2f}"
            )
            object.__setattr__(self, "description", desc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all reason fields.
        """
        return {
            "threshold_name": self.threshold_name,
            "actual_score": self.actual_score,
            "required_score": self.required_score,
            "description": self.description,
        }


@dataclass(frozen=True)
class RejectionResult:
    """Complete rejection result with pass/fail status and details.

    Immutable dataclass containing the validation outcome, any failure
    reasons, and recommended remediation steps.

    Attributes:
        passed: True if all thresholds passed, False if any failed
        reasons: Tuple of RejectionReason for each failed threshold
        recommendations: Tuple of remediation suggestions

    Example:
        >>> result = RejectionResult(
        ...     passed=False,
        ...     reasons=(RejectionReason(...),),
        ...     recommendations=("Resolve high-severity ambiguities",),
        ... )
        >>> if not result.passed:
        ...     for reason in result.reasons:
        ...         print(f"- {reason.description}")
    """

    passed: bool
    reasons: tuple[RejectionReason, ...] = ()
    recommendations: tuple[str, ...] = ()

    @property
    def failure_count(self) -> int:
        """Number of thresholds that failed."""
        return len(self.reasons)

    @property
    def has_overall_failure(self) -> bool:
        """Check if overall threshold failed."""
        return any(r.threshold_name == "overall" for r in self.reasons)

    @property
    def has_ambiguity_failure(self) -> bool:
        """Check if ambiguity threshold failed."""
        return any(r.threshold_name == "ambiguity" for r in self.reasons)

    @property
    def has_sop_failure(self) -> bool:
        """Check if SOP threshold failed."""
        return any(r.threshold_name == "sop" for r in self.reasons)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all result fields.
        """
        return {
            "passed": self.passed,
            "failure_count": self.failure_count,
            "reasons": [r.to_dict() for r in self.reasons],
            "recommendations": list(self.recommendations),
        }


# =============================================================================
# Validation Functions
# =============================================================================


def validate_quality_thresholds(
    metrics: QualityMetrics,
    thresholds: QualityThreshold | None = None,
) -> RejectionResult:
    """Validate quality metrics against configured thresholds.

    Checks each quality dimension (overall, ambiguity, SOP) against its
    corresponding threshold. All failures are collected (not just first).

    Args:
        metrics: QualityMetrics from report generation
        thresholds: QualityThreshold configuration (uses defaults if None)

    Returns:
        RejectionResult with pass/fail status and failure details

    Example:
        >>> from yolo_developer.seed.report import QualityMetrics
        >>> metrics = QualityMetrics(
        ...     ambiguity_score=0.50,
        ...     sop_score=0.70,
        ...     extraction_score=0.85,
        ...     overall_score=0.65,
        ... )
        >>> result = validate_quality_thresholds(metrics)
        >>> result.passed
        False
        >>> result.failure_count
        2
    """
    if thresholds is None:
        thresholds = QualityThreshold()

    reasons: list[RejectionReason] = []

    # Check overall score
    if metrics.overall_score < thresholds.overall:
        reasons.append(
            RejectionReason(
                threshold_name="overall",
                actual_score=metrics.overall_score,
                required_score=thresholds.overall,
            )
        )

    # Check ambiguity score
    if metrics.ambiguity_score < thresholds.ambiguity:
        reasons.append(
            RejectionReason(
                threshold_name="ambiguity",
                actual_score=metrics.ambiguity_score,
                required_score=thresholds.ambiguity,
            )
        )

    # Check SOP score
    if metrics.sop_score < thresholds.sop:
        reasons.append(
            RejectionReason(
                threshold_name="sop",
                actual_score=metrics.sop_score,
                required_score=thresholds.sop,
            )
        )

    passed = len(reasons) == 0

    logger.debug(
        "quality_threshold_validation",
        passed=passed,
        failure_count=len(reasons),
        overall_score=metrics.overall_score,
        overall_threshold=thresholds.overall,
        ambiguity_score=metrics.ambiguity_score,
        ambiguity_threshold=thresholds.ambiguity,
        sop_score=metrics.sop_score,
        sop_threshold=thresholds.sop,
    )

    return RejectionResult(
        passed=passed,
        reasons=tuple(reasons),
        recommendations=(),  # Recommendations added by generate_remediation_steps
    )


def generate_remediation_steps(
    rejection_result: RejectionResult,
    report: ValidationReport,
) -> list[str]:
    """Generate actionable remediation steps for rejected seeds.

    Analyzes the rejection reasons and validation report to produce
    specific, actionable suggestions for improving seed quality.

    Args:
        rejection_result: The RejectionResult from threshold validation
        report: The ValidationReport containing detailed quality data

    Returns:
        List of remediation step strings

    Example:
        >>> steps = generate_remediation_steps(rejection_result, report)
        >>> for i, step in enumerate(steps, 1):
        ...     print(f"{i}. {step}")
    """
    steps: list[str] = []

    # Count ambiguities by severity
    high_count = 0
    medium_count = 0
    low_count = 0

    if report.parse_result.has_ambiguities:
        for amb in report.parse_result.ambiguities:
            if amb.severity.value == "high":
                high_count += 1
            elif amb.severity.value == "medium":
                medium_count += 1
            else:
                low_count += 1

    # Count SOP conflicts by severity
    hard_count = 0
    soft_count = 0

    if report.parse_result.sop_validation:
        for conflict in report.parse_result.sop_validation.conflicts:
            if conflict.severity.value == "hard":
                hard_count += 1
            else:
                soft_count += 1

    # Generate steps based on failure reasons
    if rejection_result.has_ambiguity_failure:
        if high_count > 0:
            steps.append(f"Resolve {high_count} high-severity ambiguities")
        if medium_count > 0:
            steps.append(f"Clarify {medium_count} medium-severity ambiguities")
        if low_count > 0:
            steps.append(f"Address {low_count} low-severity ambiguities (optional)")

    if rejection_result.has_sop_failure:
        if hard_count > 0:
            steps.append(f"Fix {hard_count} hard SOP conflicts (mandatory)")
        if soft_count > 0:
            steps.append(f"Review {soft_count} soft SOP conflicts (recommended)")

    if rejection_result.has_overall_failure:
        if not rejection_result.has_ambiguity_failure and not rejection_result.has_sop_failure:
            # Overall failed but specific thresholds passed - extraction issue
            steps.append("Improve seed document clarity and structure")
            steps.append("Add more specific requirements and constraints")

    # Always add general guidance
    if steps:
        steps.append("Review and revise seed document before re-submitting")
    else:
        # No specific failures identified - shouldn't happen but handle gracefully
        steps.append("Review seed quality and address any identified issues")

    logger.debug(
        "generated_remediation_steps",
        step_count=len(steps),
        high_ambiguities=high_count,
        medium_ambiguities=medium_count,
        hard_sop_conflicts=hard_count,
        soft_sop_conflicts=soft_count,
    )

    return steps


def create_rejection_with_remediation(
    metrics: QualityMetrics,
    report: ValidationReport,
    thresholds: QualityThreshold | None = None,
) -> RejectionResult:
    """Validate thresholds and generate remediation in one step.

    Convenience function that combines threshold validation and
    remediation generation.

    Args:
        metrics: QualityMetrics from report generation
        report: ValidationReport for remediation analysis
        thresholds: QualityThreshold configuration (uses defaults if None)

    Returns:
        RejectionResult with recommendations populated

    Example:
        >>> result = create_rejection_with_remediation(metrics, report)
        >>> if not result.passed:
        ...     for rec in result.recommendations:
        ...         print(f"- {rec}")
    """
    # First validate thresholds
    validation_result = validate_quality_thresholds(metrics, thresholds)

    # If passed, return as-is
    if validation_result.passed:
        return validation_result

    # Generate remediation steps
    recommendations = generate_remediation_steps(validation_result, report)

    # Return new result with recommendations
    return RejectionResult(
        passed=validation_result.passed,
        reasons=validation_result.reasons,
        recommendations=tuple(recommendations),
    )
