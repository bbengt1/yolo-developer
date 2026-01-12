"""Confidence scoring types and functions (Story 9.4).

This module provides the data types and functions used for confidence scoring:

Types:
- ConfidenceWeight: Weight configuration for score components
- ConfidenceBreakdown: Detailed breakdown of score components
- ConfidenceResult: Complete confidence scoring result

Functions:
- get_default_weights: Get default weight configuration
- validate_weights: Validate weight configuration
- _calculate_coverage_score: Calculate coverage contribution
- _calculate_test_execution_score: Calculate test execution contribution
- _calculate_validation_score: Calculate validation findings contribution
- _apply_score_modifiers: Apply bonuses and penalties
- calculate_confidence_score: Main confidence calculation
- check_deployment_threshold: Check if score meets threshold

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.tea.scoring import (
    ...     ConfidenceWeight,
    ...     ConfidenceBreakdown,
    ...     ConfidenceResult,
    ... )
    >>>
    >>> weights = ConfidenceWeight(
    ...     coverage_weight=0.4,
    ...     test_execution_weight=0.3,
    ...     validation_weight=0.3,
    ... )
    >>> weights.to_dict()
    {'coverage_weight': 0.4, ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from yolo_developer.agents.tea.coverage import CoverageReport
    from yolo_developer.agents.tea.execution import TestExecutionResult
    from yolo_developer.agents.tea.types import (
        DeploymentRecommendation,
        Finding,
        ValidationResult,
    )

logger = structlog.get_logger(__name__)


# =============================================================================
# Default Configuration
# =============================================================================

# Default weights for confidence score components
# Rationale:
# - Coverage (40%): Test coverage is the strongest predictor of deployment quality
# - Test Execution (30%): Passing tests validate actual behavior
# - Validation Findings (30%): Static analysis catches issues tests may miss
DEFAULT_WEIGHTS = None  # Initialized after ConfidenceWeight class is defined


def get_default_weights() -> ConfidenceWeight:
    """Get the default confidence weights.

    Returns the standard weight distribution:
    - Coverage: 40%
    - Test Execution: 30%
    - Validation: 30%

    Returns:
        ConfidenceWeight with default values.
    """
    return ConfidenceWeight(
        coverage_weight=0.4,
        test_execution_weight=0.3,
        validation_weight=0.3,
    )


def validate_weights(weights: ConfidenceWeight) -> bool:
    """Validate that weights sum to 1.0 within floating point tolerance.

    Args:
        weights: ConfidenceWeight to validate.

    Returns:
        True if weights are valid, False otherwise.

    Example:
        >>> weights = ConfidenceWeight(0.4, 0.3, 0.3)
        >>> validate_weights(weights)
        True
    """
    total = weights.coverage_weight + weights.test_execution_weight + weights.validation_weight
    tolerance = 0.001
    is_valid = abs(total - 1.0) < tolerance

    if not is_valid:
        logger.warning(
            "invalid_weight_configuration",
            total=total,
            expected=1.0,
            tolerance=tolerance,
        )

    return is_valid


def get_weights_from_config() -> ConfidenceWeight:
    """Get confidence weights from YoloConfig if available.

    Attempts to load weights from the configuration system. Falls back
    to default weights if configuration is unavailable or invalid.

    Returns:
        ConfidenceWeight from config or defaults.
    """
    try:
        from yolo_developer.config import load_config
        from yolo_developer.config.loader import ConfigurationError

        config = load_config()
        # Check if quality config has custom weights
        # Note: config.quality is available for future custom weight support
        _ = config.quality

        # Use default weights - custom weights would be added here
        # if quality config supported them
        logger.debug("confidence_weights_using_defaults")
        return get_default_weights()

    except FileNotFoundError:
        logger.debug("confidence_weights_config_not_found_using_defaults")
        return get_default_weights()
    except ConfigurationError as e:
        logger.warning("confidence_weights_config_error_using_defaults", error=str(e))
        return get_default_weights()
    except ImportError:
        logger.debug("confidence_weights_config_module_unavailable_using_defaults")
        return get_default_weights()
    except Exception as e:
        logger.warning(
            "confidence_weights_unexpected_error_using_defaults",
            error=str(e),
            error_type=type(e).__name__,
        )
        return get_default_weights()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class ConfidenceWeight:
    """Weight configuration for confidence score components.

    Weights must sum to 1.0 (within floating point tolerance).
    These weights determine how much each factor contributes to the
    final confidence score.

    Attributes:
        coverage_weight: Weight for test coverage contribution (default 0.4)
        test_execution_weight: Weight for test execution results (default 0.3)
        validation_weight: Weight for validation findings (default 0.3)

    Example:
        >>> weight = ConfidenceWeight(
        ...     coverage_weight=0.4,
        ...     test_execution_weight=0.3,
        ...     validation_weight=0.3,
        ... )
        >>> weight.coverage_weight
        0.4
    """

    coverage_weight: float
    test_execution_weight: float
    validation_weight: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the weights.
        """
        return {
            "coverage_weight": self.coverage_weight,
            "test_execution_weight": self.test_execution_weight,
            "validation_weight": self.validation_weight,
        }


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Detailed breakdown of confidence score components.

    Contains the individual scores for each component, their weighted
    contributions, and any penalties or bonuses applied.

    Attributes:
        coverage_score: Raw coverage score (0-100)
        test_execution_score: Raw test execution score (0-100)
        validation_score: Raw validation score (0-100)
        weighted_coverage: Coverage contribution after weighting
        weighted_test_execution: Test execution contribution after weighting
        weighted_validation: Validation contribution after weighting
        penalties: Tuple of penalty descriptions applied
        bonuses: Tuple of bonus descriptions applied
        base_score: Score before modifiers
        final_score: Final score after all modifiers (0-100)

    Example:
        >>> breakdown = ConfidenceBreakdown(
        ...     coverage_score=85.0,
        ...     test_execution_score=90.0,
        ...     validation_score=70.0,
        ...     weighted_coverage=34.0,
        ...     weighted_test_execution=27.0,
        ...     weighted_validation=21.0,
        ...     penalties=("-10 for high finding",),
        ...     bonuses=(),
        ...     base_score=82.0,
        ...     final_score=82,
        ... )
    """

    coverage_score: float
    test_execution_score: float
    validation_score: float
    weighted_coverage: float
    weighted_test_execution: float
    weighted_validation: float
    base_score: float
    final_score: int
    penalties: tuple[str, ...] = field(default_factory=tuple)
    bonuses: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the breakdown.
        """
        return {
            "coverage_score": self.coverage_score,
            "test_execution_score": self.test_execution_score,
            "validation_score": self.validation_score,
            "weighted_coverage": self.weighted_coverage,
            "weighted_test_execution": self.weighted_test_execution,
            "weighted_validation": self.weighted_validation,
            "penalties": list(self.penalties),
            "bonuses": list(self.bonuses),
            "base_score": self.base_score,
            "final_score": self.final_score,
        }


@dataclass(frozen=True)
class ConfidenceResult:
    """Complete confidence scoring result.

    Contains the final score, detailed breakdown, threshold status,
    and deployment recommendation.

    Attributes:
        score: Final confidence score (0-100)
        breakdown: Detailed component breakdown
        passed_threshold: Whether score meets deployment threshold
        threshold_value: The threshold used for comparison
        blocking_reasons: Tuple of reasons if deployment blocked (empty if passed)
        deployment_recommendation: Recommendation (deploy/deploy_with_warnings/block)
        blocking_finding: Critical Finding generated when deployment blocked (AC4)
        created_at: ISO timestamp when result was created

    Example:
        >>> result = ConfidenceResult(
        ...     score=82,
        ...     breakdown=breakdown,
        ...     passed_threshold=False,
        ...     threshold_value=90,
        ...     blocking_reasons=("Score 82 is below threshold 90",),
        ...     deployment_recommendation="block",
        ... )
        >>> result.to_dict()
        {'score': 82, ...}
    """

    score: int
    breakdown: ConfidenceBreakdown
    passed_threshold: bool
    threshold_value: int
    deployment_recommendation: DeploymentRecommendation
    blocking_reasons: tuple[str, ...] = field(default_factory=tuple)
    blocking_finding: Finding | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested breakdown.
        """
        return {
            "score": self.score,
            "breakdown": self.breakdown.to_dict(),
            "passed_threshold": self.passed_threshold,
            "threshold_value": self.threshold_value,
            "blocking_reasons": list(self.blocking_reasons),
            "deployment_recommendation": self.deployment_recommendation,
            "blocking_finding": self.blocking_finding.to_dict() if self.blocking_finding else None,
            "created_at": self.created_at,
        }


# =============================================================================
# Score Calculation Functions
# =============================================================================


def _calculate_coverage_score(coverage_report: CoverageReport | None) -> float:
    """Calculate coverage contribution to confidence score.

    Converts coverage percentage to a 0-100 score with optional boost
    for 100% coverage on critical paths.

    Args:
        coverage_report: CoverageReport from analyze_coverage, or None.

    Returns:
        Coverage score from 0-100. Returns 50 (neutral) if report is None.

    Example:
        >>> report = CoverageReport(overall_coverage=80.0, ...)
        >>> _calculate_coverage_score(report)
        80.0
    """
    if coverage_report is None:
        logger.debug("coverage_score_no_report_using_neutral")
        return 50.0

    # Base score is the overall coverage percentage
    score = coverage_report.overall_coverage

    # Critical path coverage boost: +10 if ALL critical paths have 100% coverage
    if coverage_report.critical_files_coverage:
        all_critical_covered = all(
            coverage >= 100.0 for coverage in coverage_report.critical_files_coverage.values()
        )
        if all_critical_covered:
            score = min(100.0, score + 10.0)
            logger.debug(
                "coverage_score_critical_path_bonus",
                original_score=coverage_report.overall_coverage,
                bonus=10.0,
                final_score=score,
            )

    logger.debug(
        "coverage_score_calculated",
        overall_coverage=coverage_report.overall_coverage,
        score=score,
    )

    return score


def _calculate_test_execution_score(result: TestExecutionResult | None) -> float:
    """Calculate test execution contribution to confidence score.

    Converts test pass rate to a 0-100 score with:
    - -15 penalty per execution error (capped at -45)

    Note: The +5 bonus for perfect pass rate is applied separately in
    _apply_score_modifiers() to ensure it appears in the breakdown's bonuses list.

    Args:
        result: TestExecutionResult from execute_tests, or None.

    Returns:
        Test execution score from 0-100. Returns 50 (neutral) if result is None.

    Example:
        >>> result = TestExecutionResult(passed_count=10, failed_count=0, ...)
        >>> _calculate_test_execution_score(result)
        100.0
    """
    if result is None:
        logger.debug("test_execution_score_no_result_using_neutral")
        return 50.0

    total = result.passed_count + result.failed_count + result.error_count

    if total == 0:
        logger.debug("test_execution_score_no_tests_using_neutral")
        return 50.0

    # Base score is the pass rate
    pass_rate = result.passed_count / total
    score = pass_rate * 100.0

    # Error penalty: -15 per error, capped at -45
    if result.error_count > 0:
        error_penalty = min(result.error_count * 15, 45)
        score -= error_penalty
        logger.debug(
            "test_execution_score_error_penalty",
            error_count=result.error_count,
            penalty=error_penalty,
        )

    # Note: Perfect pass rate bonus (+5) is applied in _apply_score_modifiers()

    # Clamp to 0-100
    score = max(0.0, min(100.0, score))

    logger.debug(
        "test_execution_score_calculated",
        passed=result.passed_count,
        failed=result.failed_count,
        errors=result.error_count,
        score=score,
    )

    return score


def _calculate_validation_score(
    results: tuple[ValidationResult, ...],
) -> tuple[float, list[str]]:
    """Calculate validation findings contribution to confidence score.

    Starts at 100 and deducts penalties based on finding severity:
    - Critical: -25 per finding (capped at -75 total)
    - High: -10 per finding (capped at -40 total)
    - Medium: -5 per finding (capped at -20 total)
    - Low: -2 per finding (capped at -10 total)
    - Info: -1 per finding (capped at -5 total)

    Args:
        results: Tuple of ValidationResult from TEA validation.

    Returns:
        Tuple of (validation_score, penalty_descriptions).
        Score is 0-100. Returns (100.0, []) if no findings.

    Example:
        >>> results = (ValidationResult(findings=(), ...),)
        >>> _calculate_validation_score(results)
        (100.0, [])
    """
    if not results:
        return 100.0, []

    # Collect all findings from all results
    # Import Finding for type annotation
    from yolo_developer.agents.tea.types import Finding as FindingType

    all_findings: list[FindingType] = []
    for result in results:
        all_findings.extend(result.findings)

    if not all_findings:
        return 100.0, []

    # Count findings by severity
    severity_counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
    }

    for finding in all_findings:
        if finding.severity in severity_counts:
            severity_counts[finding.severity] += 1

    # Calculate penalties with caps
    # Severity: (penalty_per_finding, max_total_penalty)
    severity_config = {
        "critical": (25, 75),
        "high": (10, 40),
        "medium": (5, 20),
        "low": (2, 10),
        "info": (1, 5),
    }

    total_penalty = 0.0
    penalty_descriptions: list[str] = []

    for severity, (per_finding, max_penalty) in severity_config.items():
        count = severity_counts[severity]
        if count > 0:
            penalty = min(count * per_finding, max_penalty)
            total_penalty += penalty
            capped_note = " (capped)" if count * per_finding > max_penalty else ""
            penalty_descriptions.append(f"-{penalty} for {count} {severity} finding(s){capped_note}")

    score = max(0.0, 100.0 - total_penalty)

    logger.debug(
        "validation_score_calculated",
        finding_count=len(all_findings),
        severity_counts=severity_counts,
        total_penalty=total_penalty,
        score=score,
    )

    return score, penalty_descriptions


def _apply_score_modifiers(
    base_score: float,
    breakdown: ConfidenceBreakdown,
    perfect_tests: bool = False,
) -> tuple[float, list[str]]:
    """Apply score modifiers (bonuses and penalties).

    Currently applies:
    - +5 bonus for perfect test pass rate (all tests pass)

    Additional modifiers can be added here as needed.

    Args:
        base_score: Score before modifiers.
        breakdown: ConfidenceBreakdown for context.
        perfect_tests: Whether all tests passed perfectly.

    Returns:
        Tuple of (modified_score, list_of_reasons).

    Example:
        >>> _apply_score_modifiers(95.0, breakdown, perfect_tests=True)
        (100.0, ["+5 for perfect test pass rate"])
    """
    score = base_score
    reasons: list[str] = []

    # Perfect test pass rate bonus
    if perfect_tests:
        score += 5.0
        reasons.append("+5 for perfect test pass rate")
        logger.debug("score_modifier_perfect_tests_bonus", bonus=5.0)

    # Clamp to 0-100
    score = max(0.0, min(100.0, score))

    return score, reasons


def check_deployment_threshold(
    score: int,
    threshold: int = 90,
) -> tuple[bool, DeploymentRecommendation, list[str], Finding | None]:
    """Check if confidence score meets deployment threshold.

    Determines deployment recommendation based on score vs threshold:
    - score >= threshold: "deploy"
    - threshold - 10 <= score < threshold: "deploy_with_warnings"
    - score < threshold - 10: "block" (generates critical Finding per AC4)

    Args:
        score: Confidence score (0-100).
        threshold: Minimum score required for deployment (default 90).

    Returns:
        Tuple of (passed, recommendation, blocking_reasons, blocking_finding).
        The blocking_finding is a Finding with severity="critical" when
        deployment is blocked (AC4 compliance), None otherwise.

    Example:
        >>> passed, rec, reasons, finding = check_deployment_threshold(95, threshold=90)
        >>> passed, rec, finding
        (True, "deploy", None)
        >>> passed, rec, reasons, finding = check_deployment_threshold(70, threshold=90)
        >>> finding.severity
        "critical"
    """
    from yolo_developer.agents.tea.types import Finding

    if score >= threshold:
        logger.debug(
            "deployment_threshold_passed",
            score=score,
            threshold=threshold,
        )
        return True, "deploy", [], None

    # Score is below threshold
    reasons = [f"Score {score} is below threshold {threshold}"]
    blocking_finding: Finding | None = None

    if score >= threshold - 10:
        # Close to threshold - warn but allow
        recommendation: DeploymentRecommendation = "deploy_with_warnings"
        reasons = [f"Score {score} is close to threshold {threshold}"]
        logger.debug(
            "deployment_threshold_warning",
            score=score,
            threshold=threshold,
            gap=threshold - score,
        )
    else:
        # Far below threshold - block (AC4: generate critical Finding)
        recommendation = "block"
        blocking_finding = Finding(
            finding_id=f"F-CONFIDENCE-{score}",
            category="test_coverage",
            severity="critical",
            description=f"Confidence score {score} is below deployment threshold {threshold}",
            location="confidence_scoring",
            remediation=f"Increase test coverage, fix test failures, or resolve validation findings to raise confidence score above {threshold}",
        )
        logger.warning(
            "deployment_threshold_blocked",
            score=score,
            threshold=threshold,
            gap=threshold - score,
            finding_id=blocking_finding.finding_id,
        )

    return False, recommendation, reasons, blocking_finding


def calculate_confidence_score(
    validation_results: tuple[ValidationResult, ...],
    coverage_report: CoverageReport | None = None,
    test_execution_result: TestExecutionResult | None = None,
    weights: ConfidenceWeight | None = None,
    threshold: int = 90,
) -> ConfidenceResult:
    """Calculate comprehensive confidence score.

    Combines coverage, test execution, and validation scores using
    weighted factors to produce a final confidence score.

    Formula:
        weighted_score = (
            coverage_score * coverage_weight +
            test_execution_score * test_execution_weight +
            validation_score * validation_weight
        )

    Args:
        validation_results: Tuple of ValidationResult from TEA validation.
        coverage_report: CoverageReport from analyze_coverage (optional).
        test_execution_result: TestExecutionResult from execute_tests (optional).
        weights: ConfidenceWeight to use (uses defaults if None).
        threshold: Deployment threshold score (default 90).

    Returns:
        ConfidenceResult with score, breakdown, and recommendation.

    Example:
        >>> result = calculate_confidence_score(
        ...     validation_results=(validation_result,),
        ...     coverage_report=coverage_report,
        ...     test_execution_result=test_result,
        ... )
        >>> result.score
        85
    """
    # Use default weights if not provided
    if weights is None:
        weights = get_default_weights()

    # Calculate individual component scores (validation returns penalties too)
    coverage_score = _calculate_coverage_score(coverage_report)
    test_execution_score = _calculate_test_execution_score(test_execution_result)
    validation_score, validation_penalties = _calculate_validation_score(validation_results)

    # Apply weights
    weighted_coverage = coverage_score * weights.coverage_weight
    weighted_test_execution = test_execution_score * weights.test_execution_weight
    weighted_validation = validation_score * weights.validation_weight

    # Calculate base score
    base_score = weighted_coverage + weighted_test_execution + weighted_validation

    # Determine if tests were perfect
    perfect_tests = False
    if test_execution_result is not None:
        total = (
            test_execution_result.passed_count
            + test_execution_result.failed_count
            + test_execution_result.error_count
        )
        if (
            total > 0
            and test_execution_result.passed_count == total
            and test_execution_result.error_count == 0
        ):
            perfect_tests = True

    # Build initial breakdown (before modifiers)
    initial_breakdown = ConfidenceBreakdown(
        coverage_score=coverage_score,
        test_execution_score=test_execution_score,
        validation_score=validation_score,
        weighted_coverage=weighted_coverage,
        weighted_test_execution=weighted_test_execution,
        weighted_validation=weighted_validation,
        base_score=base_score,
        final_score=int(base_score),  # Placeholder, updated after modifiers
    )

    # Apply modifiers
    final_score_float, modifier_reasons = _apply_score_modifiers(
        base_score, initial_breakdown, perfect_tests=perfect_tests
    )
    final_score = int(final_score_float)

    # Build final breakdown with penalties and bonuses
    breakdown = ConfidenceBreakdown(
        coverage_score=coverage_score,
        test_execution_score=test_execution_score,
        validation_score=validation_score,
        weighted_coverage=weighted_coverage,
        weighted_test_execution=weighted_test_execution,
        weighted_validation=weighted_validation,
        penalties=tuple(validation_penalties),
        bonuses=tuple(modifier_reasons),
        base_score=base_score,
        final_score=final_score,
    )

    # Check deployment threshold (AC4: generates critical Finding on block)
    passed_threshold, deployment_recommendation, blocking_reasons, blocking_finding = (
        check_deployment_threshold(final_score, threshold)
    )

    logger.info(
        "confidence_score_calculated",
        final_score=final_score,
        coverage_score=coverage_score,
        test_execution_score=test_execution_score,
        validation_score=validation_score,
        passed_threshold=passed_threshold,
        recommendation=deployment_recommendation,
        has_blocking_finding=blocking_finding is not None,
    )

    return ConfidenceResult(
        score=final_score,
        breakdown=breakdown,
        passed_threshold=passed_threshold,
        threshold_value=threshold,
        blocking_reasons=tuple(blocking_reasons),
        deployment_recommendation=deployment_recommendation,
        blocking_finding=blocking_finding,
    )
