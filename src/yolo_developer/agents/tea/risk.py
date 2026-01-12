"""Risk categorization types and functions (Story 9.5).

This module provides the data types and functions used for risk categorization:

Types:
- RiskLevel: Literal type for risk levels (critical/high/low)
- OverallRiskLevel: Literal type for overall risk (critical/high/low/none)
- CategorizedRisk: A categorized risk derived from a validation finding
- RiskReport: Complete risk categorization report

Functions:
- _map_severity_to_risk_level: Map finding severity to risk level
- categorize_finding: Categorize a single finding into a risk
- categorize_risks: Categorize all findings from validation results
- generate_risk_report: Generate a complete risk report
- check_risk_deployment_blocking: Check if risks should block deployment
- get_acknowledgment_requirements: Get items requiring acknowledgment

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.tea.risk import (
    ...     RiskLevel,
    ...     CategorizedRisk,
    ...     RiskReport,
    ... )
    >>>
    >>> # Risk levels: critical, high, low
    >>> level: RiskLevel = "critical"
    >>> level
    'critical'

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
    from yolo_developer.agents.tea.types import (
        Finding,
        FindingCategory,
        FindingSeverity,
        ValidationResult,
    )

logger = structlog.get_logger(__name__)

# =============================================================================
# Literal Types
# =============================================================================

RiskLevel = Literal[
    "critical",
    "high",
    "low",
]
"""Risk level for categorized risks.

Values:
    critical: Blocks deployment automatically
    high: Requires explicit acknowledgment before deployment
    low: Noted but doesn't block deployment
"""

OverallRiskLevel = Literal[
    "critical",
    "high",
    "low",
    "none",
]
"""Overall risk level for a risk report.

Values:
    critical: At least one critical risk present
    high: At least one high risk present (no critical)
    low: Only low risks present
    none: No risks found
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CategorizedRisk:
    """A categorized risk derived from a validation finding.

    Represents a single risk derived from a validation finding, with
    severity-based categorization and impact assessment.

    Attributes:
        risk_id: Unique identifier for the risk (e.g., "R-F001")
        finding: The original Finding that generated this risk
        risk_level: Categorized risk level (critical/high/low)
        impact_description: Human-readable impact description
        requires_acknowledgment: Whether deployment requires acknowledging this risk

    Example:
        >>> from yolo_developer.agents.tea.types import Finding
        >>> finding = Finding(
        ...     finding_id="F001",
        ...     category="test_coverage",
        ...     severity="critical",
        ...     description="Missing unit tests",
        ...     location="src/auth.py",
        ...     remediation="Add unit tests",
        ... )
        >>> risk = CategorizedRisk(
        ...     risk_id="R-F001",
        ...     finding=finding,
        ...     risk_level="critical",
        ...     impact_description="Critical test coverage gap",
        ...     requires_acknowledgment=False,
        ... )
        >>> risk.to_dict()
        {'risk_id': 'R-F001', ...}
    """

    risk_id: str
    finding: Finding
    risk_level: RiskLevel
    impact_description: str
    requires_acknowledgment: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested finding.
        """
        return {
            "risk_id": self.risk_id,
            "finding": self.finding.to_dict(),
            "risk_level": self.risk_level,
            "impact_description": self.impact_description,
            "requires_acknowledgment": self.requires_acknowledgment,
        }


@dataclass(frozen=True)
class RiskReport:
    """Complete risk categorization report.

    Contains all categorized risks, counts by level, overall risk assessment,
    and deployment blocking information.

    Attributes:
        risks: Tuple of all categorized risks
        critical_count: Count of critical-level risks
        high_count: Count of high-level risks
        low_count: Count of low-level risks
        overall_risk_level: Highest risk level present (or "none")
        deployment_blocked: Whether deployment should be blocked
        blocking_reasons: Reasons for blocking (if blocked)
        acknowledgment_required: Items requiring explicit acknowledgment
        created_at: ISO timestamp when report was created

    Example:
        >>> report = RiskReport(
        ...     risks=(),
        ...     critical_count=0,
        ...     high_count=0,
        ...     low_count=0,
        ...     overall_risk_level="none",
        ...     deployment_blocked=False,
        ... )
        >>> report.to_dict()
        {'risks': [], 'critical_count': 0, ...}
    """

    risks: tuple[CategorizedRisk, ...]
    critical_count: int
    high_count: int
    low_count: int
    overall_risk_level: OverallRiskLevel
    deployment_blocked: bool
    blocking_reasons: tuple[str, ...] = field(default_factory=tuple)
    acknowledgment_required: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested risks.
        """
        return {
            "risks": [r.to_dict() for r in self.risks],
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "low_count": self.low_count,
            "overall_risk_level": self.overall_risk_level,
            "deployment_blocked": self.deployment_blocked,
            "blocking_reasons": list(self.blocking_reasons),
            "acknowledgment_required": list(self.acknowledgment_required),
            "created_at": self.created_at,
        }


# =============================================================================
# Impact Description Templates
# =============================================================================

# Generate impact descriptions based on category and severity
IMPACT_TEMPLATES: dict[str, dict[str, str]] = {
    "test_coverage": {
        "critical": "Critical test coverage gap may allow undetected bugs in production",
        "high": "Significant test coverage gap increases risk of regressions",
        "low": "Minor test coverage improvement recommended",
    },
    "code_quality": {
        "critical": "Critical code quality issue may cause runtime failures",
        "high": "Code quality concern could impact maintainability",
        "low": "Minor code quality improvement suggested",
    },
    "documentation": {
        "critical": "Missing critical documentation blocks understanding",
        "high": "Documentation gap affects developer experience",
        "low": "Documentation enhancement would improve clarity",
    },
    "security": {
        "critical": "Critical security vulnerability must be addressed immediately",
        "high": "Security concern requires review before deployment",
        "low": "Minor security improvement recommended",
    },
    "performance": {
        "critical": "Critical performance issue will impact user experience",
        "high": "Performance concern should be addressed",
        "low": "Minor performance optimization opportunity",
    },
    "architecture": {
        "critical": "Critical architecture violation breaks system design",
        "high": "Architecture deviation needs justification",
        "low": "Minor architecture refinement suggested",
    },
}


# =============================================================================
# Risk Categorization Functions
# =============================================================================


def _map_severity_to_risk_level(severity: FindingSeverity) -> RiskLevel:
    """Map finding severity to risk level.

    Maps the five-level finding severity to three-level risk categorization:
    - critical severity -> critical risk (blocks deployment)
    - high severity -> high risk (requires acknowledgment)
    - medium, low, info severity -> low risk (noted but doesn't block)

    This mapping is designed to:
    1. Ensure critical issues block deployment automatically
    2. Require explicit acknowledgment for high-severity issues
    3. Allow low-impact issues to be noted without blocking

    Args:
        severity: Finding severity level from validation.

    Returns:
        Corresponding risk level.

    Example:
        >>> _map_severity_to_risk_level("critical")
        'critical'
        >>> _map_severity_to_risk_level("medium")
        'low'
    """
    if severity == "critical":
        return "critical"
    elif severity == "high":
        return "high"
    else:
        # medium, low, info all map to low risk
        return "low"


def _get_impact_description(category: FindingCategory, risk_level: RiskLevel) -> str:
    """Get impact description based on category and risk level.

    Args:
        category: Finding category (test_coverage, security, etc.)
        risk_level: Categorized risk level

    Returns:
        Human-readable impact description.
    """
    category_templates = IMPACT_TEMPLATES.get(category, IMPACT_TEMPLATES["code_quality"])
    return category_templates.get(risk_level, category_templates["low"])


def categorize_finding(finding: Finding) -> CategorizedRisk:
    """Categorize a single finding into a risk.

    Creates a CategorizedRisk from a Finding by:
    1. Mapping severity to risk level
    2. Generating unique risk_id from finding_id
    3. Creating impact description from category and severity
    4. Setting acknowledgment requirements (True for high risks)

    Args:
        finding: The validation finding to categorize.

    Returns:
        CategorizedRisk derived from the finding.

    Example:
        >>> from yolo_developer.agents.tea.types import Finding
        >>> finding = Finding(
        ...     finding_id="F001",
        ...     category="security",
        ...     severity="critical",
        ...     description="SQL injection vulnerability",
        ...     location="src/db.py",
        ...     remediation="Use parameterized queries",
        ... )
        >>> risk = categorize_finding(finding)
        >>> risk.risk_level
        'critical'
    """
    risk_level = _map_severity_to_risk_level(finding.severity)
    risk_id = f"R-{finding.finding_id}"
    impact_description = _get_impact_description(finding.category, risk_level)

    # High risks require acknowledgment, critical blocks, low just notes
    requires_acknowledgment = risk_level == "high"

    logger.debug(
        "finding_categorized",
        finding_id=finding.finding_id,
        severity=finding.severity,
        risk_level=risk_level,
        requires_acknowledgment=requires_acknowledgment,
    )

    return CategorizedRisk(
        risk_id=risk_id,
        finding=finding,
        risk_level=risk_level,
        impact_description=impact_description,
        requires_acknowledgment=requires_acknowledgment,
    )


def categorize_risks(
    validation_results: tuple[ValidationResult, ...],
) -> tuple[CategorizedRisk, ...]:
    """Categorize all findings from validation results into risks.

    Extracts all findings from all validation results and categorizes
    each one into a CategorizedRisk.

    Args:
        validation_results: Tuple of validation results containing findings.

    Returns:
        Tuple of categorized risks (empty if no findings).

    Example:
        >>> from yolo_developer.agents.tea.types import ValidationResult
        >>> results = (ValidationResult(artifact_id="test.py", validation_status="passed"),)
        >>> risks = categorize_risks(results)
        >>> len(risks)
        0
    """
    if not validation_results:
        return ()

    categorized: list[CategorizedRisk] = []

    for result in validation_results:
        for finding in result.findings:
            risk = categorize_finding(finding)
            categorized.append(risk)

    logger.debug(
        "risks_categorized",
        total_validation_results=len(validation_results),
        total_risks=len(categorized),
    )

    return tuple(categorized)


def generate_risk_report(
    categorized_risks: tuple[CategorizedRisk, ...],
) -> RiskReport:
    """Generate a complete risk report from categorized risks.

    Creates a RiskReport with:
    - Counts by risk level (critical, high, low)
    - Overall risk level (highest present, or "none")
    - Deployment blocking status (True if any critical)
    - Blocking reasons for critical risks
    - Acknowledgment requirements for high risks

    Args:
        categorized_risks: Tuple of categorized risks to report on.

    Returns:
        Complete risk report.

    Example:
        >>> report = generate_risk_report(())
        >>> report.overall_risk_level
        'none'
        >>> report.deployment_blocked
        False
    """
    if not categorized_risks:
        return RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
            blocking_reasons=(),
            acknowledgment_required=(),
        )

    # Count by level
    critical_count = sum(1 for r in categorized_risks if r.risk_level == "critical")
    high_count = sum(1 for r in categorized_risks if r.risk_level == "high")
    low_count = sum(1 for r in categorized_risks if r.risk_level == "low")

    # Determine overall risk level (highest present)
    # Note: Since we return early for empty categorized_risks, and every
    # CategorizedRisk has a risk_level of critical/high/low, at least one
    # count must be > 0 when we reach this point.
    overall_risk_level: OverallRiskLevel
    if critical_count > 0:
        overall_risk_level = "critical"
    elif high_count > 0:
        overall_risk_level = "high"
    else:
        # Must be low_count > 0 since categorized_risks is non-empty
        overall_risk_level = "low"

    # Determine deployment blocking (critical risks block)
    deployment_blocked = critical_count > 0

    # Generate blocking reasons for critical risks
    blocking_reasons: list[str] = []
    if deployment_blocked:
        for risk in categorized_risks:
            if risk.risk_level == "critical":
                blocking_reasons.append(f"Critical risk {risk.risk_id}: {risk.impact_description}")

    # Generate acknowledgment requirements for high risks
    acknowledgment_required: list[str] = []
    for risk in categorized_risks:
        if risk.risk_level == "high":
            acknowledgment_required.append(
                f"{risk.risk_id}: {risk.impact_description} at {risk.finding.location}"
            )

    logger.info(
        "risk_report_generated",
        critical_count=critical_count,
        high_count=high_count,
        low_count=low_count,
        overall_risk_level=overall_risk_level,
        deployment_blocked=deployment_blocked,
    )

    return RiskReport(
        risks=categorized_risks,
        critical_count=critical_count,
        high_count=high_count,
        low_count=low_count,
        overall_risk_level=overall_risk_level,
        deployment_blocked=deployment_blocked,
        blocking_reasons=tuple(blocking_reasons),
        acknowledgment_required=tuple(acknowledgment_required),
    )


def check_risk_deployment_blocking(
    risk_report: RiskReport,
) -> tuple[bool, list[str]]:
    """Check if risks should block deployment.

    Examines the risk report to determine if deployment should be blocked.
    Deployment is blocked if there are any critical risks.

    Args:
        risk_report: The risk report to check.

    Returns:
        Tuple of (is_blocked, blocking_reasons).
        is_blocked is True if deployment should be blocked.
        blocking_reasons contains specific reasons if blocked.

    Example:
        >>> report = generate_risk_report(())
        >>> is_blocked, reasons = check_risk_deployment_blocking(report)
        >>> is_blocked
        False
    """
    if risk_report.critical_count > 0:
        logger.warning(
            "deployment_blocked_by_critical_risks",
            critical_count=risk_report.critical_count,
        )
        return True, list(risk_report.blocking_reasons)

    return False, []


def get_acknowledgment_requirements(
    risk_report: RiskReport,
) -> tuple[str, ...]:
    """Get items requiring explicit acknowledgment before deployment.

    Returns the list of high-risk items that require explicit acknowledgment
    before deployment can proceed.

    Args:
        risk_report: The risk report to extract requirements from.

    Returns:
        Tuple of acknowledgment requirement strings (empty if none).

    Example:
        >>> report = generate_risk_report(())
        >>> requirements = get_acknowledgment_requirements(report)
        >>> len(requirements)
        0
    """
    return risk_report.acknowledgment_required
