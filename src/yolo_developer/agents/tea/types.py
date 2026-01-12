"""Type definitions for TEA agent (Story 9.1, 9.3).

This module provides the data types used by the TEA (Test Engineering and Assurance) agent:

- ValidationStatus: Literal type for validation lifecycle status
- FindingSeverity: Literal type for finding severity levels
- FindingCategory: Literal type for finding categories
- Finding: A validation finding with severity and remediation
- ValidationResult: Complete validation result for an artifact
- TEAOutput: Complete output from TEA processing (includes test execution results)

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.tea.types import (
    ...     Finding,
    ...     ValidationResult,
    ...     TEAOutput,
    ... )
    >>>
    >>> finding = Finding(
    ...     finding_id="F001",
    ...     category="test_coverage",
    ...     severity="high",
    ...     description="Missing unit tests for auth module",
    ...     location="src/auth/handler.py",
    ...     remediation="Add unit tests for authenticate() function",
    ... )
    >>> finding.to_dict()
    {'finding_id': 'F001', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from yolo_developer.agents.tea.blocking import DeploymentDecisionReport
    from yolo_developer.agents.tea.execution import TestExecutionResult
    from yolo_developer.agents.tea.gap_analysis import GapAnalysisReport
    from yolo_developer.agents.tea.risk import OverallRiskLevel, RiskReport
    from yolo_developer.agents.tea.scoring import ConfidenceResult
    from yolo_developer.agents.tea.testability import TestabilityReport

# =============================================================================
# Literal Types
# =============================================================================

ValidationStatus = Literal[
    "pending",
    "passed",
    "failed",
    "warning",
]
"""Lifecycle status of a validation.

Values:
    pending: Artifact received but not yet validated
    passed: Validation passed with no blocking issues
    failed: Validation failed with blocking issues
    warning: Validation passed with non-blocking warnings
"""

FindingSeverity = Literal[
    "critical",
    "high",
    "medium",
    "low",
    "info",
]
"""Severity level for validation findings.

Values:
    critical: Blocking issue that must be fixed before deployment
    high: Major issue that should be fixed before deployment
    medium: Moderate issue that should be addressed soon
    low: Minor issue that would be nice to fix
    info: Informational finding with no action required
"""

FindingCategory = Literal[
    "test_coverage",
    "code_quality",
    "documentation",
    "security",
    "performance",
    "architecture",
]
"""Category classification for validation findings.

Values:
    test_coverage: Issues related to test coverage (missing tests, low coverage)
    code_quality: Issues related to code quality (long functions, deep nesting)
    documentation: Issues related to documentation (missing docstrings)
    security: Security-related concerns (potential vulnerabilities)
    performance: Performance-related concerns (inefficient patterns)
    architecture: Architecture violations (pattern deviations)
"""

DeploymentRecommendation = Literal[
    "deploy",
    "deploy_with_warnings",
    "block",
]
"""Deployment recommendation based on validation results.

Values:
    deploy: Safe to deploy, all validations passed
    deploy_with_warnings: Can deploy but with known issues to address
    block: Deployment should be blocked until issues resolved
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class Finding:
    """A validation finding with severity and remediation guidance.

    Represents a single issue or observation found during artifact validation.

    Attributes:
        finding_id: Unique identifier for the finding (e.g., "F001")
        category: Category classification of the finding
        severity: Severity level of the finding
        description: Human-readable description of the issue
        location: File path or code location where issue was found
        remediation: Suggested fix or action to address the finding
        created_at: ISO timestamp when finding was created

    Example:
        >>> finding = Finding(
        ...     finding_id="F001",
        ...     category="test_coverage",
        ...     severity="high",
        ...     description="Missing unit tests for auth module",
        ...     location="src/auth/handler.py",
        ...     remediation="Add unit tests for authenticate() function",
        ... )
        >>> finding.to_dict()
        {'finding_id': 'F001', ...}
    """

    finding_id: str
    category: FindingCategory
    severity: FindingSeverity
    description: str
    location: str
    remediation: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the finding.
        """
        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "location": self.location,
            "remediation": self.remediation,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Complete validation result for a single artifact.

    Contains validation status, findings, and recommendations for
    a single artifact (code file, test file, etc.).

    Attributes:
        artifact_id: ID or path of the artifact being validated
        validation_status: Overall validation status
        findings: Tuple of findings discovered during validation
        recommendations: List of general recommendations
        score: Validation score from 0-100
        created_at: ISO timestamp when result was created

    Example:
        >>> result = ValidationResult(
        ...     artifact_id="src/auth/handler.py",
        ...     validation_status="warning",
        ...     findings=(finding,),
        ...     recommendations=["Consider adding integration tests"],
        ...     score=75,
        ... )
        >>> result.to_dict()
        {'artifact_id': 'src/auth/handler.py', ...}
    """

    artifact_id: str
    validation_status: ValidationStatus
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    recommendations: tuple[str, ...] = field(default_factory=tuple)
    score: int = 100
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested findings.
        """
        return {
            "artifact_id": self.artifact_id,
            "validation_status": self.validation_status,
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": list(self.recommendations),
            "score": self.score,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TEAOutput:
    """Complete output from TEA agent processing.

    Contains all validation results generated during tea_node execution,
    plus overall confidence score, deployment recommendation, and risk report.

    Attributes:
        validation_results: Tuple of validation results per artifact
        processing_notes: Notes about the processing (stats, issues, etc.)
        overall_confidence: Weighted confidence score from 0.0 to 1.0
        deployment_recommendation: Recommendation for deployment decision
        test_execution_result: Test execution results (Story 9.3)
        confidence_result: Detailed confidence scoring breakdown (Story 9.4)
        risk_report: Risk categorization report with severity breakdown (Story 9.5)
        overall_risk_level: Highest risk level present (critical/high/low/none) (Story 9.5)
        testability_report: Testability audit report with patterns and score (Story 9.6)
        deployment_decision_report: Complete deployment decision report (Story 9.7)
        gap_analysis_report: Test gap analysis report with suggestions (Story 9.8)
        created_at: ISO timestamp when output was created

    Example:
        >>> output = TEAOutput(
        ...     validation_results=(result1, result2),
        ...     processing_notes="Validated 2 artifacts, 1 warning found",
        ...     overall_confidence=0.85,
        ...     deployment_recommendation="deploy_with_warnings",
        ... )
        >>> output.to_dict()
        {'validation_results': [...], 'overall_confidence': 0.85, ...}
    """

    validation_results: tuple[ValidationResult, ...] = field(default_factory=tuple)
    processing_notes: str = ""
    overall_confidence: float = 1.0
    deployment_recommendation: DeploymentRecommendation = "deploy"
    test_execution_result: TestExecutionResult | None = None
    confidence_result: ConfidenceResult | None = None
    risk_report: RiskReport | None = None
    overall_risk_level: OverallRiskLevel | None = None
    testability_report: TestabilityReport | None = None
    deployment_decision_report: DeploymentDecisionReport | None = None
    gap_analysis_report: GapAnalysisReport | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested validation results.
        """
        return {
            "validation_results": [r.to_dict() for r in self.validation_results],
            "processing_notes": self.processing_notes,
            "overall_confidence": self.overall_confidence,
            "deployment_recommendation": self.deployment_recommendation,
            "test_execution_result": self.test_execution_result.to_dict()
            if self.test_execution_result
            else None,
            "confidence_result": self.confidence_result.to_dict()
            if self.confidence_result
            else None,
            "risk_report": self.risk_report.to_dict() if self.risk_report else None,
            "overall_risk_level": self.overall_risk_level,
            "testability_report": self.testability_report.to_dict()
            if self.testability_report
            else None,
            "deployment_decision_report": self.deployment_decision_report.to_dict()
            if self.deployment_decision_report
            else None,
            "gap_analysis_report": self.gap_analysis_report.to_dict()
            if self.gap_analysis_report
            else None,
            "created_at": self.created_at,
        }
