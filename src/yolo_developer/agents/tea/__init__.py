"""TEA agent module for validation and quality assurance (Story 9.1, 9.3, 9.5, 9.6).

The TEA (Test Engineering and Assurance) agent is responsible for:
- Validating implementation artifacts from Dev agent
- Calculating deployment confidence scores
- Identifying quality issues and providing remediation guidance
- Blocking deployment when thresholds are not met
- Executing tests and reporting results (Story 9.3)
- Categorizing risks by severity (Critical/High/Low) with appropriate responses (Story 9.5)
- Generating risk reports with deployment blocking for critical risks (Story 9.5)
- Auditing code for testability anti-patterns (Story 9.6)

Example:
    >>> from yolo_developer.agents.tea import (
    ...     tea_node,
    ...     TEAOutput,
    ...     ValidationResult,
    ...     Finding,
    ... )
    >>>
    >>> # Create a finding
    >>> finding = Finding(
    ...     finding_id="F001",
    ...     category="test_coverage",
    ...     severity="high",
    ...     description="Missing unit tests",
    ...     location="src/auth.py",
    ...     remediation="Add unit tests for auth module",
    ... )
    >>>
    >>> # Run the TEA node
    >>> result = await tea_node(state)

Architecture:
    The tea_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O
    - Integrates with confidence_scoring gate (Story 9.1)

References:
    - ADR-001: TypedDict for internal state
    - ADR-005: LangGraph node patterns
    - ADR-006: Quality gate patterns
    - FR65-70: TEA Agent capabilities
"""

from __future__ import annotations

from yolo_developer.agents.tea.coverage import (
    CoverageReport,
    CoverageResult,
    analyze_coverage,
    check_coverage_threshold,
    get_coverage_threshold_from_config,
    get_critical_paths_from_config,
    validate_critical_paths,
)
from yolo_developer.agents.tea.execution import (
    ExecutionStatus,
    FailureType,
    TestExecutionResult,
    TestFailure,
    detect_test_issues,
    discover_tests,
    execute_tests,
    generate_test_findings,
)
from yolo_developer.agents.tea.node import tea_node
from yolo_developer.agents.tea.risk import (
    CategorizedRisk,
    OverallRiskLevel,
    RiskLevel,
    RiskReport,
    categorize_finding,
    categorize_risks,
    check_risk_deployment_blocking,
    generate_risk_report,
    get_acknowledgment_requirements,
)
from yolo_developer.agents.tea.scoring import (
    ConfidenceBreakdown,
    ConfidenceResult,
    ConfidenceWeight,
    calculate_confidence_score,
    check_deployment_threshold,
    get_default_weights,
    validate_weights,
)
from yolo_developer.agents.tea.testability import (
    TestabilityIssue,
    TestabilityMetrics,
    TestabilityPattern,
    TestabilityReport,
    TestabilityScore,
    TestabilitySeverity,
    audit_testability,
    calculate_testability_score,
    collect_testability_metrics,
    convert_testability_issues_to_findings,
    generate_testability_recommendations,
)
from yolo_developer.agents.tea.types import (
    DeploymentRecommendation,
    Finding,
    FindingCategory,
    FindingSeverity,
    TEAOutput,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "CategorizedRisk",
    "ConfidenceBreakdown",
    "ConfidenceResult",
    "ConfidenceWeight",
    "CoverageReport",
    "CoverageResult",
    "DeploymentRecommendation",
    "ExecutionStatus",
    "FailureType",
    "Finding",
    "FindingCategory",
    "FindingSeverity",
    "OverallRiskLevel",
    "RiskLevel",
    "RiskReport",
    "TEAOutput",
    "TestExecutionResult",
    "TestFailure",
    "TestabilityIssue",
    "TestabilityMetrics",
    "TestabilityPattern",
    "TestabilityReport",
    "TestabilityScore",
    "TestabilitySeverity",
    "ValidationResult",
    "ValidationStatus",
    "analyze_coverage",
    "audit_testability",
    "calculate_confidence_score",
    "calculate_testability_score",
    "categorize_finding",
    "categorize_risks",
    "check_coverage_threshold",
    "check_deployment_threshold",
    "check_risk_deployment_blocking",
    "collect_testability_metrics",
    "convert_testability_issues_to_findings",
    "detect_test_issues",
    "discover_tests",
    "execute_tests",
    "generate_risk_report",
    "generate_test_findings",
    "generate_testability_recommendations",
    "get_acknowledgment_requirements",
    "get_coverage_threshold_from_config",
    "get_critical_paths_from_config",
    "get_default_weights",
    "tea_node",
    "validate_critical_paths",
    "validate_weights",
]
