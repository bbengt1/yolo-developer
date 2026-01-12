"""TEA agent module for validation and quality assurance (Story 9.1, 9.3).

The TEA (Test Engineering and Assurance) agent is responsible for:
- Validating implementation artifacts from Dev agent
- Calculating deployment confidence scores
- Identifying quality issues and providing remediation guidance
- Blocking deployment when thresholds are not met
- Executing tests and reporting results (Story 9.3)

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
    "CoverageReport",
    "CoverageResult",
    "DeploymentRecommendation",
    "ExecutionStatus",
    "FailureType",
    "Finding",
    "FindingCategory",
    "FindingSeverity",
    "TEAOutput",
    "TestExecutionResult",
    "TestFailure",
    "ValidationResult",
    "ValidationStatus",
    "analyze_coverage",
    "check_coverage_threshold",
    "detect_test_issues",
    "discover_tests",
    "execute_tests",
    "generate_test_findings",
    "get_coverage_threshold_from_config",
    "get_critical_paths_from_config",
    "tea_node",
    "validate_critical_paths",
]
