"""Confidence scoring gate implementation.

This module implements the confidence scoring quality gate that calculates
a composite confidence score for deployable artifacts based on multiple factors:
- Test coverage assessment
- Gate results from all evaluated gates
- Risk assessment
- Documentation coverage

Example:
    >>> from yolo_developer.gates.gates.confidence_scoring import (
    ...     confidence_scoring_evaluator,
    ...     calculate_confidence_score,
    ...     calculate_coverage_factor,
    ... )
    >>> from yolo_developer.gates.types import GateContext
    >>>
    >>> context = GateContext(
    ...     state={"gate_results": [...], "code": {...}},
    ...     gate_name="confidence_scoring",
    ... )
    >>> result = await confidence_scoring_evaluator(context)

Security Note:
    This gate performs read-only validation of artifacts.
    It does not execute any user code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import structlog

from yolo_developer.gates.evaluators import register_evaluator
from yolo_developer.gates.threshold_resolver import resolve_threshold
from yolo_developer.gates.types import GateContext, GateResult

logger = structlog.get_logger(__name__)


# =============================================================================
# Types and Constants
# =============================================================================


@dataclass(frozen=True)
class ConfidenceFactor:
    """Individual factor contributing to confidence score.

    Immutable dataclass representing a single scoring factor.

    Attributes:
        name: Identifier for this factor (e.g., "test_coverage").
        score: Raw score from 0-100.
        weight: Weight in final calculation (0.0-1.0).
        description: Human-readable explanation of the score.
    """

    name: str
    score: int
    weight: float
    description: str


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Complete breakdown of confidence score calculation.

    Immutable dataclass containing all factors and final score.

    Attributes:
        factors: List of individual ConfidenceFactor objects.
        total_score: Simple average of all factor scores.
        weighted_score: Weighted average score (final confidence).
        threshold: Threshold for passing the gate.
        passed: Whether the weighted score meets the threshold.
    """

    factors: tuple[ConfidenceFactor, ...]
    total_score: float
    weighted_score: float
    threshold: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert breakdown to dictionary for logging and audit.

        Returns:
            Dictionary representation of the breakdown.
        """
        return {
            "factors": [
                {
                    "name": f.name,
                    "score": f.score,
                    "weight": f.weight,
                    "contribution": round(f.score * f.weight, 2),
                    "description": f.description,
                }
                for f in self.factors
            ],
            "total_score": round(self.total_score, 2),
            "weighted_score": round(self.weighted_score, 2),
            "threshold": self.threshold,
            "passed": self.passed,
        }


# Default factor weights (must sum to 1.0)
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "test_coverage": 0.30,  # 30% weight
    "gate_results": 0.35,  # 35% weight (most important)
    "risk_assessment": 0.20,  # 20% weight
    "documentation": 0.15,  # 15% weight
}

# Default confidence threshold for deployment
DEFAULT_CONFIDENCE_THRESHOLD = 90

# Risk severity scores (higher = more risk = lower score)
RISK_SEVERITY_IMPACT: dict[str, int] = {
    "critical": 40,
    "high": 25,
    "medium": 10,
    "low": 3,
}

# Tolerance for weight sum validation (allows for floating point precision)
WEIGHT_SUM_TOLERANCE = 0.01


def _validate_factor_weights(weights: dict[str, float]) -> tuple[bool, str]:
    """Validate that custom factor weights are valid.

    Args:
        weights: Dictionary of factor weights.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not weights:
        return True, ""

    # Check all required factors are present
    required_factors = set(DEFAULT_FACTOR_WEIGHTS.keys())
    provided_factors = set(weights.keys())

    missing = required_factors - provided_factors
    if missing:
        return False, f"Missing required factors: {', '.join(sorted(missing))}"

    # Check all values are valid floats between 0 and 1
    for factor, weight in weights.items():
        if not isinstance(weight, (int, float)):
            return False, f"Weight for '{factor}' must be a number, got {type(weight).__name__}"
        if weight < 0 or weight > 1:
            return False, f"Weight for '{factor}' must be between 0 and 1, got {weight}"

    # Check weights sum to approximately 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        return False, f"Weights must sum to 1.0 (got {weight_sum:.3f})"

    return True, ""


# =============================================================================
# Helper Functions for Analysis
# =============================================================================


def _extract_functions_from_content(content: str) -> list[dict[str, Any]]:
    """Extract function information from Python source code.

    Args:
        content: Python source code as string.

    Returns:
        List of dicts with function info (name, line_count, nesting_depth).
    """
    functions: list[dict[str, Any]] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return functions

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_info: dict[str, Any] = {
                "name": node.name,
                "is_private": node.name.startswith("_"),
                "has_docstring": ast.get_docstring(node) is not None,
                "line_count": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                "nesting_depth": _calculate_nesting_depth(node),
            }
            functions.append(func_info)

    return functions


def _calculate_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth in a function.

    Args:
        node: AST node to analyze.
        current_depth: Current nesting level.

    Returns:
        Maximum nesting depth found.
    """
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If | ast.For | ast.While | ast.With | ast.Try):
            child_depth = _calculate_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _calculate_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
    return max_depth


def _has_module_docstring(content: str) -> bool:
    """Check if content has a module-level docstring.

    Args:
        content: Python source code as string.

    Returns:
        True if module has a docstring.
    """
    try:
        tree = ast.parse(content)
        return ast.get_docstring(tree) is not None
    except SyntaxError:
        return False


def _count_test_functions(content: str) -> int:
    """Count test functions in content.

    Args:
        content: Python test file content.

    Returns:
        Number of test functions found.
    """
    count = 0
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return count

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("test_"):
                count += 1

    return count


def _get_remediation_for_factor(factor_name: str, score: int) -> str:
    """Get remediation guidance for a low-scoring factor.

    Args:
        factor_name: Name of the factor.
        score: Current score for the factor.

    Returns:
        Remediation guidance string.
    """
    remediations = {
        "test_coverage": "Add more unit tests for public functions and increase test file coverage",
        "gate_results": "Address failing gates and improve scores on gates with low scores",
        "risk_assessment": "Reduce code complexity, simplify deeply nested functions, and address security risks",
        "documentation": "Add docstrings to public functions and module-level documentation",
    }
    return remediations.get(factor_name, f"Improve {factor_name} to increase confidence score")


# =============================================================================
# Task 2: Test Coverage Factor
# =============================================================================


def calculate_coverage_factor(
    state: dict[str, Any], weights: dict[str, float] | None = None
) -> ConfidenceFactor:
    """Calculate the test coverage confidence factor.

    Extracts coverage from state["coverage"] if available, otherwise
    estimates coverage from code analysis.

    Args:
        state: State dictionary with optional "coverage" and "code" keys.
        weights: Optional custom weights (uses DEFAULT_FACTOR_WEIGHTS if None).

    Returns:
        ConfidenceFactor with test coverage score.
    """
    factor_weights = weights or DEFAULT_FACTOR_WEIGHTS
    weight = factor_weights.get("test_coverage", 0.30)

    # Try to get explicit coverage info
    coverage_info = state.get("coverage", {})

    if coverage_info and isinstance(coverage_info, dict):
        # Use explicit coverage data
        line_cov = coverage_info.get("line_coverage", 0)
        branch_cov = coverage_info.get("branch_coverage", 0)
        func_cov = coverage_info.get("function_coverage", 0)

        # Weight different coverage types
        if line_cov or branch_cov or func_cov:
            # Average available coverage metrics
            values = [v for v in [line_cov, branch_cov, func_cov] if v > 0]
            score = int(sum(values) / len(values)) if values else 0
            description = f"Coverage: {line_cov:.1f}% line, {branch_cov:.1f}% branch, {func_cov:.1f}% function"
        else:
            score = 0
            description = "Coverage data present but all values are zero"
    else:
        # Estimate coverage from code analysis
        code = state.get("code") or state.get("implementation", {})
        score, description = _estimate_coverage_from_code(code)

    logger.debug("calculate_coverage_factor", score=score, description=description)
    return ConfidenceFactor(
        name="test_coverage",
        score=score,
        weight=weight,
        description=description,
    )


def _estimate_coverage_from_code(code: dict[str, Any]) -> tuple[int, str]:
    """Estimate test coverage from code structure.

    Args:
        code: Code artifact dict with 'files' key.

    Returns:
        Tuple of (score, description).
    """
    if not code or not isinstance(code, dict):
        return 50, "No code available for analysis - using default score"

    files = code.get("files", [])
    if not files:
        return 50, "No files in code artifact - using default score"

    # Separate source and test files
    source_files = [f for f in files if not f.get("path", "").startswith("tests/")]
    test_files = [f for f in files if f.get("path", "").startswith("tests/")]

    if not source_files:
        return 80, "No source files to test - high confidence"

    # Count public functions in source files
    public_func_count = 0
    for src_file in source_files:
        content = src_file.get("content", "")
        functions = _extract_functions_from_content(content)
        public_func_count += sum(1 for f in functions if not f["is_private"])

    # Count test functions
    test_func_count = 0
    for test_file in test_files:
        content = test_file.get("content", "")
        test_func_count += _count_test_functions(content)

    # Calculate ratio-based score
    if public_func_count == 0:
        score = 80  # No public functions = high confidence
        description = "No public functions to test"
    elif test_func_count == 0:
        score = 20  # No tests at all
        description = f"No test functions found for {public_func_count} public functions"
    else:
        ratio = min(1.0, test_func_count / public_func_count)
        score = int(ratio * 100)
        description = f"Estimated {score}% coverage: {test_func_count} tests for {public_func_count} functions"

    return score, description


# =============================================================================
# Task 3: Gate Results Factor
# =============================================================================


def calculate_gate_factor(
    state: dict[str, Any], weights: dict[str, float] | None = None
) -> ConfidenceFactor:
    """Calculate the gate results confidence factor.

    Extracts gate results from state["gate_results"] and calculates
    a weighted pass rate.

    Args:
        state: State dictionary with "gate_results" key.
        weights: Optional custom weights (uses DEFAULT_FACTOR_WEIGHTS if None).

    Returns:
        ConfidenceFactor with gate results score.
    """
    factor_weights = weights or DEFAULT_FACTOR_WEIGHTS
    weight = factor_weights.get("gate_results", 0.35)

    gate_results = state.get("gate_results", [])

    if not gate_results or not isinstance(gate_results, list):
        # No gate results available - neutral score
        return ConfidenceFactor(
            name="gate_results",
            score=75,
            weight=weight,
            description="No gate results available - using neutral score",
        )

    # Calculate weighted pass rate and average score
    total_score = 0
    passed_count = 0
    failed_gates: list[str] = []

    for gate_result in gate_results:
        if not isinstance(gate_result, dict):
            continue

        gate_name = gate_result.get("gate_name", "unknown")
        passed = gate_result.get("passed", False)
        gate_score = gate_result.get("score", 100 if passed else 0)

        total_score += gate_score
        if passed:
            passed_count += 1
        else:
            failed_gates.append(f"{gate_name} ({gate_score})")

    total_gates = len([g for g in gate_results if isinstance(g, dict)])

    if total_gates == 0:
        return ConfidenceFactor(
            name="gate_results",
            score=75,
            weight=weight,
            description="No valid gate results - using neutral score",
        )

    # Calculate aggregate score (average of gate scores)
    avg_score = int(total_score / total_gates)

    if failed_gates:
        description = (
            f"Score {avg_score}/100 from {total_gates} gates. Failed: {', '.join(failed_gates[:3])}"
        )
        if len(failed_gates) > 3:
            description += f" (+{len(failed_gates) - 3} more)"
    else:
        description = f"All {total_gates} gates passed with average score {avg_score}/100"

    logger.debug(
        "calculate_gate_factor",
        score=avg_score,
        passed_count=passed_count,
        total_gates=total_gates,
    )

    return ConfidenceFactor(
        name="gate_results",
        score=avg_score,
        weight=weight,
        description=description,
    )


# =============================================================================
# Task 4: Risk Assessment Factor
# =============================================================================


def calculate_risk_factor(
    state: dict[str, Any], weights: dict[str, float] | None = None
) -> ConfidenceFactor:
    """Calculate the risk assessment confidence factor.

    Extracts risks from state["risks"] if available, otherwise
    assesses risks from code complexity.

    Args:
        state: State dictionary with optional "risks" and "code" keys.
        weights: Optional custom weights (uses DEFAULT_FACTOR_WEIGHTS if None).

    Returns:
        ConfidenceFactor with risk assessment score (100 = low risk).
    """
    factor_weights = weights or DEFAULT_FACTOR_WEIGHTS
    weight = factor_weights.get("risk_assessment", 0.20)

    # Try to get explicit risk info
    risks = state.get("risks", [])

    if risks and isinstance(risks, list):
        # Calculate score from explicit risks
        total_impact = 0
        risk_details: list[str] = []

        for risk in risks:
            if not isinstance(risk, dict):
                continue
            severity = risk.get("severity", "low").lower()
            impact = RISK_SEVERITY_IMPACT.get(severity, 3)
            total_impact += impact

            risk_type = risk.get("type", "unknown")
            risk_details.append(f"{risk_type}({severity})")

        # Score = 100 - total_impact, minimum 0
        score = max(0, 100 - total_impact)
        if risk_details:
            description = (
                f"Risk score {score}/100 based on {len(risks)} risks: {', '.join(risk_details[:3])}"
            )
            if len(risk_details) > 3:
                description += f" (+{len(risk_details) - 3} more)"
        else:
            description = "No valid risks found in risk data"
    else:
        # Estimate risks from code complexity
        code = state.get("code") or state.get("implementation", {})
        score, description = _estimate_risk_from_code(code)

    logger.debug("calculate_risk_factor", score=score)
    return ConfidenceFactor(
        name="risk_assessment",
        score=score,
        weight=weight,
        description=description,
    )


def _estimate_risk_from_code(code: dict[str, Any]) -> tuple[int, str]:
    """Estimate risk from code complexity.

    Args:
        code: Code artifact dict with 'files' key.

    Returns:
        Tuple of (score, description).
    """
    if not code or not isinstance(code, dict):
        return 70, "No code available for risk analysis - using default score"

    files = code.get("files", [])
    if not files:
        return 70, "No files in code artifact - using default score"

    # Only analyze source files
    source_files = [f for f in files if not f.get("path", "").startswith("tests/")]

    total_risk_score = 0
    risk_factors: list[str] = []

    for src_file in source_files:
        content = src_file.get("content", "")
        functions = _extract_functions_from_content(content)

        for func in functions:
            # Long functions are risky
            if func["line_count"] > 30:
                total_risk_score += 5
                risk_factors.append(f"long_function:{func['name']}")

            # Deep nesting is risky
            if func["nesting_depth"] > 4:
                total_risk_score += 10
                risk_factors.append(f"deep_nesting:{func['name']}")
            elif func["nesting_depth"] > 3:
                total_risk_score += 3

    # Score = 100 - total_risk, minimum 0
    score = max(0, 100 - total_risk_score)

    if risk_factors:
        description = f"Risk score {score}/100 from code analysis: {', '.join(risk_factors[:3])}"
        if len(risk_factors) > 3:
            description += f" (+{len(risk_factors) - 3} more)"
    else:
        description = f"Low risk code - score {score}/100"

    return score, description


# =============================================================================
# Task 5: Documentation Factor
# =============================================================================


def _is_documentation_file(path: str) -> bool:
    """Check if a file path is a documentation file.

    Args:
        path: File path to check.

    Returns:
        True if the file is a README or documentation file.
    """
    path_lower = path.lower()
    filename = path_lower.split("/")[-1]

    # Check for README files
    if filename.startswith("readme"):
        return True

    # Check for common documentation directories and files
    doc_patterns = ["docs/", "doc/", "documentation/"]
    if any(pattern in path_lower for pattern in doc_patterns):
        return True

    # Check for common documentation file extensions in root
    doc_extensions = [".md", ".rst", ".txt"]
    doc_names = ["contributing", "changelog", "license", "authors", "history"]
    for name in doc_names:
        for ext in doc_extensions:
            if filename == f"{name}{ext}":
                return True

    return False


def calculate_documentation_factor(
    state: dict[str, Any], weights: dict[str, float] | None = None
) -> ConfidenceFactor:
    """Calculate the documentation confidence factor.

    Checks for docstring presence in code files and README/documentation files.

    Args:
        state: State dictionary with "code" key.
        weights: Optional custom weights (uses DEFAULT_FACTOR_WEIGHTS if None).

    Returns:
        ConfidenceFactor with documentation score.
    """
    factor_weights = weights or DEFAULT_FACTOR_WEIGHTS
    weight = factor_weights.get("documentation", 0.15)

    code = state.get("code") or state.get("implementation", {})

    if not code or not isinstance(code, dict):
        return ConfidenceFactor(
            name="documentation",
            score=50,
            weight=weight,
            description="No code available for documentation analysis",
        )

    files = code.get("files", [])
    if not files:
        return ConfidenceFactor(
            name="documentation",
            score=50,
            weight=weight,
            description="No files in code artifact",
        )

    # Separate source files, test files, and documentation files
    source_files = []
    doc_files = []
    for f in files:
        path = f.get("path", "")
        if path.startswith("tests/"):
            continue
        if _is_documentation_file(path):
            doc_files.append(f)
        elif path.endswith(".py"):
            source_files.append(f)

    if not source_files:
        return ConfidenceFactor(
            name="documentation",
            score=80,
            weight=weight,
            description="No source files to document",
        )

    total_items = 0
    documented_items = 0
    missing_docs: list[str] = []

    # Check for README presence (bonus points)
    has_readme = any(f.get("path", "").lower().split("/")[-1].startswith("readme") for f in files)

    for src_file in source_files:
        content = src_file.get("content", "")
        path = src_file.get("path", "unknown")

        # Check module docstring
        total_items += 1
        if _has_module_docstring(content):
            documented_items += 1
        else:
            missing_docs.append(f"module:{path}")

        # Check function docstrings
        functions = _extract_functions_from_content(content)
        public_functions = [f for f in functions if not f["is_private"]]

        for func in public_functions:
            total_items += 1
            if func["has_docstring"]:
                documented_items += 1
            else:
                missing_docs.append(f"func:{func['name']}")

    # Add README as a documentation item
    total_items += 1
    if has_readme:
        documented_items += 1
    else:
        missing_docs.append("README file")

    if total_items == 0:
        score = 80
        description = "No items to document"
    else:
        score = int((documented_items / total_items) * 100)
        doc_file_info = f" ({len(doc_files)} doc files)" if doc_files else ""
        if missing_docs:
            description = f"Documentation {score}%: {documented_items}/{total_items} items{doc_file_info}. Missing: {', '.join(missing_docs[:3])}"
            if len(missing_docs) > 3:
                description += f" (+{len(missing_docs) - 3} more)"
        else:
            description = f"Fully documented: {documented_items}/{total_items} items{doc_file_info}"

    logger.debug(
        "calculate_documentation_factor",
        score=score,
        documented=documented_items,
        total=total_items,
    )
    return ConfidenceFactor(
        name="documentation",
        score=score,
        weight=weight,
        description=description,
    )


# =============================================================================
# Task 6: Weighted Score Calculation
# =============================================================================


def calculate_confidence_score(
    factors: list[ConfidenceFactor], threshold: int = DEFAULT_CONFIDENCE_THRESHOLD
) -> ConfidenceBreakdown:
    """Calculate the overall confidence score from factors.

    Applies weights to each factor and calculates the weighted average.

    Args:
        factors: List of ConfidenceFactor objects.
        threshold: Threshold for passing (default 90).

    Returns:
        ConfidenceBreakdown with complete score information.
    """
    if not factors:
        return ConfidenceBreakdown(
            factors=(),
            total_score=0.0,
            weighted_score=0.0,
            threshold=threshold,
            passed=False,
        )

    # Calculate total score (simple average)
    total_score = sum(f.score for f in factors) / len(factors)

    # Calculate weighted score
    weighted_sum = sum(f.score * f.weight for f in factors)
    weight_sum = sum(f.weight for f in factors)

    if weight_sum > 0:
        weighted_score = weighted_sum / weight_sum
    else:
        weighted_score = total_score

    passed = weighted_score >= threshold

    logger.debug(
        "calculate_confidence_score",
        total_score=round(total_score, 2),
        weighted_score=round(weighted_score, 2),
        threshold=threshold,
        passed=passed,
    )

    return ConfidenceBreakdown(
        factors=tuple(factors),
        total_score=total_score,
        weighted_score=weighted_score,
        threshold=threshold,
        passed=passed,
    )


# =============================================================================
# Task 7: Confidence Evaluator
# =============================================================================


async def confidence_scoring_evaluator(context: GateContext) -> GateResult:
    """Evaluate artifact confidence score.

    Main evaluator function that calculates all factors and returns a gate result.

    Args:
        context: Gate context with state containing various artifacts.

    Returns:
        GateResult indicating pass/fail with breakdown details.
    """
    state = context.state

    # Get threshold from config using threshold resolver
    # Note: config stores threshold as 0.0-1.0, but gate uses 0-100 scale
    threshold_decimal = resolve_threshold(
        gate_name="confidence_scoring",
        state=state,
        default=DEFAULT_CONFIDENCE_THRESHOLD / 100,  # 0.90
    )
    threshold = int(threshold_decimal * 100)  # Convert to 0-100 scale

    # Get quality config for custom weights
    config = state.get("config", {})
    quality_config = config.get("quality", {}) if isinstance(config, dict) else {}

    # Get custom weights if provided and validate them
    custom_weights = quality_config.get("factor_weights")
    weights = None
    if isinstance(custom_weights, dict):
        is_valid, error_msg = _validate_factor_weights(custom_weights)
        if is_valid:
            weights = custom_weights
        else:
            logger.warning(
                "invalid_custom_weights",
                error=error_msg,
                using_defaults=True,
            )

    logger.info(
        "confidence_scoring_evaluation_started",
        gate_name=context.gate_name,
        threshold=threshold,
        has_custom_weights=weights is not None,
    )

    # Calculate all factors
    factors: list[ConfidenceFactor] = [
        calculate_coverage_factor(state, weights),
        calculate_gate_factor(state, weights),
        calculate_risk_factor(state, weights),
        calculate_documentation_factor(state, weights),
    ]

    # Calculate overall score
    breakdown = calculate_confidence_score(factors, threshold)

    # Generate report
    report = generate_confidence_report(breakdown, threshold)

    logger.info(
        "confidence_scoring_evaluation_complete",
        gate_name=context.gate_name,
        weighted_score=round(breakdown.weighted_score, 2),
        threshold=threshold,
        passed=breakdown.passed,
    )

    if breakdown.passed:
        reason = f"Confidence score: {breakdown.weighted_score:.1f}/100 (threshold: {threshold}). All factors acceptable."
    else:
        reason = f"Confidence score: {breakdown.weighted_score:.1f}/100 (threshold: {threshold}). {report}"

    return GateResult(
        passed=breakdown.passed,
        gate_name=context.gate_name,
        reason=reason,
    )


# =============================================================================
# Task 8: Confidence Report Generation
# =============================================================================


def generate_confidence_report(breakdown: ConfidenceBreakdown, threshold: int) -> str:
    """Generate a human-readable confidence report.

    Args:
        breakdown: ConfidenceBreakdown with all factor scores.
        threshold: Threshold for passing.

    Returns:
        Formatted report string.
    """
    lines: list[str] = []

    # Status line
    status = "PASSED" if breakdown.passed else "BLOCKED"
    lines.append(f"Status: {status} (threshold: {threshold})")
    lines.append("")

    # Factor breakdown
    lines.append("Factor Breakdown:")
    for factor in breakdown.factors:
        contribution = factor.score * factor.weight
        lines.append(
            f"  {factor.name}: {factor.score}/100 (weight: {factor.weight:.2f}, contrib: {contribution:.1f})"
        )
        lines.append(f"    {factor.description}")

    lines.append("")
    lines.append(f"  Weighted Total: {breakdown.weighted_score:.1f}/100")

    # Improvement suggestions for low-scoring factors
    low_factors = [f for f in breakdown.factors if f.score < 70]
    if low_factors and not breakdown.passed:
        lines.append("")
        lines.append("Improvement Suggestions:")
        for factor in sorted(low_factors, key=lambda f: f.score):
            remediation = _get_remediation_for_factor(factor.name, factor.score)
            lines.append(f"  - {factor.name} ({factor.score}): {remediation}")

        # Estimate potential gain
        lines.append("")
        lines.append("To reach threshold, focus on:")
        for i, factor in enumerate(sorted(low_factors, key=lambda f: f.score)[:2], 1):
            potential = min(30, 100 - factor.score)  # Max 30 point improvement estimate
            lines.append(
                f"  {i}. Improve {factor.name} (+{int(potential * factor.weight)} points potential)"
            )

    return "\n".join(lines)


# =============================================================================
# Task 9: Register Evaluator
# =============================================================================


# Register on module import
register_evaluator("confidence_scoring", confidence_scoring_evaluator)
