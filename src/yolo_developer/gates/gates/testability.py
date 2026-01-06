"""Testability gate evaluator for requirement validation.

This module implements the testability quality gate that validates
requirements produced by the Analyst agent. It checks for:
- Vague or subjective terms that cannot be objectively measured
- Presence of measurable success criteria
- Quantifiable outcomes

Example:
    >>> from yolo_developer.gates.gates.testability import testability_evaluator
    >>> from yolo_developer.gates.types import GateContext
    >>>
    >>> state = {"requirements": [{"id": "req-1", "content": "API responds in 500ms"}]}
    >>> context = GateContext(state=state, gate_name="testability")
    >>> result = await testability_evaluator(context)
    >>> result.passed
    True

Security Note:
    This gate performs validation only and does not modify state.
    Requirements content is processed as untrusted input.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from yolo_developer.gates.evaluators import register_evaluator
from yolo_developer.gates.report_generator import format_report_text, generate_failure_report
from yolo_developer.gates.report_types import GateIssue, Severity
from yolo_developer.gates.threshold_resolver import resolve_threshold
from yolo_developer.gates.types import GateContext, GateResult

logger = structlog.get_logger(__name__)

# Default testability threshold (80% of requirements must be testable)
DEFAULT_TESTABILITY_THRESHOLD = 0.80

# Vague terms that indicate untestable requirements
VAGUE_TERMS: tuple[str, ...] = (
    "fast",
    "quick",
    "slow",
    "easy",
    "simple",
    "complex",
    "good",
    "bad",
    "better",
    "best",
    "user-friendly",
    "user friendly",
    "intuitive",
    "efficient",
    "effective",
    "optimal",
    "robust",
    "scalable",
    "performant",
    "nice",
    "beautiful",
    "clean",
    "appropriate",
    "reasonable",
    "adequate",
    "seamless",
    "smooth",
    "natural",
    "modern",
    "innovative",
    "cutting-edge",
    "cutting edge",
    "easy to use",
    "readable",
    "maintainable",
    "flexible",
    "powerful",
    "lightweight",
    "responsive",
)

# Patterns that indicate measurable success criteria
MEASURABLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Numbers with units (e.g., "500ms", "2 hours", "10MB")
    re.compile(r"\d+\s*(?:ms|seconds?|minutes?|hours?|days?|mb|gb|kb|bytes?)", re.IGNORECASE),
    # Percentages (e.g., "95%", "80 percent")
    re.compile(r"\d+\s*(?:%|percent)", re.IGNORECASE),
    # Specific counts (e.g., "up to 10,000", "at least 5")
    re.compile(r"(?:at least|at most|up to|exactly|maximum|minimum)\s*\d+", re.IGNORECASE),
    # Given/When/Then format (non-greedy to avoid ReDoS)
    re.compile(r"\bgiven\b.*?\bwhen\b.*?\bthen\b", re.IGNORECASE | re.DOTALL),
    # Comparison operators with numbers
    re.compile(r"(?:<|>|<=|>=|==|!=)\s*\d+", re.IGNORECASE),
    # Boolean outcomes (e.g., "succeeds", "fails", "returns true")
    re.compile(r"\b(?:succeeds?|fails?|returns?\s+(?:true|false)|completes?)\b", re.IGNORECASE),
)


def detect_vague_terms(text: str) -> list[tuple[str, int]]:
    """Detect vague terms in text.

    Searches for vague or subjective terms that cannot be objectively
    measured or tested. Detection is case-insensitive.

    Args:
        text: The text to search for vague terms.

    Returns:
        List of (term, position) tuples for each vague term found.

    Example:
        >>> detect_vague_terms("The system should be fast and easy")
        [('fast', 21), ('easy', 30)]
    """
    found_terms: list[tuple[str, int]] = []
    text_lower = text.lower()

    # Sort terms by length descending to match multi-word phrases first
    sorted_terms = sorted(VAGUE_TERMS, key=len, reverse=True)

    # Track positions already matched to avoid overlapping matches
    matched_positions: set[int] = set()

    for term in sorted_terms:
        term_lower = term.lower()
        # Use word boundary matching for single words, substring for phrases
        if " " in term_lower or "-" in term_lower:
            # Multi-word phrase: use substring match
            pattern = re.compile(re.escape(term_lower), re.IGNORECASE)
        else:
            # Single word: use word boundary match
            pattern = re.compile(r"\b" + re.escape(term_lower) + r"\b", re.IGNORECASE)

        for match in pattern.finditer(text_lower):
            pos = match.start()
            # Skip if this position overlaps with an already matched term
            if pos in matched_positions:
                continue
            # Mark all positions in this match as used
            for i in range(pos, pos + len(term)):
                matched_positions.add(i)
            # Get the original case from the text
            original_term = text[pos : pos + len(term)]
            found_terms.append((original_term, pos))

    # Sort by position
    found_terms.sort(key=lambda x: x[1])
    return found_terms


def has_success_criteria(requirement: dict[str, Any]) -> bool:
    """Check if a requirement has measurable success criteria.

    A requirement has success criteria if it contains:
    - Quantifiable metrics (numbers, percentages, time bounds)
    - Observable outcomes (Given/When/Then structure)
    - Explicit success_criteria field with measurable content

    Args:
        requirement: Dictionary with at least 'content' key.

    Returns:
        True if the requirement has measurable success criteria.

    Example:
        >>> has_success_criteria({"id": "r1", "content": "API responds in 500ms"})
        True
        >>> has_success_criteria({"id": "r2", "content": "System should be fast"})
        False
    """
    # Check explicit success_criteria field first
    if requirement.get("success_criteria"):
        criteria = requirement["success_criteria"]
        if isinstance(criteria, str) and criteria.strip():
            # Check if the criteria itself is measurable
            for pattern in MEASURABLE_PATTERNS:
                if pattern.search(criteria):
                    return True

    # Check the main content
    content = requirement.get("content", "")
    if not isinstance(content, str):
        return False

    # Check for measurable patterns in content
    for pattern in MEASURABLE_PATTERNS:
        if pattern.search(content):
            return True

    return False


async def testability_evaluator(context: GateContext) -> GateResult:
    """Evaluate requirements for testability.

    Checks each requirement in the state for:
    1. Vague or subjective terms
    2. Presence of measurable success criteria

    Uses configurable threshold to determine pass/fail. The threshold
    specifies what percentage of requirements must be testable (no issues).

    Args:
        context: Gate context containing state with requirements.

    Returns:
        GateResult indicating pass/fail with detailed reason.

    Example:
        >>> context = GateContext(
        ...     state={"requirements": [{"id": "r1", "content": "API responds in 500ms"}]},
        ...     gate_name="testability",
        ... )
        >>> result = await testability_evaluator(context)
        >>> result.passed
        True
    """
    # Get threshold from config using threshold resolver
    threshold = resolve_threshold(
        gate_name="testability",
        state=context.state,
        default=DEFAULT_TESTABILITY_THRESHOLD,
    )

    requirements = context.state.get("requirements", [])

    # Validate requirements is a list
    if not isinstance(requirements, list):
        logger.warning(
            "testability_gate_invalid_input",
            gate_name=context.gate_name,
            reason="requirements must be a list",
            actual_type=type(requirements).__name__,
        )
        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason=f"Invalid input: requirements must be a list, got {type(requirements).__name__}",
        )

    # Empty or missing requirements - nothing to validate
    if not requirements:
        logger.info(
            "testability_gate_passed",
            gate_name=context.gate_name,
            reason="No requirements to validate",
        )
        return GateResult(
            passed=True,
            gate_name=context.gate_name,
            reason=None,
        )

    issues: list[GateIssue] = []

    for idx, req in enumerate(requirements):
        # Validate each requirement is a dict
        if not isinstance(req, dict):
            issues.append(
                GateIssue(
                    location=f"index-{idx}",
                    issue_type="invalid_structure",
                    description=f"Requirement at index {idx} is not a dict (got {type(req).__name__})",
                    severity=Severity.BLOCKING,
                )
            )
            continue

        req_id = req.get("id", f"index-{idx}")
        content = req.get("content", "")

        # Check for vague terms
        vague_terms = detect_vague_terms(content)
        for term, _pos in vague_terms:
            issues.append(
                GateIssue(
                    location=req_id,
                    issue_type="vague_term",
                    description=f"Contains vague term '{term}'",
                    severity=Severity.BLOCKING,
                )
            )

        # Check for success criteria
        if not has_success_criteria(req):
            issues.append(
                GateIssue(
                    location=req_id,
                    issue_type="no_success_criteria",
                    description="No measurable success criteria found",
                    severity=Severity.BLOCKING,
                )
            )

    # Calculate testability score
    # Count requirements with issues (each unique location counts as 1 failure)
    failing_req_ids = {issue.location for issue in issues}
    passing_count = len(requirements) - len(failing_req_ids)
    score = passing_count / len(requirements) if requirements else 1.0

    # Check against threshold
    passed = score >= threshold
    threshold_percent = int(threshold * 100)
    score_percent = int(score * 100)

    if not passed:
        report = generate_failure_report(
            gate_name=context.gate_name,
            issues=issues,
            score=score,
            threshold=threshold,
        )
        formatted_report = format_report_text(report)
        failing_reqs = sorted(failing_req_ids)
        reason = (
            f"Testability score {score_percent}% below threshold {threshold_percent}%. "
            f"{len(failing_reqs)} of {len(requirements)} requirement(s) failed: "
            f"{', '.join(failing_reqs)}\n\n{formatted_report}"
        )

        logger.warning(
            "testability_gate_failed",
            gate_name=context.gate_name,
            score=score,
            threshold=threshold,
            issue_count=len(issues),
            failing_requirements=failing_reqs,
        )

        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason=reason,
        )

    logger.info(
        "testability_gate_passed",
        gate_name=context.gate_name,
        score=score,
        threshold=threshold,
        requirements_checked=len(requirements),
    )

    # Include score info even on pass
    pass_reason: str | None = None
    if issues:
        report = generate_failure_report(
            gate_name=context.gate_name,
            issues=issues,
            score=score,
            threshold=threshold,
        )
        formatted_report = format_report_text(report)
        pass_reason = (
            f"Testability score {score_percent}% meets threshold {threshold_percent}%. "
            f"Note: {len(failing_req_ids)} requirement(s) have issues.\n\n{formatted_report}"
        )

    return GateResult(
        passed=True,
        gate_name=context.gate_name,
        reason=pass_reason,
    )


# Register the evaluator when module is imported
register_evaluator("testability", testability_evaluator)
