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
from dataclasses import dataclass
from typing import Any

import structlog

from yolo_developer.gates.evaluators import register_evaluator
from yolo_developer.gates.types import GateContext, GateResult

logger = structlog.get_logger(__name__)

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


@dataclass(frozen=True)
class TestabilityIssue:
    """Represents a testability issue found in a requirement.

    Attributes:
        requirement_id: ID of the requirement with the issue.
        issue_type: Type of issue (e.g., 'vague_term', 'no_success_criteria').
        description: Human-readable description of the issue.
        severity: Severity level ('blocking' or 'warning').

    Example:
        >>> issue = TestabilityIssue(
        ...     requirement_id="req-001",
        ...     issue_type="vague_term",
        ...     description="Contains vague term 'fast'",
        ...     severity="blocking",
        ... )
    """

    requirement_id: str
    issue_type: str
    description: str
    severity: str

    def to_dict(self) -> dict[str, str]:
        """Convert issue to dictionary for serialization.

        Returns:
            Dictionary representation of the issue.
        """
        return {
            "requirement_id": self.requirement_id,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
        }


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


def generate_testability_report(issues: list[TestabilityIssue]) -> str:
    """Generate a human-readable report of testability issues.

    Creates a formatted report listing all issues by requirement,
    including severity levels and remediation guidance.

    Args:
        issues: List of testability issues found.

    Returns:
        Formatted report string.

    Example:
        >>> issues = [TestabilityIssue("r1", "vague_term", "Contains 'fast'", "blocking")]
        >>> report = generate_testability_report(issues)
        >>> "r1" in report
        True
    """
    if not issues:
        return ""

    lines: list[str] = ["Testability Gate Report", "=" * 40, ""]

    # Group issues by requirement
    issues_by_req: dict[str, list[TestabilityIssue]] = {}
    for issue in issues:
        if issue.requirement_id not in issues_by_req:
            issues_by_req[issue.requirement_id] = []
        issues_by_req[issue.requirement_id].append(issue)

    for req_id, req_issues in issues_by_req.items():
        lines.append(f"Requirement: {req_id}")
        lines.append("-" * 30)

        for issue in req_issues:
            severity_marker = "[BLOCKING]" if issue.severity == "blocking" else "[WARNING]"
            lines.append(f"  {severity_marker} {issue.description}")

            # Add remediation guidance based on issue type
            if issue.issue_type == "vague_term":
                lines.append(
                    "    Suggestion: Replace vague terms with specific, measurable criteria."
                )
                lines.append("    Example: Instead of 'fast', use 'responds within 500ms'")
            elif issue.issue_type == "no_success_criteria":
                lines.append(
                    "    Suggestion: Add quantifiable success criteria to this requirement."
                )
                lines.append(
                    "    Example: Include specific metrics, percentages, or Given/When/Then format"
                )

        lines.append("")

    return "\n".join(lines)


async def testability_evaluator(context: GateContext) -> GateResult:
    """Evaluate requirements for testability.

    Checks each requirement in the state for:
    1. Vague or subjective terms
    2. Presence of measurable success criteria

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

    issues: list[TestabilityIssue] = []

    for idx, req in enumerate(requirements):
        # Validate each requirement is a dict
        if not isinstance(req, dict):
            issues.append(
                TestabilityIssue(
                    requirement_id=f"index-{idx}",
                    issue_type="invalid_structure",
                    description=f"Requirement at index {idx} is not a dict (got {type(req).__name__})",
                    severity="blocking",
                )
            )
            continue

        req_id = req.get("id", f"index-{idx}")
        content = req.get("content", "")

        # Check for vague terms
        vague_terms = detect_vague_terms(content)
        for term, _pos in vague_terms:
            issues.append(
                TestabilityIssue(
                    requirement_id=req_id,
                    issue_type="vague_term",
                    description=f"Contains vague term '{term}'",
                    severity="blocking",
                )
            )

        # Check for success criteria
        if not has_success_criteria(req):
            issues.append(
                TestabilityIssue(
                    requirement_id=req_id,
                    issue_type="no_success_criteria",
                    description="No measurable success criteria found",
                    severity="blocking",
                )
            )

    if issues:
        report = generate_testability_report(issues)
        # Create a summary reason
        failing_reqs = sorted({issue.requirement_id for issue in issues})
        reason = f"Testability check failed for {len(failing_reqs)} requirement(s): {', '.join(failing_reqs)}\n\n{report}"

        logger.warning(
            "testability_gate_failed",
            gate_name=context.gate_name,
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
        requirements_checked=len(requirements),
    )

    return GateResult(
        passed=True,
        gate_name=context.gate_name,
        reason=None,
    )


# Register the evaluator when module is imported
register_evaluator("testability", testability_evaluator)
