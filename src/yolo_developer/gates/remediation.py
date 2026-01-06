"""Remediation suggestions for gate failure reports (Story 3.8 - Task 3).

This module provides a registry of remediation suggestions for various
issue types discovered during quality gate evaluation. Suggestions help
developers understand how to fix issues and improve their artifacts.

The module supports:
- Default remediation suggestions by issue type
- Gate-specific overrides for customized guidance
- Extensible registry for custom issue types

Example:
    >>> from yolo_developer.gates.remediation import get_remediation_suggestion
    >>>
    >>> # Get suggestion for a vague term issue
    >>> suggestion = get_remediation_suggestion("vague_term", "testability")
    >>> print(suggestion)
    Replace vague terms with specific, measurable criteria...

Security Note:
    Remediation suggestions may reference internal patterns and architecture.
    Review suggestions before exposing to external parties.
"""

from __future__ import annotations

# Default remediation suggestions by issue type
# These apply to all gates unless overridden
DEFAULT_REMEDIATION: dict[str, str] = {
    # Testability gate issues
    "vague_term": (
        "Replace vague terms with specific, measurable criteria. "
        "Example: Instead of 'fast', use 'responds within 500ms'."
    ),
    "no_success_criteria": (
        "Add quantifiable success criteria. "
        "Include specific metrics, percentages, or Given/When/Then format."
    ),
    # AC Measurability gate issues
    "unmeasurable_ac": (
        "Rewrite acceptance criteria with observable outcomes. "
        "Use concrete assertions that can be verified programmatically."
    ),
    "missing_assertion": (
        "Add explicit assertion statements. "
        "Each acceptance criterion should have testable conditions."
    ),
    "missing_gwt": (
        "Add Given/When/Then structure to acceptance criteria. "
        "Example: 'Given a logged-in user, When they click logout, Then they are redirected to login page'."
    ),
    "subjective_term": (
        "Replace subjective terms with measurable criteria. "
        "Instead of 'intuitive', use specific metrics like 'completes task in under 3 clicks'."
    ),
    "vague_outcome": (
        "Make the 'Then' clause specific and verifiable. "
        "Use observable actions like 'user sees confirmation dialog' or 'system displays error code'."
    ),
    # Architecture Validation gate issues
    "adr_violation": (
        "Review the referenced ADR and update implementation to comply "
        "with the architectural decision."
    ),
    "pattern_mismatch": (
        "Apply the required architectural pattern. "
        "Check architecture.md for pattern examples and usage guidelines."
    ),
    "missing_component": (
        "Add the required component per architecture specification. "
        "Ensure it follows the documented component patterns."
    ),
    # Definition of Done gate issues
    "tests_missing": (
        "Add unit tests covering the implemented functionality. "
        "Ensure critical paths and edge cases are tested."
    ),
    "coverage_gap": (
        "Increase test coverage to meet threshold. "
        "Focus on untested code paths and branch conditions."
    ),
    "documentation_missing": (
        "Add documentation for public APIs and complex logic. "
        "Include docstrings, type hints, and inline comments where needed."
    ),
    "dod_incomplete": (
        "Complete the remaining definition of done items. "
        "Review the DoD checklist and address each unchecked item."
    ),
    # Confidence Scoring gate issues
    "low_gate_score": (
        "Address failures in underlying gates to improve overall confidence. "
        "Review individual gate results for specific issues."
    ),
    "low_coverage": (
        "Increase test coverage, particularly branch coverage. "
        "Target uncovered functions and conditional branches."
    ),
    "high_risk": (
        "Mitigate identified risks before proceeding. "
        "Document risk acceptance if proceeding with known risks."
    ),
    "low_documentation": (
        "Add docstrings, README content, and inline comments. "
        "Ensure public interfaces are well-documented."
    ),
}


# Gate-specific remediation overrides
# These take precedence over default suggestions for specific gates
GATE_SPECIFIC_REMEDIATION: dict[str, dict[str, str]] = {
    # Example: testability gate could have specialized suggestions
    # "testability": {
    #     "vague_term": "Custom suggestion for testability gate vague terms..."
    # },
}


def get_remediation_suggestion(issue_type: str, gate_name: str) -> str | None:
    """Get remediation suggestion for an issue type.

    Looks up the remediation suggestion for the given issue type,
    checking gate-specific overrides first, then falling back to
    the default registry.

    Args:
        issue_type: The type of issue (e.g., "vague_term", "coverage_gap").
        gate_name: Name of the gate (for gate-specific overrides).

    Returns:
        Remediation suggestion string, or None if no suggestion found.

    Example:
        >>> suggestion = get_remediation_suggestion("vague_term", "testability")
        >>> "measurable" in suggestion.lower()
        True
        >>> get_remediation_suggestion("unknown_type", "any_gate")
        None
    """
    # Check gate-specific overrides first
    gate_overrides = GATE_SPECIFIC_REMEDIATION.get(gate_name, {})
    if issue_type in gate_overrides:
        return gate_overrides[issue_type]

    # Fall back to default
    return DEFAULT_REMEDIATION.get(issue_type)
