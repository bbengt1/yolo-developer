"""Testability validation for PM agent acceptance criteria (Story 6.3).

This module provides functions to validate that acceptance criteria are
testable and measurable. It checks for:

- Vague terms that make ACs unmeasurable (AC2)
- Proper Given/When/Then structure (AC1)
- Edge case coverage (AC3)
- Appropriate AC count per story (AC4)

The main entry point is `validate_story_testability()` which returns a
`TestabilityResult` with all validation findings.

Usage:
    >>> from yolo_developer.agents.pm.testability import validate_story_testability
    >>> from yolo_developer.agents.pm.types import Story
    >>>
    >>> result = validate_story_testability(story)
    >>> if not result["is_valid"]:
    ...     print(f"Found issues: {result['vague_terms_found']}")

Architecture Note:
    This module contains synchronous validation functions (no I/O needed).
    It follows ADR-006 Quality Gate Pattern as a precursor to the
    AC Measurability Gate (Story 3.3).
"""

from __future__ import annotations

import structlog

from yolo_developer.agents.pm.llm import VAGUE_TERMS
from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    Story,
    TestabilityResult,
)

logger = structlog.get_logger(__name__)

# Re-export VAGUE_TERMS for external access (imported from pm.llm for single source of truth)
__all__ = [
    "BOUNDARY_PATTERNS",
    "EMPTY_PATTERNS",
    "ERROR_PATTERNS",
    "VAGUE_TERMS",
    "validate_story_testability",
]

# Edge case patterns for detecting coverage
ERROR_PATTERNS: frozenset[str] = frozenset(
    {
        "error",
        "fail",
        "invalid",
        "exception",
        "reject",
        "denied",
    }
)

EMPTY_PATTERNS: frozenset[str] = frozenset(
    {
        "empty",
        "null",
        "none",
        "missing",
        "blank",
        "undefined",
    }
)

BOUNDARY_PATTERNS: frozenset[str] = frozenset(
    {
        "maximum",
        "minimum",
        "limit",
        "boundary",
        "overflow",
        "threshold",
    }
)


def _detect_vague_terms(ac: AcceptanceCriterion) -> list[str]:
    """Detect vague terms in an acceptance criterion.

    Scans all text fields of the AC (given, when, then, and_clauses)
    for vague terms that make the criterion unmeasurable.

    Args:
        ac: The acceptance criterion to check.

    Returns:
        List of vague terms found (empty if none).

    Example:
        >>> ac = AcceptanceCriterion(
        ...     id="AC1",
        ...     given="system is fast",
        ...     when="user acts",
        ...     then="result is good",
        ... )
        >>> _detect_vague_terms(ac)
        ['fast', 'good']
    """
    found_terms: list[str] = []

    # Combine all text fields for scanning
    text_parts = [ac.given, ac.when, ac.then]
    text_parts.extend(ac.and_clauses)

    full_text = " ".join(text_parts).lower()

    # Check each vague term
    for term in VAGUE_TERMS:
        if term in full_text:
            found_terms.append(term)

    return found_terms


def _validate_ac_structure(ac: AcceptanceCriterion) -> list[str]:
    """Validate the structure of an acceptance criterion.

    Checks that the Given/When/Then fields are non-empty and not
    whitespace-only.

    Args:
        ac: The acceptance criterion to validate.

    Returns:
        List of structural issues found (empty if valid).

    Example:
        >>> ac = AcceptanceCriterion(id="AC1", given="", when="action", then="result")
        >>> _validate_ac_structure(ac)
        ["AC AC1: 'given' field is empty or whitespace-only"]
    """
    issues: list[str] = []

    if not ac.given or not ac.given.strip():
        issues.append(f"AC {ac.id}: 'given' field is empty or whitespace-only")

    if not ac.when or not ac.when.strip():
        issues.append(f"AC {ac.id}: 'when' field is empty or whitespace-only")

    if not ac.then or not ac.then.strip():
        issues.append(f"AC {ac.id}: 'then' field is empty or whitespace-only")

    return issues


def _check_edge_cases(story: Story) -> list[str]:
    """Check for missing edge case coverage in a story's ACs.

    Scans all acceptance criteria for patterns indicating coverage of
    common edge cases: error handling, empty input, and boundary conditions.

    Args:
        story: The story to check for edge case coverage.

    Returns:
        List of missing edge case category names (e.g., "error_handling").

    Example:
        >>> # Story without error handling ACs
        >>> result = _check_edge_cases(story)
        >>> "error_handling" in result
        True
    """
    missing: list[str] = []

    # Combine all AC text for pattern matching
    all_text_parts: list[str] = []
    for ac in story.acceptance_criteria:
        all_text_parts.extend([ac.given, ac.when, ac.then])
        all_text_parts.extend(ac.and_clauses)

    full_text = " ".join(all_text_parts).lower()

    # Check for error handling coverage
    has_error_handling = any(pattern in full_text for pattern in ERROR_PATTERNS)
    if not has_error_handling:
        missing.append("error_handling")

    # Check for empty input coverage
    has_empty_handling = any(pattern in full_text for pattern in EMPTY_PATTERNS)
    if not has_empty_handling:
        missing.append("empty_input")

    # Check for boundary condition coverage
    has_boundary_handling = any(pattern in full_text for pattern in BOUNDARY_PATTERNS)
    if not has_boundary_handling:
        missing.append("boundary")

    return missing


def _validate_ac_count(story: Story) -> str | None:
    """Validate that the story has an appropriate number of ACs.

    Returns a warning message if the AC count is unusual:
    - Too few (< 2): May indicate incomplete story
    - Too many (> 8): May indicate story should be split

    Args:
        story: The story to validate.

    Returns:
        Warning message if count is unusual, None if acceptable.

    Example:
        >>> # Story with only 1 AC
        >>> result = _validate_ac_count(story_with_1_ac)
        >>> "only 1" in result.lower()
        True
    """
    ac_count = len(story.acceptance_criteria)

    if ac_count < 2:
        return f"Story has only {ac_count} AC(s); consider adding more for completeness"

    if ac_count > 8:
        return f"Story has {ac_count} ACs; consider splitting into smaller stories"

    return None


def validate_story_testability(story: Story) -> TestabilityResult:
    """Validate a story's acceptance criteria for testability.

    Performs comprehensive validation including:
    - Structural validation (Given/When/Then format)
    - Vague term detection
    - Edge case coverage analysis
    - AC count validation

    Args:
        story: The story to validate.

    Returns:
        TestabilityResult with all validation findings.

    Example:
        >>> result = validate_story_testability(story)
        >>> if not result["is_valid"]:
        ...     print("Story has testability issues")
    """
    logger.debug(
        "validating_story_testability",
        story_id=story.id,
        ac_count=len(story.acceptance_criteria),
    )

    vague_terms_found: list[tuple[str, str]] = []
    structural_issues: list[str] = []
    validation_notes: list[str] = []

    # Validate each AC
    for ac in story.acceptance_criteria:
        # Check structure
        issues = _validate_ac_structure(ac)
        structural_issues.extend(issues)

        # Check for vague terms
        vague_terms = _detect_vague_terms(ac)
        for term in vague_terms:
            vague_terms_found.append((ac.id, term))

    # Check edge case coverage
    missing_edge_cases = _check_edge_cases(story)

    # Check AC count
    ac_count_warning = _validate_ac_count(story)

    # Build validation notes
    if structural_issues:
        validation_notes.append(f"Found {len(structural_issues)} structural issue(s)")
        validation_notes.extend(structural_issues)

    if vague_terms_found:
        unique_acs_with_vague = len({ac_id for ac_id, _ in vague_terms_found})
        validation_notes.append(f"Found vague terms in {unique_acs_with_vague} AC(s)")

    if missing_edge_cases:
        validation_notes.append(f"Missing edge case coverage: {', '.join(missing_edge_cases)}")

    if ac_count_warning:
        validation_notes.append(ac_count_warning)

    if not validation_notes:
        validation_notes.append("All testability checks passed")

    # Determine overall validity
    # is_valid is False if there are vague terms OR structural issues
    is_valid = len(vague_terms_found) == 0 and len(structural_issues) == 0

    result: TestabilityResult = {
        "is_valid": is_valid,
        "vague_terms_found": vague_terms_found,
        "missing_edge_cases": missing_edge_cases,
        "ac_count_warning": ac_count_warning,
        "validation_notes": validation_notes,
    }

    logger.info(
        "story_testability_validated",
        story_id=story.id,
        is_valid=is_valid,
        vague_term_count=len(vague_terms_found),
        structural_issue_count=len(structural_issues),
        missing_edge_case_count=len(missing_edge_cases),
    )

    return result
