"""AC Measurability gate evaluator for acceptance criteria validation.

This module implements the AC measurability quality gate that validates
acceptance criteria produced by the PM agent. It checks for:
- Given/When/Then structure (blocking on missing)
- Subjective terms that cannot be objectively measured (warning)
- Concrete, verifiable conditions in the "Then" clause

Example:
    >>> from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
    >>> from yolo_developer.gates.types import GateContext
    >>>
    >>> state = {"stories": [{"id": "s-1", "acceptance_criteria": [
    ...     {"content": "Given a user, When they login, Then they see dashboard"}
    ... ]}]}
    >>> context = GateContext(state=state, gate_name="ac_measurability")
    >>> result = await ac_measurability_evaluator(context)
    >>> result.passed
    True

Security Note:
    This gate performs validation only and does not modify state.
    Story content is processed as untrusted input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

from yolo_developer.gates.evaluators import register_evaluator
from yolo_developer.gates.types import GateContext, GateResult

logger = structlog.get_logger(__name__)

# Subjective terms that indicate unmeasurable acceptance criteria
# These trigger warnings (not blocking) when GWT structure is present
SUBJECTIVE_TERMS: tuple[str, ...] = (
    "intuitive",
    "user-friendly",
    "user friendly",
    "easy",
    "simple",
    "clean",
    "appropriate",
    "reasonable",
    "good",
    "nice",
    "beautiful",
    "elegant",
    "proper",
    "adequate",
    "sufficient",
    "efficient",
    "effective",
    "optimal",
    "seamless",
    "smooth",
    "natural",
    "robust",
    "flexible",
    "powerful",
    "modern",
    "improved",
    "better",
    "enhanced",
    "usable",
    "friendly",
    "fast",
    "quick",
    "slow",
    "easy to use",
    "hard to use",
)

# Pre-sorted subjective terms by length descending for efficient multi-word phrase matching
# Sorted at module load time to avoid repeated sorting on every function call
_SUBJECTIVE_TERMS_SORTED: tuple[str, ...] = tuple(sorted(SUBJECTIVE_TERMS, key=len, reverse=True))

# Patterns for detecting Given/When/Then structure
GWT_PATTERNS: dict[str, re.Pattern[str]] = {
    "given": re.compile(r"\bgiven\b", re.IGNORECASE),
    "when": re.compile(r"\bwhen\b", re.IGNORECASE),
    "then": re.compile(r"\bthen\b", re.IGNORECASE),
}

# Patterns that indicate concrete, verifiable conditions
CONCRETE_CONDITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Specific UI outcomes (user sees, displays, shows)
    re.compile(r"\b(?:user\s+)?(?:sees?|displays?|shows?|appears?)\b", re.IGNORECASE),
    # State changes (is created, is updated, is deleted)
    re.compile(r"\bis\s+(?:created|updated|deleted|saved|removed|added)\b", re.IGNORECASE),
    # Navigation (redirected, navigated, taken to)
    re.compile(r"\b(?:redirected|navigated|taken)\s+to\b", re.IGNORECASE),
    # Error handling (error message, error is displayed)
    re.compile(r"\berror\s+(?:message|is\s+displayed)\b", re.IGNORECASE),
    # Specific values or states
    re.compile(r"\b(?:equals?|contains?|includes?|matches?)\b", re.IGNORECASE),
    # Numbers with context (at least, at most, exactly)
    re.compile(r"(?:at\s+least|at\s+most|exactly|maximum|minimum)\s*\d+", re.IGNORECASE),
    # Boolean outcomes
    re.compile(r"\b(?:succeeds?|fails?|passes?|completes?|returns?)\b", re.IGNORECASE),
    # Specific messages or text
    re.compile(r"['\"][^'\"]+['\"]", re.IGNORECASE),
    # Enabled/disabled states
    re.compile(r"\b(?:is\s+)?(?:enabled|disabled|visible|hidden|active|inactive)\b", re.IGNORECASE),
    # Count expectations
    re.compile(r"\d+\s+(?:items?|records?|results?|rows?|entries?)", re.IGNORECASE),
)


@dataclass(frozen=True)
class ACMeasurabilityIssue:
    """Represents a measurability issue found in an acceptance criterion.

    Attributes:
        story_id: ID of the story containing the AC.
        ac_index: Index of the AC within the story's acceptance_criteria list.
        issue_type: Type of issue (e.g., 'missing_gwt', 'subjective_term', 'vague_outcome').
        description: Human-readable description of the issue.
        severity: Severity level ('blocking' or 'warning').

    Example:
        >>> issue = ACMeasurabilityIssue(
        ...     story_id="story-001",
        ...     ac_index=0,
        ...     issue_type="missing_gwt",
        ...     description="Missing 'Given' clause",
        ...     severity="blocking",
        ... )
    """

    story_id: str
    ac_index: int
    issue_type: str
    description: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to dictionary for serialization.

        Returns:
            Dictionary representation of the issue.
        """
        return {
            "story_id": self.story_id,
            "ac_index": self.ac_index,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
        }


def detect_subjective_terms(text: str) -> list[tuple[str, int]]:
    """Detect subjective terms in text.

    Searches for subjective or vague terms that cannot be objectively
    measured or tested. Detection is case-insensitive.

    Args:
        text: The text to search for subjective terms.

    Returns:
        List of (term, position) tuples for each subjective term found.

    Example:
        >>> detect_subjective_terms("The UI should be intuitive and user-friendly")
        [('intuitive', 17), ('user-friendly', 31)]
    """
    found_terms: list[tuple[str, int]] = []
    text_lower = text.lower()

    # Track positions already matched to avoid overlapping matches
    matched_positions: set[int] = set()

    # Use pre-sorted terms (sorted by length descending) to match multi-word phrases first
    for term in _SUBJECTIVE_TERMS_SORTED:
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


def has_gwt_structure(ac_text: str) -> tuple[bool, list[str]]:
    """Check if acceptance criteria has Given/When/Then structure.

    Validates that the AC contains all three parts: Given, When, and Then.

    Args:
        ac_text: The acceptance criterion text to validate.

    Returns:
        Tuple of (passed, missing_parts) where passed is True if all
        parts are present, and missing_parts lists any missing keywords.

    Example:
        >>> has_gwt_structure("Given a user, When they login, Then they see dashboard")
        (True, [])
        >>> has_gwt_structure("The user sees the dashboard")
        (False, ['Given', 'When', 'Then'])
    """
    missing_parts: list[str] = []

    for part_name, pattern in GWT_PATTERNS.items():
        if not pattern.search(ac_text):
            missing_parts.append(part_name.capitalize())

    return (len(missing_parts) == 0, missing_parts)


def has_concrete_condition(ac_text: str) -> bool:
    """Check if acceptance criteria has concrete, verifiable conditions.

    Validates that the AC (particularly the "Then" clause) contains
    observable, measurable outcomes.

    Args:
        ac_text: The acceptance criterion text to validate.

    Returns:
        True if the AC contains concrete conditions, False otherwise.

    Example:
        >>> has_concrete_condition("Then they are redirected to the dashboard")
        True
        >>> has_concrete_condition("Then the experience is good")
        False
    """
    # Extract the "Then" clause if present
    then_match = re.search(r"\bthen\b(.*)$", ac_text, re.IGNORECASE | re.DOTALL)
    if then_match:
        then_clause = then_match.group(1)
    else:
        # If no "Then" clause, check the entire text
        then_clause = ac_text

    # Check for concrete condition patterns
    for pattern in CONCRETE_CONDITION_PATTERNS:
        if pattern.search(then_clause):
            return True

    return False


def generate_improvement_suggestions(issues: list[ACMeasurabilityIssue]) -> dict[str, str]:
    """Generate improvement suggestions for measurability issues.

    Creates targeted suggestions based on the type of issue found.

    Args:
        issues: List of measurability issues to generate suggestions for.

    Returns:
        Dictionary mapping issue descriptions to improvement suggestions.

    Example:
        >>> issues = [ACMeasurabilityIssue("s-1", 0, "missing_gwt", "Missing 'Given'", "blocking")]
        >>> suggestions = generate_improvement_suggestions(issues)
        >>> "Given" in suggestions.get("Missing 'Given'", "")
        True
    """
    suggestions: dict[str, str] = {}

    for issue in issues:
        if issue.issue_type == "missing_gwt":
            if "Given" in issue.description:
                suggestions[issue.description] = (
                    "Add a 'Given' clause to establish preconditions. "
                    "Example: 'Given a logged-in user with admin privileges'"
                )
            elif "When" in issue.description:
                suggestions[issue.description] = (
                    "Add a 'When' clause to describe the action. "
                    "Example: 'When the user clicks the submit button'"
                )
            elif "Then" in issue.description:
                suggestions[issue.description] = (
                    "Add a 'Then' clause to specify expected outcomes. "
                    "Example: 'Then a success message is displayed'"
                )
            else:
                suggestions[issue.description] = (
                    "Use Given/When/Then format: "
                    "'Given [precondition], When [action], Then [expected outcome]'"
                )

        elif issue.issue_type == "subjective_term":
            suggestions[issue.description] = (
                "Replace subjective terms with measurable criteria. "
                "Instead of 'intuitive', use specific metrics like "
                "'completes task in under 3 clicks' or 'shows clear error messages'"
            )

        elif issue.issue_type == "vague_outcome":
            suggestions[issue.description] = (
                "Make the outcome specific and verifiable. "
                "Instead of vague outcomes, use observable actions like "
                "'user sees confirmation dialog' or 'system displays error code'"
            )

        elif issue.issue_type == "invalid_structure":
            suggestions[issue.description] = (
                "Ensure acceptance criteria is properly structured. "
                "Each AC should be a dictionary with a 'content' key containing the AC text."
            )

    return suggestions


def generate_ac_measurability_report(issues: list[ACMeasurabilityIssue]) -> str:
    """Generate a human-readable report of AC measurability issues.

    Creates a formatted report listing all issues by story and AC,
    including severity levels and remediation guidance.

    Args:
        issues: List of measurability issues found.

    Returns:
        Formatted report string.

    Example:
        >>> issues = [ACMeasurabilityIssue("s-1", 0, "missing_gwt", "Missing structure", "blocking")]
        >>> report = generate_ac_measurability_report(issues)
        >>> "s-1" in report
        True
    """
    if not issues:
        return ""

    lines: list[str] = ["AC Measurability Gate Report", "=" * 40, ""]

    # Group issues by story
    issues_by_story: dict[str, list[ACMeasurabilityIssue]] = {}
    for issue in issues:
        if issue.story_id not in issues_by_story:
            issues_by_story[issue.story_id] = []
        issues_by_story[issue.story_id].append(issue)

    # Generate suggestions for all issues
    suggestions = generate_improvement_suggestions(issues)

    for story_id, story_issues in issues_by_story.items():
        lines.append(f"Story: {story_id}")
        lines.append("-" * 30)

        # Group by AC index within story
        issues_by_ac: dict[int, list[ACMeasurabilityIssue]] = {}
        for issue in story_issues:
            if issue.ac_index not in issues_by_ac:
                issues_by_ac[issue.ac_index] = []
            issues_by_ac[issue.ac_index].append(issue)

        for ac_index, ac_issues in sorted(issues_by_ac.items()):
            lines.append(f"  AC #{ac_index}:")

            for issue in ac_issues:
                severity_marker = "[BLOCKING]" if issue.severity == "blocking" else "[WARNING]"
                lines.append(f"    {severity_marker} {issue.description}")

                # Add suggestion if available
                if issue.description in suggestions:
                    lines.append(f"      → {suggestions[issue.description]}")

        lines.append("")

    # Add summary
    blocking_count = sum(1 for i in issues if i.severity == "blocking")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    lines.append(f"Summary: {blocking_count} blocking, {warning_count} warnings")

    return "\n".join(lines)


async def ac_measurability_evaluator(context: GateContext) -> GateResult:
    """Evaluate acceptance criteria for measurability.

    Checks each AC in stories for:
    1. Given/When/Then structure (blocking if missing)
    2. Subjective terms (warning if present)
    3. Concrete conditions in "Then" clause (warning if missing)

    Args:
        context: Gate context containing state with stories.

    Returns:
        GateResult indicating pass/fail with detailed reason.

    Example:
        >>> context = GateContext(
        ...     state={"stories": [{"id": "s-1", "acceptance_criteria": [
        ...         {"content": "Given a user, When they login, Then they see dashboard"}
        ...     ]}]},
        ...     gate_name="ac_measurability",
        ... )
        >>> result = await ac_measurability_evaluator(context)
        >>> result.passed
        True
    """
    stories = context.state.get("stories", [])

    # Validate stories is a list
    if not isinstance(stories, list):
        logger.warning(
            "ac_measurability_gate_invalid_input",
            gate_name=context.gate_name,
            reason="stories must be a list",
            actual_type=type(stories).__name__,
        )
        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason=f"Invalid input: stories must be a list, got {type(stories).__name__}",
        )

    # Empty or missing stories - nothing to validate
    if not stories:
        logger.info(
            "ac_measurability_gate_passed",
            gate_name=context.gate_name,
            reason="No stories to validate",
        )
        return GateResult(
            passed=True,
            gate_name=context.gate_name,
            reason=None,
        )

    issues: list[ACMeasurabilityIssue] = []

    for story_idx, story in enumerate(stories):
        # Validate each story is a dict
        if not isinstance(story, dict):
            issues.append(
                ACMeasurabilityIssue(
                    story_id=f"index-{story_idx}",
                    ac_index=-1,
                    issue_type="invalid_structure",
                    description=f"Story at index {story_idx} is not a dict (got {type(story).__name__})",
                    severity="blocking",
                )
            )
            continue

        story_id = story.get("id", f"index-{story_idx}")
        acceptance_criteria = story.get("acceptance_criteria", [])

        # Validate acceptance_criteria is a list
        if not isinstance(acceptance_criteria, list):
            issues.append(
                ACMeasurabilityIssue(
                    story_id=story_id,
                    ac_index=-1,
                    issue_type="invalid_structure",
                    description=f"acceptance_criteria must be a list, got {type(acceptance_criteria).__name__}",
                    severity="blocking",
                )
            )
            continue

        for ac_idx, ac in enumerate(acceptance_criteria):
            # Validate each AC is a dict
            if not isinstance(ac, dict):
                issues.append(
                    ACMeasurabilityIssue(
                        story_id=story_id,
                        ac_index=ac_idx,
                        issue_type="invalid_structure",
                        description=f"AC at index {ac_idx} is not a dict (got {type(ac).__name__})",
                        severity="blocking",
                    )
                )
                continue

            content = ac.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content else ""

            # Check for Given/When/Then structure (blocking)
            has_structure, missing_parts = has_gwt_structure(content)
            if not has_structure:
                issues.append(
                    ACMeasurabilityIssue(
                        story_id=story_id,
                        ac_index=ac_idx,
                        issue_type="missing_gwt",
                        description=f"Missing Given/When/Then structure: {', '.join(missing_parts)} not found",
                        severity="blocking",
                    )
                )

            # Check for subjective terms (warning)
            subjective_terms = detect_subjective_terms(content)
            for term, _pos in subjective_terms:
                issues.append(
                    ACMeasurabilityIssue(
                        story_id=story_id,
                        ac_index=ac_idx,
                        issue_type="subjective_term",
                        description=f"Contains subjective term '{term}'",
                        severity="warning",
                    )
                )

            # Check for concrete conditions (warning if has GWT but vague outcome)
            if has_structure and not has_concrete_condition(content):
                issues.append(
                    ACMeasurabilityIssue(
                        story_id=story_id,
                        ac_index=ac_idx,
                        issue_type="vague_outcome",
                        description="'Then' clause lacks concrete, verifiable condition",
                        severity="warning",
                    )
                )

    # Determine if there are any blocking issues
    blocking_issues = [i for i in issues if i.severity == "blocking"]

    if blocking_issues:
        report = generate_ac_measurability_report(issues)
        failing_stories = sorted({issue.story_id for issue in blocking_issues})
        reason = f"AC measurability check failed for {len(failing_stories)} story(ies): {', '.join(failing_stories)}\n\n{report}"

        logger.warning(
            "ac_measurability_gate_failed",
            gate_name=context.gate_name,
            blocking_count=len(blocking_issues),
            warning_count=len(issues) - len(blocking_issues),
            failing_stories=failing_stories,
        )

        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason=reason,
        )

    # Gate passes but may have warnings
    warning_issues = [i for i in issues if i.severity == "warning"]

    if warning_issues:
        report = generate_ac_measurability_report(warning_issues)
        logger.info(
            "ac_measurability_gate_passed_with_warnings",
            gate_name=context.gate_name,
            warning_count=len(warning_issues),
        )

        return GateResult(
            passed=True,
            gate_name=context.gate_name,
            reason=f"Passed with {len(warning_issues)} warning(s):\n\n{report}",
        )

    logger.info(
        "ac_measurability_gate_passed",
        gate_name=context.gate_name,
        stories_checked=len(stories),
    )

    return GateResult(
        passed=True,
        gate_name=context.gate_name,
        reason=None,
    )


# Register the evaluator when module is imported
register_evaluator("ac_measurability", ac_measurability_evaluator)
