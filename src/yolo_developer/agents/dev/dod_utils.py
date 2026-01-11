"""DoD validation utilities for Dev agent (Story 8.6).

This module provides programmatic Definition of Done validation:

- DoDChecklistItem: Single item in DoD checklist
- DoDValidationResult: Aggregate result of DoD validation
- validate_implementation_dod: Validate code against DoD checklist
- validate_artifact_dod: Validate ImplementationArtifact against DoD
- validate_dev_output_dod: Validate DevOutput with all artifacts

These utilities REUSE existing check functions from gates/gates/definition_of_done.py.
They provide programmatic access to DoD validation outside the gate decorator context.

Example:
    >>> from yolo_developer.agents.dev.dod_utils import (
    ...     validate_implementation_dod,
    ...     DoDValidationResult,
    ... )
    >>>
    >>> code = {"files": [{"path": "src/impl.py", "content": "..."}]}
    >>> story = {"acceptance_criteria": ["AC1: ..."]}
    >>> result = validate_implementation_dod(code, story)
    >>> result.passed
    True

Security Note:
    DoD validation reads code content but does not execute it.
    Validation results may contain file paths - do not expose to untrusted parties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog

from yolo_developer.gates.gates.definition_of_done import (
    DOD_CHECKLIST_ITEMS,
    DoDCategory,
    check_ac_coverage,
    check_code_style,
    check_documentation,
    check_test_presence,
    generate_dod_checklist,
)
from yolo_developer.gates.report_types import GateIssue, Severity

if TYPE_CHECKING:
    from yolo_developer.agents.dev.types import (
        DevOutput,
        ImplementationArtifact,
    )

logger = structlog.get_logger(__name__)


# =============================================================================
# Type Definitions (Task 1)
# =============================================================================


DoDCategoryType = Literal["tests", "documentation", "style", "ac_coverage"]
"""Category type for DoD checklist items."""

# Map DoDCategory enum to string for checklist items
_CATEGORY_MAP: dict[DoDCategory, DoDCategoryType] = {
    DoDCategory.TESTS: "tests",
    DoDCategory.DOCUMENTATION: "documentation",
    DoDCategory.STYLE: "style",
    DoDCategory.AC_COVERAGE: "ac_coverage",
}


@dataclass(frozen=True)
class DoDChecklistItem:
    """Single item in DoD checklist.

    Represents a single validation result for a DoD checklist item.
    Items are immutable for audit trail integrity.

    Attributes:
        category: Category from DOD_CHECKLIST_ITEMS (tests, documentation, style, ac_coverage).
        item_name: Specific checklist item name.
        passed: Whether this item passed validation.
        severity: Issue severity if failed (high=blocking, medium=warning, low=info).
            None if passed.
        message: Human-readable description of result.

    Example:
        >>> item = DoDChecklistItem(
        ...     category="tests",
        ...     item_name="unit_tests_present",
        ...     passed=True,
        ...     severity=None,
        ...     message="All unit tests present",
        ... )
        >>> item.passed
        True
    """

    category: DoDCategoryType
    item_name: str
    passed: bool
    severity: Literal["high", "medium", "low"] | None
    message: str


@dataclass
class DoDValidationResult:
    """Result of DoD validation for an implementation.

    Aggregates all validation results including score, pass/fail status,
    itemized checklist, and detailed issues.

    Note: Not frozen because checklist items are appended during validation.

    Attributes:
        score: Compliance score 0-100 using SEVERITY_WEIGHTS.
        passed: Whether score meets threshold (default 70).
        threshold: Score threshold used for pass/fail.
        checklist: List of individual checklist results.
        issues: List of GateIssue objects for failures.
        artifact_id: Optional story/artifact ID being validated.

    Example:
        >>> result = DoDValidationResult(score=85, passed=True)
        >>> result.to_dict()
        {'score': 85, 'passed': True, ...}
    """

    score: int = 100
    passed: bool = True
    threshold: int = 70
    checklist: list[DoDChecklistItem] = field(default_factory=list)
    issues: list[GateIssue] = field(default_factory=list)
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of validation result.
        """
        return {
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "checklist": [
                {
                    "category": item.category,
                    "item_name": item.item_name,
                    "passed": item.passed,
                    "severity": item.severity,
                    "message": item.message,
                }
                for item in self.checklist
            ],
            "issue_count": len(self.issues),
            "artifact_id": self.artifact_id,
        }


# =============================================================================
# Programmatic DoD Validation (Task 2)
# =============================================================================


def validate_implementation_dod(
    code: dict[str, Any],
    story: dict[str, Any],
    threshold: int = 70,
) -> DoDValidationResult:
    """Validate implementation against Definition of Done checklist.

    Runs all DoD checks: test presence, documentation, style, AC coverage.
    REUSES existing check functions from gates/gates/definition_of_done.py.
    Returns a complete checklist with ALL DOD_CHECKLIST_ITEMS, marking each
    as passed or failed based on validation results.

    Args:
        code: Code artifact dict with 'files' key containing list of file dicts.
              Each file dict has 'path' and 'content' keys.
        story: Story dict with optional 'acceptance_criteria' key.
        threshold: Minimum score to pass (0-100, default 70).

    Returns:
        DoDValidationResult with score, pass/fail, and complete itemized checklist.

    Example:
        >>> code = {
        ...     "files": [
        ...         {"path": "src/impl.py", "content": "def foo(): pass"},
        ...         {"path": "tests/test_impl.py", "content": "def test_foo(): pass"},
        ...     ]
        ... }
        >>> story = {"acceptance_criteria": ["AC1: foo works"]}
        >>> result = validate_implementation_dod(code, story)
        >>> result.passed
        True
    """
    logger.debug(
        "validate_implementation_dod_start",
        file_count=len(code.get("files", [])),
        has_acceptance_criteria=bool(story.get("acceptance_criteria")),
        threshold=threshold,
    )

    # Collect all issues from existing check functions with error handling
    all_issues: list[GateIssue] = []
    validation_errors: list[str] = []

    # Run test presence check (reuses existing function)
    try:
        all_issues.extend(check_test_presence(code, story))
    except Exception as e:
        validation_errors.append(f"test_presence: {e}")
        logger.warning("dod_check_test_presence_error", error=str(e))

    # Run documentation check (reuses existing function)
    try:
        all_issues.extend(check_documentation(code))
    except Exception as e:
        validation_errors.append(f"documentation: {e}")
        logger.warning("dod_check_documentation_error", error=str(e))

    # Run code style check (reuses existing function)
    try:
        all_issues.extend(check_code_style(code))
    except Exception as e:
        validation_errors.append(f"code_style: {e}")
        logger.warning("dod_check_code_style_error", error=str(e))

    # Run AC coverage check (reuses existing function)
    try:
        all_issues.extend(check_ac_coverage(code, story))
    except Exception as e:
        validation_errors.append(f"ac_coverage: {e}")
        logger.warning("dod_check_ac_coverage_error", error=str(e))

    # Add validation error issues if any checks failed
    for error_msg in validation_errors:
        all_issues.append(
            GateIssue(
                location="validation",
                issue_type="validation_error",
                description=f"DoD check failed: {error_msg}",
                severity=Severity.BLOCKING,
                context={"category": "tests", "original_severity": "high"},
            )
        )

    # Generate checklist with score (reuses existing function)
    try:
        checklist_result = generate_dod_checklist(all_issues)
        score = checklist_result["score"]
    except Exception as e:
        logger.warning("dod_generate_checklist_error", error=str(e))
        score = 0 if all_issues else 100

    # Build issue lookup by category for complete checklist
    failed_items: dict[str, set[str]] = {}
    for issue in all_issues:
        category = issue.context.get("category", "tests")
        if category not in failed_items:
            failed_items[category] = set()
        # Track the specific item type that failed
        failed_items[category].add(issue.issue_type)

    # Build COMPLETE checklist with ALL DOD_CHECKLIST_ITEMS (AC1 compliance)
    checklist_items: list[DoDChecklistItem] = []

    for dod_category, items in DOD_CHECKLIST_ITEMS.items():
        category_str = _CATEGORY_MAP.get(dod_category, "tests")
        category_failures = failed_items.get(category_str, set())

        for item_name in items:
            # Check if this specific item failed
            item_failed = any(
                item_name in failure_type or failure_type.startswith(item_name.split("_")[0])
                for failure_type in category_failures
            )

            if item_failed:
                # Find the matching issue for details
                matching_issue = next(
                    (
                        i
                        for i in all_issues
                        if i.context.get("category") == category_str
                        and (
                            item_name in i.issue_type
                            or i.issue_type.startswith(item_name.split("_")[0])
                        )
                    ),
                    None,
                )
                raw_severity = (
                    matching_issue.context.get("original_severity", "medium")
                    if matching_issue
                    else "medium"
                )
                # Validate severity is one of the allowed values and cast
                if raw_severity == "high":
                    severity: Literal["high", "medium", "low"] | None = "high"
                elif raw_severity == "low":
                    severity = "low"
                else:
                    severity = "medium"
                message = (
                    matching_issue.description if matching_issue else f"{item_name} check failed"
                )
            else:
                severity = None
                message = f"{item_name.replace('_', ' ').title()} verified"

            checklist_items.append(
                DoDChecklistItem(
                    category=category_str,
                    item_name=item_name,
                    passed=not item_failed,
                    severity=severity,
                    message=message,
                )
            )

    # Determine pass/fail
    passed = score >= threshold

    result = DoDValidationResult(
        score=score,
        passed=passed,
        threshold=threshold,
        checklist=checklist_items,
        issues=all_issues,
        artifact_id=None,
    )

    # Audit trail logging (AC5 compliance)
    logger.info(
        "dod_validation_audit",
        score=score,
        passed=passed,
        threshold=threshold,
        issue_count=len(all_issues),
        checklist_item_count=len(checklist_items),
        passed_items=sum(1 for item in checklist_items if item.passed),
        failed_items=sum(1 for item in checklist_items if not item.passed),
        validation_errors=validation_errors if validation_errors else None,
    )

    logger.debug(
        "validate_implementation_dod_complete",
        score=score,
        passed=passed,
        issue_count=len(all_issues),
    )

    return result


# Alias for API consistency with AC1 documentation
validate_dod = validate_implementation_dod


# =============================================================================
# Artifact-Level Validation (Task 3)
# =============================================================================


def _artifact_to_code_dict(artifact: ImplementationArtifact) -> dict[str, Any]:
    """Convert ImplementationArtifact to dict format for DoD checks.

    Args:
        artifact: Implementation artifact with code_files and test_files.

    Returns:
        Dict with 'files' key containing list of file dicts.
    """
    files: list[dict[str, str]] = []

    # Add code files
    for cf in artifact.code_files:
        files.append(
            {
                "path": cf.file_path,
                "content": cf.content,
            }
        )

    # Add test files
    for tf in artifact.test_files:
        files.append(
            {
                "path": tf.file_path,
                "content": tf.content,
            }
        )

    return {"files": files}


def validate_artifact_dod(
    artifact: ImplementationArtifact,
    story: dict[str, Any],
    threshold: int = 70,
) -> DoDValidationResult:
    """Validate ImplementationArtifact against Definition of Done checklist.

    Converts artifact to code dict format and runs DoD validation.

    Args:
        artifact: Implementation artifact with code_files and test_files.
        story: Story dict with optional 'acceptance_criteria' key.
        threshold: Minimum score to pass (0-100, default 70).

    Returns:
        DoDValidationResult with artifact_id set to story_id.

    Example:
        >>> from yolo_developer.agents.dev.types import ImplementationArtifact
        >>> artifact = ImplementationArtifact(story_id="story-001", ...)
        >>> result = validate_artifact_dod(artifact, {})
        >>> result.artifact_id
        'story-001'
    """
    logger.debug(
        "validate_artifact_dod_start",
        story_id=artifact.story_id,
        code_file_count=len(artifact.code_files),
        test_file_count=len(artifact.test_files),
    )

    # Convert artifact to code dict format
    code = _artifact_to_code_dict(artifact)

    # Run validation
    result = validate_implementation_dod(code, story, threshold)

    # Set artifact_id
    result.artifact_id = artifact.story_id

    logger.debug(
        "validate_artifact_dod_complete",
        story_id=artifact.story_id,
        score=result.score,
        passed=result.passed,
    )

    return result


# =============================================================================
# DevOutput Aggregate Validation (Task 4)
# =============================================================================


def validate_dev_output_dod(
    output: DevOutput,
    state: dict[str, Any],
    threshold: int = 70,
) -> list[DoDValidationResult]:
    """Validate DevOutput with all artifacts against Definition of Done.

    Validates each ImplementationArtifact in the output and returns
    a list of validation results.

    Args:
        output: DevOutput containing implementations tuple.
        state: Orchestration state dict with 'story' key.
        threshold: Minimum score to pass (0-100, default 70).

    Returns:
        List of DoDValidationResult (one per artifact).
        Empty list if output has no implementations.

    Example:
        >>> from yolo_developer.agents.dev.types import DevOutput
        >>> output = DevOutput(implementations=(...), ...)
        >>> results = validate_dev_output_dod(output, {"story": {}})
        >>> len(results)
        2
    """
    logger.debug(
        "validate_dev_output_dod_start",
        artifact_count=len(output.implementations),
    )

    # Extract story from state for AC validation
    story = state.get("story", {})
    if not isinstance(story, dict):
        story = {}

    results: list[DoDValidationResult] = []

    for artifact in output.implementations:
        result = validate_artifact_dod(artifact, story, threshold)
        results.append(result)

    logger.debug(
        "validate_dev_output_dod_complete",
        artifact_count=len(results),
        passed_count=sum(1 for r in results if r.passed),
    )

    return results
