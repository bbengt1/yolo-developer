# Story 3.8: Generate Gate Failure Reports

Status: done

## Story

As a developer,
I want detailed failure reports when gates block,
So that I understand exactly what needs to be fixed.

## Acceptance Criteria

1. **AC1: Specific Issues Listed**
   - **Given** a quality gate fails
   - **When** the failure report is generated
   - **Then** specific issues causing the failure are itemized
   - **And** each issue includes the location/context where it was found
   - **And** the issue type is clearly identified

2. **AC2: Severity Indication**
   - **Given** issues are identified during gate evaluation
   - **When** the failure report is generated
   - **Then** each issue indicates severity (blocking vs warning)
   - **And** blocking issues are clearly distinguished from advisory warnings
   - **And** the overall gate mode (blocking/advisory) is reflected

3. **AC3: Remediation Suggestions**
   - **Given** a gate failure report is generated
   - **When** issues are listed
   - **Then** actionable remediation suggestions are provided per issue type
   - **And** suggestions include concrete examples where applicable
   - **And** suggestions are specific to the gate that failed

4. **AC4: Human-Readable Reports**
   - **Given** a gate evaluation completes (pass or fail)
   - **When** the report is generated
   - **Then** the report is human-readable and actionable
   - **And** the report can be displayed in CLI or log output
   - **And** the report supports structured logging format

5. **AC5: Consistent Report Structure**
   - **Given** any gate generates a failure report
   - **When** reports from different gates are compared
   - **Then** all reports follow the same structural format
   - **And** a common report generator utility is used
   - **And** the structure integrates with existing GateResult.reason field

## Tasks / Subtasks

- [x] Task 1: Create Report Data Models (AC: 1, 2, 5)
  - [x] Create `src/yolo_developer/gates/report_types.py` module
  - [x] Define `GateIssue` dataclass with fields: `location`, `issue_type`, `description`, `severity`
  - [x] Define `GateFailureReport` dataclass with fields: `gate_name`, `issues`, `score`, `threshold`, `summary`
  - [x] Add `to_dict()` method for structured logging compatibility
  - [x] Add `Severity` enum: `BLOCKING`, `WARNING`
  - [x] Add Pydantic validators if needed for field constraints

- [x] Task 2: Create Report Generator Utility (AC: 3, 4, 5)
  - [x] Create `src/yolo_developer/gates/report_generator.py` module
  - [x] Implement `generate_failure_report(gate_name: str, issues: list[GateIssue], score: float, threshold: float) -> GateFailureReport`
  - [x] Implement `format_report_text(report: GateFailureReport) -> str` for human-readable output
  - [x] Add remediation suggestions registry per issue type
  - [x] Support gate-specific remediation overrides
  - [x] Use structlog for report generation logging

- [x] Task 3: Define Remediation Suggestions (AC: 3)
  - [x] Create `src/yolo_developer/gates/remediation.py` module
  - [x] Define `REMEDIATION_SUGGESTIONS: dict[str, str]` mapping issue_type to suggestion
  - [x] Include suggestions for: `vague_term`, `no_success_criteria`, `missing_metric`, `architecture_violation`, `coverage_gap`, `dod_incomplete`
  - [x] Add concrete examples for each suggestion type
  - [x] Support custom suggestions via configuration

- [x] Task 4: Refactor Testability Gate (AC: 1, 2, 3, 4, 5)
  - [x] Update `testability.py` to use new `GateIssue` type (migrate from `TestabilityIssue`)
  - [x] Use `generate_failure_report()` instead of `generate_testability_report()`
  - [x] Ensure backward compatibility with existing test patterns
  - [x] Verify report format matches new structure

- [x] Task 5: Refactor AC Measurability Gate (AC: 1, 2, 3, 4, 5)
  - [x] Update `ac_measurability.py` to use `GateIssue` and report generator
  - [x] Add issue identification for AC validation failures
  - [x] Include remediation suggestions for measurability issues
  - [x] Update failure reason to use formatted report

- [x] Task 6: Refactor Architecture Validation Gate (AC: 1, 2, 3, 4, 5)
  - [x] Update `architecture_validation.py` to use `GateIssue` and report generator
  - [x] Add issue identification for architecture violations
  - [x] Map violation types to remediation suggestions
  - [x] Update failure reason to use formatted report

- [x] Task 7: Refactor Definition of Done Gate (AC: 1, 2, 3, 4, 5)
  - [x] Update `definition_of_done.py` to use `GateIssue` and report generator
  - [x] Add issue identification for incomplete DoD items
  - [x] Map DoD check types to remediation suggestions
  - [x] Update failure reason to use formatted report

- [x] Task 8: Refactor Confidence Scoring Gate (AC: 1, 2, 3, 4, 5)
  - [x] Update `confidence_scoring.py` to use `GateIssue` and report generator
  - [x] Identify issues from low-scoring factors
  - [x] Add factor-specific remediation suggestions
  - [x] Update failure reason to use formatted report

- [x] Task 9: Update GateResult Integration (AC: 4, 5)
  - [x] Embed formatted reports in `GateResult.reason` field (design choice)
  - [x] Ensure decorator logs structured report data via structlog
  - [x] Maintain backward compatibility with string-only `reason` field
  - Note: Design changed from adding `report` field to embedding in `reason` for simplicity

- [x] Task 10: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/gates/test_report_types.py`
  - [x] Create `tests/unit/gates/test_report_generator.py`
  - [x] Create `tests/unit/gates/test_remediation.py`
  - [x] Test GateIssue creation and serialization
  - [x] Test report generation with various issue combinations
  - [x] Test remediation suggestion lookup
  - [x] Test formatted text output

- [x] Task 11: Write Integration Tests (AC: all)
  - [x] Create `tests/integration/test_gate_failure_reports.py`
  - [x] Test each gate produces properly formatted reports
  - [x] Test report consistency across gates
  - [x] Test remediation suggestions appear correctly
  - [x] Test structured logging output contains report data

- [x] Task 12: Update Exports and Documentation (AC: 5)
  - [x] Export report types from `gates/__init__.py`
  - [x] Export report generator from `gates/__init__.py`
  - [x] Update module docstring with report generation examples
  - [x] Add inline documentation for remediation patterns

## Dev Notes

### Architecture Compliance

- **ADR-006 (Quality Gate Pattern):** Reports integrate with decorator-based gates, maintaining blocking/advisory semantics
- **FR25:** System can generate quality gate failure reports with remediation guidance
- **FR26/FR90:** Threshold configuration affects report score/threshold display

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses for `GateIssue` and `GateFailureReport`
- **Backward Compatibility:** Gates must continue working with existing test patterns
- **Structured Logging:** All report data must be logged via structlog
- **Format Flexibility:** Reports should support both text and structured (dict) formats

### Existing Pattern Reference

The testability gate already implements a solid report pattern that should be generalized:

```python
# Current pattern in testability.py (to be generalized)
@dataclass(frozen=True)
class TestabilityIssue:
    requirement_id: str
    issue_type: str
    description: str
    severity: str  # "blocking" or "warning"

    def to_dict(self) -> dict[str, str]:
        return {
            "requirement_id": self.requirement_id,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
        }


def generate_testability_report(issues: list[TestabilityIssue]) -> str:
    """Generate a human-readable report of testability issues."""
    # Groups issues by requirement
    # Adds severity markers [BLOCKING] or [WARNING]
    # Includes remediation suggestions per issue type
```

### Proposed Report Data Models

```python
# src/yolo_developer/gates/report_types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    """Severity level for gate issues."""
    BLOCKING = "blocking"
    WARNING = "warning"


@dataclass(frozen=True)
class GateIssue:
    """Represents a single issue found during gate evaluation.

    Attributes:
        location: Where the issue was found (e.g., requirement ID, file path)
        issue_type: Category of issue (e.g., "vague_term", "coverage_gap")
        description: Human-readable description of the issue
        severity: Whether this issue is blocking or advisory
        context: Optional additional context about the issue
    """
    location: str
    issue_type: str
    description: str
    severity: Severity
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity.value,
            "context": self.context,
        }


@dataclass(frozen=True)
class GateFailureReport:
    """Structured report of gate evaluation results.

    Attributes:
        gate_name: Name of the gate that generated the report
        issues: List of issues found during evaluation
        score: Numeric score (0.0-1.0) achieved
        threshold: Required threshold for passing
        summary: Brief summary of the evaluation result
    """
    gate_name: str
    issues: tuple[GateIssue, ...]  # Immutable tuple for frozen dataclass
    score: float
    threshold: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "issues": [issue.to_dict() for issue in self.issues],
            "score": self.score,
            "threshold": self.threshold,
            "summary": self.summary,
        }
```

### Proposed Report Generator

```python
# src/yolo_developer/gates/report_generator.py

from yolo_developer.gates.report_types import GateFailureReport, GateIssue, Severity
from yolo_developer.gates.remediation import get_remediation_suggestion


def generate_failure_report(
    gate_name: str,
    issues: list[GateIssue],
    score: float,
    threshold: float,
) -> GateFailureReport:
    """Generate a structured failure report for a gate.

    Args:
        gate_name: Name of the gate
        issues: List of issues found
        score: Achieved score (0.0-1.0)
        threshold: Required threshold

    Returns:
        Structured GateFailureReport
    """
    blocking_count = sum(1 for i in issues if i.severity == Severity.BLOCKING)
    warning_count = sum(1 for i in issues if i.severity == Severity.WARNING)

    score_pct = int(score * 100)
    threshold_pct = int(threshold * 100)

    summary = (
        f"{gate_name} score {score_pct}% below threshold {threshold_pct}%. "
        f"Found {blocking_count} blocking issue(s) and {warning_count} warning(s)."
    )

    return GateFailureReport(
        gate_name=gate_name,
        issues=tuple(issues),
        score=score,
        threshold=threshold,
        summary=summary,
    )


def format_report_text(report: GateFailureReport) -> str:
    """Format a failure report as human-readable text.

    Args:
        report: The failure report to format

    Returns:
        Formatted text string suitable for CLI/log output
    """
    lines = [
        f"{report.gate_name.replace('_', ' ').title()} Gate Report",
        "=" * 50,
        "",
        report.summary,
        "",
    ]

    # Group issues by location
    issues_by_location: dict[str, list[GateIssue]] = {}
    for issue in report.issues:
        if issue.location not in issues_by_location:
            issues_by_location[issue.location] = []
        issues_by_location[issue.location].append(issue)

    for location, location_issues in sorted(issues_by_location.items()):
        lines.append(f"Location: {location}")
        lines.append("-" * 40)

        for issue in location_issues:
            severity_marker = (
                "[BLOCKING]" if issue.severity == Severity.BLOCKING else "[WARNING]"
            )
            lines.append(f"  {severity_marker} {issue.description}")

            # Add remediation suggestion
            suggestion = get_remediation_suggestion(issue.issue_type, report.gate_name)
            if suggestion:
                lines.append(f"    Suggestion: {suggestion}")

        lines.append("")

    return "\n".join(lines)
```

### Proposed Remediation Registry

```python
# src/yolo_developer/gates/remediation.py

# Default remediation suggestions by issue type
DEFAULT_REMEDIATION: dict[str, str] = {
    # Testability gate
    "vague_term": "Replace vague terms with specific, measurable criteria. Example: Instead of 'fast', use 'responds within 500ms'.",
    "no_success_criteria": "Add quantifiable success criteria. Include specific metrics, percentages, or Given/When/Then format.",

    # AC Measurability gate
    "unmeasurable_ac": "Rewrite acceptance criteria with observable outcomes. Use concrete assertions.",
    "missing_assertion": "Add explicit assertion statements. Each AC should have testable conditions.",

    # Architecture Validation gate
    "adr_violation": "Review the referenced ADR and update implementation to comply with the decision.",
    "pattern_mismatch": "Apply the required architectural pattern. Check architecture.md for examples.",
    "missing_component": "Add the required component per architecture specification.",

    # Definition of Done gate
    "tests_missing": "Add unit tests covering the implemented functionality.",
    "coverage_gap": "Increase test coverage to meet threshold. Focus on untested code paths.",
    "documentation_missing": "Add documentation for public APIs and complex logic.",

    # Confidence Scoring gate
    "low_gate_score": "Address failures in underlying gates to improve overall confidence.",
    "low_coverage": "Increase test coverage, particularly branch coverage.",
    "high_risk": "Mitigate identified risks before proceeding.",
    "low_documentation": "Add docstrings, README, and inline comments.",
}


# Gate-specific overrides (optional)
GATE_SPECIFIC_REMEDIATION: dict[str, dict[str, str]] = {
    "testability": {
        # Can override default suggestions for this gate
    },
}


def get_remediation_suggestion(issue_type: str, gate_name: str) -> str | None:
    """Get remediation suggestion for an issue type.

    Args:
        issue_type: The type of issue
        gate_name: Name of the gate (for gate-specific overrides)

    Returns:
        Remediation suggestion string, or None if not found
    """
    # Check gate-specific first
    gate_overrides = GATE_SPECIFIC_REMEDIATION.get(gate_name, {})
    if issue_type in gate_overrides:
        return gate_overrides[issue_type]

    # Fall back to default
    return DEFAULT_REMEDIATION.get(issue_type)
```

### File Structure

```
src/yolo_developer/gates/
├── __init__.py              # UPDATE: Export report types and generator
├── report_types.py          # NEW: GateIssue, GateFailureReport, Severity
├── report_generator.py      # NEW: generate_failure_report, format_report_text
├── remediation.py           # NEW: Remediation suggestions registry
├── decorator.py             # No changes expected
├── evaluators.py            # No changes expected
├── threshold_resolver.py    # No changes expected
├── types.py                 # UPDATE: Add optional report field to GateResult
└── gates/
    ├── testability.py           # UPDATE: Use GateIssue, report generator
    ├── ac_measurability.py      # UPDATE: Use GateIssue, report generator
    ├── architecture_validation.py # UPDATE: Use GateIssue, report generator
    ├── definition_of_done.py    # UPDATE: Use GateIssue, report generator
    └── confidence_scoring.py    # UPDATE: Use GateIssue, report generator
```

### Previous Story Intelligence (from Story 3.7)

**Patterns to Apply:**
1. Use frozen dataclasses for all data types (immutable)
2. Evaluators are async callable: `async def evaluator(ctx: GateContext) -> GateResult`
3. State accessible via `context.state`
4. Config accessible via `state.get("config", {})`
5. Use structured logging for all operations via structlog
6. Validate input types before processing
7. Provide clear error messages for invalid inputs
8. Add autouse fixture in tests to ensure consistent state
9. All thresholds use 0.0-1.0 decimal format
10. Export new types and functions from `__init__.py`

**Key Files to Reference:**
- `src/yolo_developer/gates/gates/testability.py` - Existing report pattern (TestabilityIssue, generate_testability_report)
- `src/yolo_developer/gates/types.py` - GateResult, GateContext definitions
- `src/yolo_developer/gates/threshold_resolver.py` - Pattern for utility modules
- `tests/unit/gates/test_testability.py` - Testing patterns for gate evaluators

**Code Review Learnings from Story 3.7:**
- Export new constants from `__init__.py`
- Remove unused imports
- Use `()` literal instead of `tuple()` for empty tuples
- Use tuple for immutable collections in frozen dataclasses (not list)

### Testing Standards

- Use pytest with pytest-asyncio for async tests
- Create fixtures for various report scenarios (empty, single issue, multiple issues)
- Test serialization (to_dict) methods
- Test text formatting output
- Test remediation suggestion lookup
- Verify structured logging output
- Test backward compatibility with existing gate tests

### Implementation Approach

1. **Data Models First:** Create report_types.py with GateIssue, Severity, GateFailureReport
2. **Remediation Registry:** Create remediation.py with suggestion mappings
3. **Report Generator:** Create report_generator.py using the data models
4. **Gate Refactoring:** Update each gate to use new report utilities (one at a time)
5. **GateResult Enhancement:** Add optional report field after gates are updated
6. **Backward Compatible:** All existing tests must continue to pass

### References

- [Source: architecture.md#ADR-006] - Quality Gate Pattern
- [Source: prd.md#FR25] - System can generate quality gate failure reports
- [Source: epics.md#Story-3.8] - Generate Gate Failure Reports requirements
- [Story 3.7 Implementation] - Threshold resolver patterns
- [testability.py] - Existing TestabilityIssue and generate_testability_report patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

1. **Tasks 1-3 (Core Infrastructure):** Created report_types.py, report_generator.py, and remediation.py with frozen dataclasses for immutability.
2. **Tasks 4-8 (Gate Refactoring):** All 5 gates refactored to use shared `GateIssue` type and `generate_failure_report()`/`format_report_text()` utilities.
3. **Task 9 (Integration Design):** Embedded reports in `GateResult.reason` field rather than adding separate `report` field for simplicity and backward compatibility.
4. **Tasks 10-11 (Testing):** Unit tests created for all new modules. Integration tests verify cross-gate report consistency.
5. **Task 12 (Exports):** All new types and functions exported from `gates/__init__.py` with docstring examples.
6. **Code Review (2026-01-05):** Fixed story file documentation, created missing integration tests, updated task checkboxes.

### File List

**New Files Created:**
- `src/yolo_developer/gates/report_types.py` - GateIssue, GateFailureReport, Severity types
- `src/yolo_developer/gates/report_generator.py` - generate_failure_report(), format_report_text()
- `src/yolo_developer/gates/remediation.py` - Remediation suggestion registry
- `tests/unit/gates/test_report_types.py` - Unit tests for report types
- `tests/unit/gates/test_report_generator.py` - Unit tests for report generator
- `tests/unit/gates/test_remediation.py` - Unit tests for remediation module
- `tests/integration/test_gate_failure_reports.py` - Integration tests for cross-gate reports

**Modified Files:**
- `src/yolo_developer/gates/__init__.py` - Added exports for new types and functions
- `src/yolo_developer/gates/gates/__init__.py` - Updated exports, removed deprecated types
- `src/yolo_developer/gates/gates/testability.py` - Refactored to use GateIssue and report generator
- `src/yolo_developer/gates/gates/ac_measurability.py` - Refactored to use GateIssue and report generator
- `src/yolo_developer/gates/gates/architecture_validation.py` - Refactored to use GateIssue and report generator
- `src/yolo_developer/gates/gates/definition_of_done.py` - Refactored to use GateIssue and report generator
- `src/yolo_developer/gates/gates/confidence_scoring.py` - Refactored to use GateIssue and report generator
- `tests/unit/gates/test_testability.py` - Updated tests for new report structure
- `tests/unit/gates/test_ac_measurability.py` - Updated tests for new report structure
- `tests/unit/gates/test_architecture_validation.py` - Updated tests for new report structure
- `tests/unit/gates/test_definition_of_done.py` - Updated tests for new report structure
