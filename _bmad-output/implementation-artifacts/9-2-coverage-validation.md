# Story 9.2: Coverage Validation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want test coverage validated against thresholds,
So that inadequate testing is caught before deployment.

## Acceptance Criteria

1. **AC1: Coverage Measurement**
   - **Given** code with tests exists in the project
   - **When** coverage validation runs
   - **Then** overall coverage percentage is calculated
   - **And** the percentage is computed as lines_covered / lines_total * 100
   - **And** results are stored in a CoverageResult dataclass

2. **AC2: Critical Path Coverage**
   - **Given** code files are marked as critical (e.g., core business logic)
   - **When** coverage is measured
   - **Then** critical paths are validated for 100% coverage
   - **And** files in `orchestrator/`, `gates/`, `agents/` are considered critical by default
   - **And** critical path failures are reported as `critical` severity findings

3. **AC3: Threshold Blocking**
   - **Given** a configured coverage threshold (default 80%)
   - **When** coverage falls below the threshold
   - **Then** deployment blocking is triggered via ValidationResult status="failed"
   - **And** the finding includes the current coverage percentage and required threshold
   - **And** uncovered lines are listed in the finding details

4. **AC4: Uncovered Lines Reporting**
   - **Given** coverage analysis is complete
   - **When** uncovered lines are detected
   - **Then** each uncovered file/line range is reported
   - **And** the report includes file path and line numbers
   - **And** findings are categorized as "test_coverage" with appropriate severity

5. **AC5: Coverage Integration with TEA Node**
   - **Given** the TEA agent receives artifacts for validation
   - **When** tea_node processes code files
   - **Then** it calls coverage validation for each code file
   - **And** coverage results are integrated into ValidationResult
   - **And** overall_confidence is adjusted based on coverage

6. **AC6: Configuration Support**
   - **Given** project configuration specifies coverage thresholds
   - **When** coverage validation runs
   - **Then** it uses configured thresholds from YoloConfig
   - **And** default threshold is 80% if not configured
   - **And** critical path list is configurable

## Tasks / Subtasks

- [x] Task 1: Create Coverage Types (AC: 1, 4)
  - [x] Create `CoverageResult` frozen dataclass with: file_path, lines_total, lines_covered, coverage_percentage, uncovered_lines
  - [x] Create `CoverageReport` frozen dataclass with: results, overall_coverage, threshold, passed, critical_files_coverage
  - [x] Add `to_dict()` methods for serialization
  - [x] Add types to `agents/tea/types.py` or create `agents/tea/coverage.py`

- [x] Task 2: Implement Coverage Analyzer (AC: 1, 2, 4)
  - [x] Create `_analyze_coverage(code_files: list, test_files: list) -> CoverageReport`
  - [x] For MVP, implement stub analysis (heuristic-based on code/test file presence)
  - [x] Calculate lines_total from code file content
  - [x] Estimate coverage based on test file presence and assertions
  - [x] Identify critical paths from file paths (orchestrator/, gates/, agents/)
  - [x] Note: Full pytest-cov integration deferred to future story

- [x] Task 3: Implement Uncovered Lines Detection (AC: 4)
  - [x] Create `_detect_uncovered_lines(code_content: str, test_content: str) -> list[tuple[int, int]]`
  - [x] Stub implementation: identify functions without corresponding test functions
  - [x] Return line ranges for uncovered sections
  - [x] Include in CoverageResult.uncovered_lines

- [x] Task 4: Implement Critical Path Validation (AC: 2)
  - [x] Create `_validate_critical_paths(report: CoverageReport) -> list[Finding]`
  - [x] Check if critical files have 100% coverage
  - [x] Generate critical severity findings for any gaps
  - [x] Default critical paths: `orchestrator/`, `gates/`, `agents/core`

- [x] Task 5: Implement Threshold Checking (AC: 3, 6)
  - [x] Create `_check_coverage_threshold(report: CoverageReport, threshold: float) -> tuple[bool, list[Finding]]`
  - [x] Compare overall_coverage against threshold
  - [x] Generate appropriate findings with severity based on gap size
  - [x] Load threshold from config or use 80% default

- [x] Task 6: Integrate Coverage into TEA Node (AC: 5)
  - [x] Update `_validate_artifact()` to call coverage analysis for code files
  - [x] Include coverage findings in ValidationResult
  - [x] Adjust validation score based on coverage percentage
  - [x] Update `_calculate_overall_confidence()` to weight coverage heavily

- [x] Task 7: Add Configuration Support (AC: 6)
  - [x] Check for coverage threshold in state config or YoloConfig
  - [x] Support configurable critical path list
  - [x] Document configuration options

- [x] Task 8: Write Unit Tests for Coverage Types (AC: 1)
  - [x] Test CoverageResult creation and to_dict()
  - [x] Test CoverageReport creation and to_dict()
  - [x] Test immutability (frozen dataclass)
  - [x] Test edge cases (0% coverage, 100% coverage)

- [x] Task 9: Write Unit Tests for Coverage Analyzer (AC: 1, 2)
  - [x] Test coverage calculation for code with tests
  - [x] Test coverage calculation for code without tests
  - [x] Test critical path identification
  - [x] Test empty input handling

- [x] Task 10: Write Unit Tests for Threshold Checking (AC: 3)
  - [x] Test threshold pass (85% coverage, 80% threshold)
  - [x] Test threshold fail (75% coverage, 80% threshold)
  - [x] Test boundary case (exactly 80%)
  - [x] Test custom threshold override

- [x] Task 11: Write Integration Tests (AC: 5)
  - [x] Test coverage integration in tea_node
  - [x] Test ValidationResult includes coverage findings
  - [x] Test overall_confidence reflects coverage
  - [x] Test blocking behavior when coverage too low

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for CoverageResult, CoverageReport
- **ADR-006 (Quality Gates):** Coverage validation is part of confidence scoring gate
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from Story 9.1 TEA implementation
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- Maintain backward compatibility with existing ValidationResult structure

### Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| coverage.py | 7.13.1 | Latest coverage library (for future integration) |
| pytest-cov | latest | Coverage plugin for pytest (for future integration) |
| structlog | latest | Structured logging |

**Note:** For MVP, we implement stub coverage analysis based on heuristics (test file presence, assertion counts). Full pytest-cov programmatic API integration will be a future enhancement.

### Coverage Calculation Strategy (MVP)

Since we're validating artifacts in state (not running actual tests), the MVP approach:

1. **Estimate coverage** based on:
   - Presence of corresponding test files (e.g., `module.py` has `test_module.py`)
   - Count of test functions vs code functions
   - Assertion density in test files

2. **Critical path validation**:
   - Files in `orchestrator/`, `gates/`, `agents/` considered critical
   - These require higher coverage (100% target)

3. **Future enhancement**: Integrate with pytest-cov API to run actual coverage

### Finding Severity Mapping for Coverage

| Scenario | Severity | Description |
|----------|----------|-------------|
| Coverage < 50% | `critical` | Severely under-tested |
| Coverage 50-79% | `high` | Below threshold |
| Coverage 80-89% | `medium` | Acceptable but could improve |
| Coverage 90-99% | `low` | Good coverage |
| Coverage 100% | N/A | No finding needed |
| Critical path < 100% | `critical` | Critical path must have full coverage |

### Project Structure Notes

- **Module Location:** `src/yolo_developer/agents/tea/`
- **New File (optional):** `src/yolo_developer/agents/tea/coverage.py` (or add to types.py)
- **Modified File:** `src/yolo_developer/agents/tea/node.py`
- **Test Location:** `tests/unit/agents/tea/`

### Existing Code to Integrate With

The Story 9.1 implementation provides:
- `_validate_artifact()` - Will be extended to include coverage
- `ValidationResult` - Already has findings field for coverage findings
- `Finding` with `test_coverage` category already defined
- `_calculate_overall_confidence()` - Will be updated to weight coverage

### Key Integration Points

```python
# In _validate_artifact() add coverage analysis
coverage_result = _analyze_coverage_for_artifact(artifact)
if coverage_result.coverage_percentage < threshold:
    findings.append(Finding(
        finding_id=f"COV-{artifact_id}-001",
        category="test_coverage",
        severity="high" if coverage_result.coverage_percentage < 80 else "medium",
        description=f"Coverage {coverage_result.coverage_percentage}% below threshold {threshold}%",
        location=artifact_id,
        remediation=f"Add tests to improve coverage. Uncovered lines: {coverage_result.uncovered_lines}",
    ))
```

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR73 | TEA Agent can validate test coverage meets configured thresholds | CoverageReport with threshold checking |
| FR78 | TEA Agent can block deployment when confidence score < 90% | Coverage below threshold = blocking finding |
| FR79 | TEA Agent can generate test coverage reports with gap analysis | CoverageResult with uncovered_lines |

### Previous Story Learnings Applied

From Story 9.1:
- Use frozen dataclasses for all data structures with to_dict() methods
- Integrate with existing `_validate_artifact()` function
- Add new findings to existing ValidationResult.findings tuple
- Update processing_notes with coverage statistics
- Follow structured logging pattern with structlog

### Git Commit Pattern

```
feat: Implement coverage validation with code review fixes (Story 9.2)
```

### Web Research Findings

- **coverage.py 7.13.1** is the latest version (Dec 2025), supporting Python 3.10-3.15
- **pytest-cov** provides convenient pytest integration but underlying coverage.py API is more flexible
- Programmatic API recommended for clean reports and control
- Branch coverage (not just statement coverage) is important for comprehensive testing
- Configuration can be in pyproject.toml (recommended) or .coveragerc

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-9.2] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR73] - Coverage threshold validation
- [Source: _bmad-output/planning-artifacts/prd.md#FR78] - Deployment blocking
- [Source: _bmad-output/planning-artifacts/prd.md#FR79] - Coverage gap analysis
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] - State management patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate patterns
- [Source: src/yolo_developer/agents/tea/types.py] - Existing TEA types
- [Source: src/yolo_developer/agents/tea/node.py] - Existing TEA node implementation
- [Source: _bmad-output/implementation-artifacts/9-1-create-tea-agent-node.md] - Previous story learnings
- [Web: coverage.py Documentation](https://coverage.readthedocs.io/) - Coverage.py 7.13.1 docs
- [Web: pytest-cov GitHub](https://github.com/pytest-dev/pytest-cov) - pytest-cov plugin

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 125 TEA unit tests pass
- ruff check passes
- mypy strict mode passes

### Completion Notes List

- Implemented `CoverageResult` and `CoverageReport` frozen dataclasses with to_dict() methods
- Implemented heuristic-based coverage analyzer using function-level test matching
- Implemented critical path validation for orchestrator/, gates/, agents/ paths
- Implemented threshold checking with severity mapping (critical < 50%, high < 80%)
- Integrated coverage analysis into _validate_artifact() in tea_node
- Added configuration support via YoloConfig.quality.test_coverage_threshold and quality.critical_paths
- All 11 tasks completed with comprehensive test coverage
- Pre-existing test failures (uv path issues, log capture) are unrelated to this implementation

**Code Review Fixes (8 issues resolved):**
- Fixed docstring parsing bug in `_count_executable_lines()` - multiline docstrings now handled correctly
- Fixed weak test assertion in `test_returns_passed_for_good_code` - now requires score >= 80
- Fixed broad exception swallowing in `get_coverage_threshold_from_config()` - catches specific exceptions
- Fixed tab/indent detection in `_detect_uncovered_functions()` - uses robust indent level calculation
- Fixed useless assertion in `test_coverage_integration.py` - now verifies untested code generates findings
- Added sprint-status.yaml to File List
- Implemented configurable critical paths via `YoloConfig.quality.critical_paths`
- Fixed finding ID collisions - uses path hash for unique IDs

### Change Log

- 2026-01-11: Implemented coverage validation for Story 9.2
- 2026-01-11: Completed adversarial code review - fixed 8 issues

### File List

**New Files:**
- `src/yolo_developer/agents/tea/coverage.py` - Coverage types and analysis functions
- `tests/unit/agents/tea/test_coverage_types.py` - Tests for CoverageResult/CoverageReport
- `tests/unit/agents/tea/test_coverage_analyzer.py` - Tests for analyze_coverage
- `tests/unit/agents/tea/test_critical_path_validation.py` - Tests for validate_critical_paths
- `tests/unit/agents/tea/test_threshold_checking.py` - Tests for check_coverage_threshold
- `tests/unit/agents/tea/test_coverage_integration.py` - Integration tests for tea_node

**Modified Files:**
- `src/yolo_developer/agents/tea/__init__.py` - Added new exports
- `src/yolo_developer/agents/tea/node.py` - Integrated coverage analysis into _validate_artifact()
- `src/yolo_developer/config/schema.py` - Added critical_paths field to QualityConfig
- `tests/unit/agents/tea/test_node.py` - Updated test for coverage integration
- `tests/unit/agents/tea/test_threshold_checking.py` - Fixed unused variable warnings
- `tests/unit/agents/tea/test_coverage_integration.py` - Fixed useless assertion
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status to done

