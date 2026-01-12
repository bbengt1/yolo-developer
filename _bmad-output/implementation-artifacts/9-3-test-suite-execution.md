# Story 9.3: Test Suite Execution

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want tests executed and results reported,
So that I know if the code works correctly.

## Acceptance Criteria

1. **AC1: Test Execution**
   - **Given** code with tests exists in dev_output artifacts
   - **When** the test suite execution runs
   - **Then** all tests are executed (or simulated for MVP)
   - **And** tests are identified from test_file artifacts
   - **And** the execution is logged with structlog

2. **AC2: Pass/Fail Reporting**
   - **Given** test execution completes
   - **When** results are compiled
   - **Then** pass/fail counts are reported accurately
   - **And** the TestExecutionResult dataclass contains passed_count, failed_count, error_count
   - **And** overall status is determined: "passed" (all pass), "failed" (any fail), "error" (execution errors)

3. **AC3: Failure Details**
   - **Given** tests have failures
   - **When** failure details are recorded
   - **Then** each failure includes test name, file path, and error message
   - **And** failure details are stored in TestFailure dataclass
   - **And** failures are included in TestExecutionResult.failures tuple

4. **AC4: Test Duration Tracking**
   - **Given** test execution occurs
   - **When** timing is recorded
   - **Then** test duration is recorded in milliseconds
   - **And** duration is stored in TestExecutionResult.duration_ms
   - **And** start_time and end_time ISO timestamps are captured

5. **AC5: Integration with TEA Node**
   - **Given** tea_node is processing artifacts
   - **When** test files are present in artifacts
   - **Then** test execution is triggered automatically
   - **And** TestExecutionResult is included in ValidationResult or TEAOutput
   - **And** overall_confidence is adjusted based on test pass rate

6. **AC6: Finding Generation**
   - **Given** test failures occur
   - **When** findings are generated
   - **Then** each failure generates a Finding with category="test_coverage"
   - **And** severity is "critical" for test errors, "high" for failures
   - **And** remediation includes the error message and suggested fix

## Tasks / Subtasks

- [x] Task 1: Create Test Execution Types (AC: 2, 3, 4)
  - [x] Create `TestFailure` frozen dataclass with: test_name, file_path, error_message, failure_type
  - [x] Create `TestExecutionResult` frozen dataclass with: status, passed_count, failed_count, error_count, failures, duration_ms, start_time, end_time
  - [x] Add `to_dict()` methods for serialization
  - [x] Add types to `agents/tea/execution.py` (new file)

- [x] Task 2: Implement Test Discovery (AC: 1)
  - [x] Create `discover_tests(test_content: str) -> list[str]` function
  - [x] Extract test function names matching `test_*` or `Test*` patterns
  - [x] Handle async test functions (`async def test_*`)
  - [x] Return list of discovered test names

- [x] Task 3: Implement Test Execution Simulation (AC: 1, 2, 4)
  - [x] Create `execute_tests(test_files: list[dict]) -> TestExecutionResult` function
  - [x] For MVP: Simulate execution based on test file analysis (heuristic-based)
  - [x] Count test functions as passed_count baseline
  - [x] Detect obvious issues (syntax errors, missing imports) as failures
  - [x] Track execution duration with time.monotonic()
  - [x] Note: Real pytest execution deferred to future story

- [x] Task 4: Implement Failure Detection Heuristics (AC: 3)
  - [x] Create `detect_test_issues(test_content: str, file_path: str) -> list[TestFailure]` function
  - [x] Check for common issues: missing assertions, incomplete tests, TODO markers
  - [x] Check for obvious syntax issues that would cause test failures
  - [x] Generate TestFailure for each detected issue

- [x] Task 5: Implement Finding Generation (AC: 6)
  - [x] Create `generate_test_findings(result: TestExecutionResult) -> list[Finding]` function
  - [x] Convert each TestFailure to a Finding
  - [x] Map failure_type to severity (error -> critical, failure -> high)
  - [x] Include remediation guidance in finding

- [x] Task 6: Integrate into TEA Node (AC: 5)
  - [x] Update `tea_node()` to call test execution for test artifacts
  - [x] Add test execution results to TEAOutput
  - [x] Adjust confidence scoring based on test pass rate
  - [x] Include test findings in validation results

- [x] Task 7: Add TEAOutput Extension (AC: 5)
  - [x] Add `test_execution_result: TestExecutionResult | None` field to TEAOutput
  - [x] Update TEAOutput.to_dict() to include test execution data
  - [x] Handle None case when no tests are present

- [x] Task 8: Write Unit Tests for Execution Types (AC: 2, 3)
  - [x] Test TestFailure creation and to_dict()
  - [x] Test TestExecutionResult creation and to_dict()
  - [x] Test immutability (frozen dataclass)
  - [x] Test edge cases (0 tests, all pass, all fail)

- [x] Task 9: Write Unit Tests for Test Discovery (AC: 1)
  - [x] Test discovery of standard test functions
  - [x] Test discovery of async test functions
  - [x] Test discovery of test classes
  - [x] Test empty file handling

- [x] Task 10: Write Unit Tests for Execution Simulation (AC: 1, 2, 4)
  - [x] Test execution with passing tests
  - [x] Test execution with failing tests
  - [x] Test duration tracking
  - [x] Test status determination logic

- [x] Task 11: Write Integration Tests (AC: 5, 6)
  - [x] Test test execution integration in tea_node
  - [x] Test finding generation for failures
  - [x] Test confidence adjustment based on pass rate
  - [x] Test TEAOutput includes test results

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for TestFailure, TestExecutionResult
- **ADR-006 (Quality Gates):** Test execution feeds into confidence scoring gate
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from Story 9.1 and 9.2 TEA implementations
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- Maintain backward compatibility with existing TEAOutput structure

### Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| pytest | 8.x | Test framework (for future real execution) |
| structlog | latest | Structured logging |
| time | stdlib | Duration tracking with monotonic() |

**Note:** For MVP, we implement heuristic-based test execution simulation that analyzes test file content. Real pytest programmatic execution will be a future enhancement.

### Test Execution Strategy (MVP)

Since we're validating artifacts in state (not running actual pytest), the MVP approach:

1. **Discover tests** from test file content:
   - Extract `def test_*` and `async def test_*` functions
   - Extract `class Test*` test classes
   - Count total discovered tests

2. **Simulate execution** via heuristics:
   - Assume well-formed tests pass
   - Detect obvious issues that would cause failures:
     - Missing assertions (test does nothing)
     - TODO/FIXME markers indicating incomplete tests
     - Obvious syntax issues
     - Missing required imports

3. **Future enhancement**: Integrate with pytest API for real execution

### TestExecutionResult Status Logic

| Condition | Status |
|-----------|--------|
| All tests pass | `passed` |
| Any test fails (but no errors) | `failed` |
| Execution errors occurred | `error` |
| No tests found | `passed` (vacuously true) |

### Confidence Adjustment Formula

```python
# Pass rate impacts confidence
pass_rate = passed_count / (passed_count + failed_count + error_count)
confidence_adjustment = pass_rate * 0.3  # Up to 30% weight from test results

# Errors have higher penalty
if error_count > 0:
    confidence_adjustment -= 0.1 * error_count  # -10% per error
```

### Finding Severity Mapping for Test Results

| Scenario | Severity | Description |
|----------|----------|-------------|
| Test execution error | `critical` | Test couldn't run (syntax error, import failure) |
| Test failure | `high` | Test ran but assertion failed |
| Missing assertions | `medium` | Test exists but doesn't verify anything |
| Incomplete test (TODO) | `low` | Test marked as incomplete |

### Project Structure Notes

- **Module Location:** `src/yolo_developer/agents/tea/`
- **New File:** `src/yolo_developer/agents/tea/execution.py`
- **Modified Files:**
  - `src/yolo_developer/agents/tea/node.py` - Add test execution call
  - `src/yolo_developer/agents/tea/types.py` - Add test_execution_result to TEAOutput
  - `src/yolo_developer/agents/tea/__init__.py` - Export new types
- **Test Location:** `tests/unit/agents/tea/`

### Existing Code to Integrate With

The Story 9.1 and 9.2 implementations provide:
- `_extract_artifacts_for_validation()` - Already separates test files
- `_validate_artifact()` - Will call test execution for test files
- `ValidationResult` - Already has findings field
- `Finding` with `test_coverage` category already defined
- `_calculate_overall_confidence()` - Will be updated to weight test results
- `CoverageResult`, `CoverageReport` - Similar pattern for new types

### Key Integration Points

```python
# In tea_node(), after artifact extraction:
test_artifacts = [a for a in artifacts if a.get("type") == "test_file"]

if test_artifacts:
    test_result = execute_tests(test_artifacts)
    test_findings = _generate_test_findings(test_result)

    # Add to validation findings
    findings.extend(test_findings)

    # Adjust confidence based on test pass rate
    if test_result.passed_count + test_result.failed_count > 0:
        pass_rate = test_result.passed_count / (test_result.passed_count + test_result.failed_count)
        confidence_adjustment = pass_rate * 0.3
```

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR74 | TEA Agent can run automated test suites and report results | TestExecutionResult with pass/fail counts |
| FR75 | TEA Agent can calculate deployment confidence scores | Confidence adjusted by test pass rate |
| FR79 | TEA Agent can generate test coverage reports with gap analysis | Test failure details with remediation |

### Previous Story Learnings Applied

From Story 9.1 and 9.2:
- Use frozen dataclasses for all data structures with to_dict() methods
- Integrate with existing `tea_node()` function flow
- Add new findings to existing ValidationResult.findings tuple
- Update processing_notes with test execution statistics
- Follow structured logging pattern with structlog
- Handle missing/empty content gracefully
- Use hash-based IDs for unique finding identification

### Git Commit Pattern

```
feat: Implement test suite execution with code review fixes (Story 9.3)
```

### Sample TestExecutionResult Output

```python
TestExecutionResult(
    status="failed",
    passed_count=8,
    failed_count=2,
    error_count=0,
    failures=(
        TestFailure(
            test_name="test_invalid_input",
            file_path="tests/test_validation.py",
            error_message="Missing assertion - test does not verify anything",
            failure_type="no_assertion",
        ),
        TestFailure(
            test_name="test_edge_case",
            file_path="tests/test_validation.py",
            error_message="Test marked as TODO - incomplete implementation",
            failure_type="incomplete",
        ),
    ),
    duration_ms=15,
    start_time="2026-01-12T10:00:00.000Z",
    end_time="2026-01-12T10:00:00.015Z",
)
```

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-9.3] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR74] - Test suite execution
- [Source: _bmad-output/planning-artifacts/prd.md#FR75] - Confidence scoring
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] - State management patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate patterns
- [Source: src/yolo_developer/agents/tea/types.py] - Existing TEA types
- [Source: src/yolo_developer/agents/tea/node.py] - Existing TEA node implementation
- [Source: src/yolo_developer/agents/tea/coverage.py] - Coverage patterns to follow
- [Source: _bmad-output/implementation-artifacts/9-1-create-tea-agent-node.md] - Story 9.1 patterns
- [Source: _bmad-output/implementation-artifacts/9-2-coverage-validation.md] - Story 9.2 patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 182 TEA agent tests pass
- mypy type checking passes on all TEA module files
- ruff linting passes

### Completion Notes List

- Created `execution.py` with TestFailure, TestExecutionResult, FailureType, ExecutionStatus types
- Implemented `discover_tests()` for test function and class discovery using regex patterns
- Implemented `execute_tests()` for heuristic-based test execution simulation
- Implemented `detect_test_issues()` to find missing assertions, TODO markers, and stub tests
- Implemented `generate_test_findings()` to convert TestFailure to Finding with severity mapping
- Extended TEAOutput with optional `test_execution_result` field
- Integrated test execution into `tea_node()` with confidence adjustment based on pass rate
- Updated `__init__.py` to export all new types and functions
- Added comprehensive unit tests for all new functionality
- Added integration tests for TEA node test execution

### Change Log

- 2026-01-12: Implemented Story 9.3 test suite execution feature
  - Created new execution.py module with types and functions
  - Extended TEAOutput with test_execution_result field
  - Integrated test execution into tea_node workflow
  - Added 57 new tests across 5 test files

### File List

**New Files:**
- src/yolo_developer/agents/tea/execution.py
- tests/unit/agents/tea/test_execution_types.py
- tests/unit/agents/tea/test_test_discovery.py
- tests/unit/agents/tea/test_test_execution.py
- tests/unit/agents/tea/test_finding_generation.py
- tests/unit/agents/tea/test_tea_integration.py

**Modified Files:**
- src/yolo_developer/agents/tea/__init__.py
- src/yolo_developer/agents/tea/node.py
- src/yolo_developer/agents/tea/types.py
- src/yolo_developer/agents/tea/coverage.py (ruff formatting only)
