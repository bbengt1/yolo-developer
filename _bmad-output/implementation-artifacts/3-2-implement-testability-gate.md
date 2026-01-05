# Story 3.2: Implement Testability Gate

Status: done

## Story

As a system user,
I want requirements validated for testability,
So that only requirements that can be verified proceed to implementation.

## Acceptance Criteria

1. **AC1: Measurability Check**
   - **Given** the Analyst agent produces requirements
   - **When** the testability gate evaluates them
   - **Then** each requirement is checked for measurability
   - **And** requirements with concrete success criteria pass
   - **And** requirements without measurable outcomes fail

2. **AC2: Vague Term Detection**
   - **Given** requirements containing vague or subjective terms
   - **When** the testability gate evaluates them
   - **Then** terms like "fast", "easy", "simple", "user-friendly", "intuitive" are flagged
   - **And** the specific vague terms are listed in the failure report
   - **And** the location of each vague term is identified

3. **AC3: Success Criteria Validation**
   - **Given** a requirement without clear success criteria
   - **When** the testability gate evaluates it
   - **Then** the gate fails with reason "No clear success criteria"
   - **And** the failure report suggests what kind of criteria would make it testable
   - **And** examples of testable alternatives are provided where possible

4. **AC4: Failure Report Generation**
   - **Given** one or more requirements fail the testability gate
   - **When** the failure report is generated
   - **Then** each untestable requirement is listed with its failure reason
   - **And** the report includes severity level (blocking vs warning)
   - **And** remediation guidance is provided for each failure

5. **AC5: Gate Evaluator Registration**
   - **Given** the testability gate evaluator is implemented
   - **When** the gates module is loaded
   - **Then** the evaluator is registered with name "testability"
   - **And** the evaluator is available via `get_evaluator("testability")`
   - **And** the evaluator follows the GateEvaluator protocol

6. **AC6: State Integration**
   - **Given** requirements exist in the state under `requirements` key
   - **When** the testability gate is applied via @quality_gate("testability")
   - **Then** the gate reads requirements from `state["requirements"]`
   - **And** untestable requirements are identified by index or ID
   - **And** the gate result includes which specific requirements failed

## Tasks / Subtasks

- [x] Task 1: Define Testability Types (AC: 1, 2, 3, 4)
  - [x] Create `src/yolo_developer/gates/gates/testability.py` module
  - [x] Define `TestabilityIssue` dataclass (requirement_id, issue_type, description, severity)
  - [x] Define `TestabilityResult` dataclass (passed, issues, suggestions) - Note: Using GateResult from types.py instead per architecture
  - [x] Define `VAGUE_TERMS` constant with common vague terms list (38 terms)
  - [x] Export types from `gates/gates/__init__.py`

- [x] Task 2: Implement Vague Term Detection (AC: 2)
  - [x] Create `detect_vague_terms(text: str) -> list[tuple[str, int]]` function
  - [x] Match against VAGUE_TERMS list (case-insensitive)
  - [x] Return list of (term, position) tuples for each match
  - [x] Handle multi-word vague phrases (e.g., "user friendly", "easy to use")

- [x] Task 3: Implement Success Criteria Detection (AC: 1, 3)
  - [x] Create `has_success_criteria(requirement: dict) -> bool` function
  - [x] Check for presence of measurable keywords (numbers, percentages, timeframes)
  - [x] Check for Given/When/Then or similar structured format
  - [x] Detect quantifiable outcomes vs vague outcomes

- [x] Task 4: Implement Testability Evaluator (AC: 1, 2, 3, 5, 6)
  - [x] Create async `testability_evaluator(context: GateContext) -> GateResult` function
  - [x] Extract requirements from `context.state["requirements"]`
  - [x] Iterate through each requirement and check testability
  - [x] Collect all issues across all requirements
  - [x] Return GateResult with passed=True only if all requirements pass

- [x] Task 5: Implement Failure Report Generation (AC: 4)
  - [x] Create `generate_testability_report(issues: list[TestabilityIssue]) -> str` function
  - [x] Format issues by requirement with severity levels
  - [x] Include remediation suggestions for each issue type
  - [x] Provide examples of testable alternatives where applicable

- [x] Task 6: Register Evaluator (AC: 5)
  - [x] Register `testability_evaluator` in module initialization
  - [x] Use `register_evaluator("testability", testability_evaluator)`
  - [x] Update `gates/gates/__init__.py` to auto-register on import
  - [x] Verify registration in `gates/__init__.py` exports

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/gates/test_testability.py`
  - [x] Test vague term detection with various inputs
  - [x] Test success criteria detection
  - [x] Test full evaluator with passing requirements
  - [x] Test full evaluator with failing requirements (vague terms)
  - [x] Test full evaluator with failing requirements (no success criteria)
  - [x] Test failure report generation format

- [x] Task 8: Write Integration Tests (AC: 5, 6)
  - [x] Create `tests/integration/test_testability_gate.py`
  - [x] Test gate decorator integration with testability evaluator
  - [x] Test state reading from `state["requirements"]`
  - [x] Test gate blocking behavior with untestable requirements
  - [x] Test gate passing behavior with testable requirements

## Dev Notes

### Architecture Compliance

- **ADR-006 (Quality Gate Pattern):** Decorator-based gates per architecture specification
- **FR19:** System can assess testability of requirements produced by Analyst
- **Epic 3 Goal:** Validate artifacts at every agent boundary and block low-quality handoffs

### Technical Requirements

- **Evaluator Protocol:** Must implement `GateEvaluator` protocol from Story 3.1
- **Async Pattern:** Evaluator must be async function per architecture
- **State Access:** Read requirements from `state["requirements"]` key
- **Structured Logging:** Use structlog for all log messages

### Vague Terms to Detect (Initial List)

```python
VAGUE_TERMS = [
    "fast", "quick", "slow",
    "easy", "simple", "complex",
    "good", "bad", "better", "best",
    "user-friendly", "user friendly", "intuitive",
    "efficient", "effective", "optimal",
    "robust", "scalable", "performant",
    "nice", "beautiful", "clean",
    "appropriate", "reasonable", "adequate",
    "seamless", "smooth", "natural",
    "modern", "innovative", "cutting-edge",
]
```

### Testability Criteria (Reference)

A requirement is considered testable if it has:
1. **Quantifiable metrics:** Numbers, percentages, time bounds
2. **Observable outcomes:** Clear what success looks like
3. **Verifiable conditions:** Given/When/Then or equivalent structure
4. **No subjective qualifiers:** No vague terms that require human judgment

### Example Requirements

**Testable (PASS):**
- "The API responds within 500ms for 95% of requests"
- "User login succeeds when valid credentials are provided"
- "The system stores up to 10,000 records per project"

**Untestable (FAIL):**
- "The system should be fast" (vague: "fast")
- "User experience should be intuitive" (vague: "intuitive")
- "The code should be clean and readable" (vague: "clean", "readable")

### File Structure

```
src/yolo_developer/gates/
├── __init__.py           # UPDATE: Export testability gate
├── types.py              # From Story 3.1
├── decorator.py          # From Story 3.1
├── evaluators.py         # From Story 3.1
└── gates/
    ├── __init__.py       # NEW: Gate implementations package
    └── testability.py    # NEW: Testability gate implementation
```

### Previous Story Intelligence (from Story 3.1)

**Patterns to Apply:**
1. Use frozen dataclasses for result types (immutable)
2. Evaluator is async callable: `async def evaluator(ctx: GateContext) -> GateResult`
3. Register via `register_evaluator(gate_name, evaluator)`
4. State is accessible via `context.state`
5. Use `GateResult.to_dict()` for state serialization

**Key Files to Reference:**
- `src/yolo_developer/gates/types.py` - GateResult, GateContext dataclasses
- `src/yolo_developer/gates/evaluators.py` - GateEvaluator protocol, registration functions
- `src/yolo_developer/gates/decorator.py` - @quality_gate decorator usage
- `tests/unit/gates/test_decorator.py` - Test patterns for evaluators

### Testing Standards

- Use pytest-asyncio for async tests
- Create mock requirements in test fixtures
- Test both passing and failing scenarios
- Test edge cases (empty requirements, missing keys)
- Verify structured logging output

### Requirement Schema Expected

```python
# Requirements are expected to be list of dicts in state
# Minimum structure for testability evaluation:
Requirement = TypedDict("Requirement", {
    "id": str,           # Unique identifier
    "content": str,      # The requirement text
    "success_criteria": str | None,  # Optional explicit criteria
})
```

### References

- [Source: architecture.md#ADR-006] - Quality Gate Pattern
- [Source: epics.md#Story-3.2] - Implement Testability Gate requirements
- [Source: prd.md#FR19] - System can assess testability of requirements produced by Analyst
- [Story 3.1 Implementation] - Gate decorator framework

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Test suite: 774 tests passing
- Unit tests: 29 tests in tests/unit/gates/test_testability.py
- Integration tests: 9 tests in tests/integration/test_testability_gate.py

### Completion Notes List

1. **VAGUE_TERMS expanded**: Added 38 vague terms (vs 30 in dev notes) including additional multi-word phrases like "cutting edge", "easy to use"

2. **MEASURABLE_PATTERNS**: Implemented 6 regex patterns for detecting:
   - Numbers with units (ms, seconds, MB, etc.)
   - Percentages
   - Specific counts (at least, at most, up to, etc.)
   - Given/When/Then format
   - Comparison operators with numbers
   - Boolean outcomes (succeeds, fails, completes)

3. **TestabilityResult not needed**: Used existing `GateResult` from types.py per architecture pattern. `TestabilityIssue` is used internally for collecting issues.

4. **Test isolation fix**: Added autouse fixture in both unit and integration tests to re-register evaluator after `clear_evaluators()` calls from other tests

5. **Multi-word phrase matching**: detect_vague_terms handles overlapping matches by prioritizing longer phrases first and tracking matched positions

6. **Success criteria field support**: has_success_criteria checks both the main content and an optional explicit `success_criteria` field

### File List

**Created:**
- `src/yolo_developer/gates/gates/testability.py` - Main implementation (411 lines)
- `tests/unit/gates/test_testability.py` - Unit tests (480 lines, 33 tests)
- `tests/integration/test_testability_gate.py` - Integration tests (259 lines, 9 tests)

**Modified:**
- `src/yolo_developer/gates/gates/__init__.py` - Added exports for testability components

### Code Review Fixes Applied

**Review Date:** 2026-01-05
**Reviewer:** Claude Opus 4.5 (Adversarial Code Review)

**Issues Fixed:**

1. **[HIGH] Input Validation** - Added validation that `requirements` is a list and each item is a dict. Prevents runtime crashes on malformed input.

2. **[MEDIUM] ReDoS Prevention** - Changed greedy `.*` to non-greedy `.*?` in Given/When/Then regex pattern.

3. **[MEDIUM] Test Coverage** - Added 4 new tests for invalid input handling:
   - `test_evaluator_rejects_non_list_requirements`
   - `test_evaluator_handles_non_dict_requirement`
   - `test_evaluator_handles_missing_content_key`

4. **[MEDIUM] Documentation** - Updated file line counts to reflect actual values.
