# Story 3.3: Implement AC Measurability Gate

Status: done

## Story

As a system user,
I want acceptance criteria validated for measurability,
So that every story has criteria that can be objectively tested.

## Acceptance Criteria

1. **AC1: Concrete Condition Check**
   - **Given** the PM agent produces stories with acceptance criteria
   - **When** the AC measurability gate evaluates them
   - **Then** each AC is checked for concrete, verifiable conditions
   - **And** ACs containing observable behaviors pass (e.g., "user sees confirmation message")
   - **And** ACs with vague outcomes fail (e.g., "user has good experience")

2. **AC2: Subjective Term Detection**
   - **Given** acceptance criteria containing subjective language
   - **When** the AC measurability gate evaluates them
   - **Then** subjective terms trigger warnings (not blocking failures)
   - **And** terms like "intuitive", "user-friendly", "clean", "appropriate" are flagged
   - **And** the specific subjective terms are listed in the warning report
   - **And** suggestions for objective alternatives are provided

3. **AC3: Given/When/Then Structure Validation**
   - **Given** acceptance criteria without proper Given/When/Then structure
   - **When** the AC measurability gate evaluates them
   - **Then** missing structure fails the gate (blocking)
   - **And** the failure report identifies which part is missing (Given, When, or Then)
   - **And** examples of proper structure are provided in remediation guidance

4. **AC4: Improvement Suggestions**
   - **Given** one or more ACs fail measurability checks
   - **When** the failure report is generated
   - **Then** specific suggestions for improvement are provided
   - **And** suggestions are tailored to the type of issue (missing structure vs subjective terms)
   - **And** examples of measurable alternatives are included where possible

5. **AC5: Gate Evaluator Registration**
   - **Given** the AC measurability gate evaluator is implemented
   - **When** the gates module is loaded
   - **Then** the evaluator is registered with name "ac_measurability"
   - **And** the evaluator is available via `get_evaluator("ac_measurability")`
   - **And** the evaluator follows the GateEvaluator protocol

6. **AC6: State Integration**
   - **Given** stories exist in the state under `stories` key
   - **When** the AC measurability gate is applied via @quality_gate("ac_measurability")
   - **Then** the gate reads stories from `state["stories"]`
   - **And** each story's acceptance_criteria list is evaluated
   - **And** unmeasurable ACs are identified by story ID and AC index
   - **And** the gate result includes which specific ACs failed

## Tasks / Subtasks

- [x] Task 1: Define AC Measurability Types (AC: 1, 2, 3, 4)
  - [x] Create `src/yolo_developer/gates/gates/ac_measurability.py` module
  - [x] Define `ACMeasurabilityIssue` dataclass (story_id, ac_index, issue_type, description, severity)
  - [x] Define `SUBJECTIVE_TERMS` constant with common subjective terms list
  - [x] Define `GWT_PATTERNS` constant for Given/When/Then detection
  - [x] Export types from `gates/gates/__init__.py`

- [x] Task 2: Implement Subjective Term Detection (AC: 2)
  - [x] Create `detect_subjective_terms(text: str) -> list[tuple[str, int]]` function
  - [x] Match against SUBJECTIVE_TERMS list (case-insensitive)
  - [x] Return list of (term, position) tuples for each match
  - [x] Handle multi-word subjective phrases (e.g., "user friendly", "easy to use")
  - [x] Reuse pattern from testability gate where applicable

- [x] Task 3: Implement Given/When/Then Validation (AC: 3)
  - [x] Create `has_gwt_structure(ac_text: str) -> tuple[bool, list[str]]` function
  - [x] Check for presence of Given, When, and Then keywords (case-insensitive)
  - [x] Return (passed, missing_parts) tuple
  - [x] Support variations: "given", "when", "then" and "GIVEN", "WHEN", "THEN"
  - [x] Handle multi-line AC text

- [x] Task 4: Implement Concrete Condition Detection (AC: 1)
  - [x] Create `has_concrete_condition(ac_text: str) -> bool` function
  - [x] Check for observable outcomes (action verbs with objects)
  - [x] Detect measurable patterns (numbers, states, specific actions)
  - [x] Return True only if AC has verifiable condition in "Then" clause

- [x] Task 5: Implement Improvement Suggestion Generation (AC: 4)
  - [x] Create `generate_improvement_suggestions(issues: list[ACMeasurabilityIssue]) -> dict[str, str]` function
  - [x] Generate targeted suggestions based on issue type
  - [x] For missing GWT: provide template with examples
  - [x] For subjective terms: suggest objective alternatives
  - [x] For vague outcomes: suggest measurable conditions

- [x] Task 6: Implement AC Measurability Evaluator (AC: 1, 2, 3, 5, 6)
  - [x] Create async `ac_measurability_evaluator(context: GateContext) -> GateResult` function
  - [x] Extract stories from `context.state["stories"]`
  - [x] Iterate through each story's acceptance_criteria
  - [x] Collect all issues across all ACs
  - [x] Return GateResult with passed=True only if no blocking issues
  - [x] Include warning-only issues in metadata

- [x] Task 7: Implement Failure Report Generation (AC: 4)
  - [x] Create `generate_ac_measurability_report(issues: list[ACMeasurabilityIssue]) -> str` function
  - [x] Format issues by story ID with AC index
  - [x] Include severity levels (blocking vs warning)
  - [x] Include remediation suggestions for each issue type
  - [x] Provide examples of measurable ACs

- [x] Task 8: Register Evaluator (AC: 5)
  - [x] Register `ac_measurability_evaluator` in module initialization
  - [x] Use `register_evaluator("ac_measurability", ac_measurability_evaluator)`
  - [x] Update `gates/gates/__init__.py` to auto-register on import
  - [x] Verify registration in `gates/__init__.py` exports

- [x] Task 9: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/gates/test_ac_measurability.py`
  - [x] Test subjective term detection with various inputs
  - [x] Test Given/When/Then structure validation
  - [x] Test concrete condition detection
  - [x] Test full evaluator with passing stories (proper GWT structure)
  - [x] Test full evaluator with failing stories (missing GWT)
  - [x] Test full evaluator with warning-only issues (subjective terms)
  - [x] Test improvement suggestion generation
  - [x] Test failure report generation format
  - [x] Test input validation (non-list stories, non-dict story)

- [x] Task 10: Write Integration Tests (AC: 5, 6)
  - [x] Create `tests/integration/test_ac_measurability_gate.py`
  - [x] Test gate decorator integration with ac_measurability evaluator
  - [x] Test state reading from `state["stories"]`
  - [x] Test gate blocking behavior with unmeasurable ACs
  - [x] Test gate passing behavior with measurable ACs
  - [x] Test advisory mode with warnings only

## Dev Notes

### Architecture Compliance

- **ADR-006 (Quality Gate Pattern):** Decorator-based gates per architecture specification
- **FR20:** System can verify acceptance criteria measurability from PM output
- **Epic 3 Goal:** Validate artifacts at every agent boundary and block low-quality handoffs

### Technical Requirements

- **Evaluator Protocol:** Must implement `GateEvaluator` protocol from Story 3.1
- **Async Pattern:** Evaluator must be async function per architecture
- **State Access:** Read stories from `state["stories"]` key
- **Structured Logging:** Use structlog for all log messages

### Subjective Terms to Detect (Initial List)

```python
SUBJECTIVE_TERMS = [
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
]
```

### Given/When/Then Structure Patterns

```python
# Patterns for detecting GWT structure
GWT_PATTERNS = {
    "given": re.compile(r"\bgiven\b", re.IGNORECASE),
    "when": re.compile(r"\bwhen\b", re.IGNORECASE),
    "then": re.compile(r"\bthen\b", re.IGNORECASE),
}
```

### Difference from Testability Gate (Story 3.2)

- **Testability Gate (3.2):** Validates **requirements** produced by Analyst
- **AC Measurability Gate (3.3):** Validates **acceptance criteria** in stories produced by PM
- The testability gate checks the raw requirement text
- The AC measurability gate checks the structured acceptance criteria within stories
- This gate enforces Given/When/Then structure which testability gate does not

### Example Stories

**Measurable AC (PASS):**
```python
{
    "id": "story-001",
    "title": "User Login",
    "acceptance_criteria": [
        {
            "content": "Given a user with valid credentials, When they submit the login form, Then they are redirected to the dashboard"
        },
        {
            "content": "Given an invalid password, When the user submits login, Then an error message 'Invalid credentials' is displayed"
        }
    ]
}
```

**Unmeasurable AC (FAIL - missing structure):**
```python
{
    "id": "story-002",
    "title": "Improve UX",
    "acceptance_criteria": [
        {
            "content": "The user interface should be intuitive"  # Missing GWT, subjective
        }
    ]
}
```

**Warning-only (PASS with warnings - subjective but has structure):**
```python
{
    "id": "story-003",
    "title": "Dashboard Display",
    "acceptance_criteria": [
        {
            "content": "Given a logged-in user, When they view the dashboard, Then the layout is clean and user-friendly"  # Has GWT but subjective terms
        }
    ]
}
```

### File Structure

```
src/yolo_developer/gates/
├── __init__.py           # UPDATE: Export ac_measurability gate
├── types.py              # From Story 3.1
├── decorator.py          # From Story 3.1
├── evaluators.py         # From Story 3.1
└── gates/
    ├── __init__.py       # UPDATE: Add ac_measurability exports
    ├── testability.py    # From Story 3.2
    └── ac_measurability.py  # NEW: AC measurability gate implementation
```

### Previous Story Intelligence (from Story 3.2)

**Patterns to Apply:**
1. Use frozen dataclasses for issue types (immutable)
2. Evaluator is async callable: `async def evaluator(ctx: GateContext) -> GateResult`
3. Register via `register_evaluator(gate_name, evaluator)`
4. State is accessible via `context.state`
5. Use `GateResult.to_dict()` for state serialization
6. Add autouse fixture in tests to re-register evaluator after `clear_evaluators()` calls
7. Validate input types before processing (requirements/stories must be list, items must be dict)
8. Use non-greedy regex `.*?` to avoid ReDoS vulnerabilities

**Key Files to Reference:**
- `src/yolo_developer/gates/types.py` - GateResult, GateContext dataclasses
- `src/yolo_developer/gates/evaluators.py` - GateEvaluator protocol, registration functions
- `src/yolo_developer/gates/decorator.py` - @quality_gate decorator usage
- `src/yolo_developer/gates/gates/testability.py` - Pattern for detect_vague_terms, evaluator structure
- `tests/unit/gates/test_testability.py` - Test patterns including autouse fixture
- `tests/integration/test_testability_gate.py` - Integration test patterns

### Story Schema Expected

```python
# Stories are expected to be list of dicts in state
# Minimum structure for AC measurability evaluation:
Story = TypedDict("Story", {
    "id": str,                    # Unique identifier
    "title": str,                 # Story title
    "acceptance_criteria": list,  # List of AC dicts
})

AcceptanceCriterion = TypedDict("AcceptanceCriterion", {
    "content": str,               # The AC text (should be GWT format)
})
```

### Severity Levels

- **blocking:** Missing Given/When/Then structure (gate fails, handoff blocked)
- **warning:** Subjective terms present but structure is valid (gate passes with warnings)

### Testing Standards

- Use pytest-asyncio for async tests
- Create mock stories in test fixtures
- Test both passing and failing scenarios
- Test edge cases (empty stories, empty ACs, missing keys)
- Verify structured logging output
- Add autouse fixture to ensure evaluator registration

### References

- [Source: architecture.md#ADR-006] - Quality Gate Pattern
- [Source: epics.md#Story-3.3] - Implement AC Measurability Gate requirements
- [Source: prd.md#FR20] - System can verify acceptance criteria measurability from PM output
- [Story 3.1 Implementation] - Gate decorator framework
- [Story 3.2 Implementation] - Testability gate pattern (similar implementation)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Test suite: 843 tests passing
- Unit tests: 55 tests in tests/unit/gates/test_ac_measurability.py
- Integration tests: 11 tests in tests/integration/test_ac_measurability_gate.py

### Completion Notes List

1. **SUBJECTIVE_TERMS expanded**: Added 35 subjective terms (vs 29 in dev notes) including additional terms like "fast", "quick", "slow", "hard to use", "friendly"

2. **CONCRETE_CONDITION_PATTERNS**: Implemented 10 regex patterns for detecting:
   - Specific UI outcomes (sees, displays, shows, appears)
   - State changes (is created, updated, deleted, saved)
   - Navigation (redirected to, navigated to, taken to)
   - Error handling (error message, error is displayed)
   - Specific values (equals, contains, includes, matches)
   - Numbers with context (at least, at most, exactly)
   - Boolean outcomes (succeeds, fails, passes, completes, returns)
   - Specific quoted text messages
   - Enabled/disabled states
   - Count expectations (X items, records, results)

3. **Input validation**: Validates that stories is a list, each story is a dict, acceptance_criteria is a list, and each AC is a dict with content key

4. **Test isolation**: Added autouse fixture in both unit and integration tests to re-register evaluator after `clear_evaluators()` calls from other tests

5. **Multi-word phrase matching**: detect_subjective_terms handles overlapping matches by sorting terms by length descending and tracking matched positions

6. **Severity differentiation**: Missing GWT structure is blocking (gate fails), subjective terms are warnings (gate passes with warnings)

7. **Vague outcome detection**: When GWT structure is present but Then clause lacks concrete conditions, generates a warning (not blocking)

### File List

**Created:**
- `src/yolo_developer/gates/gates/ac_measurability.py` - Main implementation (~600 lines)
- `tests/unit/gates/test_ac_measurability.py` - Unit tests (55 tests)
- `tests/integration/test_ac_measurability_gate.py` - Integration tests (11 tests)

**Modified:**
- `src/yolo_developer/gates/gates/__init__.py` - Added exports for ac_measurability components
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status

### Code Review Fixes Applied

1. **Performance optimization**: Pre-sorted SUBJECTIVE_TERMS at module level (`_SUBJECTIVE_TERMS_SORTED`) to avoid redundant sorting on every `detect_subjective_terms()` call

2. **Export cleanup**: Fixed `__all__` list in `__init__.py` to follow isort-style sorting and exported `CONCRETE_CONDITION_PATTERNS` constant

3. **Test coverage**: Added tests for `CONCRETE_CONDITION_PATTERNS` constant (2 tests) and additional negative tests for `has_concrete_condition` (5 tests)
