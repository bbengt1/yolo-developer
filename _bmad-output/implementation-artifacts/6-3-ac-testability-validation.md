# Story 6.3: AC Testability Validation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want acceptance criteria that are testable,
So that I know exactly when a story is complete.

## Acceptance Criteria

1. **AC1: ACs Use Given/When/Then Format**
   - **Given** a story is being created with acceptance criteria
   - **When** testability is validated
   - **Then** each AC follows Given/When/Then format
   - **And** ACs are stored as `AcceptanceCriterion` objects with `given`, `when`, `then` fields
   - **And** ACs missing any of these fields fail validation

2. **AC2: Conditions Are Concrete and Measurable**
   - **Given** generated acceptance criteria text
   - **When** testability validation runs
   - **Then** ACs containing vague terms are flagged (fast, easy, simple, intuitive, etc.)
   - **And** vague term detection uses the same `VAGUE_TERMS` frozenset from analyst module
   - **And** flagged ACs are returned with specific vague terms identified

3. **AC3: Edge Cases Are Included**
   - **Given** a story with acceptance criteria
   - **When** edge case analysis runs
   - **Then** each story is checked for common edge case patterns (error handling, empty input, boundary conditions)
   - **And** missing edge case patterns are identified as warnings (not blocking)
   - **And** suggestions for missing edge cases are provided

4. **AC4: AC Count Is Appropriate for Story Size**
   - **Given** a story with acceptance criteria
   - **When** AC count is validated
   - **Then** stories with fewer than 2 ACs generate a warning
   - **And** stories with more than 8 ACs generate a warning (consider splitting)
   - **And** the typical range is 2-5 ACs per story

5. **AC5: Validation Returns Structured Results**
   - **Given** validation is performed on a story
   - **When** results are returned
   - **Then** results include a `TestabilityResult` with:
     - `is_valid`: bool (True if all critical checks pass)
     - `vague_terms_found`: list of (ac_id, term) tuples
     - `missing_edge_cases`: list of suggested edge case patterns
     - `ac_count_warning`: str | None (warning message if count is unusual)
     - `validation_notes`: list of str (detailed findings)

## Tasks / Subtasks

- [x] Task 1: Create Testability Types (AC: 5)
  - [x] Create `TestabilityResult` TypedDict in `src/yolo_developer/agents/pm/types.py`
  - [x] ~~Add `TestabilitySeverity` enum~~ (Removed during code review - unused dead code)
  - [x] TypedDict used for TestabilityResult (no to_dict needed)

- [x] Task 2: Implement VAGUE_TERMS Detection (AC: 2)
  - [x] Import `VAGUE_TERMS` from pm.llm module (code review fix: single source of truth)
  - [x] Create `_detect_vague_terms(ac: AcceptanceCriterion) -> list[str]` function
  - [x] Check `given`, `when`, `then`, and `and_clauses` fields
  - [x] Return list of vague terms found in each AC

- [x] Task 3: Implement Given/When/Then Format Validation (AC: 1)
  - [x] Create `_validate_ac_structure(ac: AcceptanceCriterion) -> list[str]` function
  - [x] Check that `given`, `when`, `then` are non-empty strings
  - [x] Flag ACs where any field is empty or whitespace-only
  - [x] Return list of structural issues

- [x] Task 4: Implement Edge Case Detection (AC: 3)
  - [x] Create `_check_edge_cases(story: Story) -> list[str]` function
  - [x] Define common edge case patterns as constants:
    - Error/exception handling: "error", "fail", "invalid", "exception", "reject", "denied"
    - Empty/null input: "empty", "null", "none", "missing", "blank", "undefined"
    - Boundary conditions: "maximum", "minimum", "limit", "boundary", "overflow", "threshold"
  - [x] Scan AC text for presence of these patterns
  - [x] Return list of missing edge case categories

- [x] Task 5: Implement AC Count Validation (AC: 4)
  - [x] Create `_validate_ac_count(story: Story) -> str | None` function
  - [x] Return warning if count < 2: "Story has only {n} AC(s); consider adding more for completeness"
  - [x] Return warning if count > 8: "Story has {n} ACs; consider splitting into smaller stories"
  - [x] Return None for acceptable range (2-8)

- [x] Task 6: Create Main Validation Function (AC: all)
  - [x] Create `validate_story_testability(story: Story) -> TestabilityResult` function
  - [x] Orchestrate calls to all validation sub-functions
  - [x] Aggregate results into `TestabilityResult`
  - [x] Set `is_valid` to False if any vague terms found OR structural issues exist
  - [x] Add structured logging for validation steps

- [x] Task 7: Integrate Validation into pm_node (AC: all)
  - [x] Import `validate_story_testability` in `node.py`
  - [x] After story transformation, validate each story
  - [x] Add validation results to processing notes
  - [x] Log validation warnings but don't block story creation

- [x] Task 8: Write Unit Tests for Vague Term Detection (AC: 2)
  - [x] Test detection of each vague term category
  - [x] Test clean AC passes validation
  - [x] Test multiple vague terms in single AC
  - [x] Test vague terms in `and_clauses`

- [x] Task 9: Write Unit Tests for Structure Validation (AC: 1)
  - [x] Test valid AC structure passes
  - [x] Test empty `given` field fails
  - [x] Test whitespace-only fields fail
  - [x] Test all three fields validated

- [x] Task 10: Write Unit Tests for Edge Case Detection (AC: 3)
  - [x] Test story with error handling AC passes
  - [x] Test story missing all edge cases returns suggestions
  - [x] Test partial edge case coverage
  - [x] Test edge case detection is case-insensitive

- [x] Task 11: Write Unit Tests for AC Count Validation (AC: 4)
  - [x] Test 0 ACs returns warning
  - [x] Test 1 AC returns warning
  - [x] Test 2-8 ACs returns None
  - [x] Test 9+ ACs returns splitting warning

- [x] Task 12: Write Unit Tests for Main Validation (AC: 5)
  - [x] Test fully valid story returns is_valid=True
  - [x] Test story with vague terms returns is_valid=False
  - [x] Test story with structural issues returns is_valid=False
  - [x] Test all result fields are populated correctly

- [x] Task 13: Write Integration Test (AC: all)
  - [x] Test pm_node with validation enabled
  - [x] Test validation results appear in processing notes
  - [x] Test validation doesn't block story creation

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use TypedDict for `TestabilityResult` (internal state)
- **ADR-006 (Quality Gate Pattern):** This validation function is a precursor to the AC Measurability Gate (Story 3.3)
- **ARCH-QUALITY-5:** No I/O operations needed; validation is purely computational
- **ARCH-QUALITY-6:** Use structlog for all validation logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- All validation functions are synchronous (no async needed - no I/O)
- Follow existing patterns from `agents/pm/llm.py` for vague term detection

### VAGUE_TERMS Reference

From analyst module (`agents/analyst/node.py`) - copy or import:
```python
VAGUE_TERMS: frozenset[str] = frozenset({
    # Quantifier vagueness
    "fast", "quick", "slow", "efficient", "scalable", "responsive", "real-time",
    # Ease vagueness
    "easy", "simple", "straightforward", "intuitive", "user-friendly", "seamless",
    # Certainty vagueness
    "should", "might", "could", "may", "possibly", "probably", "maybe",
    # Quality vagueness
    "good", "better", "best", "nice", "beautiful", "clean", "modern", "robust",
})
```

### Edge Case Pattern Constants

Define as module-level constants:
```python
ERROR_PATTERNS: frozenset[str] = frozenset({
    "error", "fail", "invalid", "exception", "reject", "denied"
})
EMPTY_PATTERNS: frozenset[str] = frozenset({
    "empty", "null", "none", "missing", "blank", "undefined"
})
BOUNDARY_PATTERNS: frozenset[str] = frozenset({
    "maximum", "minimum", "limit", "boundary", "overflow", "threshold"
})
```

### TestabilityResult TypedDict

```python
from typing import TypedDict

class TestabilityResult(TypedDict):
    """Result of AC testability validation."""
    is_valid: bool
    vague_terms_found: list[tuple[str, str]]  # (ac_id, term)
    missing_edge_cases: list[str]  # categories: "error_handling", "empty_input", "boundary"
    ac_count_warning: str | None
    validation_notes: list[str]
```

### File Structure (ARCH-STRUCT)

```
src/yolo_developer/agents/pm/
├── __init__.py          # Add TestabilityResult export
├── types.py             # MODIFY: Add TestabilityResult, TestabilityIssue
├── testability.py       # NEW: Testability validation functions
├── llm.py               # Existing - no changes needed
└── node.py              # MODIFY: Import and use validate_story_testability

tests/unit/agents/pm/
├── test_testability.py  # NEW: Tests for testability validation
└── test_node.py         # MODIFY: Add validation integration tests
```

### Existing Code Patterns to Follow

**From `agents/pm/llm.py` (vague term pattern):**
```python
# The llm.py already has vague term detection for AC generation
# Use similar pattern but make it reusable for validation
```

**From `agents/pm/types.py` (TypedDict pattern):**
```python
@dataclass(frozen=True)
class AcceptanceCriterion:
    id: str
    given: str
    when: str
    then: str
    and_clauses: tuple[str, ...] = ()
```

### Previous Story Intelligence (Story 6.2)

Key learnings to apply:
1. **Feature flag pattern:** Not needed here (no LLM calls)
2. **Structured logging:** Use structlog for all validation steps
3. **Graceful degradation:** Validation warns but doesn't block
4. **Comprehensive testing:** 47+ tests pattern - aim for similar coverage

Code review fixes from 6.2 to avoid:
- Don't use simple regex for complex parsing
- Handle edge cases in string validation (whitespace, empty)
- Test all code paths including fallbacks

### Integration Points

**Input (from Story transformation):**
- `Story` object with `acceptance_criteria` tuple of `AcceptanceCriterion`
- Each AC has: `id`, `given`, `when`, `then`, `and_clauses`

**Output (to pm_node):**
- `TestabilityResult` dict with validation findings
- Results added to `PMOutput.processing_notes`
- `Decision` record updated with validation summary

### Testing Strategy

**Unit Tests (synchronous):**
- No mocking needed (no external calls)
- Direct function testing with fixture data
- Edge case coverage for each validation function

**Integration Tests:**
- Test full pm_node flow includes validation
- Verify validation results in processing notes
- Test validation doesn't block story creation

### Relationship to Quality Gate (Story 3.3)

This story implements the validation logic that will later be used by:
- **Story 3.3: AC Measurability Gate** - Uses this validation as blocking gate
- The `validate_story_testability()` function becomes the core of the gate

### Project Structure Notes

- PM module at `src/yolo_developer/agents/pm/`
- New `testability.py` alongside existing `types.py`, `node.py`, `llm.py`
- Tests at `tests/unit/agents/pm/`
- No circular imports - PM only imports from config (if needed)

### References

- [Source: _bmad-output/planning-artifacts/epics.md - Story 6.3: AC Testability Validation]
- [Source: _bmad-output/planning-artifacts/epics.md - Epic 6: PM Agent overview]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-006: Quality Gate Pattern]
- [Source: _bmad-output/planning-artifacts/prd.md - FR43: Ensure all acceptance criteria are testable and measurable]
- [Source: src/yolo_developer/agents/pm/types.py - PM type definitions]
- [Source: src/yolo_developer/agents/pm/llm.py - Vague term detection pattern]
- [Source: _bmad-output/implementation-artifacts/6-2-transform-requirements-to-stories.md - Previous story patterns]
- [Source: _bmad-output/implementation-artifacts/6-1-create-pm-agent-node.md - PM agent node patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - All tests passed without debugging needed.

### Completion Notes List

- **Task 1:** Created `TestabilityResult` TypedDict and `TestabilitySeverity` enum in `types.py`
- **Task 2:** Implemented `_detect_vague_terms()` with VAGUE_TERMS frozenset containing 25+ vague terms
- **Task 3:** Implemented `_validate_ac_structure()` to check for empty/whitespace-only Given/When/Then fields
- **Task 4:** Implemented `_check_edge_cases()` with ERROR_PATTERNS, EMPTY_PATTERNS, BOUNDARY_PATTERNS constants
- **Task 5:** Implemented `_validate_ac_count()` returning warnings for <2 or >8 ACs
- **Task 6:** Implemented `validate_story_testability()` as main orchestration function with structlog logging
- **Task 7:** Integrated validation into `pm_node` - validates each story and adds results to processing notes
- **Tasks 8-12:** Added 39 tests in `test_testability.py` covering all validation functions
- **Task 13:** Added 3 integration tests in `test_node.py` for pm_node validation integration

### Change Log

- 2026-01-09: Story 6.3 implementation complete. All 161 PM tests passing.
- 2026-01-09: Code review fixes applied:
  - M1: Import VAGUE_TERMS from pm.llm instead of duplicating (single source of truth)
  - M2: Removed unused TestabilitySeverity enum (dead code)
  - M3: Removed unused validation_notes variable; added validation summary to processing notes
  - M4: Added validation summary to Decision rationale for audit trail

### File List

**New Files:**
- `src/yolo_developer/agents/pm/testability.py` - Testability validation functions (290 lines)
- `tests/unit/agents/pm/test_testability.py` - Unit tests for testability validation (39 tests)

**Modified Files:**
- `src/yolo_developer/agents/pm/types.py` - Added TestabilityResult TypedDict and TestabilitySeverity enum
- `src/yolo_developer/agents/pm/__init__.py` - Added exports for new types and validate_story_testability
- `src/yolo_developer/agents/pm/node.py` - Integrated validate_story_testability after story transformation
- `tests/unit/agents/pm/test_node.py` - Added 3 integration tests for validation (3 tests)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Story status updated to review

