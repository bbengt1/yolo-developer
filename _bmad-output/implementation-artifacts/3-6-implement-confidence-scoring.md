# Story 3.6: Implement Confidence Scoring

Status: done

## Story

As a system user,
I want a confidence score for deployable artifacts,
So that I know how certain the system is about the quality.

## Acceptance Criteria

1. **AC1: Confidence Score Calculation**
   - **Given** code passes through all gates
   - **When** the confidence scorer evaluates it
   - **Then** a score between 0-100 is calculated
   - **And** the score reflects overall artifact quality confidence

2. **AC2: Multi-Factor Scoring**
   - **Given** an artifact to evaluate
   - **When** the confidence score is calculated
   - **Then** score factors in test coverage assessment
   - **And** gate results from all evaluated gates are considered
   - **And** risk assessment contributes to the score
   - **And** each factor has configurable weighting

3. **AC3: Deployment Blocking Threshold**
   - **Given** a confidence score is calculated
   - **When** the score falls below 90%
   - **Then** deployment is blocked
   - **And** the blocking reason includes which factors caused the low score
   - **And** the threshold is configurable via state config

4. **AC4: Score Breakdown**
   - **Given** a confidence score evaluation completes
   - **When** results are generated
   - **Then** score breakdown shows all contributing factors
   - **And** each factor shows its individual contribution (0-100)
   - **And** weights for each factor are displayed
   - **And** the calculation method is transparent

5. **AC5: Confidence Evaluator Registration**
   - **Given** the confidence scoring evaluator is implemented
   - **When** the gates module is loaded
   - **Then** the evaluator is registered with name "confidence_scoring"
   - **And** the evaluator is available via `get_evaluator("confidence_scoring")`
   - **And** the evaluator follows the GateEvaluator protocol

6. **AC6: State Integration**
   - **Given** gate results and code artifacts exist in state
   - **When** confidence scoring is applied via @quality_gate("confidence_scoring")
   - **Then** the scorer reads gate results from `state["gate_results"]`
   - **And** code coverage info is read from `state["coverage"]` or estimated
   - **And** risk factors are read from `state["risks"]` or assessed
   - **And** the result includes detailed factor breakdown

## Tasks / Subtasks

- [x] Task 1: Define Confidence Scoring Types (AC: 1, 4)
  - [x] Create `src/yolo_developer/gates/gates/confidence_scoring.py` module
  - [x] Define `ConfidenceFactor` dataclass (name, score, weight, description)
  - [x] Define `ConfidenceBreakdown` dataclass (factors, total_score, weighted_score)
  - [x] Define `DEFAULT_FACTOR_WEIGHTS` constant with configurable weights
  - [x] Export types from `gates/gates/__init__.py`

- [x] Task 2: Implement Test Coverage Factor (AC: 2)
  - [x] Create `calculate_coverage_factor(state: dict) -> ConfidenceFactor` function
  - [x] Extract coverage from `state["coverage"]` if available
  - [x] Estimate coverage from code analysis if not available
  - [x] Consider: test file presence, test function count, code-to-test ratio
  - [x] Score 0-100 based on coverage percentage

- [x] Task 3: Implement Gate Results Factor (AC: 2)
  - [x] Create `calculate_gate_factor(state: dict) -> ConfidenceFactor` function
  - [x] Extract gate results from `state["gate_results"]`
  - [x] Calculate weighted pass rate across all gates
  - [x] Consider gate severity (blocking gates weighted higher)
  - [x] Score 0-100 based on aggregate gate performance

- [x] Task 4: Implement Risk Assessment Factor (AC: 2)
  - [x] Create `calculate_risk_factor(state: dict) -> ConfidenceFactor` function
  - [x] Extract risks from `state["risks"]` if available
  - [x] Assess risks from code complexity if not available
  - [x] Consider: cyclomatic complexity, function length, nesting depth
  - [x] Score 0-100 (100 = low risk, 0 = high risk)

- [x] Task 5: Implement Documentation Factor (AC: 2)
  - [x] Create `calculate_documentation_factor(state: dict) -> ConfidenceFactor` function
  - [x] Check for docstring presence in code files
  - [x] Check for README or documentation files
  - [x] Score 0-100 based on documentation coverage

- [x] Task 6: Implement Weighted Score Calculation (AC: 1, 4)
  - [x] Create `calculate_confidence_score(factors: list[ConfidenceFactor]) -> ConfidenceBreakdown` function
  - [x] Apply configurable weights to each factor
  - [x] Calculate weighted average score
  - [x] Return breakdown with individual and total scores

- [x] Task 7: Implement Confidence Evaluator (AC: 1, 2, 3, 5, 6)
  - [x] Create async `confidence_scoring_evaluator(context: GateContext) -> GateResult` function
  - [x] Extract all required data from context.state
  - [x] Extract threshold from `context.state.get("config", {}).get("quality", {}).get("confidence_threshold", 90)`
  - [x] Run all factor calculations
  - [x] Calculate overall confidence score
  - [x] Return GateResult with passed=True only if score >= threshold
  - [x] Include breakdown in result metadata

- [x] Task 8: Implement Confidence Report Generation (AC: 4)
  - [x] Create `generate_confidence_report(breakdown: ConfidenceBreakdown, threshold: int) -> str` function
  - [x] Format each factor with score, weight, and contribution
  - [x] Show weighted calculation method
  - [x] Include pass/fail status with threshold comparison
  - [x] Provide remediation suggestions for low-scoring factors

- [x] Task 9: Register Evaluator (AC: 5)
  - [x] Register `confidence_scoring_evaluator` in module initialization
  - [x] Use `register_evaluator("confidence_scoring", confidence_scoring_evaluator)`
  - [x] Update `gates/gates/__init__.py` to export confidence scoring types and functions
  - [x] Verify registration in `gates/__init__.py` exports

- [x] Task 10: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/gates/test_confidence_scoring.py`
  - [x] Test coverage factor calculation with various scenarios
  - [x] Test gate results factor with mixed pass/fail gates
  - [x] Test risk factor with different complexity levels
  - [x] Test documentation factor with missing/present docs
  - [x] Test weighted score calculation
  - [x] Test confidence evaluator with passing artifacts
  - [x] Test confidence evaluator with failing artifacts
  - [x] Test threshold configuration integration
  - [x] Test report generation format
  - [x] Test input validation (missing keys)

- [x] Task 11: Write Integration Tests (AC: 5, 6)
  - [x] Create `tests/integration/test_confidence_scoring_gate.py`
  - [x] Test gate decorator integration with confidence_scoring evaluator
  - [x] Test state reading from various state keys
  - [x] Test deployment blocking with low scores
  - [x] Test passing behavior with high confidence scores
  - [x] Test threshold override via configuration

## Dev Notes

### Architecture Compliance

- **ADR-006 (Quality Gate Pattern):** Decorator-based gates per architecture specification
- **FR23:** System can calculate confidence scores for deployable artifacts
- **Epic 3 Goal:** Validate artifacts at every agent boundary and block low-quality handoffs

### Technical Requirements

- **Evaluator Protocol:** Must implement `GateEvaluator` protocol from Story 3.1
- **Async Pattern:** Evaluator must be async function per architecture
- **State Access:** Read from multiple state keys (gate_results, coverage, risks, code)
- **Structured Logging:** Use structlog for all log messages

### Confidence Score Factors

```python
@dataclass(frozen=True)
class ConfidenceFactor:
    name: str           # e.g., "test_coverage", "gate_results"
    score: int          # 0-100 raw score
    weight: float       # 0.0-1.0 weight in final calculation
    description: str    # Human-readable explanation

DEFAULT_FACTOR_WEIGHTS = {
    "test_coverage": 0.30,      # 30% weight
    "gate_results": 0.35,       # 35% weight (most important)
    "risk_assessment": 0.20,    # 20% weight
    "documentation": 0.15,      # 15% weight
}
```

### Expected State Structure

```python
# State structure for confidence scoring
state = {
    # Required for gate factor
    "gate_results": [
        {"gate_name": "testability", "passed": True, "score": 85},
        {"gate_name": "ac_measurability", "passed": True, "score": 90},
        {"gate_name": "architecture_validation", "passed": False, "score": 65},
        {"gate_name": "definition_of_done", "passed": True, "score": 78},
    ],

    # Optional - coverage info
    "coverage": {
        "line_coverage": 85.5,
        "branch_coverage": 72.0,
        "function_coverage": 90.0,
    },

    # Optional - risk info
    "risks": [
        {"type": "complexity", "severity": "medium", "location": "module.py"},
        {"type": "security", "severity": "low", "location": "api.py"},
    ],

    # Optional - code for analysis
    "code": {
        "files": [{"path": "...", "content": "..."}],
    },

    # Optional - config override
    "config": {
        "quality": {
            "confidence_threshold": 90,
            "factor_weights": {...},
        }
    },
}
```

### Weighted Score Calculation

```python
# Weighted score formula
# total_score = sum(factor.score * factor.weight) / sum(weights)

# Example:
# coverage: 85 * 0.30 = 25.5
# gates: 78 * 0.35 = 27.3
# risk: 70 * 0.20 = 14.0
# docs: 90 * 0.15 = 13.5
# weighted_total = 80.3

# Default threshold = 90
# If weighted_total < 90: deployment blocked
```

### File Structure

```
src/yolo_developer/gates/
├── __init__.py                  # UPDATE: Export confidence_scoring gate
├── types.py                     # From Story 3.1
├── decorator.py                 # From Story 3.1
├── evaluators.py                # From Story 3.1
└── gates/
    ├── __init__.py              # UPDATE: Add confidence_scoring exports
    ├── testability.py           # From Story 3.2
    ├── ac_measurability.py      # From Story 3.3
    ├── architecture_validation.py  # From Story 3.4
    ├── definition_of_done.py    # From Story 3.5
    └── confidence_scoring.py    # NEW: Confidence scoring implementation
```

### Previous Story Intelligence (from Story 3.5)

**Patterns to Apply:**
1. Use frozen dataclasses for factor types (immutable)
2. Evaluator is async callable: `async def evaluator(ctx: GateContext) -> GateResult`
3. Register via `register_evaluator(gate_name, evaluator)`
4. State is accessible via `context.state`
5. Use `GateResult.to_dict()` for state serialization
6. Add autouse fixture in tests to re-register evaluator after `clear_evaluators()` calls
7. Validate input types before processing
8. Pre-sort constants at module level for performance
9. Include factor breakdown in metadata
10. Provide remediation guidance for all low-scoring factors

**Key Files to Reference:**
- `src/yolo_developer/gates/types.py` - GateResult, GateContext dataclasses
- `src/yolo_developer/gates/evaluators.py` - GateEvaluator protocol, registration functions
- `src/yolo_developer/gates/decorator.py` - @quality_gate decorator usage
- `src/yolo_developer/gates/gates/definition_of_done.py` - Latest gate implementation pattern
- `tests/unit/gates/test_definition_of_done.py` - Test patterns including autouse fixture

**Code Review Learnings from Previous Stories:**
- Include remediation guidance for ALL issues/factors
- Track actual factor IDs from input, don't hardcode generic values
- Add tests for all new functions added during implementation
- Ensure constants have proper structure (dicts with all required fields)
- Add structured logging verification test
- Export DEFAULT_* constants from __init__.py

### Testing Standards

- Use pytest-asyncio for async tests
- Create mock state data in test fixtures with various scenarios
- Test both passing (high confidence) and failing (low confidence) scenarios
- Test edge cases (empty state, missing keys, partial data)
- Verify structured logging output
- Add autouse fixture to ensure evaluator registration
- Follow TDD: write failing tests first, then implement

### Implementation Approach

1. **Graceful Degradation:** If state keys are missing, estimate/default values
2. **Configurable Weights:** All factor weights configurable via state config
3. **Transparent Calculation:** Show exactly how the score was computed
4. **Actionable Feedback:** Every low factor should suggest how to improve

### Example Output

```
Confidence Score Report
=======================
Overall Score: 78.3 / 100
Status: BLOCKED (threshold: 90)

Factor Breakdown:
┌─────────────────────┬───────┬────────┬──────────────┐
│ Factor              │ Score │ Weight │ Contribution │
├─────────────────────┼───────┼────────┼──────────────┤
│ Test Coverage       │   85  │  0.30  │    25.5      │
│ Gate Results        │   72  │  0.35  │    25.2      │
│ Risk Assessment     │   70  │  0.20  │    14.0      │
│ Documentation       │   90  │  0.15  │    13.5      │
├─────────────────────┼───────┼────────┼──────────────┤
│ Weighted Total      │       │  1.00  │    78.3      │
└─────────────────────┴───────┴────────┴──────────────┘

Improvement Suggestions:
- Gate Results (72): Address failing gates - architecture_validation has score 65
- Risk Assessment (70): Reduce complexity in module.py (high nesting depth)

To reach threshold (90), focus on:
1. Fix architecture_validation gate issues (+10 points potential)
2. Reduce code complexity in identified modules (+5 points potential)
```

### References

- [Source: architecture.md#ADR-006] - Quality Gate Pattern
- [Source: epics.md#Story-3.6] - Implement Confidence Scoring requirements
- [Source: prd.md#FR23] - System can calculate confidence scores for deployable artifacts
- [Story 3.1 Implementation] - Gate decorator framework
- [Story 3.5 Implementation] - Definition of Done gate (latest pattern)
- [Epic 9.4] - TEA agent confidence scoring (related requirements)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- Implemented multi-factor confidence scoring with 4 factors: test_coverage (30%), gate_results (35%), risk_assessment (20%), documentation (15%)
- Used frozen dataclasses for immutability (ConfidenceFactor, ConfidenceBreakdown)
- Added AST-based analysis for estimating coverage/risk when state data is unavailable
- Implemented graceful degradation when state keys are missing
- Report generation includes remediation guidance for low-scoring factors
- All 78 tests pass (61 unit + 17 integration)
- Full test suite passes (1054 tests total)

**Code Review Fixes (2026-01-05):**
- Added README/documentation file checking to `calculate_documentation_factor()` per Task 5 requirements
- Added `_is_documentation_file()` helper to detect README, CONTRIBUTING, CHANGELOG, etc.
- Added `_validate_factor_weights()` with validation that custom weights sum to 1.0
- Removed unused imports (`re`, `field` from dataclasses)
- Added confidence_scoring import example to `gates/__init__.py` docstring
- Exported `WEIGHT_SUM_TOLERANCE` constant
- Fixed unused variable assignments (`location`, `path`)
- Changed `tuple()` to `()` literal per ruff C408
- Updated test fixtures to include README files for documentation factor tests

### File List

- `src/yolo_developer/gates/gates/confidence_scoring.py` (NEW - 840 lines)
- `src/yolo_developer/gates/gates/__init__.py` (MODIFIED - added confidence_scoring exports)
- `tests/unit/gates/test_confidence_scoring.py` (NEW - 61 tests)
- `tests/integration/test_confidence_scoring_gate.py` (NEW - 17 tests)
