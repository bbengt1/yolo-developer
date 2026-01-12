# Story 9.4: Confidence Scoring

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a confidence score for deployability,
So that I can trust the system's quality assessment.

## Acceptance Criteria

1. **AC1: Score Calculation**
   - **Given** completed validation results from TEA agent
   - **When** confidence score is calculated
   - **Then** score ranges from 0-100 (integer) representing deployment confidence
   - **And** score is calculated using weighted factors (coverage, gate results, test results)
   - **And** the calculation formula is documented and deterministic

2. **AC2: Factor Weighting**
   - **Given** multiple validation inputs (coverage, gate results, test execution)
   - **When** factors are weighted into final score
   - **Then** test coverage contributes up to 40% of score
   - **And** test execution pass rate contributes up to 30% of score
   - **And** validation findings severity contributes up to 30% of score
   - **And** weights are configurable via constants or config

3. **AC3: Score Breakdown**
   - **Given** a calculated confidence score
   - **When** score breakdown is requested
   - **Then** breakdown shows individual factor contributions
   - **And** breakdown includes: coverage_score, test_execution_score, validation_score
   - **And** breakdown includes penalty reasons if score was reduced
   - **And** breakdown is stored in ConfidenceBreakdown frozen dataclass

4. **AC4: Deployment Blocking Threshold**
   - **Given** a confidence score below 90%
   - **When** deployment decision is made
   - **Then** deployment is blocked (returns DeploymentRecommendation="block")
   - **And** blocking reason includes the specific score and threshold
   - **And** threshold is configurable (default 90%)
   - **And** blocking generates a Finding with severity="critical"

5. **AC5: Score Modifiers**
   - **Given** validation results with specific patterns
   - **When** score modifiers are applied
   - **Then** critical findings reduce score by 25% each (capped at -75%)
   - **And** high findings reduce score by 10% each (capped at -40%)
   - **And** test execution errors reduce score by 15% each (capped at -45%)
   - **And** perfect test pass rate adds 5% bonus (capped at 100 total)

6. **AC6: Integration with TEA Output**
   - **Given** TEA node produces output
   - **When** confidence scoring runs
   - **Then** ConfidenceResult is included in TEAOutput
   - **And** overall_confidence field uses the new scoring system
   - **And** score breakdown is accessible via to_dict() method
   - **And** legacy overall_confidence behavior is preserved for compatibility

## Tasks / Subtasks

- [x] Task 1: Create Confidence Scoring Types (AC: 2, 3)
  - [x] Create `ConfidenceWeight` frozen dataclass with: coverage_weight, test_execution_weight, validation_weight (all floats summing to 1.0)
  - [x] Create `ConfidenceBreakdown` frozen dataclass with: coverage_score, test_execution_score, validation_score, penalties, bonuses, total_score
  - [x] Create `ConfidenceResult` frozen dataclass with: score (0-100), breakdown, passed_threshold, threshold_value, blocking_reasons
  - [x] Add `to_dict()` methods for serialization
  - [x] Add types to `agents/tea/scoring.py` (new file created instead of confidence.py)

- [x] Task 2: Implement Weight Configuration (AC: 2)
  - [x] Create `get_default_weights()` returning ConfidenceWeight(coverage=0.4, test_execution=0.3, validation=0.3)
  - [x] Add `get_weights_from_config()` for optional config loading
  - [x] Add `validate_weights()` to validate weights sum to 1.0 with tolerance
  - [x] Add docstring explaining weight rationale

- [x] Task 3: Implement Coverage Score Calculation (AC: 1, 2)
  - [x] Create `_calculate_coverage_score(coverage_report: CoverageReport | None) -> float` function
  - [x] Return 0-100 score based on average coverage percentage
  - [x] Handle None case (no coverage data) by returning neutral score (50)
  - [x] Critical path coverage boost: +10 if all critical paths covered

- [x] Task 4: Implement Test Execution Score Calculation (AC: 1, 2)
  - [x] Create `_calculate_test_execution_score(result: TestExecutionResult | None) -> float` function
  - [x] Return 0-100 score based on pass rate
  - [x] Formula: (passed / total) * 100, with error penalty of -15 per error (capped at -45)
  - [x] Handle None case (no test execution) by returning neutral score (50)
  - [x] Perfect pass rate (100%) adds 5 bonus points (capped at 100)

- [x] Task 5: Implement Validation Score Calculation (AC: 1, 2)
  - [x] Create `_calculate_validation_score(results: tuple[ValidationResult, ...]) -> float` function
  - [x] Start at 100, deduct based on finding severity
  - [x] Critical: -25 per finding (capped at -75 total)
  - [x] High: -10 per finding (capped at -40 total)
  - [x] Medium: -5 per finding (capped at -20 total)
  - [x] Low: -2 per finding (capped at -10 total)
  - [x] Info: -1 per finding (capped at -5 total)

- [x] Task 6: Implement Score Modifier System (AC: 5)
  - [x] Create `_apply_score_modifiers(base_score: float, breakdown: ConfidenceBreakdown, perfect_tests: bool) -> tuple[float, list[str]]`
  - [x] Apply perfect test pass bonus (+5 if all tests pass)
  - [x] Return modified score and list of modification reasons
  - [x] Cap final score at 0-100 range

- [x] Task 7: Implement Main Confidence Calculation (AC: 1, 3)
  - [x] Create `calculate_confidence_score(validation_results, coverage_report, test_execution_result, weights, threshold) -> ConfidenceResult`
  - [x] Calculate individual component scores using helper functions
  - [x] Apply weights to get weighted average
  - [x] Build ConfidenceBreakdown with all components
  - [x] Apply score modifiers
  - [x] Build and return ConfidenceResult

- [x] Task 8: Implement Threshold Checking (AC: 4)
  - [x] Create `check_deployment_threshold(score: int, threshold: int = 90) -> tuple[bool, DeploymentRecommendation, list[str]]`
  - [x] Return (passed: bool, recommendation: DeploymentRecommendation, reasons: list[str])
  - [x] If score >= threshold: return (True, "deploy", [])
  - [x] If score within 10 points of threshold: return (False, "deploy_with_warnings", [reason])
  - [x] If score < threshold - 10: return (False, "block", [blocking_reason])

- [x] Task 9: Integrate with TEA Node (AC: 6)
  - [x] Update `tea_node()` to call `calculate_confidence_score()`
  - [x] Replace existing confidence calculation with new system
  - [x] Add `confidence_result: ConfidenceResult | None` field to TEAOutput
  - [x] Ensure backward compatibility with `overall_confidence` float field
  - [x] Update processing_notes to include confidence breakdown summary

- [x] Task 10: Write Unit Tests for Types (AC: 3)
  - [x] Test ConfidenceWeight creation and validation
  - [x] Test ConfidenceBreakdown creation and to_dict()
  - [x] Test ConfidenceResult creation and to_dict()
  - [x] Test immutability (frozen dataclass)

- [x] Task 11: Write Unit Tests for Score Calculations (AC: 1, 2, 5)
  - [x] Test coverage score with various coverage percentages
  - [x] Test test execution score with pass/fail/error combinations
  - [x] Test validation score with various finding severities
  - [x] Test score modifiers (bonuses and penalties)
  - [x] Test edge cases (empty inputs, all zeros, all perfect)

- [x] Task 12: Write Unit Tests for Threshold (AC: 4)
  - [x] Test threshold passing (score >= 90)
  - [x] Test threshold failing (score < 90)
  - [x] Test custom threshold values
  - [x] Test blocking reason generation

- [x] Task 13: Write Integration Tests (AC: 6)
  - [x] Test full confidence calculation flow
  - [x] Test TEA node integration with confidence scoring
  - [x] Test ConfidenceResult included in TEAOutput
  - [x] Test backward compatibility with overall_confidence

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for ConfidenceWeight, ConfidenceBreakdown, ConfidenceResult
- **ADR-006 (Quality Gates):** Confidence score feeds into deployment blocking gate
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions
- **ARCH-QUALITY-5:** Async patterns not required for pure calculation functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from Story 9.1, 9.2, 9.3 TEA implementations
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- Maintain backward compatibility with existing TEAOutput.overall_confidence

### Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| structlog | latest | Structured logging |
| dataclasses | stdlib | Frozen dataclasses for types |

### Confidence Score Formula

The confidence score is calculated using weighted components:

```python
# Component weights (must sum to 1.0)
COVERAGE_WEIGHT = 0.40    # Test coverage importance
TEST_EXEC_WEIGHT = 0.30   # Test execution results importance
VALIDATION_WEIGHT = 0.30  # Validation findings importance

# Base calculation
coverage_score = calculate_coverage_contribution(coverage_report)      # 0-100
test_exec_score = calculate_test_execution_contribution(test_result)   # 0-100
validation_score = calculate_validation_contribution(findings)          # 0-100

# Weighted average
weighted_score = (
    coverage_score * COVERAGE_WEIGHT +
    test_exec_score * TEST_EXEC_WEIGHT +
    validation_score * VALIDATION_WEIGHT
)

# Apply modifiers
final_score = apply_modifiers(weighted_score, context)  # Bonuses/penalties
final_score = max(0, min(100, final_score))            # Clamp to 0-100
```

### Severity Penalty Matrix

| Finding Severity | Penalty | Max Total Penalty |
|-----------------|---------|-------------------|
| Critical | -25 per finding | -75 total |
| High | -10 per finding | -40 total |
| Medium | -5 per finding | -20 total |
| Low | -2 per finding | -10 total |
| Info | -1 per finding | -5 total |

### Test Execution Impact

| Scenario | Score Impact |
|----------|--------------|
| All tests pass | +5 bonus (capped at 100) |
| Some tests fail | Score = (passed/total) * 100 |
| Test errors | -15 per error (capped at -45) |
| No tests | Neutral score (50) |

### Deployment Threshold Logic

```python
def check_deployment_threshold(score: int, threshold: int = 90) -> tuple:
    if score >= threshold:
        return (True, "deploy", [])
    elif score >= threshold - 10:  # Within 10 points
        return (False, "deploy_with_warnings", [f"Score {score} is close to threshold {threshold}"])
    else:
        return (False, "block", [f"Score {score} is below threshold {threshold}"])
```

### ConfidenceResult Data Structure

```python
@dataclass(frozen=True)
class ConfidenceResult:
    """Complete confidence scoring result."""

    score: int                              # 0-100 final confidence score
    breakdown: ConfidenceBreakdown          # Component contributions
    passed_threshold: bool                  # Whether deployment threshold passed
    threshold_value: int                    # The threshold used (default 90)
    blocking_reasons: tuple[str, ...]       # Reasons if blocked (empty if passed)
    deployment_recommendation: DeploymentRecommendation  # deploy/deploy_with_warnings/block
    created_at: str                         # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "breakdown": self.breakdown.to_dict(),
            "passed_threshold": self.passed_threshold,
            "threshold_value": self.threshold_value,
            "blocking_reasons": list(self.blocking_reasons),
            "deployment_recommendation": self.deployment_recommendation,
            "created_at": self.created_at,
        }
```

### Project Structure Notes

- **Module Location:** `src/yolo_developer/agents/tea/`
- **Modified Files:**
  - `src/yolo_developer/agents/tea/scoring.py` - Add new types and functions (NEW FILE)
  - `src/yolo_developer/agents/tea/types.py` - Add confidence_result to TEAOutput
  - `src/yolo_developer/agents/tea/node.py` - Integrate confidence scoring
  - `src/yolo_developer/agents/tea/__init__.py` - Export new types
- **Test Location:** `tests/unit/agents/tea/`

### Existing Code to Integrate With

The Story 9.1-9.3 implementations provide:
- `CoverageReport` - From Story 9.2, provides coverage data for scoring
- `TestExecutionResult` - From Story 9.3, provides test pass/fail data
- `ValidationResult` - From Story 9.1, contains findings for severity analysis
- `Finding` with severity levels - Used for penalty calculation
- `_calculate_overall_confidence()` - **REPLACE** with new system
- `TEAOutput` - Extend with `confidence_result` field

### Key Integration Points

```python
# In tea_node(), replace existing confidence calculation:

# OLD (to be replaced):
overall_confidence, deployment_recommendation = _calculate_overall_confidence(validation_results)

# NEW:
confidence_result = calculate_confidence_score(
    validation_results=tuple(validation_results),
    coverage_report=coverage_report,  # From Story 9.2
    test_execution_result=test_execution_result,  # From Story 9.3
    weights=None,  # Use defaults, or load from config
)

# Use confidence_result.score for overall_confidence (backward compat)
overall_confidence = confidence_result.score / 100.0  # Convert to 0-1 range
deployment_recommendation = confidence_result.deployment_recommendation
```

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR75 | TEA Agent can calculate deployment confidence scores | ConfidenceResult with 0-100 score |
| FR23 | System can calculate confidence scores for deployable artifacts | calculate_confidence_score() function |
| FR24 | System can block handoffs when quality gates fail | check_deployment_threshold() with blocking |
| FR78 | TEA Agent can block deployment when confidence score < 90% | Threshold checking with configurable value |

### Previous Story Learnings Applied

From Story 9.1, 9.2, 9.3:
- Use frozen dataclasses for all data structures with to_dict() methods
- Integrate with existing `tea_node()` function flow
- Follow structured logging pattern with structlog
- Handle missing/empty content gracefully (return neutral scores)
- Use hash-based IDs for unique finding identification
- Test execution results feed into confidence via pass rate

### Git Commit Pattern

```
feat: Implement confidence scoring with code review fixes (Story 9.4)
```

### Sample ConfidenceResult Output

```python
ConfidenceResult(
    score=82,
    breakdown=ConfidenceBreakdown(
        coverage_score=85.0,
        test_execution_score=90.0,
        validation_score=70.0,
        weighted_coverage=34.0,      # 85 * 0.4
        weighted_test_execution=27.0, # 90 * 0.3
        weighted_validation=21.0,     # 70 * 0.3
        penalties=("-10 for 1 high finding",),
        bonuses=(),
        base_score=82.0,
        final_score=82,
    ),
    passed_threshold=False,
    threshold_value=90,
    blocking_reasons=("Score 82 is below threshold 90",),
    deployment_recommendation="block",
    created_at="2026-01-12T10:00:00.000Z",
)
```

### Confidence Scoring Best Practices (2025)

Based on industry research:
- Use calibrated confidence scores that correlate with actual correctness
- Provide score breakdowns for transparency and actionability
- Support configurable thresholds for different project needs
- Include penalty reasons to guide remediation efforts
- Design for active learning loops where low-confidence results trigger review

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-9.4] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR75] - Confidence scoring
- [Source: _bmad-output/planning-artifacts/prd.md#FR23] - Confidence calculation
- [Source: _bmad-output/planning-artifacts/prd.md#FR78] - Deployment blocking
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] - State management patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate patterns
- [Source: src/yolo_developer/agents/tea/types.py] - Existing TEA types
- [Source: src/yolo_developer/agents/tea/node.py] - Existing TEA node implementation
- [Source: src/yolo_developer/agents/tea/coverage.py] - Coverage analysis (Story 9.2)
- [Source: src/yolo_developer/agents/tea/execution.py] - Test execution (Story 9.3)
- [Source: _bmad-output/implementation-artifacts/9-3-test-suite-execution.md] - Story 9.3 patterns
- [Ultralytics - Confidence Score Explained](https://www.ultralytics.com/glossary/confidence)
- [MLOps Deployment Best Practices 2025](https://dasroot.net/posts/2025/12/mlops-deploying-monitoring-ml-models-2025/)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- All 13 tasks completed with TDD (RED-GREEN-REFACTOR) approach
- 65 new tests added (20 type tests, 38 scoring tests, 7 integration tests)
- 248 total TEA agent tests passing
- Type checking (mypy) passing
- Linting (ruff) passing
- Backward compatibility maintained with overall_confidence float field

### File List

**New Files:**
- `src/yolo_developer/agents/tea/scoring.py` - Confidence scoring types and functions
- `tests/unit/agents/tea/test_confidence_types.py` - Unit tests for types
- `tests/unit/agents/tea/test_confidence_scoring.py` - Unit tests for calculations
- `tests/unit/agents/tea/test_confidence_integration.py` - Integration tests

**Modified Files:**
- `src/yolo_developer/agents/tea/types.py` - Added confidence_result field to TEAOutput
- `src/yolo_developer/agents/tea/node.py` - Integrated confidence scoring
- `src/yolo_developer/agents/tea/__init__.py` - Exported new types and functions
- `tests/unit/agents/tea/test_node.py` - Updated test expectation for neutral confidence

