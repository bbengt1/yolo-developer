# Story 4.7: Quality Threshold Rejection

Status: done

## Story

As a developer,
I want seeds below minimum quality rejected with clear explanation,
So that I don't waste processing time on inadequate requirements.

## Acceptance Criteria

1. **AC1: Threshold Configuration**
   - **Given** a YOLO Developer project configuration
   - **When** quality thresholds are defined
   - **Then** minimum overall score can be configured (default: 0.7)
   - **And** minimum ambiguity score can be configured (default: 0.6)
   - **And** minimum SOP score can be configured (default: 0.8)
   - **And** threshold values are validated to be between 0.0 and 1.0

2. **AC2: Automatic Rejection**
   - **Given** a seed with quality score below configured threshold
   - **When** validation completes
   - **Then** processing is automatically halted
   - **And** a clear rejection message is displayed
   - **And** the specific threshold failures are listed
   - **And** exit code is non-zero (1)

3. **AC3: Remediation Guidance**
   - **Given** a seed that fails quality thresholds
   - **When** rejection occurs
   - **Then** specific remediation steps are provided
   - **And** the failing scores vs thresholds are shown
   - **And** actionable suggestions for improvement are listed
   - **And** user can provide a revised seed

4. **AC4: CLI Override Option**
   - **Given** a seed that would be rejected
   - **When** user provides `--force` flag
   - **Then** processing continues despite low quality score
   - **And** a warning is displayed about the override
   - **And** the warning is logged to audit trail

## Tasks / Subtasks

- [x] Task 1: Design Rejection Types (AC: 1, 2, 3)
  - [x] Create `QualityThreshold` dataclass: overall, ambiguity, sop minimum scores
  - [x] Create `RejectionReason` dataclass: threshold_name, actual_score, required_score
  - [x] Create `RejectionResult` dataclass: passed, reasons, recommendations
  - [x] Add `to_dict()` methods for JSON serialization

- [x] Task 2: Implement Threshold Validation (AC: 1, 2)
  - [x] Create `validate_quality_thresholds(metrics: QualityMetrics, thresholds: QualityThreshold) -> RejectionResult`
  - [x] Check overall_score against overall threshold
  - [x] Check ambiguity_score against ambiguity threshold
  - [x] Check sop_score against SOP threshold
  - [x] Collect all failing thresholds (not just first)

- [x] Task 3: Implement Remediation Generator (AC: 3)
  - [x] Create `generate_remediation_steps(result: RejectionResult, report: ValidationReport) -> list[str]`
  - [x] Generate specific suggestions based on failing threshold
  - [x] For ambiguity failures: "Resolve N high-severity ambiguities"
  - [x] For SOP failures: "Address N SOP conflicts (X hard, Y soft)"
  - [x] For overall failures: "Improve seed quality before proceeding"

- [x] Task 4: Update CLI with Threshold Enforcement (AC: 2, 4)
  - [x] Add `--force` flag to bypass threshold rejection
  - [x] Integrate threshold validation after report generation
  - [x] Display rejection message with Rich formatting
  - [x] Exit with code 1 on rejection (0 on success)
  - [x] Log override to structlog when --force used

- [x] Task 5: Add Threshold Configuration Support (AC: 1)
  - [x] Add `quality_thresholds` section to YoloConfig
  - [x] Define default threshold values (0.7, 0.6, 0.8)
  - [x] Support environment variable overrides (YOLO_QUALITY__*)
  - [x] Validate threshold values on config load

- [x] Task 6: Write Unit Tests (AC: all)
  - [x] Test `QualityThreshold`, `RejectionReason`, `RejectionResult` dataclasses
  - [x] Test `validate_quality_thresholds()` with various scores
  - [x] Test `generate_remediation_steps()` output
  - [x] Test edge cases: exact threshold, all pass, all fail

- [x] Task 7: Write Integration Tests (AC: all)
  - [x] Test CLI rejects low-quality seed with exit code 1
  - [x] Test CLI displays rejection reasons
  - [x] Test `--force` flag allows processing
  - [x] Test configured thresholds are applied
  - [x] Test rejection includes remediation guidance

- [x] Task 8: Update Exports and Documentation (AC: all)
  - [x] Export `QualityThreshold`, `RejectionResult` from `seed/__init__.py`
  - [x] Export `validate_quality_thresholds()` function
  - [x] Update module docstring with usage examples
  - [x] Add inline documentation for threshold behavior

## Dev Notes

### Architecture Compliance

- **ADR-008 (Pydantic Settings):** Quality thresholds configured via YoloConfig
- **ADR-005 (CLI Framework):** Use Typer + Rich for rejection display
- **FR8:** System can reject seeds that fail minimum quality thresholds with explanatory feedback
- [Source: architecture.md#Configuration] - Configuration via Pydantic Settings
- [Source: epics.md#Story-4.7] - Quality threshold rejection requirements

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses for `QualityThreshold`, `RejectionResult`
- **Pure Functions:** Threshold validation must be deterministic, no side effects
- **Configuration Integration:** Thresholds loaded from YoloConfig
- **Exit Codes:** Non-zero (1) on rejection, zero on success
- **Audit Trail:** Override actions logged via structlog

### Previous Story Intelligence (Story 4.6)

**Files Created/Modified in Story 4.6:**
- `src/yolo_developer/seed/report.py` (742 lines) - Report types, generator, formatters
- `src/yolo_developer/cli/commands/seed.py` - Report display integration
- `src/yolo_developer/cli/main.py` - --report-format, --report-output flags
- Tests: 43 passing (33 unit + 10 integration)

**Key Patterns from Story 4.6:**

```python
# Data model pattern (from report.py)
@dataclass(frozen=True)
class QualityMetrics:
    ambiguity_score: float
    sop_score: float
    extraction_score: float
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguity_score": self.ambiguity_score,
            "sop_score": self.sop_score,
            "extraction_score": self.extraction_score,
            "overall_score": self.overall_score,
        }

# Score calculation pattern
def calculate_quality_score(result: SeedParseResult) -> QualityMetrics:
    ambiguity_score = _calculate_ambiguity_score(result)
    sop_score = _calculate_sop_score(result)
    extraction_score = _calculate_extraction_score(result)
    overall_score = (
        ambiguity_score * AMBIGUITY_WEIGHT +
        sop_score * SOP_WEIGHT +
        extraction_score * EXTRACTION_WEIGHT
    )
    return QualityMetrics(...)
```

**Critical Learnings from 4.6:**
1. Use `from __future__ import annotations` in all files
2. Frozen dataclasses for immutability
3. `to_dict()` methods for JSON serialization consistency
4. Rich Panel/Table for structured CLI output
5. Exit codes via `raise typer.Exit(code=1)`

### Git Intelligence (Recent Commits)

**Story 4.6 Commit (ea491a6):**
- feat: Implement semantic validation reports with code review fixes (Story 4.6)
- 9 files changed
- Pattern: Report generation integrated with CLI

**Commit Message Pattern:**
```
feat: <description> (Story X.Y)

- Bullet point 1
- Bullet point 2
...

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Implementation Approach

1. **Types First:** Create all dataclasses in `seed/rejection.py`
2. **Validation Logic:** Implement threshold checking
3. **Remediation:** Build remediation suggestion generator
4. **Configuration:** Add thresholds to YoloConfig
5. **CLI Integration:** Add --force flag, exit code handling
6. **Tests:** Unit tests for all components, integration tests for CLI
7. **Exports:** Update `seed/__init__.py` with new public API

### Default Threshold Values

| Threshold | Default | Rationale |
|-----------|---------|-----------|
| Overall Score | 0.70 | Allows moderate issues but catches severe quality problems |
| Ambiguity Score | 0.60 | More lenient - ambiguities can be clarified later |
| SOP Score | 0.80 | Stricter - SOP conflicts are harder to resolve |

### Rejection Message Structure

```
┌─────────────────── Seed Rejected ───────────────────┐
│                                                      │
│  Quality Score: 0.52 (minimum: 0.70)                │
│                                                      │
│  ❌ Failed Thresholds:                               │
│    • Overall Score: 0.52 < 0.70 required            │
│    • Ambiguity Score: 0.45 < 0.60 required          │
│                                                      │
│  📋 Remediation Steps:                               │
│    1. Resolve 3 high-severity ambiguities           │
│    2. Clarify 5 medium-severity ambiguities         │
│    3. Review and revise seed document               │
│                                                      │
│  💡 Tip: Use --force to proceed despite low quality │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Project Structure Notes

**Files to Create:**
```
src/yolo_developer/seed/
└── rejection.py               # NEW: Rejection types and validation
```

**Files to Modify:**
```
src/yolo_developer/seed/
└── __init__.py                # UPDATE: Export rejection types and functions

src/yolo_developer/config/
└── schema.py                  # UPDATE: Add QualityThreshold to YoloConfig

src/yolo_developer/cli/
├── commands/seed.py           # UPDATE: Add threshold enforcement logic
└── main.py                    # UPDATE: Add --force flag

tests/unit/seed/
└── test_quality_rejection.py  # NEW: Unit tests for rejection
```

### Dependencies

**Depends On:**
- Story 4.6 (Semantic Validation Reports) - QualityMetrics for threshold comparison

**Downstream Dependencies:**
- None (this completes Epic 4)

### External Dependencies

- **rich** (installed) - Console formatting and tables
- **structlog** (installed) - Logging override actions
- **typer** (installed) - CLI flag handling
- No new dependencies required

### References

- [Source: architecture.md#ADR-008] - Pydantic Settings configuration
- [Source: architecture.md#FR8] - Quality threshold rejection requirement
- [Source: epics.md#Story-4.7] - Rejection requirements
- [Source: seed/report.py] - QualityMetrics data model
- [Source: config/schema.py] - YoloConfig patterns
- [Source: cli/commands/seed.py] - CLI patterns

### Files to Consult (MUST READ Before Implementation)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `seed/report.py` | QualityMetrics model | 103-141 |
| `config/schema.py` | YoloConfig pattern | Full file |
| `cli/commands/seed.py` | CLI integration pattern | 46-58, 980-1021 |
| `cli/main.py` | Flag definition pattern | 96-113 |
| `gates/types.py` | GateResult pattern | 50-107 |

### Existing Code to Reuse (CRITICAL - Follow These Patterns)

**From `seed/report.py` - QualityMetrics Model:**
```python
# REUSE THIS PATTERN for QualityThreshold
@dataclass(frozen=True)
class QualityMetrics:
    ambiguity_score: float
    sop_score: float
    extraction_score: float
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        return {...}
```

**From `config/schema.py` - Configuration Pattern:**
```python
# REUSE THIS PATTERN for quality threshold config
class QualityConfig(BaseModel):
    """Quality gate configuration."""
    test_coverage_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum test coverage percentage",
    )
```

**From `cli/commands/seed.py` - Exit Code Pattern:**
```python
# REUSE THIS PATTERN for rejection exit
if not result.passed:
    console.print(Panel(error_text, title="Error", border_style="red"))
    raise typer.Exit(code=1)
```

**From `gates/types.py` - Result Pattern:**
```python
# REUSE THIS PATTERN for RejectionResult
@dataclass(frozen=True)
class GateResult:
    passed: bool
    score: float
    issues: tuple[GateIssue, ...] = ()
    suggestions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {...}
```

### Anti-Patterns to Avoid

- **DO NOT** block on first threshold failure - collect ALL failures
- **DO NOT** modify QualityMetrics - thresholds are separate concern
- **DO NOT** add new commands - extend existing `yolo seed`
- **DO NOT** skip remediation when --force used - still show warnings
- **DO NOT** use mutable dataclasses - all types must be frozen

### Testing Strategy

**Unit Tests:**
```python
def test_threshold_validation_all_pass() -> None:
    """All scores above thresholds should pass."""
    metrics = QualityMetrics(
        ambiguity_score=0.9, sop_score=0.95,
        extraction_score=0.85, overall_score=0.9
    )
    thresholds = QualityThreshold()  # defaults
    result = validate_quality_thresholds(metrics, thresholds)
    assert result.passed is True
    assert len(result.reasons) == 0

def test_threshold_validation_overall_fail() -> None:
    """Below overall threshold should fail."""
    metrics = QualityMetrics(
        ambiguity_score=0.9, sop_score=0.95,
        extraction_score=0.5, overall_score=0.65
    )
    result = validate_quality_thresholds(metrics, QualityThreshold())
    assert result.passed is False
    assert any(r.threshold_name == "overall" for r in result.reasons)
```

**Integration Tests:**
```python
def test_cli_rejects_low_quality_seed(cli_runner: CliRunner, low_quality_seed: Path) -> None:
    result = cli_runner.invoke(app, ["seed", str(low_quality_seed)])
    assert result.exit_code == 1
    assert "Seed Rejected" in result.output
    assert "Failed Thresholds" in result.output

def test_cli_force_flag_bypasses_rejection(cli_runner: CliRunner, low_quality_seed: Path) -> None:
    result = cli_runner.invoke(app, ["seed", str(low_quality_seed), "--force"])
    assert result.exit_code == 0
    assert "Warning" in result.output
```

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- **AC3 Note:** "User can provide revised seed" is implemented as: user re-runs `yolo seed` command with revised file path after addressing remediation guidance. This follows CLI best practices (non-blocking exit) rather than inline re-prompting.
- Created `src/yolo_developer/seed/rejection.py` with frozen dataclasses for QualityThreshold, RejectionReason, RejectionResult
- Implemented `validate_quality_thresholds()` to check metrics against configurable thresholds
- Implemented `generate_remediation_steps()` with specific suggestions based on failure types
- Added `create_rejection_with_remediation()` convenience function
- Added `--force`/`-f` flag to CLI to bypass threshold rejection
- Integrated threshold validation after report generation in CLI
- Added `_display_rejection()` and `_display_threshold_warning()` Rich panel functions
- Added `SeedThresholdConfig` to config/schema.py with configurable threshold values
- Added threshold loading from config with fallback to defaults
- Override actions logged via structlog for audit trail
- All tests passing: 41 unit tests (rejection), 8 config tests (thresholds), 8 integration tests
- Exports added to `seed/__init__.py` with usage examples in module docstring

### File List

**Created:**
- `src/yolo_developer/seed/rejection.py` - Rejection types and validation
- `tests/unit/seed/test_quality_rejection.py` - Unit tests for rejection

**Modified:**
- `src/yolo_developer/seed/__init__.py` - Added rejection exports and docstring examples
- `src/yolo_developer/cli/commands/seed.py` - Added threshold enforcement, --force flag, display functions
- `src/yolo_developer/cli/main.py` - Added --force/-f flag definition to seed command
- `src/yolo_developer/config/schema.py` - Added SeedThresholdConfig class
- `src/yolo_developer/config/__init__.py` - Exported SeedThresholdConfig
- `tests/unit/config/test_schema.py` - Added 8 tests for SeedThresholdConfig
- `tests/integration/test_cli_seed.py` - Added TestQualityThresholdRejection class (8 tests)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status

