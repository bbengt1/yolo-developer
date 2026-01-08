# Story 4.6: Semantic Validation Reports

Status: done

## Story

As a developer,
I want comprehensive validation reports from seed parsing,
So that I can understand all issues before proceeding.

## Acceptance Criteria

1. **AC1: Report Generation**
   - **Given** a parsed seed with validation results
   - **When** a report is requested
   - **Then** a comprehensive ValidationReport is generated
   - **And** the report includes all ambiguities with priorities
   - **And** the report includes all SOP conflicts with severities
   - **And** the report includes parse quality metrics

2. **AC2: Report Formats**
   - **Given** a generated validation report
   - **When** output is requested
   - **Then** JSON format is supported for machine consumption
   - **And** Markdown format is supported for human readability
   - **And** Rich console format is supported for CLI display

3. **AC3: Quality Score Calculation**
   - **Given** validation results from parsing
   - **When** quality score is calculated
   - **Then** score reflects ambiguity count and severity
   - **And** score reflects SOP conflict count and severity
   - **And** score reflects extraction confidence
   - **And** score is normalized to 0.0-1.0 range

4. **AC4: CLI Report Command**
   - **Given** a user runs `yolo seed` with `--report`
   - **When** parsing completes
   - **Then** validation report is displayed
   - **And** `--report-format` allows selecting JSON/Markdown/Rich
   - **And** `--report-file` allows saving report to file

## Tasks / Subtasks

- [x] Task 1: Design Report Data Model (AC: 1, 3)
  - [x] Create `ValidationReport` dataclass with: parse_result, quality_score, generated_at, report_id
  - [x] Create `QualityMetrics` dataclass: ambiguity_score, sop_score, extraction_score, overall_score
  - [x] Create `ReportFormat` enum: JSON, MARKDOWN, RICH
  - [x] Add `to_dict()` methods for JSON serialization

- [x] Task 2: Implement Quality Score Calculator (AC: 3)
  - [x] Create `calculate_quality_score(result: SeedParseResult) -> QualityMetrics`
  - [x] Implement ambiguity scoring: base 1.0, deduct per ambiguity weighted by severity
  - [x] Implement SOP scoring: base 1.0, deduct per conflict weighted by severity
  - [x] Implement extraction scoring: based on ambiguity confidence
  - [x] Combine into overall score with configurable weights (0.3/0.3/0.4)
  - [x] Handle edge cases: no ambiguities (1.0), no SOP validation (1.0)

- [x] Task 3: Implement Report Generator (AC: 1, 2)
  - [x] Create `generate_validation_report(result: SeedParseResult) -> ValidationReport`
  - [x] Generate parse summary section (goals, features, constraints counts)
  - [x] Generate ambiguity section with prioritized list
  - [x] Generate SOP conflicts section with severity indicators
  - [x] Generate quality metrics section with score breakdown
  - [x] Generate recommendations section based on issues found

- [x] Task 4: Implement Report Formatters (AC: 2)
  - [x] Implement `format_report_json(report: ValidationReport) -> str`
  - [x] Implement `format_report_markdown(report: ValidationReport) -> str`
  - [x] Implement `format_report_rich(report: ValidationReport, console: Console) -> None`
  - [x] Ensure consistent structure across all formats
  - [x] Add color coding for severity levels in Rich output

- [x] Task 5: Update CLI with Report Options (AC: 4)
  - [x] Add `--report-format` option with choices: json, markdown, rich
  - [x] Add `--report-output PATH` option for file output
  - [x] Integrate report generation after successful parse
  - [x] Support `-r` short flag for report format

- [x] Task 6: Write Unit Tests (AC: all)
  - [x] Test `ValidationReport`, `QualityMetrics` dataclasses
  - [x] Test `calculate_quality_score()` with various inputs
  - [x] Test `generate_validation_report()` output structure
  - [x] Test all formatters produce valid output
  - [x] Test edge cases: empty results, perfect scores, all failures

- [x] Task 7: Write Integration Tests (AC: all)
  - [x] Test CLI `--report-format` displays validation report
  - [x] Test `--report-format json` produces valid JSON
  - [x] Test `--report-format markdown` produces valid Markdown
  - [x] Test `--report-output` writes to specified path
  - [x] Test report includes all validation data from parse

- [x] Task 8: Update Exports and Documentation (AC: all)
  - [x] Export `ValidationReport`, `QualityMetrics`, `ReportFormat` from `seed/__init__.py`
  - [x] Export `calculate_quality_score()` and `generate_validation_report()` functions
  - [x] Update module docstring with usage examples
  - [x] Add inline documentation for report structure

## Dev Notes

### Architecture Compliance

- **ADR-003 (LLM Abstraction):** Report generation is deterministic, no LLM calls needed
- **ADR-005 (CLI Framework):** Use Typer + Rich for CLI options and report display
- **FR6:** Semantic validation reports consolidate all validation results
- [Source: architecture.md#Seed Input] - `seed/` module handles FR1-8
- [Source: epics.md#Story-4.6] - Report generation requirements

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses for `ValidationReport`, `QualityMetrics`
- **Pure Functions:** Report generation and scoring must be deterministic, no side effects
- **Format Consistency:** All output formats must contain identical data, just rendered differently
- **Backward Compatibility:** Existing `yolo seed` behavior unchanged without --report flag

### Previous Story Intelligence (Story 4.5)

**Files Created/Modified in Story 4.5:**
- `src/yolo_developer/seed/sop.py` (700 lines) - SOP types, store, validation
- `src/yolo_developer/seed/types.py` - Added sop_validation to SeedParseResult
- `src/yolo_developer/seed/api.py` - Added validate_sop parameter
- `src/yolo_developer/cli/commands/seed.py` - SOP display, override handling
- Tests: 49 passing

**Key Patterns from Story 4.5:**

```python
# Data model pattern (from sop.py)
@dataclass(frozen=True)
class SOPConstraint:
    id: str
    rule_text: str
    category: SOPCategory
    source: str
    severity: ConflictSeverity
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule_text": self.rule_text,
            "category": self.category.value,
            "source": self.source,
            "severity": self.severity.value,
            "created_at": self.created_at,
        }

# CLI display pattern (from seed.py)
def _display_sop_conflicts(sop_result: SOPValidationResult, verbose: bool = False) -> None:
    table = Table(title="SOP Conflicts Detected", show_header=True, header_style="bold red")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Severity", justify="center")
    # ... add rows from conflicts
    console.print(table)

# Enum pattern
class ConflictSeverity(Enum):
    HARD = "hard"
    SOFT = "soft"
```

**Critical Learnings from 4.5:**
1. Use `from __future__ import annotations` in all files
2. Frozen dataclasses for immutability (except when mutation needed like override_applied)
3. `to_dict()` methods for JSON serialization consistency
4. Rich Table for structured CLI output with color styling
5. Enum values use lowercase strings for JSON compatibility
6. Properties for computed values (e.g., `has_conflicts`, `hard_conflict_count`)

### Git Intelligence (Recent Commits)

**Story 4.5 Commit (9cc85db):**
- feat: Implement SOP constraint validation with code review fixes (Story 4.5)
- 10 files changed
- Pattern: SOP validation integrated into parse flow

**Commit Message Pattern:**
```
feat: <description> (Story X.Y)

- Bullet point 1
- Bullet point 2
...

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Implementation Approach

1. **Types First:** Create all dataclasses in `seed/report.py`
2. **Calculator:** Implement quality score calculation logic
3. **Generator:** Build report generation from SeedParseResult
4. **Formatters:** Implement JSON, Markdown, Rich output formatters
5. **CLI Integration:** Add flags and integrate with seed command
6. **Tests:** Unit tests for all components, integration tests for CLI
7. **Exports:** Update `seed/__init__.py` with new public API

### Quality Score Weights

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Ambiguity Score | 0.3 | 1.0 - (HIGH*0.15 + MEDIUM*0.08 + LOW*0.03) |
| SOP Score | 0.3 | 1.0 - (HARD*0.20 + SOFT*0.08) |
| Extraction Score | 0.4 | Average confidence of parsed components |

**Overall Score Formula:**
```python
overall = (ambiguity * 0.3) + (sop * 0.3) + (extraction * 0.4)
```

### Report Sections Structure

```python
ValidationReport:
├── metadata (report_id, generated_at, source_file)
├── summary
│   ├── goals_count, features_count, constraints_count
│   ├── ambiguity_count, sop_conflict_count
│   └── quality_score (overall)
├── quality_metrics
│   ├── ambiguity_score (0.0-1.0)
│   ├── sop_score (0.0-1.0)
│   ├── extraction_score (0.0-1.0)
│   └── overall_score (0.0-1.0)
├── ambiguities (prioritized list)
│   └── [priority, type, severity, description, question]
├── sop_conflicts (severity-sorted list)
│   └── [severity, category, rule, conflict, options]
├── parse_result (full SeedParseResult data)
└── recommendations (actionable suggestions)
```

### Project Structure Notes

**Files to Create:**
```
src/yolo_developer/seed/
└── report.py                  # NEW: Report types, generator, formatters

tests/unit/seed/
└── test_validation_report.py  # NEW: Unit tests for report generation
```

**Files to Modify:**
```
src/yolo_developer/seed/
└── __init__.py                # UPDATE: Export report types and functions

src/yolo_developer/cli/
├── commands/seed.py           # UPDATE: Add report display after parse
└── main.py                    # UPDATE: Add --report, --report-format, --report-file
```

### Dependencies

**Depends On:**
- Story 4.3 (Ambiguity Detection) - Ambiguity data for reports
- Story 4.4 (Clarification Questions) - Priority scoring for ambiguities
- Story 4.5 (SOP Constraint Validation) - SOP conflict data for reports

**Downstream Dependencies:**
- Story 4.7 (Quality Threshold Rejection) - Uses quality_score to block low-quality seeds

### External Dependencies

- **rich** (installed) - Console formatting and tables
- **structlog** (installed) - Logging
- **typer** (installed) - CLI option handling
- No new dependencies required

### References

- [Source: architecture.md#ADR-005] - Typer + Rich for CLI
- [Source: architecture.md#FR6] - Semantic validation requirement
- [Source: epics.md#Story-4.6] - Report generation requirements
- [Source: seed/sop.py] - Data model patterns
- [Source: seed/ambiguity.py] - Priority calculation patterns
- [Source: cli/commands/seed.py] - Rich display patterns

### Files to Consult (MUST READ Before Implementation)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `gates/report_generator.py` | Report generation pattern | 39-94 |
| `gates/gates/confidence_scoring.py` | Weighted scoring pattern | 801-853, 959-1006 |
| `gates/metrics_types.py` | Frozen dataclass patterns | 59-218 |
| `gates/types.py` | GateResult pattern | 50-107 |
| `seed/types.py` | to_dict() patterns | 133-403 |
| `seed/sop.py` | SOPValidationResult pattern | 280-300 |
| `cli/commands/seed.py` | Rich table/panel patterns | 151-323, 594-601 |

### Existing Code to Reuse (CRITICAL - Follow These Patterns)

**From `gates/report_generator.py` (161 lines) - Report Generation Pattern:**
```python
# REUSE THIS PATTERN for generate_validation_report()
def generate_failure_report(
    gate_name: str,
    issues: list[GateIssue],
    score: float,
    threshold: float,
) -> GateFailureReport:
    """Generate a structured failure report"""
    blocking_count = sum(1 for i in issues if i.severity == Severity.BLOCKING)
    summary = f"{gate_name} score {score_pct}% below threshold..."
    return GateFailureReport(
        gate_name=gate_name,
        issues=tuple(issues),
        score=score,
        summary=summary,
    )
```

**From `gates/gates/confidence_scoring.py` (1016 lines) - Scoring Pattern:**
```python
# REUSE THIS PATTERN for QualityMetrics and calculate_quality_score()
@dataclass(frozen=True)
class ConfidenceFactor:
    name: str
    score: int
    weight: float
    description: str

@dataclass(frozen=True)
class ConfidenceBreakdown:
    factors: tuple[ConfidenceFactor, ...]
    total_score: float
    weighted_score: float
    threshold: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": [{"name": f.name, "score": f.score, ...} for f in self.factors],
            "total_score": round(self.total_score, 2),
            "passed": self.passed,
        }

def calculate_confidence_score(factors: list[ConfidenceFactor], threshold: int) -> ConfidenceBreakdown:
    weighted_sum = sum(f.score * f.weight for f in factors)
    weighted_score = weighted_sum / sum(f.weight for f in factors)
    passed = weighted_score >= threshold
    return ConfidenceBreakdown(...)
```

**From `gates/metrics_types.py` (218 lines) - Immutable Dataclass Pattern:**
```python
# REUSE THIS PATTERN for ValidationReport timestamp handling
@dataclass(frozen=True)
class GateMetricRecord:
    gate_name: str
    passed: bool
    score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),  # ISO 8601 format
            ...
        }
```

**From `cli/commands/seed.py` (996 lines) - Rich Display Patterns:**
```python
# Summary Panel pattern (line 65-80)
summary_text = (
    f"[bold green]Goals:[/bold green] {result.goal_count}\n"
    f"[bold blue]Features:[/bold blue] {result.feature_count}"
)
console.print(Panel(summary_text, title="Seed Parse Summary", border_style="cyan"))

# Table pattern (line 165-205)
table = Table(title="Ambiguities Detected", show_header=True, header_style="bold red")
table.add_column("#", justify="right", style="dim")
table.add_column("Severity", justify="center")
severity_style = {"low": "green", "medium": "yellow", "high": "red"}.get(...)
table.add_row(str(i), f"[{severity_style}]{severity}[/{severity_style}]", ...)
console.print(table)

# JSON output pattern (line 594-601)
def _output_json(result: SeedParseResult) -> None:
    result_dict = result.to_dict()
    console.print_json(json.dumps(result_dict, indent=2))
```

**From `seed/types.py` - SeedParseResult Properties:**
```python
@property
def has_ambiguities(self) -> bool:
    return len(self.ambiguities) > 0

@property
def has_sop_conflicts(self) -> bool:
    return self.sop_validation is not None and self.sop_validation.has_conflicts
```

### Anti-Patterns to Avoid

- **DO NOT** make LLM calls in report generation - this is pure data transformation
- **DO NOT** modify SeedParseResult - reports are read-only views
- **DO NOT** add new CLI commands - extend existing `yolo seed` with flags
- **DO NOT** break existing --json output - report is additive feature
- **DO NOT** use mutable dataclasses - all report types must be frozen

### Testing Strategy

**Unit Tests:**
```python
def test_quality_score_perfect() -> None:
    """Score should be 1.0 with no issues."""
    result = SeedParseResult(goals=(...), features=(...), ...)
    metrics = calculate_quality_score(result)
    assert metrics.overall_score == 1.0

def test_quality_score_with_high_ambiguity() -> None:
    """HIGH severity ambiguity should significantly reduce score."""
    # Create result with HIGH severity ambiguity
    metrics = calculate_quality_score(result)
    assert metrics.ambiguity_score < 0.9
    assert metrics.overall_score < 1.0
```

**Integration Tests:**
```python
def test_cli_report_flag(cli_runner: CliRunner, temp_seed_file: Path) -> None:
    result = cli_runner.invoke(app, ["seed", str(temp_seed_file), "--report"])
    assert result.exit_code == 0
    assert "Quality Score" in result.output
    assert "Validation Report" in result.output

def test_cli_report_json_format(cli_runner: CliRunner, temp_seed_file: Path) -> None:
    result = cli_runner.invoke(
        app, ["seed", str(temp_seed_file), "--report", "--report-format", "json"]
    )
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert "quality_metrics" in report
    assert "overall_score" in report["quality_metrics"]
```

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2026-01-08 | Claude Opus 4.5 | Initial story creation via create-story workflow |
