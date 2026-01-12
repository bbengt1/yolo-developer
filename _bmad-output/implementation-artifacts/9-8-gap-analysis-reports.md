# Story 9.8: Gap Analysis Reports

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want test gap analysis reports,
So that I know where testing is insufficient and can prioritize test improvements.

## Acceptance Criteria

1. **AC1: Identify Untested Functionality**
   - **Given** coverage data from Story 9.2 (CoverageReport)
   - **When** gap analysis runs
   - **Then** untested code is identified with:
     - file_path (path to untested file/module)
     - function_names (tuple of untested function/method names)
     - uncovered_lines (line ranges not covered by tests)
     - gap_type ("no_tests", "partial_coverage", "untested_branch")
   - **And** `TestGap` frozen dataclass captures each gap
   - **And** gaps are stored as immutable tuple

2. **AC2: Risk-Based Gap Prioritization**
   - **Given** identified test gaps
   - **When** prioritization runs
   - **Then** gaps are scored by:
     - severity (critical/high/medium/low based on code location)
     - risk_score (0-100 based on: cyclomatic complexity, file importance, change frequency)
     - priority_rank (integer ranking, 1 = highest priority)
   - **And** critical paths from config get highest priority
   - **And** recently changed files get higher priority
   - **And** `GapPriority` frozen dataclass captures scoring

3. **AC3: Suggested Tests Generation**
   - **Given** prioritized test gaps
   - **When** test suggestions are generated
   - **Then** each suggestion includes:
     - suggestion_id (unique identifier)
     - target_gap_id (links to TestGap)
     - test_type ("unit", "integration", "e2e")
     - description (what the test should verify)
     - estimated_impact (how much coverage will improve)
     - example_signature (suggested test function name)
   - **And** `TestSuggestion` frozen dataclass captures each suggestion
   - **And** suggestions are actionable and specific

4. **AC4: Gap Analysis Report Structure**
   - **Given** gap analysis is complete
   - **When** report is generated
   - **Then** `GapAnalysisReport` includes:
     - gaps (tuple of TestGap with priorities)
     - suggestions (tuple of TestSuggestion)
     - summary (GapAnalysisSummary with aggregates)
     - coverage_baseline (float - current coverage %)
     - projected_coverage (float - coverage if all suggestions implemented)
     - created_at (ISO timestamp)
   - **And** report has `to_dict()` method for serialization
   - **And** report is immutable (frozen dataclass)

5. **AC5: Summary Statistics**
   - **Given** gap analysis report
   - **When** summary is generated
   - **Then** `GapAnalysisSummary` includes:
     - total_gaps (count of all gaps)
     - critical_gaps (count of critical priority gaps)
     - high_gaps (count of high priority gaps)
     - medium_gaps (count of medium priority gaps)
     - low_gaps (count of low priority gaps)
     - total_suggestions (count of test suggestions)
     - estimated_effort (estimated time to address all gaps)
   - **And** summary provides quick overview of testing debt

6. **AC6: Export Capability**
   - **Given** a complete gap analysis report
   - **When** export is requested
   - **Then** report can be exported as:
     - JSON (full structured data)
     - Markdown (human-readable format)
     - CSV (gaps only, for tracking)
   - **And** export functions are provided for each format
   - **And** exports include all relevant data

7. **AC7: Integration with TEA Node**
   - **Given** TEA node processes artifacts
   - **When** gap analysis runs
   - **Then** `GapAnalysisReport` is included in TEAOutput
   - **And** `gap_analysis_report` field is accessible via to_dict()
   - **And** processing_notes includes gap analysis summary
   - **And** integration is backward compatible

## Tasks / Subtasks

- [x] Task 1: Create Gap Analysis Types (AC: 1, 2, 3, 4, 5)
  - [x] Create `GapType` Literal type: "no_tests", "partial_coverage", "untested_branch"
  - [x] Create `GapSeverity` Literal type: "critical", "high", "medium", "low"
  - [x] Create `SuggestionTestType` Literal type: "unit", "integration", "e2e"
  - [x] Create `TestGap` frozen dataclass with: gap_id, file_path, function_names, uncovered_lines, gap_type, description
  - [x] Create `GapPriority` frozen dataclass with: gap_id, severity, risk_score, priority_rank
  - [x] Create `TestSuggestion` frozen dataclass with: suggestion_id, target_gap_id, test_type, description, estimated_impact, example_signature
  - [x] Create `GapAnalysisSummary` frozen dataclass with: total_gaps, critical_gaps, high_gaps, medium_gaps, low_gaps, total_suggestions, estimated_effort
  - [x] Create `GapAnalysisReport` frozen dataclass with: gaps, priorities, suggestions, summary, coverage_baseline, projected_coverage, created_at
  - [x] Add `to_dict()` methods for all dataclasses
  - [x] Add types to `agents/tea/gap_analysis.py` (NEW FILE)

- [x] Task 2: Implement Gap Identification (AC: 1)
  - [x] Create `_generate_gap_id(file_path: str, sequence: int) -> str`
    - Format: "GAP-{hash(file_path)[:6]}-{seq:03d}" (e.g., "GAP-a1b2c3-001")
  - [x] Create `identify_untested_functions(coverage_report: CoverageReport) -> list[TestGap]`
    - Analyze coverage results to find files with 0% coverage
  - [x] Create `identify_partial_coverage_gaps(coverage_report: CoverageReport) -> list[TestGap]`
    - Find functions/methods with < threshold coverage
  - [x] Create `identify_untested_branches(coverage_report: CoverageReport) -> list[TestGap]`
    - Find conditional branches without coverage (if available)
  - [x] Create `identify_gaps(coverage_report: CoverageReport) -> tuple[TestGap, ...]`
    - Combines all gap identification, returns deduplicated tuple

- [x] Task 3: Implement Gap Prioritization (AC: 2)
  - [x] Create `_calculate_risk_score(gap: TestGap, critical_paths: tuple[str, ...]) -> int`
    - Score 0-100 based on code location, complexity hints, path criticality
  - [x] Create `_determine_severity(gap: TestGap, risk_score: int, critical_paths: tuple[str, ...]) -> GapSeverity`
    - Map risk score ranges to severity levels
    - Critical paths always get "critical" severity
  - [x] Create `prioritize_gaps(gaps: tuple[TestGap, ...], critical_paths: tuple[str, ...] | None = None) -> tuple[GapPriority, ...]`
    - Calculate priority for each gap
    - Sort by severity then risk_score
    - Assign priority_rank (1 = highest)

- [x] Task 4: Implement Test Suggestion Generation (AC: 3)
  - [x] Create `_generate_suggestion_id(sequence: int) -> str`
    - Format: "SUG-{seq:03d}" (e.g., "SUG-001")
  - [x] Create `_determine_test_type(gap: TestGap) -> SuggestionTestType`
    - "unit" for most cases, "integration" for cross-module, "e2e" for user-facing
  - [x] Create `_generate_test_signature(gap: TestGap) -> str`
    - Generate suggested test function name (e.g., "test_calculate_coverage_returns_percentage")
  - [x] Create `_estimate_impact(gap: TestGap, coverage_baseline: float) -> float`
    - Estimate coverage improvement if test is added
  - [x] Create `generate_test_suggestions(gaps: tuple[TestGap, ...], priorities: tuple[GapPriority, ...]) -> tuple[TestSuggestion, ...]`
    - Generate suggestions for highest priority gaps first
    - Limit to top 20 suggestions by default

- [x] Task 5: Implement Summary Generation (AC: 5)
  - [x] Create `_count_gaps_by_severity(priorities: tuple[GapPriority, ...]) -> dict[GapSeverity, int]`
  - [x] Create `_estimate_total_effort(gaps: tuple[TestGap, ...], suggestions: tuple[TestSuggestion, ...]) -> str`
    - Returns human-readable estimate (e.g., "2-4 hours", "1-2 days")
  - [x] Create `generate_summary(gaps: tuple[TestGap, ...], priorities: tuple[GapPriority, ...], suggestions: tuple[TestSuggestion, ...]) -> GapAnalysisSummary`

- [x] Task 6: Implement Report Generation (AC: 4)
  - [x] Create `generate_gap_analysis_report(coverage_report: CoverageReport, critical_paths: tuple[str, ...] | None = None, max_suggestions: int = 20) -> GapAnalysisReport`
    - Run full gap analysis pipeline
    - Calculate projected coverage
    - Build and return complete report

- [x] Task 7: Implement Export Functions (AC: 6)
  - [x] Create `export_to_json(report: GapAnalysisReport) -> str`
    - Full JSON serialization
  - [x] Create `export_to_markdown(report: GapAnalysisReport) -> str`
    - Human-readable markdown format with sections
  - [x] Create `export_to_csv(report: GapAnalysisReport) -> str`
    - CSV with gaps and priorities only

- [x] Task 8: Integrate with TEA Node (AC: 7)
  - [x] Add `gap_analysis_report: GapAnalysisReport | None` field to TEAOutput
  - [x] Update `tea_node()` to call `generate_gap_analysis_report()`
  - [x] Update processing_notes to include gap analysis summary
  - [x] Update `to_dict()` to serialize gap_analysis_report

- [x] Task 9: Write Unit Tests for Types (AC: 1, 2, 3, 4, 5)
  - [x] Test TestGap creation and to_dict()
  - [x] Test GapPriority creation and to_dict()
  - [x] Test TestSuggestion creation and to_dict()
  - [x] Test GapAnalysisSummary creation and to_dict()
  - [x] Test GapAnalysisReport creation and to_dict()
  - [x] Test immutability (frozen dataclass)

- [x] Task 10: Write Unit Tests for Gap Identification (AC: 1)
  - [x] Test gap ID generation format
  - [x] Test identify_untested_functions with various coverage data
  - [x] Test identify_partial_coverage_gaps
  - [x] Test identify_untested_branches
  - [x] Test combined gap identification

- [x] Task 11: Write Unit Tests for Prioritization (AC: 2)
  - [x] Test risk score calculation
  - [x] Test severity determination
  - [x] Test priority ranking
  - [x] Test critical path handling

- [x] Task 12: Write Unit Tests for Suggestions (AC: 3)
  - [x] Test suggestion ID generation
  - [x] Test test type determination
  - [x] Test signature generation
  - [x] Test impact estimation
  - [x] Test suggestion generation with max limit

- [x] Task 13: Write Unit Tests for Report Generation (AC: 4, 5)
  - [x] Test full report generation
  - [x] Test summary statistics
  - [x] Test projected coverage calculation

- [x] Task 14: Write Unit Tests for Export (AC: 6)
  - [x] Test JSON export format
  - [x] Test Markdown export format
  - [x] Test CSV export format

- [x] Task 15: Write Unit Tests for TEA Integration (AC: 7)
  - [x] Test gap_analysis_report field in TEAOutput
  - [x] Test to_dict() serialization
  - [x] Test tea_node() integration

## Dev Notes

### Architecture Compliance

- **ADR-001**: Use frozen dataclasses for all gap analysis types (immutable state)
- **ADR-008**: Read critical paths from config via `get_critical_paths_from_config()`
- Use tuples for all collections (immutable)
- Use `from __future__ import annotations` for forward references
- Follow existing TEA module patterns from coverage.py, risk.py, blocking.py

### Existing TEA Implementation Patterns

Follow patterns established in previous TEA stories:

```python
# Pattern from coverage.py - config loading with graceful fallback
def get_critical_paths_from_config() -> tuple[str, ...]:
    try:
        from yolo_developer.config import load_config
        config = load_config()
        return tuple(config.quality.critical_paths)
    except (FileNotFoundError, ConfigurationError, ImportError):
        return DEFAULT_CRITICAL_PATH_PATTERNS

# Pattern from risk.py - ID generation
def _generate_gap_id(file_path: str, sequence: int) -> str:
    """Generate unique gap ID from file path hash."""
    import hashlib
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:6]
    return f"GAP-{path_hash}-{sequence:03d}"

# Pattern from blocking.py - structured report generation
def generate_gap_analysis_report(
    coverage_report: CoverageReport,
    critical_paths: tuple[str, ...] | None = None,
    max_suggestions: int = 20,
) -> GapAnalysisReport:
    """Generate complete gap analysis report."""
```

### Source Tree Components to Touch

- `src/yolo_developer/agents/tea/gap_analysis.py` (NEW FILE)
- `src/yolo_developer/agents/tea/types.py` (add gap_analysis_report field to TEAOutput)
- `src/yolo_developer/agents/tea/node.py` (integrate gap analysis into tea_node)
- `src/yolo_developer/agents/tea/__init__.py` (export new types and functions)
- `tests/unit/agents/tea/test_gap_analysis.py` (NEW FILE)

### Testing Standards

- Use pytest with pytest-asyncio
- Follow RED-GREEN-REFACTOR cycle
- Test coverage target: 100% for new module
- Use frozen dataclass assertions for immutability tests
- Mock coverage data for unit tests (don't require actual coverage runs)

### Previous Story Learnings (from 9-7-deployment-blocking)

1. **Extract configurable thresholds early** - Don't hardcode values, create getter functions
2. **Whitespace validation** - Test empty and whitespace-only strings
3. **Edge cases** - Test empty inputs (empty coverage report, no gaps)
4. **Sequential ID generation** - Use consistent patterns across modules

### File Structure Notes

```
src/yolo_developer/agents/tea/
├── __init__.py          # Export new types and functions
├── blocking.py          # Story 9.7 (dependency)
├── coverage.py          # Story 9.2 (dependency - CoverageReport)
├── execution.py         # Story 9.3 (dependency - TestExecutionResult)
├── gap_analysis.py      # NEW - Story 9.8 implementation
├── node.py             # TEA node (integrate gap analysis)
├── risk.py             # Story 9.5 (pattern reference)
├── scoring.py          # Story 9.4 (pattern reference)
├── testability.py      # Story 9.6 (pattern reference)
└── types.py            # TEA types (add gap_analysis_report field)
```

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9-Story-9.8] - FR79: TEA Agent can generate test coverage reports with gap analysis
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] - Frozen dataclasses for internal state
- [Source: src/yolo_developer/agents/tea/coverage.py] - CoverageReport type and config loading patterns
- [Source: src/yolo_developer/agents/tea/blocking.py] - Report generation patterns
- [Source: 9-7-deployment-blocking.md] - Previous story patterns and learnings

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- All 7 ACs implemented and verified with tests
- 65 unit tests in test_gap_analysis.py covering all functionality
- All 523 TEA tests pass with no regressions
- Ruff linting clean (auto-fixed import sorting)
- Mypy type checking passes
- Implementation follows established patterns from coverage.py, risk.py, blocking.py

### Change Log

- Created `src/yolo_developer/agents/tea/gap_analysis.py` (~700 lines)
  - GapType, GapSeverity, SuggestionTestType Literal types
  - TestGap, GapPriority, TestSuggestion, GapAnalysisSummary, GapAnalysisReport frozen dataclasses
  - identify_gaps, prioritize_gaps, generate_test_suggestions functions
  - generate_summary, generate_gap_analysis_report functions
  - export_to_json, export_to_markdown, export_to_csv functions
- Modified `src/yolo_developer/agents/tea/types.py`
  - Added gap_analysis_report field to TEAOutput
  - Added TYPE_CHECKING import for GapAnalysisReport
- Modified `src/yolo_developer/agents/tea/node.py`
  - Integrated gap analysis generation after deployment decision
  - Added gap_summary to processing_notes
- Modified `src/yolo_developer/agents/tea/__init__.py`
  - Exported 8 new types and 9 new functions
  - Updated module docstring for Story 9.8
- Created `tests/unit/agents/tea/test_gap_analysis.py` (~700 lines)
  - 67 tests covering all AC requirements (including code review additions)

### Code Review Fixes (2026-01-12)

1. **S324: Insecure hash function** - Fixed: Changed MD5 to SHA256 in `_generate_gap_id()`
2. **ARG001: Unused argument** - Fixed: Removed unused `gaps` param from `_estimate_total_effort()`, updated signature
3. **PLR2004: Magic numbers** - Fixed: Extracted time constants (MINUTES_PER_HOUR, MINUTES_4_HOURS, etc.)
4. **Missing tests** - Added: `test_prioritize_gaps_empty_tuple`, `test_test_gap_whitespace_description`
5. **D413: Docstring formatting** - Fixed: Auto-fixed by ruff

