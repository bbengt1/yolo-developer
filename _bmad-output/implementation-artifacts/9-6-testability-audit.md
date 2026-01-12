# Story 9.6: Testability Audit

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want testability issues highlighted,
So that I can improve test coverage and code maintainability.

## Acceptance Criteria

1. **AC1: Testability Pattern Detection**
   - **Given** code files from Dev agent implementation
   - **When** testability audit runs
   - **Then** hard-to-test patterns are identified:
     - Global state/singletons
     - Tightly coupled dependencies (no dependency injection)
     - Hidden dependencies (imports inside functions)
     - Complex conditionals (cyclomatic complexity > 10)
     - Long methods (> 50 lines)
     - Deep nesting (> 4 levels)
   - **And** each pattern is logged with specific location
   - **And** patterns are categorized by impact on testability

2. **AC2: Testability Issue Creation**
   - **Given** detected testability patterns
   - **When** issues are generated
   - **Then** each issue includes:
     - issue_id (unique, format "T-{file_hash}-{seq}")
     - pattern_type (e.g., "global_state", "tight_coupling", "complex_conditional")
     - severity (critical, high, medium, low)
     - location (file path and line number range)
     - description (human-readable explanation)
     - impact (how it affects testability)
     - remediation (specific improvement suggestion)
   - **And** issues are stored in `TestabilityIssue` frozen dataclass

3. **AC3: Testability Score Calculation**
   - **Given** all testability issues for a codebase
   - **When** testability score is calculated
   - **Then** score ranges from 0-100 (integer)
   - **And** score formula considers:
     - Base score of 100
     - Critical patterns: -20 per occurrence (capped at -60)
     - High patterns: -10 per occurrence (capped at -40)
     - Medium patterns: -5 per occurrence (capped at -20)
     - Low patterns: -2 per occurrence (capped at -10)
   - **And** score is deterministic and reproducible
   - **And** score is stored in `TestabilityScore` frozen dataclass

4. **AC4: Testability Report Generation**
   - **Given** testability analysis is complete
   - **When** report is generated
   - **Then** `TestabilityReport` includes:
     - issues (tuple of all TestabilityIssue)
     - score (TestabilityScore)
     - metrics (TestabilityMetrics with counts per pattern type)
     - recommendations (prioritized improvement suggestions)
     - created_at (ISO timestamp)
   - **And** report has `to_dict()` method for serialization
   - **And** report is immutable (frozen dataclass)

5. **AC5: Testability Recommendations**
   - **Given** testability issues have been identified
   - **When** recommendations are generated
   - **Then** recommendations are prioritized by impact
   - **And** each recommendation includes:
     - The pattern type being addressed
     - Specific refactoring suggestion
     - Expected testability improvement
   - **And** recommendations are actionable and specific

6. **AC6: Integration with TEA Node**
   - **Given** TEA node processes artifacts
   - **When** testability audit runs
   - **Then** `TestabilityReport` is included in TEAOutput
   - **And** `testability_score` field is accessible via to_dict()
   - **And** testability issues generate Findings for validation results
   - **And** overall confidence scoring considers testability score

7. **AC7: Testability Metrics Tracking**
   - **Given** testability audit runs on artifacts
   - **When** metrics are collected
   - **Then** metrics include:
     - total_issues (count of all issues)
     - issues_by_severity (dict with counts per severity level)
     - issues_by_pattern (dict with counts per pattern type)
     - files_analyzed (count of files processed)
     - files_with_issues (count of files having issues)
   - **And** metrics are stored in `TestabilityMetrics` frozen dataclass

## Tasks / Subtasks

- [x] Task 1: Create Testability Types (AC: 2, 3, 4, 7)
  - [x] Create `TestabilityPattern` Literal type: "global_state", "tight_coupling", "hidden_dependency", "complex_conditional", "long_method", "deep_nesting"
  - [x] Create `TestabilitySeverity` Literal type: "critical", "high", "medium", "low"
  - [x] Create `TestabilityIssue` frozen dataclass with: issue_id, pattern_type, severity, location, line_start, line_end, description, impact, remediation
  - [x] Create `TestabilityScore` frozen dataclass with: score (0-100), breakdown (penalties per category), base_score
  - [x] Create `TestabilityMetrics` frozen dataclass with: total_issues, issues_by_severity, issues_by_pattern, files_analyzed, files_with_issues
  - [x] Create `TestabilityReport` frozen dataclass with: issues, score, metrics, recommendations, created_at
  - [x] Add `to_dict()` methods for all dataclasses
  - [x] Add types to `agents/tea/testability.py` (NEW FILE)

- [x] Task 2: Implement Pattern Detection Functions (AC: 1)
  - [x] Create `_detect_global_state(content: str, file_path: str) -> list[TestabilityIssue]`
    - Detect module-level mutable variables
    - Detect singleton patterns
  - [x] Create `_detect_tight_coupling(content: str, file_path: str) -> list[TestabilityIssue]`
    - Detect direct instantiation in methods (no DI)
    - Detect classes that create their own dependencies
  - [x] Create `_detect_hidden_dependencies(content: str, file_path: str) -> list[TestabilityIssue]`
    - Detect imports inside functions
    - Detect dynamic imports
  - [x] Create `_detect_complex_conditionals(content: str, file_path: str) -> list[TestabilityIssue]`
    - Detect methods with cyclomatic complexity > 10
    - Use AST analysis for accuracy
  - [x] Create `_detect_long_methods(content: str, file_path: str) -> list[TestabilityIssue]`
    - Detect functions > 50 lines
    - Use AST to get accurate line counts
  - [x] Create `_detect_deep_nesting(content: str, file_path: str) -> list[TestabilityIssue]`
    - Detect nesting > 4 levels deep
    - Track if/for/while/with nesting

- [x] Task 3: Implement Issue Generation (AC: 2)
  - [x] Create `_generate_issue_id(file_path: str, sequence: int) -> str`
    - Format: "T-{file_hash[:8]}-{seq:03d}"
  - [x] Create `_map_pattern_to_severity(pattern: TestabilityPattern) -> TestabilitySeverity`
    - global_state -> critical (hardest to test)
    - tight_coupling -> high
    - hidden_dependency -> high
    - complex_conditional -> medium
    - long_method -> medium
    - deep_nesting -> low
  - [x] Create `_get_impact_description(pattern: TestabilityPattern) -> str`
  - [x] Create `_get_remediation_suggestion(pattern: TestabilityPattern) -> str`

- [x] Task 4: Implement Testability Score Calculation (AC: 3)
  - [x] Create `calculate_testability_score(issues: tuple[TestabilityIssue, ...]) -> TestabilityScore`
  - [x] Apply severity penalties:
    - Critical: -20 per occurrence (max -60)
    - High: -10 per occurrence (max -40)
    - Medium: -5 per occurrence (max -20)
    - Low: -2 per occurrence (max -10)
  - [x] Clamp score to 0-100 range
  - [x] Track penalty breakdown in score

- [x] Task 5: Implement Metrics Collection (AC: 7)
  - [x] Create `collect_testability_metrics(issues: tuple[TestabilityIssue, ...], files_analyzed: int) -> TestabilityMetrics`
  - [x] Count issues by severity level
  - [x] Count issues by pattern type
  - [x] Count unique files with issues

- [x] Task 6: Implement Recommendation Generation (AC: 5)
  - [x] Create `generate_testability_recommendations(issues: tuple[TestabilityIssue, ...]) -> tuple[str, ...]`
  - [x] Prioritize by severity (critical first)
  - [x] Group similar issues for consolidated recommendations
  - [x] Generate actionable, specific suggestions

- [x] Task 7: Implement Main Audit Function (AC: 1, 4)
  - [x] Create `audit_testability(code_files: list[dict[str, Any]]) -> TestabilityReport`
  - [x] Run all pattern detectors on each file
  - [x] Aggregate issues across all files
  - [x] Calculate testability score
  - [x] Collect metrics
  - [x] Generate recommendations
  - [x] Build and return TestabilityReport

- [x] Task 8: Implement Finding Conversion (AC: 6)
  - [x] Create `convert_testability_issues_to_findings(issues: tuple[TestabilityIssue, ...]) -> tuple[Finding, ...]`
  - [x] Map TestabilitySeverity to FindingSeverity
  - [x] Map TestabilityPattern to FindingCategory (use "code_quality")
  - [x] Generate Finding with appropriate remediation

- [x] Task 9: Integrate with TEA Node (AC: 6)
  - [x] Update `tea_node()` to call `audit_testability()` on code files
  - [x] Add `testability_report: TestabilityReport | None` field to TEAOutput
  - [x] Add testability findings to validation results
  - [x] Update processing_notes to include testability summary
  - [x] Factor testability score into confidence calculation

- [x] Task 10: Write Unit Tests for Types (AC: 2, 3, 4, 7)
  - [x] Test TestabilityIssue creation and to_dict()
  - [x] Test TestabilityScore creation and to_dict()
  - [x] Test TestabilityMetrics creation and to_dict()
  - [x] Test TestabilityReport creation and to_dict()
  - [x] Test immutability (frozen dataclass)

- [x] Task 11: Write Unit Tests for Pattern Detection (AC: 1)
  - [x] Test global state detection
  - [x] Test tight coupling detection
  - [x] Test hidden dependency detection
  - [x] Test complex conditional detection
  - [x] Test long method detection
  - [x] Test deep nesting detection
  - [x] Test edge cases (empty files, malformed code)

- [x] Task 12: Write Unit Tests for Score Calculation (AC: 3)
  - [x] Test score with various issue combinations
  - [x] Test penalty caps are applied correctly
  - [x] Test edge cases (no issues, all critical issues)
  - [x] Test score breakdown accuracy

- [x] Task 13: Write Unit Tests for Recommendations (AC: 5)
  - [x] Test recommendation generation
  - [x] Test prioritization by severity
  - [x] Test recommendation specificity

- [x] Task 14: Write Integration Tests (AC: 6)
  - [x] Test full testability audit flow
  - [x] Test TEA node integration with testability report
  - [x] Test TestabilityReport included in TEAOutput
  - [x] Test testability findings in validation results

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for TestabilityIssue, TestabilityScore, TestabilityMetrics, TestabilityReport
- **ADR-006 (Quality Gates):** Testability score feeds into overall confidence scoring
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions
- **ARCH-QUALITY-5:** Async patterns not required for pure analysis functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from Story 9.1, 9.2, 9.3, 9.4, 9.5 TEA implementations
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- Use tuples for immutable collections (not lists)
- Use Python's `ast` module for accurate code analysis
- Handle syntax errors gracefully (don't crash on malformed code)

### Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| structlog | latest | Structured logging |
| dataclasses | stdlib | Frozen dataclasses for types |
| ast | stdlib | Abstract syntax tree analysis |
| hashlib | stdlib | Hash-based ID generation |

### Pattern Detection Heuristics

The testability audit uses the following detection heuristics:

| Pattern | Detection Method | Impact on Testability |
|---------|-----------------|----------------------|
| Global State | Module-level mutable variables, singleton patterns | Cannot isolate tests, shared state causes flaky tests |
| Tight Coupling | Direct instantiation in methods (e.g., `self.db = Database()`) | Cannot mock dependencies, requires real implementations |
| Hidden Dependencies | Imports inside functions | Difficult to mock, surprises in test setup |
| Complex Conditionals | Cyclomatic complexity > 10 | Many test cases needed, hard to achieve coverage |
| Long Methods | Functions > 50 lines | Too many behaviors per test, unclear test boundaries |
| Deep Nesting | > 4 levels of if/for/while/with | Complex test setup, hard to reason about states |

### Severity Mapping

| Pattern Type | Severity | Penalty | Rationale |
|-------------|----------|---------|-----------|
| global_state | critical | -20 | Hardest to test, causes flaky tests |
| tight_coupling | high | -10 | Prevents mocking, requires integration tests |
| hidden_dependency | high | -10 | Surprises in test setup |
| complex_conditional | medium | -5 | Many test cases needed |
| long_method | medium | -5 | Unclear test boundaries |
| deep_nesting | low | -2 | Complex but manageable |

### TestabilityIssue Data Structure

```python
@dataclass(frozen=True)
class TestabilityIssue:
    """A testability issue found during code audit.

    Attributes:
        issue_id: Unique identifier (e.g., "T-a1b2c3d4-001")
        pattern_type: The testability anti-pattern detected
        severity: Impact severity on testability
        location: File path where issue was found
        line_start: Starting line number
        line_end: Ending line number
        description: Human-readable description
        impact: How this affects testability
        remediation: Suggested fix
        created_at: ISO timestamp when issue was created
    """

    issue_id: str
    pattern_type: TestabilityPattern
    severity: TestabilitySeverity
    location: str
    line_start: int
    line_end: int
    description: str
    impact: str
    remediation: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "pattern_type": self.pattern_type,
            "severity": self.severity,
            "location": self.location,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "description": self.description,
            "impact": self.impact,
            "remediation": self.remediation,
            "created_at": self.created_at,
        }
```

### TestabilityReport Data Structure

```python
@dataclass(frozen=True)
class TestabilityReport:
    """Complete testability audit report.

    Attributes:
        issues: Tuple of all testability issues found
        score: Testability score (0-100)
        metrics: Aggregated metrics about issues
        recommendations: Prioritized improvement suggestions
        created_at: ISO timestamp when report was created
    """

    issues: tuple[TestabilityIssue, ...]
    score: TestabilityScore
    metrics: TestabilityMetrics
    recommendations: tuple[str, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "score": self.score.to_dict(),
            "metrics": self.metrics.to_dict(),
            "recommendations": list(self.recommendations),
            "created_at": self.created_at,
        }
```

### AST-Based Pattern Detection Example

```python
import ast

def _detect_long_methods(content: str, file_path: str) -> list[TestabilityIssue]:
    """Detect functions longer than 50 lines using AST analysis."""
    issues: list[TestabilityIssue] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Handle malformed code gracefully
        return issues

    seq = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Calculate function length
            if node.end_lineno and node.lineno:
                length = node.end_lineno - node.lineno + 1
                if length > 50:
                    seq += 1
                    issues.append(
                        TestabilityIssue(
                            issue_id=_generate_issue_id(file_path, seq),
                            pattern_type="long_method",
                            severity="medium",
                            location=file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                            description=f"Function '{node.name}' is {length} lines long (> 50)",
                            impact="Long methods have unclear test boundaries and require too many test cases",
                            remediation=f"Split '{node.name}' into smaller, focused functions",
                        )
                    )

    return issues
```

### Cyclomatic Complexity Calculation

```python
def _calculate_cyclomatic_complexity(func_node: ast.FunctionDef) -> int:
    """Calculate McCabe cyclomatic complexity for a function.

    Complexity = 1 + number of decision points
    Decision points: if, elif, for, while, except, with, and, or, assert, comprehension
    """
    complexity = 1  # Base complexity

    for node in ast.walk(func_node):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # Count number of additional boolean operations
            complexity += len(node.values) - 1
        elif isinstance(node, ast.Assert):
            complexity += 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1

    return complexity
```

### Project Structure Notes

- **Module Location:** `src/yolo_developer/agents/tea/`
- **New Files:**
  - `src/yolo_developer/agents/tea/testability.py` - Testability audit types and functions
- **Modified Files:**
  - `src/yolo_developer/agents/tea/types.py` - Add testability_report to TEAOutput
  - `src/yolo_developer/agents/tea/node.py` - Integrate testability audit
  - `src/yolo_developer/agents/tea/__init__.py` - Export new types
- **Test Location:** `tests/unit/agents/tea/`

### Existing Code to Integrate With

The Story 9.1-9.5 implementations provide:
- `Finding` - From Story 9.1, convert testability issues to findings
- `FindingSeverity` - Map testability severity to finding severity
- `FindingCategory` - Use "code_quality" for testability findings
- `ValidationResult` - Include testability findings
- `TEAOutput` - Extend with `testability_report` field
- `tea_node()` - Call `audit_testability()` on code files
- `ConfidenceResult` - Consider testability in overall confidence

### Key Integration Points

```python
# In tea_node(), after artifact extraction:

# Audit testability on code files
code_artifacts = [a for a in artifacts if a.get("type") == "code_file"]
testability_report = None
testability_findings: list[Finding] = []

if code_artifacts:
    testability_report = audit_testability(code_artifacts)
    testability_findings = list(convert_testability_issues_to_findings(testability_report.issues))

    logger.info(
        "testability_audit_complete",
        score=testability_report.score.score,
        issue_count=testability_report.metrics.total_issues,
        files_with_issues=testability_report.metrics.files_with_issues,
    )

# Add testability findings to validation
if testability_findings:
    testability_validation_result = ValidationResult(
        artifact_id="testability_audit",
        validation_status="warning" if testability_findings else "passed",
        findings=tuple(testability_findings),
        recommendations=testability_report.recommendations if testability_report else (),
        score=testability_report.score.score if testability_report else 100,
    )
    validation_results.append(testability_validation_result)

# Include in TEAOutput
tea_output = TEAOutput(
    validation_results=validation_results,
    overall_confidence=overall_confidence,
    deployment_recommendation=deployment_recommendation,
    confidence_result=confidence_result,
    risk_report=risk_report,
    overall_risk_level=risk_report.overall_risk_level,
    testability_report=testability_report,  # NEW
    processing_notes=f"{processing_notes}\nTestability: score={testability_report.score.score if testability_report else 100}, issues={len(testability_report.issues) if testability_report else 0}",
)
```

### Impact Description Templates

```python
IMPACT_TEMPLATES: dict[TestabilityPattern, str] = {
    "global_state": "Global state cannot be isolated between tests, causing flaky tests and making parallel test execution unsafe",
    "tight_coupling": "Tight coupling prevents mocking dependencies, requiring integration tests instead of fast unit tests",
    "hidden_dependency": "Hidden dependencies make test setup surprising and difficult to mock correctly",
    "complex_conditional": "Complex conditionals require many test cases to achieve coverage and are prone to untested edge cases",
    "long_method": "Long methods have unclear test boundaries, requiring overly complex test setups",
    "deep_nesting": "Deep nesting creates complex state combinations that are difficult to test exhaustively",
}
```

### Remediation Templates

```python
REMEDIATION_TEMPLATES: dict[TestabilityPattern, str] = {
    "global_state": "Extract global state into a class that can be injected, or use dependency injection pattern",
    "tight_coupling": "Accept dependencies as constructor parameters instead of creating them internally",
    "hidden_dependency": "Move imports to module level and accept dependencies as parameters",
    "complex_conditional": "Extract conditional branches into separate methods with clear responsibilities",
    "long_method": "Split into smaller, focused methods following single responsibility principle",
    "deep_nesting": "Use early returns, extract methods, or use guard clauses to reduce nesting",
}
```

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR77 | TEA can identify hard-to-test code patterns | Pattern detection for 6 testability anti-patterns |
| FR79 | TEA can generate test coverage reports with gap analysis | Testability metrics complement coverage analysis |
| FR65 | TEA can validate all Dev deliverables | Testability audit extends validation |

### Previous Story Learnings Applied

From Story 9.1, 9.2, 9.3, 9.4, 9.5:
- Use frozen dataclasses for all data structures with to_dict() methods
- Integrate with existing `tea_node()` function flow
- Follow structured logging pattern with structlog
- Handle empty/missing content gracefully
- Use tuples (not lists) for immutable collections
- Add TYPE_CHECKING imports for circular import prevention
- Use hash-based IDs for unique identification
- Convert domain issues to Findings for validation integration

### Git Commit Pattern

```
feat: Implement testability audit with code review fixes (Story 9.6)
```

### Sample TestabilityReport Output

```python
TestabilityReport(
    issues=(
        TestabilityIssue(
            issue_id="T-a1b2c3d4-001",
            pattern_type="global_state",
            severity="critical",
            location="src/yolo_developer/config.py",
            line_start=15,
            line_end=15,
            description="Module-level mutable dictionary '_config_cache'",
            impact="Global state cannot be isolated between tests, causing flaky tests",
            remediation="Extract global state into a class that can be injected",
        ),
        TestabilityIssue(
            issue_id="T-e5f6g7h8-001",
            pattern_type="long_method",
            severity="medium",
            location="src/yolo_developer/agents/dev/node.py",
            line_start=50,
            line_end=120,
            description="Function 'dev_node' is 71 lines long (> 50)",
            impact="Long methods have unclear test boundaries",
            remediation="Split 'dev_node' into smaller, focused methods",
        ),
    ),
    score=TestabilityScore(
        score=70,
        base_score=100,
        breakdown={
            "critical_penalty": -20,
            "high_penalty": 0,
            "medium_penalty": -5,
            "low_penalty": 0,
        },
    ),
    metrics=TestabilityMetrics(
        total_issues=2,
        issues_by_severity={"critical": 1, "high": 0, "medium": 1, "low": 0},
        issues_by_pattern={"global_state": 1, "long_method": 1},
        files_analyzed=10,
        files_with_issues=2,
    ),
    recommendations=(
        "CRITICAL: Refactor src/yolo_developer/config.py to remove global state pattern",
        "MEDIUM: Split long function 'dev_node' in src/yolo_developer/agents/dev/node.py",
    ),
    created_at="2026-01-12T10:00:00.000Z",
)
```

### Testability Best Practices (2025-2026)

Based on industry research:
- Static analysis for testability is increasingly automated in CI/CD pipelines
- Testability metrics correlate with defect density and maintenance costs
- Pattern-based detection scales better than manual code review
- Actionable recommendations improve adoption of testability improvements
- Testability scoring provides objective quality metrics for teams

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-9.6] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR77] - Hard-to-test code patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] - State management patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate patterns
- [Source: src/yolo_developer/agents/tea/types.py] - Existing TEA types
- [Source: src/yolo_developer/agents/tea/node.py] - Existing TEA node implementation
- [Source: src/yolo_developer/agents/tea/risk.py] - Risk categorization (Story 9.5)
- [Source: src/yolo_developer/agents/tea/scoring.py] - Confidence scoring (Story 9.4)
- [Source: _bmad-output/implementation-artifacts/9-5-risk-categorization.md] - Story 9.5 patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- Implemented complete testability audit system with 6 pattern detectors
- Created frozen dataclasses: TestabilityIssue, TestabilityScore, TestabilityMetrics, TestabilityReport
- Pattern detection using Python AST: global_state, tight_coupling, hidden_dependency, complex_conditional, long_method, deep_nesting
- Score calculation with severity-based penalties and caps (0-100 scale)
- Metrics collection with aggregation by severity and pattern type
- Recommendation generation with severity-based prioritization
- Finding conversion for TEA validation integration
- Full integration with tea_node() - testability_report in TEAOutput, findings in validation results
- 79 new tests: 15 type tests, 30 detection tests, 22 scoring/metrics/recommendations tests, 12 integration tests
- All 404 TEA tests passing
- mypy and ruff checks passing

### File List

**New Files:**
- src/yolo_developer/agents/tea/testability.py
- tests/unit/agents/tea/test_testability_types.py
- tests/unit/agents/tea/test_testability_detection.py
- tests/unit/agents/tea/test_testability_scoring.py
- tests/unit/agents/tea/test_testability_integration.py

**Modified Files:**
- src/yolo_developer/agents/tea/__init__.py
- src/yolo_developer/agents/tea/types.py
- src/yolo_developer/agents/tea/node.py

