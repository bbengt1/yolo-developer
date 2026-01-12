# Story 9.5: Risk Categorization

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want risks categorized by severity,
So that I can focus on the most important issues.

## Acceptance Criteria

1. **AC1: Risk Categorization System**
   - **Given** identified issues from TEA validation
   - **When** risk categorization runs
   - **Then** issues are tagged with severity levels: Critical, High, or Low
   - **And** categorization is based on finding severity and impact assessment
   - **And** categorization is deterministic and reproducible

2. **AC2: Critical Risk Handling**
   - **Given** issues tagged as Critical
   - **When** deployment decision is made
   - **Then** deployment is blocked automatically
   - **And** blocking message includes the specific critical issues
   - **And** critical risks are prominently displayed in output
   - **And** a `RiskReport` includes count of critical risks

3. **AC3: High Risk Handling**
   - **Given** issues tagged as High
   - **When** deployment decision is made
   - **Then** deployment requires explicit acknowledgment
   - **And** high risks are summarized with their locations
   - **And** acknowledgment workflow is clearly communicated
   - **And** a `RiskReport` includes count of high risks

4. **AC4: Low Risk Handling**
   - **Given** issues tagged as Low
   - **When** deployment decision is made
   - **Then** risks are noted in output but don't block deployment
   - **And** low risks are logged for informational purposes
   - **And** a `RiskReport` includes count of low risks

5. **AC5: Risk Report Generation**
   - **Given** risk categorization has completed
   - **When** a risk report is generated
   - **Then** report includes summary by risk level (critical_count, high_count, low_count)
   - **And** report includes list of all categorized risks with details
   - **And** report includes overall risk_level (critical/high/low/none)
   - **And** report includes deployment_blocked boolean
   - **And** report is stored in `RiskReport` frozen dataclass with `to_dict()` method

6. **AC6: Integration with TEA Output**
   - **Given** TEA node produces output
   - **When** risk categorization runs
   - **Then** `RiskReport` is included in TEAOutput
   - **And** risk_report field is accessible via `to_dict()` method
   - **And** overall_risk_level field is added to TEAOutput for quick access
   - **And** existing confidence scoring and deployment recommendation integrate with risk data

## Tasks / Subtasks

- [x] Task 1: Create Risk Categorization Types (AC: 1, 5)
  - [x] Create `RiskLevel` literal type: "critical", "high", "low"
  - [x] Create `OverallRiskLevel` literal type: "critical", "high", "low", "none"
  - [x] Create `CategorizedRisk` frozen dataclass with: risk_id, finding, risk_level, impact_description, requires_acknowledgment
  - [x] Create `RiskReport` frozen dataclass with: risks (tuple), critical_count, high_count, low_count, overall_risk_level, deployment_blocked
  - [x] Add `to_dict()` methods for serialization
  - [x] Add types to `agents/tea/risk.py` (NEW FILE)

- [x] Task 2: Implement Finding-to-Risk Mapping (AC: 1)
  - [x] Create `_map_severity_to_risk_level(severity: FindingSeverity) -> RiskLevel`
  - [x] Critical severity -> Critical risk
  - [x] High severity -> High risk
  - [x] Medium, Low, Info severity -> Low risk
  - [x] Add docstring explaining the mapping rationale

- [x] Task 3: Implement Risk Categorization Logic (AC: 1, 2, 3, 4)
  - [x] Create `categorize_finding(finding: Finding) -> CategorizedRisk` function
  - [x] Generate unique risk_id from finding_id (e.g., "R-{finding_id}")
  - [x] Map finding severity to risk level
  - [x] Generate impact description based on category and severity
  - [x] Set requires_acknowledgment=True for High risks, False otherwise

- [x] Task 4: Implement Bulk Risk Categorization (AC: 1, 5)
  - [x] Create `categorize_risks(validation_results: tuple[ValidationResult, ...]) -> tuple[CategorizedRisk, ...]`
  - [x] Extract all findings from all validation results
  - [x] Categorize each finding into a CategorizedRisk
  - [x] Return tuple of all categorized risks
  - [x] Handle empty validation results gracefully

- [x] Task 5: Implement Risk Report Generation (AC: 5)
  - [x] Create `generate_risk_report(categorized_risks: tuple[CategorizedRisk, ...]) -> RiskReport`
  - [x] Count risks by level (critical_count, high_count, low_count)
  - [x] Determine overall_risk_level (highest severity present, or "none" if empty)
  - [x] Determine deployment_blocked (True if any critical risks)
  - [x] Build and return RiskReport

- [x] Task 6: Implement Deployment Blocking Logic (AC: 2)
  - [x] Create `check_risk_deployment_blocking(risk_report: RiskReport) -> tuple[bool, list[str]]`
  - [x] Return (is_blocked: bool, blocking_reasons: list[str])
  - [x] Block if critical_count > 0
  - [x] Include specific critical risk descriptions in blocking reasons

- [x] Task 7: Implement Acknowledgment Requirements (AC: 3)
  - [x] Create `get_acknowledgment_requirements(risk_report: RiskReport) -> tuple[str, ...]`
  - [x] Return tuple of high-risk items requiring acknowledgment
  - [x] Format each item with risk_id, location, and description
  - [x] Return empty tuple if no high risks

- [x] Task 8: Integrate with TEA Node (AC: 6)
  - [x] Update `tea_node()` to call `categorize_risks()` after validation
  - [x] Update `tea_node()` to call `generate_risk_report()`
  - [x] Add `risk_report: RiskReport | None` field to TEAOutput
  - [x] Add `overall_risk_level: OverallRiskLevel` field to TEAOutput
  - [x] Integrate risk blocking with deployment recommendation
  - [x] Update processing_notes to include risk summary

- [x] Task 9: Write Unit Tests for Types (AC: 5)
  - [x] Test CategorizedRisk creation and to_dict()
  - [x] Test RiskReport creation and to_dict()
  - [x] Test immutability (frozen dataclass)
  - [x] Test edge cases (empty risks, all same level)

- [x] Task 10: Write Unit Tests for Categorization (AC: 1, 2, 3, 4)
  - [x] Test severity-to-risk mapping for all severities
  - [x] Test categorize_finding for each finding category
  - [x] Test categorize_risks with mixed findings
  - [x] Test categorize_risks with empty validation results

- [x] Task 11: Write Unit Tests for Risk Report (AC: 5)
  - [x] Test report generation with various risk combinations
  - [x] Test overall_risk_level determination
  - [x] Test deployment_blocked flag
  - [x] Test counts accuracy

- [x] Task 12: Write Unit Tests for Blocking and Acknowledgment (AC: 2, 3)
  - [x] Test deployment blocking with critical risks
  - [x] Test no blocking with only high/low risks
  - [x] Test acknowledgment requirements generation
  - [x] Test empty acknowledgment when no high risks

- [x] Task 13: Write Integration Tests (AC: 6)
  - [x] Test full risk categorization flow
  - [x] Test TEA node integration with risk report
  - [x] Test RiskReport included in TEAOutput
  - [x] Test risk blocking integrates with confidence scoring

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for CategorizedRisk, RiskReport
- **ADR-006 (Quality Gates):** Risk categorization feeds into deployment blocking gate
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions
- **ARCH-QUALITY-5:** Async patterns not required for pure calculation functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from Story 9.1, 9.2, 9.3, 9.4 TEA implementations
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- Use tuples for immutable collections (not lists)

### Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| structlog | latest | Structured logging |
| dataclasses | stdlib | Frozen dataclasses for types |

### Risk Level Mapping

The risk categorization maps finding severity to risk levels:

| Finding Severity | Risk Level | Deployment Impact |
|-----------------|------------|-------------------|
| critical | Critical | Blocks deployment |
| high | High | Requires acknowledgment |
| medium | Low | Noted, doesn't block |
| low | Low | Noted, doesn't block |
| info | Low | Noted, doesn't block |

### Impact Description Templates

```python
# Generate impact descriptions based on category and severity
IMPACT_TEMPLATES = {
    "test_coverage": {
        "critical": "Critical test coverage gap may allow undetected bugs in production",
        "high": "Significant test coverage gap increases risk of regressions",
        "low": "Minor test coverage improvement recommended",
    },
    "code_quality": {
        "critical": "Critical code quality issue may cause runtime failures",
        "high": "Code quality concern could impact maintainability",
        "low": "Minor code quality improvement suggested",
    },
    "documentation": {
        "critical": "Missing critical documentation blocks understanding",
        "high": "Documentation gap affects developer experience",
        "low": "Documentation enhancement would improve clarity",
    },
    "security": {
        "critical": "Critical security vulnerability must be addressed immediately",
        "high": "Security concern requires review before deployment",
        "low": "Minor security improvement recommended",
    },
    "performance": {
        "critical": "Critical performance issue will impact user experience",
        "high": "Performance concern should be addressed",
        "low": "Minor performance optimization opportunity",
    },
    "architecture": {
        "critical": "Critical architecture violation breaks system design",
        "high": "Architecture deviation needs justification",
        "low": "Minor architecture refinement suggested",
    },
}
```

### CategorizedRisk Data Structure

```python
@dataclass(frozen=True)
class CategorizedRisk:
    """A categorized risk derived from a validation finding.

    Attributes:
        risk_id: Unique identifier for the risk (e.g., "R-F001")
        finding: The original Finding that generated this risk
        risk_level: Categorized risk level (critical/high/low)
        impact_description: Human-readable impact description
        requires_acknowledgment: Whether deployment requires acknowledging this risk
    """

    risk_id: str
    finding: Finding
    risk_level: RiskLevel
    impact_description: str
    requires_acknowledgment: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_id": self.risk_id,
            "finding": self.finding.to_dict(),
            "risk_level": self.risk_level,
            "impact_description": self.impact_description,
            "requires_acknowledgment": self.requires_acknowledgment,
        }
```

### RiskReport Data Structure

```python
@dataclass(frozen=True)
class RiskReport:
    """Complete risk categorization report.

    Attributes:
        risks: Tuple of all categorized risks
        critical_count: Count of critical-level risks
        high_count: Count of high-level risks
        low_count: Count of low-level risks
        overall_risk_level: Highest risk level present (or "none")
        deployment_blocked: Whether deployment should be blocked
        blocking_reasons: Reasons for blocking (if blocked)
        acknowledgment_required: Items requiring explicit acknowledgment
        created_at: ISO timestamp when report was created
    """

    risks: tuple[CategorizedRisk, ...]
    critical_count: int
    high_count: int
    low_count: int
    overall_risk_level: OverallRiskLevel
    deployment_blocked: bool
    blocking_reasons: tuple[str, ...] = field(default_factory=tuple)
    acknowledgment_required: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "risks": [r.to_dict() for r in self.risks],
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "low_count": self.low_count,
            "overall_risk_level": self.overall_risk_level,
            "deployment_blocked": self.deployment_blocked,
            "blocking_reasons": list(self.blocking_reasons),
            "acknowledgment_required": list(self.acknowledgment_required),
            "created_at": self.created_at,
        }
```

### Project Structure Notes

- **Module Location:** `src/yolo_developer/agents/tea/`
- **New Files:**
  - `src/yolo_developer/agents/tea/risk.py` - Risk categorization types and functions
- **Modified Files:**
  - `src/yolo_developer/agents/tea/types.py` - Add risk_report and overall_risk_level to TEAOutput
  - `src/yolo_developer/agents/tea/node.py` - Integrate risk categorization
  - `src/yolo_developer/agents/tea/__init__.py` - Export new types
- **Test Location:** `tests/unit/agents/tea/`

### Existing Code to Integrate With

The Story 9.1-9.4 implementations provide:
- `Finding` - From Story 9.1, provides findings for risk categorization
- `FindingSeverity` - Literal type used for severity mapping
- `FindingCategory` - Literal type used for impact templates
- `ValidationResult` - From Story 9.1, contains findings for processing
- `TEAOutput` - Extend with `risk_report` and `overall_risk_level` fields
- `DeploymentRecommendation` - Already exists, integrate with risk blocking
- `ConfidenceResult` - From Story 9.4, coordinate blocking logic

### Key Integration Points

```python
# In tea_node(), after validation and confidence scoring:

# Categorize risks from validation results
categorized_risks = categorize_risks(validation_results)

# Generate risk report
risk_report = generate_risk_report(categorized_risks)

# Check if risk should block deployment (in addition to confidence score)
risk_blocked, risk_blocking_reasons = check_risk_deployment_blocking(risk_report)

# Update deployment recommendation if risk blocks
if risk_blocked:
    deployment_recommendation = "block"
    # Merge blocking reasons

# Include in TEAOutput
tea_output = TEAOutput(
    validation_results=validation_results,
    overall_confidence=overall_confidence,
    deployment_recommendation=deployment_recommendation,
    confidence_result=confidence_result,
    risk_report=risk_report,  # NEW
    overall_risk_level=risk_report.overall_risk_level,  # NEW
    processing_notes=f"{processing_notes}\nRisk: {risk_report.critical_count} critical, {risk_report.high_count} high, {risk_report.low_count} low",
)
```

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR76 | TEA Agent can categorize risks (Critical, High, Low) with appropriate responses | RiskLevel categorization with CategorizedRisk |
| FR78 | TEA Agent can block deployment when confidence score < 90% | Risk blocking integrates with confidence blocking |
| FR79 | TEA Agent can generate test coverage reports with gap analysis | RiskReport includes detailed risk breakdown |

### Previous Story Learnings Applied

From Story 9.1, 9.2, 9.3, 9.4:
- Use frozen dataclasses for all data structures with to_dict() methods
- Integrate with existing `tea_node()` function flow
- Follow structured logging pattern with structlog
- Handle empty/missing content gracefully
- Use tuples (not lists) for immutable collections
- Add TYPE_CHECKING imports for circular import prevention
- Use hash-based IDs for unique identification

### Git Commit Pattern

```
feat: Implement risk categorization with code review fixes (Story 9.5)
```

### Sample RiskReport Output

```python
RiskReport(
    risks=(
        CategorizedRisk(
            risk_id="R-F001",
            finding=Finding(finding_id="F001", severity="critical", ...),
            risk_level="critical",
            impact_description="Critical test coverage gap may allow undetected bugs in production",
            requires_acknowledgment=False,
        ),
        CategorizedRisk(
            risk_id="R-F002",
            finding=Finding(finding_id="F002", severity="high", ...),
            risk_level="high",
            impact_description="Code quality concern could impact maintainability",
            requires_acknowledgment=True,
        ),
    ),
    critical_count=1,
    high_count=1,
    low_count=0,
    overall_risk_level="critical",
    deployment_blocked=True,
    blocking_reasons=("Critical risk R-F001: Critical test coverage gap may allow undetected bugs in production",),
    acknowledgment_required=("R-F002: Code quality concern could impact maintainability at src/module.py",),
    created_at="2026-01-12T10:00:00.000Z",
)
```

### Risk Categorization Best Practices (2025)

Based on industry research:
- Use consistent severity-to-risk mapping for predictable behavior
- Provide clear impact descriptions for actionability
- Differentiate blocking vs acknowledgment requirements
- Support aggregated risk reporting for executive summaries
- Integrate risk data with confidence scoring for comprehensive assessment

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-9.5] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR76] - Risk categorization
- [Source: _bmad-output/planning-artifacts/prd.md#FR78] - Deployment blocking
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] - State management patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate patterns
- [Source: src/yolo_developer/agents/tea/types.py] - Existing TEA types (Finding, ValidationResult, TEAOutput)
- [Source: src/yolo_developer/agents/tea/node.py] - Existing TEA node implementation
- [Source: src/yolo_developer/agents/tea/scoring.py] - Confidence scoring (Story 9.4)
- [Source: _bmad-output/implementation-artifacts/9-4-confidence-scoring.md] - Story 9.4 patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 320 TEA agent tests pass
- mypy: Success, no issues found in 7 source files
- ruff: All checks passed

### Completion Notes List

- Implemented all 13 tasks using TDD (red-green-refactor)
- Created new `risk.py` module with types and functions
- Extended `TEAOutput` with `risk_report` and `overall_risk_level` fields
- Integrated risk categorization into `tea_node()` flow
- Risk blocking now supplements confidence scoring for deployment decisions
- All acceptance criteria verified through comprehensive unit and integration tests

**Code Review Fixes Applied:**
- [M1] Removed unreachable else branch in `generate_risk_report()` (dead code)
- [M2] Added 5 unit tests for `_get_impact_description()` including unknown category fallback
- [M3] Updated `__init__.py` docstring to include Story 9.5 and risk categorization
- [M5] Updated `TEAOutput` docstring with `risk_report` and `overall_risk_level` fields

### File List

**New Files:**
- `src/yolo_developer/agents/tea/risk.py` - Risk categorization types and functions
- `tests/unit/agents/tea/test_risk_types.py` - Unit tests for risk types (21 tests)
- `tests/unit/agents/tea/test_risk_categorization.py` - Unit tests for categorization functions (38 tests)
- `tests/unit/agents/tea/test_risk_integration.py` - Integration tests (10 tests)

**Modified Files:**
- `src/yolo_developer/agents/tea/types.py` - Added risk_report and overall_risk_level to TEAOutput
- `src/yolo_developer/agents/tea/node.py` - Integrated risk categorization into tea_node()
- `src/yolo_developer/agents/tea/__init__.py` - Exported new risk types and functions
