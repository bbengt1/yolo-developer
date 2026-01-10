# Story 7.7: ATAM Review

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want designs pass basic architectural review,
so that fundamental issues are caught before implementation.

## Acceptance Criteria

1. **Given** a complete design with ArchitectOutput
   **When** the `run_atam_review()` function is called with design decisions, quality evaluations, and risk reports
   **Then** architectural approaches are evaluated against ATAM criteria
   **And** results are returned as an `ATAMReviewResult` frozen dataclass
   **And** the function is importable from `yolo_developer.agents.architect`

2. **Given** design decisions and quality attribute evaluations
   **When** ATAM review runs
   **Then** quality attribute trade-offs are analyzed for feasibility
   **And** conflicting trade-offs are identified
   **And** each conflict includes affected attributes and severity

3. **Given** technical risk reports from Story 7.5
   **When** ATAM review integrates risk analysis
   **Then** risks are categorized by impact on quality attributes
   **And** risk mitigation feasibility is assessed
   **And** unmitigated critical risks are flagged

4. **Given** a complete ATAM review
   **When** review completes
   **Then** a pass/fail decision is made based on configured thresholds
   **And** the decision includes a confidence score (0.0-1.0)
   **And** specific failure reasons are documented when failed

5. **Given** the architect_node processing stories
   **When** ATAM review is performed
   **Then** it runs after tech stack constraint validation (Story 7.6)
   **And** it uses results from quality evaluation (Story 7.4) and risk identification (Story 7.5)
   **And** review results are included in ArchitectOutput
   **And** summary is logged via structlog

6. **Given** LLM-powered ATAM analysis
   **When** analyzing complex architectural scenarios
   **Then** it uses litellm with configurable model via env var
   **And** it includes tenacity retry with exponential backoff
   **And** it handles LLM failures gracefully with rule-based fallback

7. **Given** the ATAMReviewResult dataclass
   **When** all analysis is complete
   **Then** it is frozen (immutable) per ADR-001
   **And** it has to_dict() method for serialization
   **And** it includes overall_pass (True/False) based on review criteria

## Tasks / Subtasks

- [x] Task 1: Create ATAM Review Type Definitions (AC: 1, 7)
  - [x] Create `ATAMScenario` frozen dataclass with: scenario_id, quality_attribute, stimulus, response, analysis
  - [x] Create `ATAMTradeOffConflict` frozen dataclass with: attribute_a, attribute_b, description, severity, resolution_strategy
  - [x] Create `ATAMRiskAssessment` frozen dataclass with: risk_id, quality_impact, mitigation_feasibility, unmitigated
  - [x] Create `ATAMReviewResult` frozen dataclass with: overall_pass, confidence, scenarios_evaluated, trade_off_conflicts, risk_assessments, failure_reasons, summary, to_dict()
  - [x] Add type exports to `architect/__init__.py`

- [x] Task 2: Implement Scenario Generation (AC: 1, 2)
  - [x] Create `src/yolo_developer/agents/architect/atam_reviewer.py` module
  - [x] Implement `_generate_atam_scenarios(design_decisions, quality_eval) -> list[ATAMScenario]`
  - [x] Generate scenarios for each quality attribute affected by design decisions
  - [x] Include stimulus-response pairs for architectural evaluation
  - [x] Add structlog logging for scenario generation

- [x] Task 3: Implement Trade-Off Conflict Detection (AC: 2)
  - [x] Implement `_detect_trade_off_conflicts(trade_offs, design_decisions) -> list[ATAMTradeOffConflict]`
  - [x] Analyze QualityTradeOff objects from quality evaluation
  - [x] Detect conflicting resolutions between trade-offs
  - [x] Assign severity based on conflict impact (critical, high, medium, low)
  - [x] Suggest resolution strategies for conflicts

- [x] Task 4: Implement Risk Impact Assessment (AC: 3)
  - [x] Implement `_assess_risk_impact(risk_report, quality_eval) -> list[ATAMRiskAssessment]`
  - [x] Map technical risks to affected quality attributes
  - [x] Evaluate mitigation feasibility based on effort and priority
  - [x] Flag unmitigated critical risks for review failure

- [x] Task 5: Implement Pass/Fail Decision Logic (AC: 4)
  - [x] Implement `_make_review_decision(scenarios, conflicts, risk_assessments) -> tuple[bool, float, list[str]]`
  - [x] Calculate confidence score based on scenario coverage and risk levels
  - [x] Apply configurable thresholds (via YoloConfig or defaults)
  - [x] Generate specific failure reasons when review fails
  - [x] Default thresholds: fail if any critical unmitigated risks OR confidence < 0.6

- [x] Task 6: Implement LLM-Powered ATAM Analysis (AC: 6)
  - [x] Create `_analyze_atam_with_llm(design_decisions, quality_eval, risk_report) -> ATAMReviewResult | None`
  - [x] Design prompt template for ATAM-style architectural analysis
  - [x] Add tenacity @retry decorator with exponential backoff (3 attempts)
  - [x] Use configurable model via YOLO_LLM__ROUTINE_MODEL env var
  - [x] Implement graceful fallback to rule-based analysis on LLM failure
  - [x] Parse LLM JSON response to typed objects

- [x] Task 7: Create Main Review Function (AC: 1, 4, 5, 7)
  - [x] Create `run_atam_review(design_decisions, quality_eval?, risk_report?, config?) -> ATAMReviewResult` async function
  - [x] Orchestrate scenario generation, conflict detection, and risk assessment
  - [x] Calculate overall_pass and confidence score
  - [x] Generate summary text describing review outcome
  - [x] Add structlog logging for review start/complete

- [x] Task 8: Integrate with architect_node (AC: 5)
  - [x] Update `architect_node` to call `run_atam_review` after tech stack validation
  - [x] Add `atam_review` field to ArchitectOutput dataclass
  - [x] Include review summary in processing_notes
  - [x] Update ArchitectOutput.to_dict() to include ATAM review results

- [x] Task 9: Write Unit Tests for Types (AC: 7)
  - [x] Test ATAMScenario dataclass creation and to_dict()
  - [x] Test ATAMTradeOffConflict dataclass creation and to_dict()
  - [x] Test ATAMRiskAssessment dataclass creation and to_dict()
  - [x] Test ATAMReviewResult dataclass creation and to_dict()
  - [x] Test immutability of frozen dataclasses

- [x] Task 10: Write Unit Tests for Scenario Generation (AC: 1, 2)
  - [x] Test scenario generation from design decisions
  - [x] Test scenario generation per quality attribute
  - [x] Test stimulus-response pair generation
  - [x] Test empty scenarios when no decisions

- [x] Task 11: Write Unit Tests for Conflict Detection (AC: 2)
  - [x] Test trade-off conflict detection
  - [x] Test severity assignment for conflicts
  - [x] Test resolution strategy suggestion
  - [x] Test no conflicts when trade-offs are compatible

- [x] Task 12: Write Unit Tests for Risk Assessment (AC: 3)
  - [x] Test risk to quality attribute mapping
  - [x] Test mitigation feasibility evaluation
  - [x] Test critical unmitigated risk flagging
  - [x] Test empty assessments when no risks

- [x] Task 13: Write Unit Tests for Pass/Fail Decision (AC: 4)
  - [x] Test pass decision with good scores
  - [x] Test fail decision with critical risks
  - [x] Test confidence score calculation
  - [x] Test failure reason generation

- [x] Task 14: Write Unit Tests for LLM Integration (AC: 6)
  - [x] Test LLM analysis with mocked LLM
  - [x] Test retry behavior on transient failures
  - [x] Test fallback to rule-based on LLM failure
  - [x] Test JSON parsing of LLM response

- [x] Task 15: Write Integration Tests (AC: 5)
  - [x] Test architect_node includes atam_review
  - [x] Test end-to-end flow with mock design and evaluations
  - [x] Test integration with quality evaluation and risk reports
  - [x] Test ArchitectOutput serialization with ATAM review results

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for ATAMScenario, ATAMTradeOffConflict, ATAMRiskAssessment, ATAMReviewResult (internal state)
- **ADR-003 (LLM Abstraction):** Use litellm for LLM calls with configurable model
- **ADR-005 (LangGraph Communication):** Return state update dict, don't mutate state directly
- **ADR-007 (Error Handling):** Use tenacity with exponential backoff for LLM calls
- **ARCH-QUALITY-5:** All I/O operations (LLM calls) must be async/await
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### ATAM Background

ATAM (Architecture Tradeoff Analysis Method) is a risk-mitigation process for evaluating software architectures. Key concepts:

| Concept | Description | Application in Story |
|---------|-------------|---------------------|
| Quality Attribute Scenarios | Concrete expressions of how system should respond | Generated from design decisions |
| Trade-off Points | Decisions affecting multiple quality attributes | Detected from QualityTradeOff objects |
| Sensitivity Points | Architectural decisions critical to one attribute | Identified in scenarios |
| Risks | Potential problems with architectural decisions | Integrated from TechnicalRiskReport |

### Review Decision Criteria

| Criterion | Pass Threshold | Fail Condition |
|-----------|----------------|----------------|
| Critical Unmitigated Risks | 0 | Any critical risk without feasible mitigation |
| High Severity Conflicts | ≤ 2 | > 2 unresolved high-severity trade-off conflicts |
| Scenario Coverage | ≥ 60% | < 60% of quality attributes with scenarios |
| Confidence Score | ≥ 0.6 | < 0.6 overall confidence |

### Integration with Prior Stories

This story builds on outputs from:

| Story | Output Used | Integration Point |
|-------|-------------|-------------------|
| 7.4 Quality Evaluator | QualityAttributeEvaluation | Trade-offs, attribute scores |
| 7.5 Risk Identifier | TechnicalRiskReport | Risk assessments |
| 7.6 Tech Stack Validator | TechStackValidation | (Context, not direct input) |

### LLM Prompt Template (suggested)

```python
ATAM_REVIEW_PROMPT = """Perform an ATAM-style architectural review of this design.

Design Decisions:
{design_decisions}

Quality Attribute Evaluation:
- Scores: {quality_scores}
- Trade-offs: {trade_offs}
- Risks: {quality_risks}

Technical Risk Report:
- Risks: {technical_risks}
- Overall Risk Level: {overall_risk_level}

Analyze:
1. Architectural scenarios for each quality attribute
2. Trade-off conflicts between quality attributes
3. Risk impact on quality attributes and mitigation feasibility
4. Overall pass/fail recommendation with confidence

Respond in JSON format:
{{
  "overall_pass": true/false,
  "confidence": 0.85,
  "scenarios_evaluated": [
    {{
      "scenario_id": "ATAM-001",
      "quality_attribute": "performance",
      "stimulus": "100 concurrent API requests",
      "response": "95th percentile < 500ms",
      "analysis": "Design supports async processing and caching"
    }}
  ],
  "trade_off_conflicts": [
    {{
      "attribute_a": "performance",
      "attribute_b": "security",
      "description": "Encryption adds latency",
      "severity": "medium",
      "resolution_strategy": "Use async encryption with caching"
    }}
  ],
  "risk_assessments": [
    {{
      "risk_id": "RISK-001",
      "quality_impact": ["reliability", "performance"],
      "mitigation_feasibility": "high",
      "unmitigated": false
    }}
  ],
  "failure_reasons": [],
  "summary": "Design passes ATAM review with 85% confidence"
}}
"""
```

### Project Structure Notes

- **New Module:** `src/yolo_developer/agents/architect/atam_reviewer.py`
- **Type Additions:** Add to `src/yolo_developer/agents/architect/types.py`
- **Test Location:** `tests/unit/agents/architect/test_atam_reviewer.py`

### Module Structure After This Story

```
src/yolo_developer/agents/architect/
├── __init__.py              # Add ATAMReviewResult, run_atam_review exports
├── types.py                 # Add ATAMScenario, ATAMTradeOffConflict, ATAMRiskAssessment, ATAMReviewResult
├── node.py                  # Update to integrate ATAM review after tech stack validation
├── twelve_factor.py         # Existing 12-Factor analysis (Story 7.2)
├── adr_generator.py         # Existing ADR generation (Story 7.3)
├── quality_evaluator.py     # Existing quality evaluation (Story 7.4)
├── risk_identifier.py       # Existing risk identification (Story 7.5)
├── tech_stack_validator.py  # Existing tech stack validation (Story 7.6)
└── atam_reviewer.py         # NEW: ATAM architectural review
```

### Story Dependencies

- **Depends on:** Story 7.1 (architect_node, ArchitectOutput), Story 7.4 (QualityAttributeEvaluation), Story 7.5 (TechnicalRiskReport), Story 7.6 (execution order)
- **Enables:** Story 7.8 (Pattern Matching to Codebase - final architect story)
- **FR Covered:** FR54: Architect Agent can validate designs pass basic ATAM review criteria

### Previous Story Context (7.6)

From Story 7.6 implementation:
- `validate_tech_stack_constraints()` is async and returns TechStackValidation
- LLM integration uses pattern with @retry decorator and JSON parsing
- Pattern-based fallback when LLM fails
- Integration happens in architect_node after risk identification

Follow the same patterns for ATAM review. Execute after tech stack validation in architect_node.

### Git Intelligence (Recent Commits)

Recent commit pattern: `feat: Implement X with code review fixes (Story X.X)`

Files from Story 7.6 to reference:
- `src/yolo_developer/agents/architect/tech_stack_validator.py` - LLM integration pattern with fallback
- `src/yolo_developer/agents/architect/types.py` - Type definition patterns
- `src/yolo_developer/agents/architect/node.py` - Integration point for new modules
- `tests/unit/agents/architect/test_tech_stack_validator.py` - Test patterns to follow

### Key Types From Prior Stories

```python
# From types.py - used as inputs to ATAM review
QualityAttributeEvaluation:
    attribute_scores: dict[str, float]  # 0.0-1.0 scores
    trade_offs: tuple[QualityTradeOff, ...]
    risks: tuple[QualityRisk, ...]
    overall_score: float

TechnicalRiskReport:
    risks: tuple[TechnicalRisk, ...]
    overall_risk_level: RiskSeverity  # "critical", "high", "medium", "low"
    summary: str

DesignDecision:
    id: str
    story_id: str
    decision_type: DesignDecisionType
    description: str
    rationale: str
```

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.7] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR54] - FR54: ATAM review criteria
- [Source: _bmad-output/planning-artifacts/architecture.md#Quality-Attributes] - Quality attribute list
- [Source: src/yolo_developer/agents/architect/quality_evaluator.py] - Quality evaluation patterns
- [Source: src/yolo_developer/agents/architect/risk_identifier.py] - Risk identification patterns
- [Source: src/yolo_developer/agents/architect/types.py] - Type definitions to extend
- [Source: src/yolo_developer/agents/architect/node.py] - Current architect implementation
- [FR54: Architect Agent can validate designs pass basic ATAM review criteria]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- Implemented ATAM (Architecture Tradeoff Analysis Method) review for the Architect agent
- Created 4 frozen dataclasses: ATAMScenario, ATAMTradeOffConflict, ATAMRiskAssessment, ATAMReviewResult
- Added MitigationFeasibility literal type
- Implemented scenario generation from design decisions with quality attribute mapping
- Implemented trade-off conflict detection with severity assignment and resolution strategies
- Implemented risk impact assessment with mitigation feasibility evaluation
- Implemented pass/fail decision logic with configurable thresholds
- Added LLM-powered ATAM analysis with tenacity retry and rule-based fallback
- Integrated ATAM review into architect_node after tech stack validation
- Added atam_reviews field to ArchitectOutput
- 393 architect tests pass (including 29 new ATAM tests + 6 integration tests)
- Full type annotations, structlog logging, async/await patterns

### File List

- src/yolo_developer/agents/architect/types.py - Added ATAMScenario, ATAMTradeOffConflict, ATAMRiskAssessment, ATAMReviewResult, MitigationFeasibility; added atam_reviews to ArchitectOutput
- src/yolo_developer/agents/architect/atam_reviewer.py - NEW: Main ATAM review module (~850 lines)
- src/yolo_developer/agents/architect/node.py - Integrated run_atam_review after tech stack validation
- src/yolo_developer/agents/architect/__init__.py - Added exports for ATAM types and run_atam_review
- tests/unit/agents/architect/test_atam_types.py - NEW: Unit tests for ATAM types (20 tests)
- tests/unit/agents/architect/test_atam_reviewer.py - NEW: Unit and integration tests (31 tests)
- _bmad-output/implementation-artifacts/sprint-status.yaml - Updated story status to in-progress
