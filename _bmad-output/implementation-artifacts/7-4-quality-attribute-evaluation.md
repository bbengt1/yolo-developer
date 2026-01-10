# Story 7.4: Quality Attribute Evaluation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want designs evaluated against quality requirements,
so that NFRs are properly addressed.

## Acceptance Criteria

1. **Given** NFRs from the PRD (performance, security, reliability, scalability, etc.)
   **When** a design is evaluated for quality attributes
   **Then** each quality attribute category is assessed against the story design
   **And** results are returned in a QualityAttributeEvaluation dataclass
   **And** the function is importable from `yolo_developer.agents.architect`

2. **Given** a story with design decisions
   **When** quality attribute evaluation runs
   **Then** trade-offs between quality attributes are documented
   **And** conflicts (e.g., performance vs security) are identified
   **And** the trade-off analysis is included in the evaluation result

3. **Given** NFRs defined in project configuration or PRD
   **When** a design is evaluated
   **Then** risks to meeting each NFR are identified
   **And** each risk is categorized by severity (critical, high, medium, low)
   **And** risks are linked to specific quality attributes

4. **Given** identified NFR risks
   **When** evaluation completes
   **Then** mitigation strategies are suggested for each risk
   **And** mitigations are actionable and specific to the design
   **And** mitigation effort is estimated (high/medium/low)

5. **Given** the architect_node processing stories
   **When** quality attribute evaluation is performed
   **Then** it is called after design decisions are generated
   **And** evaluation results are included in ArchitectOutput
   **And** summary is logged via structlog

6. **Given** LLM-powered quality evaluation
   **When** analyzing complex quality trade-offs
   **Then** it uses litellm with configurable model via env var
   **And** it includes tenacity retry with exponential backoff
   **And** it handles LLM failures gracefully with fallback

7. **Given** the QualityAttributeEvaluation dataclass
   **When** all evaluations are complete
   **Then** they are frozen (immutable) per ADR-001
   **And** they have to_dict() method for serialization
   **And** they include overall_score (0.0-1.0) based on weighted attributes

## Tasks / Subtasks

- [x] Task 1: Create Quality Attribute Type Definitions (AC: 1, 7)
  - [x] Create `QualityAttribute` enum or Literal type (performance, security, reliability, scalability, maintainability, etc.)
  - [x] Create `QualityRisk` frozen dataclass with: attribute, description, severity, mitigation, mitigation_effort
  - [x] Create `QualityTradeOff` frozen dataclass with: attribute_a, attribute_b, description, resolution
  - [x] Create `QualityAttributeEvaluation` frozen dataclass with: attribute_scores, trade_offs, risks, overall_score, to_dict()
  - [x] Add type exports to `architect/__init__.py`

- [x] Task 2: Implement Quality Attribute Scoring (AC: 1, 5)
  - [x] Create `src/yolo_developer/agents/architect/quality_evaluator.py` module
  - [x] Implement `_score_performance(story, design_decisions) -> float` function
  - [x] Implement `_score_security(story, design_decisions) -> float` function
  - [x] Implement `_score_reliability(story, design_decisions) -> float` function
  - [x] Implement `_score_scalability(story, design_decisions) -> float` function
  - [x] Implement `_score_maintainability(story, design_decisions) -> float` function
  - [x] Calculate weighted overall_score from individual scores

- [x] Task 3: Implement Trade-Off Detection (AC: 2)
  - [x] Create `_detect_trade_offs(story, design_decisions) -> list[QualityTradeOff]`
  - [x] Detect performance vs security trade-offs (e.g., encryption overhead)
  - [x] Detect performance vs reliability trade-offs (e.g., caching vs consistency)
  - [x] Detect scalability vs maintainability trade-offs (e.g., distributed complexity)
  - [x] Document resolution approach for each trade-off

- [x] Task 4: Implement Risk Identification (AC: 3)
  - [x] Create `_identify_risks(story, design_decisions, attribute_scores) -> list[QualityRisk]`
  - [x] Map low scores to corresponding risks
  - [x] Categorize risks by severity based on score thresholds
  - [x] Link risks to specific NFRs from PRD

- [x] Task 5: Implement Mitigation Suggestions (AC: 4)
  - [x] Create `_suggest_mitigations(risks) -> list[QualityRisk]` (updates risks with mitigations)
  - [x] Generate actionable mitigation strategies per risk
  - [x] Estimate mitigation effort (high/medium/low)
  - [x] Ensure mitigations are design-specific, not generic

- [x] Task 6: Implement LLM-Powered Evaluation (AC: 6)
  - [x] Create `_evaluate_quality_with_llm(story, design_decisions) -> QualityAttributeEvaluation`
  - [x] Design prompt template for quality attribute evaluation
  - [x] Add tenacity @retry decorator with exponential backoff
  - [x] Use configurable model via YOLO_LLM__ROUTINE_MODEL env var
  - [x] Implement graceful fallback to pattern-based evaluation on LLM failure

- [x] Task 7: Create Main Evaluation Function (AC: 1, 5)
  - [x] Create `evaluate_quality_attributes(story, design_decisions) -> QualityAttributeEvaluation` async function
  - [x] Orchestrate scoring, trade-off detection, and risk identification
  - [x] Add structlog logging for evaluation start/complete
  - [x] Include overall_score calculation

- [x] Task 8: Integrate with architect_node (AC: 5)
  - [x] Update `architect_node` to call quality attribute evaluation
  - [x] Add `quality_evaluations` field to ArchitectOutput dataclass
  - [x] Include quality evaluation summary in processing_notes
  - [x] Update ArchitectOutput.to_dict() to include evaluations

- [x] Task 9: Write Unit Tests for Types (AC: 7)
  - [x] Test QualityRisk dataclass creation and to_dict()
  - [x] Test QualityTradeOff dataclass creation and to_dict()
  - [x] Test QualityAttributeEvaluation dataclass creation and to_dict()
  - [x] Test overall_score calculation from attribute scores
  - [x] Test immutability of frozen dataclasses

- [x] Task 10: Write Unit Tests for Scoring (AC: 1)
  - [x] Test each _score_* function with various story patterns
  - [x] Test weighted overall_score calculation
  - [x] Test score ranges are 0.0-1.0

- [x] Task 11: Write Unit Tests for Trade-Offs and Risks (AC: 2, 3, 4)
  - [x] Test trade-off detection for known conflict patterns
  - [x] Test risk identification from low scores
  - [x] Test risk severity categorization
  - [x] Test mitigation suggestion generation

- [x] Task 12: Write Unit Tests for LLM Integration (AC: 6)
  - [x] Test LLM evaluation with mocked LLM
  - [x] Test retry behavior on transient failures
  - [x] Test fallback to pattern-based on LLM failure

- [x] Task 13: Write Integration Tests (AC: 5)
  - [x] Test architect_node includes quality evaluations
  - [x] Test end-to-end flow with mock stories and design decisions
  - [x] Test ArchitectOutput serialization with evaluations

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for QualityAttributeEvaluation, QualityRisk, QualityTradeOff (internal state)
- **ADR-003 (LLM Abstraction):** Use litellm for LLM calls with configurable model
- **ADR-005 (LangGraph Communication):** Return state update dict, don't mutate state directly
- **ADR-007 (Error Handling):** Use tenacity with exponential backoff for LLM calls
- **ARCH-QUALITY-5:** All I/O operations (LLM calls) must be async/await
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Quality Attributes Reference (from PRD NFRs)

| Attribute | Key Metrics | PRD Reference |
|-----------|-------------|---------------|
| Performance | <5s handoff, <10s gate eval, <4hr sprint | NFR-PERF-1 to NFR-PERF-6 |
| Security | API key management, project isolation, SAST | NFR-SEC-1 to NFR-SEC-6 |
| Reliability | >95% completion, 3-retry, checkpoints | NFR-REL-1 to NFR-REL-6 |
| Scalability | 5-10 stories MVP, 100MB memory growth | NFR-SCALE-1 to NFR-SCALE-5 |
| Maintainability | YAML config, structured logging, >80% coverage | NFR-MAINT-1 to NFR-MAINT-6 |
| Integration | Multi-LLM, MCP 1.0+, OpenTelemetry | NFR-INT-1 to NFR-INT-6 |
| Cost Efficiency | 70% cheap models, >50% cache hit | NFR-COST-1 to NFR-COST-5 |

### Scoring Thresholds

Based on score ranges, map to risk severity:
- 0.0 - 0.3: Critical risk
- 0.3 - 0.5: High risk
- 0.5 - 0.7: Medium risk
- 0.7 - 1.0: Low risk or acceptable

### Common Trade-Off Patterns

| Conflict | Description | Common Resolution |
|----------|-------------|-------------------|
| Performance vs Security | Encryption/auth adds latency | Cache auth tokens, async encryption |
| Performance vs Reliability | Caching may serve stale data | TTL tuning, cache invalidation |
| Scalability vs Maintainability | Distributed systems are complex | Start simple, document scaling path |
| Security vs Usability | MFA adds friction | Risk-based auth, remember trusted devices |

### LLM Prompt Template (suggested)

```python
QUALITY_EVALUATION_PROMPT = """Evaluate the following design decisions for quality attributes.

Story:
{story_content}

Design Decisions:
{design_decisions}

Evaluate against these quality attributes:
1. Performance: Response time, throughput, resource efficiency
2. Security: Authentication, authorization, data protection
3. Reliability: Fault tolerance, recovery, consistency
4. Scalability: Horizontal scaling, load handling
5. Maintainability: Code clarity, testability, documentation

For each attribute, provide:
- Score (0.0-1.0): How well does the design address this attribute?
- Risks: What could prevent meeting NFRs?
- Mitigations: How to address the risks?

Also identify any trade-offs between attributes.

Respond in JSON format:
{
  "attribute_scores": {
    "performance": 0.8,
    "security": 0.7,
    ...
  },
  "trade_offs": [
    {
      "attribute_a": "performance",
      "attribute_b": "security",
      "description": "...",
      "resolution": "..."
    }
  ],
  "risks": [
    {
      "attribute": "reliability",
      "description": "...",
      "severity": "medium",
      "mitigation": "...",
      "mitigation_effort": "low"
    }
  ]
}
"""
```

### Project Structure Notes

- **New Module:** `src/yolo_developer/agents/architect/quality_evaluator.py`
- **Type Additions:** Add to `src/yolo_developer/agents/architect/types.py`
- **Test Location:** `tests/unit/agents/architect/test_quality_evaluator.py`

### Module Structure After This Story

```
src/yolo_developer/agents/architect/
├── __init__.py              # Add QualityAttributeEvaluation, evaluate_quality_attributes exports
├── types.py                 # Add QualityRisk, QualityTradeOff, QualityAttributeEvaluation
├── node.py                  # Update to integrate quality evaluation
├── twelve_factor.py         # Existing 12-Factor analysis (Story 7.2)
├── adr_generator.py         # Existing ADR generation (Story 7.3)
└── quality_evaluator.py     # NEW: Quality attribute evaluation logic
```

### Story Dependencies

- **Depends on:** Story 7.1 (architect_node, ArchitectOutput), Story 7.2 (LLM integration pattern), Story 7.3 (async pattern)
- **Enables:** Story 7.5 (Risk Identification), Story 7.7 (ATAM Review)
- **FR Covered:** FR51: Architect Agent can evaluate designs against quality attribute requirements

### Previous Story Context (7.3)

From Story 7.3 implementation:
- `generate_adrs()` is async and returns list of ADR dataclasses
- LLM integration uses `_call_adr_llm()` with @retry decorator
- Pattern-based fallback when LLM fails
- Configurable model via `YOLO_LLM__ROUTINE_MODEL` env var

Follow the same LLM integration pattern for quality evaluation.

### Git Intelligence (Recent Commits)

Recent commit pattern: `feat: Implement X with code review fixes (Story X.X)`

Files from Story 7.3:
- `src/yolo_developer/agents/architect/adr_generator.py` - LLM integration pattern with fallback
- `src/yolo_developer/agents/architect/node.py` - Integration point for new modules
- `tests/unit/agents/architect/test_adr_content.py` - Test patterns to follow

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.4] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional-Requirements] - NFR specifications
- [Source: _bmad-output/planning-artifacts/architecture.md#Cross-Cutting-Concerns-Identified] - Quality concerns
- [Source: src/yolo_developer/agents/architect/types.py] - Existing type definitions
- [Source: src/yolo_developer/agents/architect/node.py] - Current architect implementation
- [Source: src/yolo_developer/agents/architect/adr_generator.py] - LLM integration pattern
- [FR51: Architect Agent can evaluate designs against quality attribute requirements]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- All 7 quality attributes implemented: performance, security, reliability, scalability, maintainability, integration, cost_efficiency
- Weighted scoring with configurable ATTRIBUTE_WEIGHTS dictionary
- Pattern-based evaluation analyzes story text and design decisions for keywords
- Trade-off detection for 5 common conflict patterns (performance-security, performance-reliability, scalability-maintainability, security-usability, reliability-performance)
- Risk identification with severity mapping: critical (<0.3), high (0.3-0.5), medium (0.5-0.7), low (>0.7)
- LLM integration with tenacity @retry decorator (3 attempts, exponential backoff)
- Graceful fallback to pattern-based evaluation when LLM fails or returns invalid JSON
- All types are frozen dataclasses per ADR-001
- 253 architect tests passing (49 new tests for quality evaluation after code review)
- Integration with architect_node adds quality_evaluations to ArchitectOutput

### Code Review Fixes Applied

**HIGH Issues Fixed:**
1. Made mitigations design-specific: Added `_generate_design_specific_mitigation()` function that generates context-aware mitigation strategies based on story title, description, and decision types instead of generic boilerplate text.
2. Fixed weak test assertion: Updated `test_detects_performance_security_tradeoff` to properly assert trade-off detection with meaningful assertions.

**MEDIUM Issues Fixed:**
3. Added `DEFAULT_BASELINE_SCORE = 0.7` constant to replace magic numbers across all 7 scoring functions.
4. Added missing test classes `TestScoreIntegration` and `TestScoreCostEfficiency` with 4 new tests.
5. Fixed type annotation in test helper: Changed `decision_type: str` to `decision_type: DesignDecisionType` and imported the Literal type.
6. Updated LLM prompt to list all 7 quality attributes (was only listing 5 but example showed 7).

### File List

**Created:**
- `src/yolo_developer/agents/architect/quality_evaluator.py` - Main quality evaluation module (~450 lines)
- `tests/unit/agents/architect/test_quality_types.py` - 15 unit tests for quality types
- `tests/unit/agents/architect/test_quality_evaluator.py` - 30 unit tests for evaluator functions
- `tests/integration/agents/architect/test_quality_integration.py` - 12 integration tests

**Modified:**
- `src/yolo_developer/agents/architect/types.py` - Added QUALITY_ATTRIBUTES, RiskSeverity, MitigationEffort, QualityRisk, QualityTradeOff, QualityAttributeEvaluation, updated ArchitectOutput
- `src/yolo_developer/agents/architect/__init__.py` - Added exports for new types and evaluate_quality_attributes function
- `src/yolo_developer/agents/architect/node.py` - Integrated quality evaluation into architect_node, added quality_evaluations to output
