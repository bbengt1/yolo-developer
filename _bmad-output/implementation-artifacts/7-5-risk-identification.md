# Story 7.5: Risk Identification

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want technical risks identified proactively,
so that I can plan mitigations early.

## Acceptance Criteria

1. **Given** a proposed technical design with design decisions
   **When** risk analysis runs via `identify_technical_risks()`
   **Then** technology risks are identified (e.g., immature libraries, version conflicts, deprecated APIs)
   **And** results are returned as a `TechnicalRiskReport` frozen dataclass
   **And** the function is importable from `yolo_developer.agents.architect`

2. **Given** a design that involves external services or APIs
   **When** risk analysis runs
   **Then** integration risks are flagged (e.g., API instability, rate limiting, authentication complexity)
   **And** each risk includes severity level and affected integration points

3. **Given** a design with scalability implications
   **When** risk analysis runs
   **Then** scalability concerns are noted (e.g., single points of failure, stateful components, database bottlenecks)
   **And** each concern links to specific design decisions

4. **Given** identified technical risks
   **When** analysis completes
   **Then** mitigation strategies are suggested for each risk
   **And** mitigations are actionable and design-specific (not generic boilerplate)
   **And** mitigation effort is estimated (high/medium/low)
   **And** mitigation priority is determined based on severity and effort

5. **Given** the architect_node processing stories
   **When** technical risk identification is performed
   **Then** it runs after quality attribute evaluation (Story 7.4)
   **And** it can consume quality evaluation risks as additional input
   **And** risk reports are included in ArchitectOutput
   **And** summary is logged via structlog

6. **Given** LLM-powered risk identification
   **When** analyzing complex technical risks
   **Then** it uses litellm with configurable model via env var
   **And** it includes tenacity retry with exponential backoff
   **And** it handles LLM failures gracefully with pattern-based fallback

7. **Given** the TechnicalRiskReport dataclass
   **When** all analysis is complete
   **Then** it is frozen (immutable) per ADR-001
   **And** it has to_dict() method for serialization
   **And** it includes overall_risk_level (critical/high/medium/low) based on highest severity risk

## Tasks / Subtasks

- [x] Task 1: Create Technical Risk Type Definitions (AC: 1, 7)
  - [x] Create `TechnicalRiskCategory` Literal type (technology, integration, scalability, compatibility, operational)
  - [x] Create `TechnicalRisk` frozen dataclass with: category, description, severity, affected_components, mitigation, mitigation_effort, mitigation_priority
  - [x] Create `TechnicalRiskReport` frozen dataclass with: risks, overall_risk_level, summary, to_dict()
  - [x] Add type exports to `architect/__init__.py`

- [x] Task 2: Implement Technology Risk Detection (AC: 1)
  - [x] Create `src/yolo_developer/agents/architect/risk_identifier.py` module
  - [x] Implement `_identify_technology_risks(story, design_decisions) -> list[TechnicalRisk]`
  - [x] Detect immature/experimental library usage patterns
  - [x] Detect deprecated technology patterns
  - [x] Detect version compatibility concerns
  - [x] Add structlog logging for risk detection

- [x] Task 3: Implement Integration Risk Detection (AC: 2)
  - [x] Implement `_identify_integration_risks(story, design_decisions) -> list[TechnicalRisk]`
  - [x] Detect external API dependency patterns
  - [x] Detect authentication complexity patterns
  - [x] Detect rate limiting concerns
  - [x] Flag vendor lock-in risks
  - [x] Identify affected integration points for each risk

- [x] Task 4: Implement Scalability Risk Detection (AC: 3)
  - [x] Implement `_identify_scalability_risks(story, design_decisions) -> list[TechnicalRisk]`
  - [x] Detect single point of failure patterns
  - [x] Detect stateful component patterns
  - [x] Detect database bottleneck patterns
  - [x] Link scalability concerns to specific design decisions

- [x] Task 5: Implement Mitigation Suggestion Engine (AC: 4)
  - [x] Create `_generate_mitigations(risks, story, decisions) -> list[TechnicalRisk]`
  - [x] Generate design-specific mitigation strategies (not generic boilerplate)
  - [x] Estimate mitigation effort based on risk category and severity
  - [x] Calculate mitigation priority from severity and effort
  - [x] Ensure mitigations reference specific components from the design

- [x] Task 6: Implement LLM-Powered Risk Analysis (AC: 6)
  - [x] Create `_analyze_risks_with_llm(story, design_decisions) -> TechnicalRiskReport | None`
  - [x] Design prompt template for technical risk identification
  - [x] Add tenacity @retry decorator with exponential backoff (3 attempts)
  - [x] Use configurable model via YOLO_LLM__ROUTINE_MODEL env var
  - [x] Implement graceful fallback to pattern-based analysis on LLM failure
  - [x] Parse LLM JSON response to TechnicalRisk objects

- [x] Task 7: Create Main Risk Identification Function (AC: 1, 5, 7)
  - [x] Create `identify_technical_risks(story, design_decisions, quality_risks?) -> TechnicalRiskReport` async function
  - [x] Orchestrate technology, integration, and scalability risk detection
  - [x] Incorporate quality_risks from Story 7.4 as additional input (optional parameter)
  - [x] Calculate overall_risk_level from highest severity risk
  - [x] Generate summary text describing key risks
  - [x] Add structlog logging for analysis start/complete

- [x] Task 8: Integrate with architect_node (AC: 5)
  - [x] Update `architect_node` to call `identify_technical_risks` after quality evaluation
  - [x] Pass quality evaluation risks to risk identifier
  - [x] Add `technical_risk_reports` field to ArchitectOutput dataclass
  - [x] Include risk summary in processing_notes
  - [x] Update ArchitectOutput.to_dict() to include risk reports

- [x] Task 9: Write Unit Tests for Types (AC: 7)
  - [x] Test TechnicalRisk dataclass creation and to_dict()
  - [x] Test TechnicalRiskReport dataclass creation and to_dict()
  - [x] Test overall_risk_level calculation from risk list
  - [x] Test immutability of frozen dataclasses

- [x] Task 10: Write Unit Tests for Risk Detection (AC: 1, 2, 3)
  - [x] Test technology risk detection with various patterns
  - [x] Test integration risk detection with external API patterns
  - [x] Test scalability risk detection with stateful patterns
  - [x] Test severity categorization for each risk type

- [x] Task 11: Write Unit Tests for Mitigations (AC: 4)
  - [x] Test mitigation generation is design-specific
  - [x] Test mitigation effort estimation
  - [x] Test mitigation priority calculation
  - [x] Test mitigations reference specific design components

- [x] Task 12: Write Unit Tests for LLM Integration (AC: 6)
  - [x] Test LLM analysis with mocked LLM
  - [x] Test retry behavior on transient failures
  - [x] Test fallback to pattern-based on LLM failure
  - [x] Test JSON parsing of LLM response

- [x] Task 13: Write Integration Tests (AC: 5)
  - [x] Test architect_node includes technical_risk_reports
  - [x] Test end-to-end flow with mock stories and design decisions
  - [x] Test integration with quality evaluation risks
  - [x] Test ArchitectOutput serialization with risk reports

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for TechnicalRisk, TechnicalRiskReport (internal state)
- **ADR-003 (LLM Abstraction):** Use litellm for LLM calls with configurable model
- **ADR-005 (LangGraph Communication):** Return state update dict, don't mutate state directly
- **ADR-007 (Error Handling):** Use tenacity with exponential backoff for LLM calls
- **ARCH-QUALITY-5:** All I/O operations (LLM calls) must be async/await
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Risk Categories

| Category | Description | Example Patterns |
|----------|-------------|------------------|
| Technology | Library/framework concerns | deprecated API, experimental feature, version conflict |
| Integration | External service concerns | rate limiting, API instability, auth complexity, vendor lock-in |
| Scalability | Growth/load concerns | single point of failure, stateful session, database bottleneck |
| Compatibility | Cross-system concerns | version mismatch, protocol incompatibility |
| Operational | Runtime concerns | monitoring gaps, deployment complexity, configuration drift |

### Risk Severity Mapping (consistent with Story 7.4)

Based on impact and likelihood:
- **Critical:** High impact + high likelihood, blocks functionality
- **High:** Significant impact, likely to occur
- **Medium:** Moderate impact or lower likelihood
- **Low:** Minor impact or unlikely to occur

### Mitigation Priority Calculation

Priority = f(severity, effort):
| Severity | High Effort | Medium Effort | Low Effort |
|----------|-------------|---------------|------------|
| Critical | P1 (urgent) | P1 (urgent) | P1 (urgent) |
| High | P2 (high) | P1 (urgent) | P1 (urgent) |
| Medium | P3 (medium) | P2 (high) | P2 (high) |
| Low | P4 (low) | P3 (medium) | P3 (medium) |

### Pattern Detection Keywords

**Technology Risks:**
- Deprecated: "deprecated", "legacy", "end-of-life", "sunset"
- Experimental: "beta", "alpha", "experimental", "unstable"
- Compatibility: "breaking change", "migration required", "incompatible"

**Integration Risks:**
- API stability: "external api", "third-party", "vendor"
- Rate limiting: "rate limit", "throttle", "quota"
- Authentication: "oauth", "api key", "credential management"
- Lock-in: "proprietary", "vendor-specific", "platform-dependent"

**Scalability Risks:**
- Single point: "single instance", "monolithic", "centralized"
- Stateful: "session state", "in-memory", "sticky session"
- Database: "single database", "no replication", "shared database"

### LLM Prompt Template (suggested)

```python
RISK_IDENTIFICATION_PROMPT = """Identify technical risks in the following design.

Story:
Title: {story_title}
Description: {story_description}

Design Decisions:
{design_decisions}

Identify risks in these categories:
1. Technology: Library maturity, deprecation, version conflicts
2. Integration: External API stability, rate limits, auth complexity, vendor lock-in
3. Scalability: Single points of failure, stateful components, bottlenecks
4. Compatibility: Version mismatches, protocol issues
5. Operational: Monitoring gaps, deployment complexity

For each risk, provide:
- Category: One of the 5 categories above
- Description: What is the risk?
- Severity: critical, high, medium, or low
- Affected components: Which parts of the design are affected?
- Mitigation: How to address this risk (be specific to this design)
- Mitigation effort: high, medium, or low

Respond in JSON format:
{{
  "risks": [
    {{
      "category": "integration",
      "description": "External API has no SLA guarantee",
      "severity": "high",
      "affected_components": ["AuthService", "UserAPI"],
      "mitigation": "Implement circuit breaker and local cache fallback",
      "mitigation_effort": "medium"
    }}
  ],
  "overall_risk_level": "high",
  "summary": "Brief summary of key risks"
}}
"""
```

### Project Structure Notes

- **New Module:** `src/yolo_developer/agents/architect/risk_identifier.py`
- **Type Additions:** Add to `src/yolo_developer/agents/architect/types.py`
- **Test Location:** `tests/unit/agents/architect/test_risk_identifier.py`

### Module Structure After This Story

```
src/yolo_developer/agents/architect/
├── __init__.py              # Add TechnicalRisk, TechnicalRiskReport, identify_technical_risks exports
├── types.py                 # Add TechnicalRiskCategory, TechnicalRisk, TechnicalRiskReport, update ArchitectOutput
├── node.py                  # Update to integrate risk identification after quality evaluation
├── twelve_factor.py         # Existing 12-Factor analysis (Story 7.2)
├── adr_generator.py         # Existing ADR generation (Story 7.3)
├── quality_evaluator.py     # Existing quality evaluation (Story 7.4)
└── risk_identifier.py       # NEW: Technical risk identification logic
```

### Relationship to Story 7.4 QualityRisk

Story 7.4 introduced `QualityRisk` for NFR-related risks. Story 7.5 `TechnicalRisk` focuses on:
- **Technology concerns** that aren't NFRs (library maturity, deprecation)
- **Integration complexities** beyond quality attributes (API stability, vendor lock-in)
- **Scalability architecture** issues (not just NFR scoring)

The `identify_technical_risks` function can optionally consume `QualityRisk` items from Story 7.4 to avoid duplicate identification and provide comprehensive coverage.

### Story Dependencies

- **Depends on:** Story 7.1 (architect_node, ArchitectOutput), Story 7.2 (LLM integration pattern), Story 7.3 (async pattern), Story 7.4 (QualityRisk, quality evaluation flow)
- **Enables:** Story 7.7 (ATAM Review - uses risk identification as input)
- **FR Covered:** FR52: Architect Agent can identify technical risks and mitigation strategies

### Previous Story Context (7.4)

From Story 7.4 implementation:
- `evaluate_quality_attributes()` is async and returns QualityAttributeEvaluation
- `QualityRisk` dataclass exists with: attribute, description, severity, mitigation, mitigation_effort
- Risk severity levels: critical, high, medium, low (reuse `RiskSeverity` type)
- Mitigation effort levels: high, medium, low (reuse `MitigationEffort` type)
- LLM integration uses pattern with @retry decorator and JSON parsing
- Pattern-based fallback when LLM fails

Follow the same patterns for technical risk identification. Reuse `RiskSeverity` and `MitigationEffort` types from Story 7.4.

### Git Intelligence (Recent Commits)

Recent commit pattern: `feat: Implement X with code review fixes (Story X.X)`

Files from Story 7.4 to reference:
- `src/yolo_developer/agents/architect/quality_evaluator.py` - LLM integration pattern with fallback
- `src/yolo_developer/agents/architect/types.py` - Type definition patterns (RiskSeverity, MitigationEffort)
- `src/yolo_developer/agents/architect/node.py` - Integration point for new modules
- `tests/unit/agents/architect/test_quality_evaluator.py` - Test patterns to follow

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.5] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR52] - FR52: Risk identification requirement
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007] - Error handling with tenacity
- [Source: src/yolo_developer/agents/architect/types.py] - Existing type definitions (RiskSeverity, MitigationEffort)
- [Source: src/yolo_developer/agents/architect/quality_evaluator.py] - LLM integration pattern
- [Source: src/yolo_developer/agents/architect/node.py] - Current architect implementation
- [FR52: Architect Agent can identify technical risks and mitigation strategies]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- All 13 tasks completed using Red-Green-Refactor TDD cycle
- Implemented pattern-based risk detection for 5 categories: technology, integration, scalability, compatibility, operational
- Each risk category has keyword-severity mappings for accurate risk detection
- Mitigation strategies are design-specific with effort and priority calculations
- LLM-powered analysis uses litellm with tenacity retry (3 attempts, exponential backoff)
- Graceful fallback to pattern-based analysis when LLM fails
- Integrated with architect_node after quality evaluation, consuming QualityRisk from Story 7.4
- All 324 architect tests passing
- No mypy or ruff errors

### File List

**Created:**
- `src/yolo_developer/agents/architect/risk_identifier.py` - Main risk identification module (~900 lines)
- `tests/unit/agents/architect/test_risk_types.py` - Unit tests for type definitions (26 tests)
- `tests/unit/agents/architect/test_risk_identifier.py` - Unit tests for risk detection (31 tests)
- `tests/integration/agents/architect/test_risk_integration.py` - Integration tests (18 tests)

**Modified:**
- `src/yolo_developer/agents/architect/types.py` - Added TechnicalRiskCategory, MitigationPriority, TechnicalRisk, TechnicalRiskReport, calculate_mitigation_priority(), calculate_overall_risk_level(), updated ArchitectOutput
- `src/yolo_developer/agents/architect/node.py` - Integrated risk identification after quality evaluation
- `src/yolo_developer/agents/architect/__init__.py` - Added new exports for risk types and functions

## Senior Developer Review (AI)

**Reviewer:** Claude Opus 4.5 (Adversarial Code Review)
**Date:** 2026-01-10
**Outcome:** APPROVED with fixes applied

### Issues Found and Fixed

| # | Severity | Issue | Fix Applied |
|---|----------|-------|-------------|
| 1 | MEDIUM | Fragile JSON extraction from LLM response (split-based) | Used regex for robust extraction |
| 2 | MEDIUM | Type ignore suppression in test helper | Added proper type validation |
| 3 | MEDIUM | Conditional assertions in tests (no failure if condition false) | Added explicit assertions |
| 4 | MEDIUM | Import inside function body (node.py) | Moved to module top level |
| 5 | LOW | Heuristic component extraction with weak comment | Documented limitation in comments |
| 6 | LOW | Type annotation mismatch on effort variable | Fixed with proper validation flow |
| 7 | LOW | Missing docstrings on detection functions | Added comprehensive docstrings |

### Verification

- All 71 risk identification tests pass
- mypy: no issues found
- ruff: all checks passed
- All acceptance criteria verified as implemented

### Files Modified During Review

- `src/yolo_developer/agents/architect/risk_identifier.py` - JSON regex, docstrings, type annotations
- `src/yolo_developer/agents/architect/node.py` - Import moved to top level
- `tests/integration/agents/architect/test_risk_integration.py` - Fixed type validation and assertions
