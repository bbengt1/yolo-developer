# Story 7.2: 12-Factor Design Generation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want designs following 12-Factor principles,
So that applications are scalable and maintainable.

## Acceptance Criteria

1. **AC1: Twelve-Factor Analyzer Function**
   - **Given** a story requiring design decisions
   - **When** `analyze_twelve_factor(story: dict) -> TwelveFactorAnalysis` is called
   - **Then** it analyzes the story against each of the 12 factors
   - **And** it returns a TwelveFactorAnalysis dataclass with compliance status per factor
   - **And** it identifies applicable factors based on story content
   - **And** the function is importable from `yolo_developer.agents.architect`

2. **AC2: Configuration Externalization Check**
   - **Given** a story describing application behavior
   - **When** the analyzer processes it
   - **Then** it checks if configuration should be externalized
   - **And** it flags stories that hardcode environment-specific values
   - **And** it recommends environment variable patterns per Factor I (Codebase) and Factor III (Config)

3. **AC3: Stateless Process Detection**
   - **Given** a story that involves data handling
   - **When** the analyzer processes it
   - **Then** it detects patterns that require persistent state
   - **And** it recommends stateless alternatives per Factor VI (Processes)
   - **And** it identifies backing services for state per Factor IV (Backing Services)

4. **AC4: Backing Services as Attached Resources**
   - **Given** a story involving external services (databases, caches, queues)
   - **When** the analyzer processes it
   - **Then** it validates services are treated as attached resources
   - **And** it recommends connection string externalization per Factor IV
   - **And** it flags any direct service dependencies that should be abstracted

5. **AC5: LLM-Powered Analysis Integration**
   - **Given** the twelve-factor analyzer
   - **When** analyzing complex stories
   - **Then** it uses LLM to interpret story intent and map to 12-Factor principles
   - **And** it uses tenacity retry with exponential backoff for LLM calls
   - **And** it routes to appropriate model tier based on complexity

6. **AC6: Design Decision Generation Enhancement**
   - **Given** the existing `_generate_design_decisions` function from Story 7.1
   - **When** architect_node processes stories
   - **Then** it calls the twelve-factor analyzer for each story
   - **And** twelve-factor compliance is included in DesignDecision rationale
   - **And** violations generate specific remediation recommendations

7. **AC7: TwelveFactorAnalysis Type**
   - **Given** the architect types module
   - **When** twelve-factor analysis is needed
   - **Then** TwelveFactorAnalysis frozen dataclass exists with:
     - factor_results: dict[str, FactorResult] (one per 12 factors)
     - applicable_factors: tuple[str, ...] (factors relevant to this story)
     - overall_compliance: float (0.0-1.0)
     - recommendations: tuple[str, ...] (specific guidance)
     - to_dict() method for serialization
   - **And** FactorResult dataclass exists with: factor_name, applies, compliant, finding, recommendation

## Tasks / Subtasks

- [x] Task 1: Create Twelve-Factor Type Definitions (AC: 7)
  - [x] Create `FactorResult` frozen dataclass with: factor_name, applies, compliant, finding, recommendation
  - [x] Create `TwelveFactorAnalysis` frozen dataclass with: factor_results, applicable_factors, overall_compliance, recommendations, to_dict()
  - [x] Define TWELVE_FACTORS constant list with all factor names
  - [x] Add type exports to `architect/__init__.py`

- [x] Task 2: Implement Factor Analysis Framework (AC: 1)
  - [x] Create `src/yolo_developer/agents/architect/twelve_factor.py` module
  - [x] Implement `analyze_twelve_factor(story: dict) -> TwelveFactorAnalysis` async function
  - [x] Add structlog logging for analysis start/complete
  - [x] Create helper `_analyze_factor(story: dict, factor_name: str) -> FactorResult`

- [x] Task 3: Implement Configuration Factor Checks (AC: 2)
  - [x] Implement Factor I (Codebase) analysis - single codebase patterns
  - [x] Implement Factor III (Config) analysis - config externalization
  - [x] Create keyword detection for hardcoded config patterns
  - [x] Generate recommendations for environment variable usage

- [x] Task 4: Implement Stateless Process Checks (AC: 3)
  - [x] Implement Factor VI (Processes) analysis - stateless execution
  - [x] Detect session state, in-memory caching, file storage patterns
  - [x] Recommend backing services for persistent state
  - [x] Flag sticky session patterns

- [x] Task 5: Implement Backing Services Checks (AC: 4)
  - [x] Implement Factor IV (Backing Services) analysis
  - [x] Detect database, cache, message queue references
  - [x] Validate attached resource patterns
  - [x] Recommend connection string externalization

- [x] Task 6: Implement Additional Factor Analyzers (AC: 1)
  - [x] Factor II (Dependencies) - explicit dependency declaration
  - [x] Factor V (Build, Release, Run) - separation of stages
  - [x] Factor VII (Port Binding) - self-contained services
  - [x] Factor VIII (Concurrency) - scale via process model
  - [x] Factor IX (Disposability) - fast startup and graceful shutdown
  - [x] Factor X (Dev/Prod Parity) - keep environments similar
  - [x] Factor XI (Logs) - treat logs as event streams
  - [x] Factor XII (Admin Processes) - run admin tasks as one-off processes

- [x] Task 7: Integrate LLM Analysis (AC: 5)
  - [x] Create `_analyze_with_llm(story: dict, factors: list[str]) -> dict[str, FactorResult]`
  - [x] Design prompt template for 12-Factor analysis
  - [x] Add tenacity @retry decorator with exponential backoff
  - [x] Route to "routine" tier model for analysis
  - [x] Parse LLM response into FactorResult dataclasses

- [x] Task 8: Integrate with architect_node (AC: 6)
  - [x] Update `_generate_design_decisions` to call twelve-factor analyzer
  - [x] Include TwelveFactorAnalysis in DesignDecision rationale
  - [x] Add twelve_factor_analysis field to ArchitectOutput
  - [x] Update architect_node to return twelve-factor results in state

- [x] Task 9: Write Unit Tests for Types (AC: 7)
  - [x] Test FactorResult dataclass creation and to_dict()
  - [x] Test TwelveFactorAnalysis dataclass creation and to_dict()
  - [x] Test overall_compliance calculation
  - [x] Test immutability of frozen dataclasses

- [x] Task 10: Write Unit Tests for Analyzers (AC: 1, 2, 3, 4)
  - [x] Test analyze_twelve_factor returns TwelveFactorAnalysis
  - [x] Test Factor III (Config) detection of hardcoded values
  - [x] Test Factor VI (Processes) detection of stateful patterns
  - [x] Test Factor IV (Backing Services) detection

- [x] Task 11: Write Unit Tests for LLM Integration (AC: 5)
  - [x] Test LLM analysis with mocked LLM
  - [x] Test retry behavior on transient failures
  - [x] Test prompt construction
  - [x] Test response parsing

- [x] Task 12: Write Integration Tests (AC: 6)
  - [x] Test architect_node includes twelve-factor analysis
  - [x] Test DesignDecision rationale includes 12-Factor compliance
  - [x] Test end-to-end flow with mock stories

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for TwelveFactorAnalysis, FactorResult (internal state)
- **ADR-003 (LLM Abstraction):** Use LLMRouter for LLM calls with model tiering
- **ADR-005 (LangGraph Communication):** Return state update dict, don't mutate state directly
- **ADR-007 (Error Handling):** Use tenacity with exponential backoff for LLM calls
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Twelve-Factor App Principles Reference

| Factor | Name | Key Principle |
|--------|------|---------------|
| I | Codebase | One codebase tracked in revision control, many deploys |
| II | Dependencies | Explicitly declare and isolate dependencies |
| III | Config | Store config in the environment |
| IV | Backing Services | Treat backing services as attached resources |
| V | Build, Release, Run | Strictly separate build and run stages |
| VI | Processes | Execute the app as one or more stateless processes |
| VII | Port Binding | Export services via port binding |
| VIII | Concurrency | Scale out via the process model |
| IX | Disposability | Maximize robustness with fast startup and graceful shutdown |
| X | Dev/Prod Parity | Keep development, staging, and production as similar as possible |
| XI | Logs | Treat logs as event streams |
| XII | Admin Processes | Run admin/management tasks as one-off processes |

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/architect/node.py`
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- Use async/await for LLM calls

### Project Structure Notes

- **New Module:** `src/yolo_developer/agents/architect/twelve_factor.py`
- **Type Additions:** Add to `src/yolo_developer/agents/architect/types.py`
- **Test Location:** `tests/unit/agents/architect/test_twelve_factor.py`

### Module Structure

```
src/yolo_developer/agents/architect/
├── __init__.py         # Add TwelveFactorAnalysis, FactorResult exports
├── types.py            # Add TwelveFactorAnalysis, FactorResult dataclasses
├── node.py             # Update to integrate twelve-factor analysis
└── twelve_factor.py    # NEW: Twelve-factor analysis logic
```

### Key Imports Pattern

```python
from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.architect.types import (
    FactorResult,
    TwelveFactorAnalysis,
)
from yolo_developer.llm import LLMRouter  # or direct litellm usage

logger = structlog.get_logger(__name__)
```

### LLM Prompt Design

```python
TWELVE_FACTOR_PROMPT = """
Analyze the following user story for 12-Factor App compliance.

Story:
{story_content}

For each of the following factors, determine:
1. Does this factor apply to this story? (yes/no)
2. If applicable, is the story compliant? (yes/no/partial)
3. What specific finding led to this assessment?
4. What recommendation would improve compliance?

Factors to analyze: {factors_list}

Respond in JSON format:
{
  "factors": [
    {
      "factor_name": "...",
      "applies": true/false,
      "compliant": true/false/null,
      "finding": "...",
      "recommendation": "..."
    }
  ]
}
"""
```

### Story Dependencies

- **Depends on:** Story 7.1 (Create Architect Agent Node) - architect_node, DesignDecision, ArchitectOutput
- **Enables:** Story 7.3 (ADR Generation) - will use twelve-factor analysis for ADR context
- **Related:** Story 7.4 (Quality Attribute Evaluation) - similar evaluation pattern

### Previous Story Context (7.1)

From Story 7.1, the following are already implemented:
- `architect_node` function with @quality_gate and @retry decorators
- `DesignDecision` dataclass with to_dict()
- `ArchitectOutput` dataclass with design_decisions and adrs
- `_generate_design_decisions` function (stub with keyword-based inference)
- `_infer_decision_type` function for type classification

This story enhances `_generate_design_decisions` to include LLM-powered 12-Factor analysis.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.2] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-003] - LLM abstraction patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007] - Error handling patterns
- [Source: src/yolo_developer/agents/architect/node.py] - Existing architect implementation
- [Source: src/yolo_developer/agents/architect/types.py] - Existing type definitions
- [12factor.net](https://12factor.net) - Twelve-Factor App methodology
- [FR49: Architect Agent can design system architecture following 12-Factor principles]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 143 architect tests pass
- ruff check passes
- mypy passes with no issues

### Completion Notes List

1. Created comprehensive pattern-based 12-Factor analysis without requiring LLM for basic cases
2. Implemented LLM-powered analysis for complex cases using litellm with retry logic
3. All 12 factors are analyzed: codebase, dependencies, config, backing_services, build_release_run, processes, port_binding, concurrency, disposability, dev_prod_parity, logs, admin_processes
4. Updated `_generate_design_decisions` to be async and return tuple (decisions, analyses)
5. Design decision rationale now includes 12-Factor compliance percentage
6. ArchitectOutput includes twelve_factor_analyses dict mapping story IDs to analysis results
7. Fixed pattern matching for hardcoded URLs to include `postgresql://` variant
8. Fixed session state detection patterns for more comprehensive coverage

### File List

**New Files:**
- `src/yolo_developer/agents/architect/twelve_factor.py` - Complete 12-Factor analysis module (~400 lines)
- `tests/unit/agents/architect/test_twelve_factor_types.py` - 16 type tests
- `tests/unit/agents/architect/test_twelve_factor.py` - 28 analyzer tests
- `tests/unit/agents/architect/test_twelve_factor_llm.py` - 13 LLM integration tests
- `tests/unit/agents/architect/test_architect_node_twelve_factor.py` - 7 integration tests

**Modified Files:**
- `src/yolo_developer/agents/architect/types.py` - Added FactorResult, TwelveFactorAnalysis, TWELVE_FACTORS, updated ArchitectOutput with twelve_factor_analyses field
- `src/yolo_developer/agents/architect/__init__.py` - Added exports for new types, analyze_twelve_factor, and analyze_twelve_factor_with_llm
- `src/yolo_developer/agents/architect/node.py` - Updated to import analyze_twelve_factor, made _generate_design_decisions async, integrated twelve-factor analysis in rationale
- `tests/unit/agents/architect/test_design_decisions.py` - Updated tests to be async and handle tuple return value
- `tests/unit/agents/architect/test_adr_generation.py` - Removed unused import (ruff cleanup)
- `tests/unit/agents/architect/test_architect_node.py` - Removed unused import (ruff cleanup)
- `tests/unit/agents/architect/test_story_extraction.py` - Removed unused import (ruff cleanup)

## Senior Developer Review (AI)

**Review Date:** 2026-01-10
**Reviewer:** Claude Opus 4.5 (Adversarial Code Review)
**Outcome:** APPROVED with fixes applied

### Issues Found and Fixed

| Severity | Issue | Resolution |
|----------|-------|------------|
| MEDIUM | M1: Hardcoded LLM model in `_call_llm` | Made configurable via `YOLO_LLM__ROUTINE_MODEL` env var |
| MEDIUM | M2: File List incomplete (3 files undocumented) | Updated File List to include all modified files |
| MEDIUM | M3: Outdated docstring referencing "Story 7.2" as future | Updated docstring text |
| MEDIUM | M4: `analyze_twelve_factor_with_llm` not exported | Added to `__init__.py` exports |

### Verification
- All 143 architect tests pass
- ruff check passes
- All 7 Acceptance Criteria verified as implemented

### Change Log
- 2026-01-10: Code review fixes applied, status → done
