# Story 6.2: Transform Requirements to Stories

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want requirements converted to user stories,
So that they have clear user value and implementation scope.

## Acceptance Criteria

1. **AC1: Story Follows User Story Format**
   - **Given** validated requirements from Analyst
   - **When** the PM transforms them using LLM
   - **Then** each story follows "As a {role} / I want {action} / So that {benefit}" format
   - **And** the role is extracted appropriately from requirement context (not hardcoded to "user")
   - **And** the action is specific and bounded to a single capability
   - **And** the benefit clearly articulates the user value

2. **AC2: Stories Have Appropriate Acceptance Criteria**
   - **Given** a requirement is being transformed into a story
   - **When** acceptance criteria are generated
   - **Then** each AC follows Given/When/Then format
   - **And** ACs are concrete and measurable (no vague terms)
   - **And** edge cases relevant to the requirement are included
   - **And** AC count is appropriate for story complexity (2-5 ACs typically)

3. **AC3: LLM Integration Follows Project Patterns**
   - **Given** the PM agent needs to call LLM for transformation
   - **When** LLM calls are made
   - **Then** calls use LiteLLM's acompletion async API
   - **And** calls use the cheap_model from config for routine tasks
   - **And** retry logic with exponential backoff is applied (3 attempts max)
   - **And** LLM responses are parsed and validated

4. **AC4: Untransformable Requirements Are Handled**
   - **Given** some requirements cannot be transformed to stories
   - **When** transformation fails or requirement is unsuitable
   - **Then** the requirement ID is added to unprocessed_requirements
   - **And** constraint-type requirements are handled appropriately (as constraints on other stories)
   - **And** failed transformations don't block processing of other requirements

5. **AC5: Story Output Contains All Required Fields**
   - **Given** a successful transformation
   - **When** a Story object is created
   - **Then** all Story fields are populated appropriately:
     - id: unique identifier (e.g., "story-001")
     - title: descriptive title from requirement
     - role/action/benefit: from LLM extraction
     - acceptance_criteria: from LLM generation
     - priority: based on requirement category and context
     - status: DRAFT (initial status)
     - source_requirements: linked to source requirement IDs
     - dependencies: empty tuple (dependency detection is Story 6.5)
     - estimated_complexity: S/M/L/XL based on requirement analysis

## Tasks / Subtasks

- [x] Task 1: Create LLM Integration Utilities for PM Agent (AC: 3)
  - [x] Create `src/yolo_developer/agents/pm/llm.py` for LLM integration
  - [x] Implement `_call_llm()` function with retry logic (copy pattern from analyst)
  - [x] Add system prompt constant `PM_SYSTEM_PROMPT` for requirement transformation
  - [x] Add user prompt template `PM_USER_PROMPT_TEMPLATE` with placeholders

- [x] Task 2: Implement LLM-Powered Story Transformation (AC: 1, 4, 5)
  - [x] Replace stub `_transform_requirements_to_stories()` with LLM-powered version
  - [x] Create `_extract_story_components()` to extract role/action/benefit from requirement
  - [x] Handle constraint requirements separately (exclude from stories, note as constraints)
  - [x] Implement fallback for LLM failures (use stub logic as fallback)
  - [x] Extract priority based on requirement category and content analysis

- [x] Task 3: Implement LLM-Powered AC Generation (AC: 2)
  - [x] Replace stub `_generate_acceptance_criteria()` with LLM-powered version
  - [x] Create `AC_GENERATION_PROMPT` template for AC generation
  - [x] Ensure ACs use Given/When/Then format
  - [x] Validate generated ACs have no vague terms (use VAGUE_TERMS pattern from analyst)
  - [x] Generate 2-5 ACs per story based on complexity

- [x] Task 4: Implement Response Parsing and Validation (AC: 3, 5)
  - [x] Create `_parse_story_response()` to parse LLM JSON response
  - [x] Create `_parse_ac_response()` to parse AC generation response
  - [x] Validate required fields are present
  - [x] Handle malformed JSON gracefully with fallback
  - [x] Log parsing failures for debugging

- [x] Task 5: Implement Complexity Estimation (AC: 5)
  - [x] Create `_estimate_complexity()` function
  - [x] Analyze requirement text for complexity indicators
  - [x] Consider AC count and scope in estimation
  - [x] Return S/M/L/XL based on analysis

- [x] Task 6: Add Feature Flag for LLM Usage (AC: 3)
  - [x] Add `_USE_LLM: bool = False` flag (same pattern as analyst)
  - [x] When False, use stub implementation (for testing)
  - [x] When True, use LLM-powered transformation
  - [x] Document flag usage in module docstring

- [x] Task 7: Update pm_node to Use New Transformation (AC: all)
  - [x] Update imports in node.py to use new LLM functions
  - [x] Ensure config is loaded and passed to transformation
  - [x] Add logging for LLM transformation steps
  - [x] Update Decision record with actual transformation details

- [x] Task 8: Write Unit Tests for LLM Integration (AC: 3)
  - [x] Test `_call_llm()` function signature and retry behavior
  - [x] Test prompt templates are well-formed
  - [x] Test with mocked LLM responses
  - [x] Test retry exhaustion handling

- [x] Task 9: Write Unit Tests for Story Transformation (AC: 1, 4, 5)
  - [x] Test `_extract_story_components()` with various requirement types
  - [x] Test role extraction is appropriate (not hardcoded "user")
  - [x] Test constraint requirements are filtered out
  - [x] Test fallback behavior when LLM fails
  - [x] Test priority assignment logic

- [x] Task 10: Write Unit Tests for AC Generation (AC: 2)
  - [x] Test `_generate_acceptance_criteria()` produces valid AC structure
  - [x] Test ACs use Given/When/Then format
  - [x] Test vague term detection
  - [x] Test AC count ranges (2-5)

- [x] Task 11: Write Unit Tests for Response Parsing (AC: 3, 5)
  - [x] Test `_parse_story_response()` with valid JSON
  - [x] Test `_parse_story_response()` with malformed JSON (fallback)
  - [x] Test `_parse_ac_response()` with valid JSON
  - [x] Test field validation

- [x] Task 12: Write Integration Test for Full Transformation (AC: all)
  - [x] Test pm_node with _USE_LLM=False produces valid output
  - [x] Test transformation preserves requirement traceability
  - [x] Test all Story fields are populated correctly

## Dev Notes

### Architecture Compliance

- **ADR-003 (LLM Provider Abstraction):** Use LiteLLM for LLM calls with config-driven model selection
- **ADR-007 (Error Handling Strategy):** Use Tenacity retry with exponential backoff (3 attempts)
- **ADR-008 (Configuration):** Load config to get cheap_model for routine transformation tasks
- **ARCH-QUALITY-5:** All LLM calls must use async/await (acompletion)
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing LLM call pattern from `agents/analyst/node.py`
- Use `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))`
- Import `acompletion` from `litellm` (not `completion`)

### LLM Prompt Design Guidelines

The PM agent needs prompts for:

1. **Story Component Extraction:**
   - Input: Crystallized requirement text
   - Output: JSON with {role, action, benefit, title}
   - Should identify appropriate user role from context
   - Should extract specific, bounded action
   - Should articulate clear user value/benefit

2. **Acceptance Criteria Generation:**
   - Input: Story components + requirement context
   - Output: JSON array of {given, when, then, and_clauses}
   - Should generate 2-5 ACs based on complexity
   - Should include relevant edge cases
   - Should avoid vague terms

### File Structure (ARCH-STRUCT)

New file to create:
```
src/yolo_developer/agents/pm/
├── llm.py              # NEW: LLM integration utilities
└── node.py             # MODIFY: Update to use LLM-powered transformation

tests/unit/agents/pm/
├── test_llm.py         # NEW: Tests for LLM integration
└── test_node.py        # MODIFY: Add tests for LLM-powered transformation
```

### Existing Code Patterns to Follow

**From Analyst Module (`agents/analyst/node.py`):**

LLM call pattern (lines 3200-3235):
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _call_llm(prompt: str, system: str) -> str:
    """Call LLM with retry logic."""
    from litellm import acompletion
    from yolo_developer.config import load_config

    config = load_config()
    model = config.llm.cheap_model

    logger.info("calling_llm", model=model, prompt_length=len(prompt))

    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
```

Feature flag pattern:
```python
# Flag to enable/disable actual LLM calls (for testing)
_USE_LLM: bool = False
```

### Previous Story Intelligence (Story 6.1)

From Story 6.1 implementation:

1. **Existing stub functions to replace:**
   - `_generate_acceptance_criteria()` - Returns single placeholder AC
   - `_transform_requirements_to_stories()` - Returns placeholder stories with hardcoded values

2. **Code review fixes from 6.1 to maintain:**
   - Sequential story IDs (use separate counter, not enumerate index)
   - Extract gaps and contradictions from analyst_output for context
   - Constraint requirements should be filtered but noted

3. **Existing types to use:**
   - `AcceptanceCriterion(id, given, when, then, and_clauses)`
   - `Story(id, title, role, action, benefit, acceptance_criteria, priority, status, ...)`
   - `PMOutput(stories, unprocessed_requirements, escalations_to_analyst, processing_notes)`

### Vague Terms to Avoid in ACs

Use the same vague term detection from analyst (VAGUE_TERMS frozenset):
- Quantifier vagueness: fast, quick, slow, efficient, scalable, responsive, real-time
- Ease vagueness: easy, simple, straightforward, intuitive, user-friendly, seamless
- Certainty vagueness: should, might, could, may, possibly, probably, maybe
- Quality vagueness: good, better, best, nice, beautiful, clean, modern, robust

### Integration Points

**Input from Analyst (via state):**
- `state["analyst_output"]["requirements"]` - List of crystallized requirements with:
  - `id`: str
  - `refined_text`: str (crystallized requirement text)
  - `original_text`: str
  - `category`: str (functional, non_functional, constraint)
  - `sub_category`: str | None
  - `is_testable`: bool
  - `confidence`: float

**Output to Orchestrator:**
- `pm_output`: PMOutput.to_dict() with transformed stories
- `decisions`: Decision records for audit trail
- `messages`: AIMessage for conversation history

### Testing Strategy

**Unit Tests:**
- Mock LiteLLM calls for deterministic testing
- Test prompt template formatting
- Test JSON response parsing
- Test fallback behavior on LLM failures
- Test vague term detection in generated ACs

**Integration Tests:**
- Test full pm_node flow with _USE_LLM=False
- Test that all Story fields are populated
- Test requirement traceability is preserved

### Complexity Estimation Heuristics

For `_estimate_complexity()`:
- **S (Small):** Simple CRUD operation, single entity, no external dependencies
- **M (Medium):** Multiple entities, basic validation, standard patterns
- **L (Large):** Complex business logic, multiple integrations, error handling
- **XL (Extra Large):** Cross-cutting concerns, security requirements, performance requirements

Indicators to look for:
- Number of entities/concepts mentioned
- Presence of "and" conjunctions (multiple capabilities)
- Integration keywords (API, external, service)
- Security keywords (authentication, authorization, encryption)
- Performance keywords (scalable, optimized, cached)

### Project Structure Notes

- PM module at `src/yolo_developer/agents/pm/`
- New llm.py alongside existing types.py and node.py
- Tests at `tests/unit/agents/pm/`
- No circular imports - PM imports from config, orchestrator/context, gates

### References

- [Source: _bmad-output/planning-artifacts/epics.md - Story 6.2: Transform Requirements to Stories]
- [Source: _bmad-output/planning-artifacts/epics.md - Epic 6: PM Agent overview]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-003: LLM Provider Abstraction]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-007: Error Handling Strategy]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-008: Configuration Management]
- [Source: _bmad-output/planning-artifacts/prd.md - FR42: Transform requirements to user stories]
- [Source: _bmad-output/planning-artifacts/prd.md - FR48: Story documentation following templates]
- [Source: src/yolo_developer/agents/analyst/node.py - LLM integration pattern (lines 3200-3345)]
- [Source: src/yolo_developer/agents/pm/node.py - Stub implementations to replace]
- [Source: src/yolo_developer/agents/pm/types.py - PM type definitions]
- [Source: _bmad-output/implementation-artifacts/6-1-create-pm-agent-node.md - Previous story patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - All tests passed without debugging needed.

### Completion Notes List

- **Task 1:** Created `llm.py` with LLM integration utilities including `_call_llm()`, prompt templates, and vague term detection
- **Task 2:** Replaced stub `_transform_requirements_to_stories()` with async LLM-powered version; added `_transform_single_requirement()` helper
- **Task 3:** Replaced stub `_generate_acceptance_criteria()` with async LLM-powered version using `_generate_acceptance_criteria_llm()`
- **Task 4:** Implemented `_parse_story_response()` and `_parse_ac_response()` with JSON extraction and validation
- **Task 5:** Implemented `_estimate_complexity()` with heuristics for S/M/L/XL estimation based on keywords and AC count
- **Task 6:** Added `_USE_LLM: bool = False` feature flag with documentation
- **Task 7:** Updated `pm_node` to use async transformation, updated Decision rationale to mention LLM-powered extraction
- **Task 8-12:** Added 47 tests in `test_llm.py` and updated 36 tests in `test_node.py` (total 119 tests passing after code review fixes)

### Change Log

- 2026-01-09: Story 6.2 implementation complete. Added LLM-powered story transformation with fallback to stub mode.
- 2026-01-09: Code review complete. Fixed 7 issues (3 HIGH, 4 MEDIUM):
  - HIGH #1: Added role extraction heuristics in stub mode (role_keywords dict)
  - HIGH #2: Updated stub to return 2 ACs minimum (added error-handling scenario AC)
  - HIGH #3: Added retry exhaustion test using tenacity.RetryError
  - MEDIUM #4: Improved JSON parsing with brace counting instead of simple regex
  - MEDIUM #5: Changed unused gaps/contradictions variables to just counts
  - MEDIUM #6: Added test for _transform_single_requirement exception handling
  - MEDIUM #7: Added string validation for and_clauses items

### File List

**New Files:**
- `src/yolo_developer/agents/pm/llm.py` - LLM integration utilities (647 lines)
- `tests/unit/agents/pm/test_llm.py` - Unit tests for LLM integration (47 tests)

**Modified Files:**
- `src/yolo_developer/agents/pm/node.py` - Updated to use async LLM-powered transformation
- `tests/unit/agents/pm/test_node.py` - Updated tests to handle async functions (36 tests)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Story status updated to review
