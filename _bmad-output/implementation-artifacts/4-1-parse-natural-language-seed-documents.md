# Story 4.1: Parse Natural Language Seed Documents

Status: done

## Story

As a developer,
I want to provide requirements in natural language,
So that I don't need to learn a special format to get started.

## Acceptance Criteria

1. **AC1: Structured Component Extraction**
   - **Given** I have a text document describing what I want to build
   - **When** I provide it as a seed
   - **Then** the system parses it into structured components
   - **And** each component has a type (goal, feature, constraint, etc.)
   - **And** components are returned as typed data structures

2. **AC2: High-Level Goal Identification**
   - **Given** a seed document with project description
   - **When** parsing completes
   - **Then** high-level goals are identified and extracted
   - **And** goals capture the "what" and "why" of the project
   - **And** goals are ordered by apparent priority

3. **AC3: Feature Description Extraction**
   - **Given** a seed document mentioning capabilities
   - **When** parsing extracts features
   - **Then** feature descriptions are extracted as discrete items
   - **And** each feature is self-contained and actionable
   - **And** implicit features are surfaced when obvious

4. **AC4: Constraint Recognition**
   - **Given** a seed document with technical or business constraints
   - **When** parsing identifies constraints
   - **Then** constraints are recognized and categorized
   - **And** constraint types include: technical, business, timeline, resource
   - **And** constraints are linked to affected goals/features when possible

5. **AC5: Multiple Input Formats**
   - **Given** seed content in various formats
   - **When** the parser processes the input
   - **Then** plain text (.txt) is supported
   - **And** markdown (.md) is supported with structure awareness
   - **And** the source format is detected or specified

6. **AC6: Parse Result Structure**
   - **Given** parsing completes successfully
   - **When** results are returned
   - **Then** results are returned as `SeedParseResult` dataclass
   - **And** results include: goals, features, constraints, raw_content, metadata
   - **And** results support `to_dict()` for serialization

## Tasks / Subtasks

- [x] Task 1: Create Seed Types Module (AC: 1, 6)
  - [x] Create `src/yolo_developer/seed/types.py` module
  - [x] Define `SeedSource` enum: `FILE`, `TEXT`, `URL`
  - [x] Define `ComponentType` enum: `GOAL`, `FEATURE`, `CONSTRAINT`, `CONTEXT`, `UNKNOWN`
  - [x] Define `ConstraintCategory` enum: `TECHNICAL`, `BUSINESS`, `TIMELINE`, `RESOURCE`, `COMPLIANCE`
  - [x] Define `SeedComponent` frozen dataclass with fields: `component_type`, `content`, `confidence`, `source_line`, `metadata`
  - [x] Define `SeedGoal` frozen dataclass with fields: `title`, `description`, `priority`, `rationale`
  - [x] Define `SeedFeature` frozen dataclass with fields: `name`, `description`, `user_value`, `related_goals`
  - [x] Define `SeedConstraint` frozen dataclass with fields: `category`, `description`, `impact`, `related_items`
  - [x] Define `SeedParseResult` frozen dataclass with fields: `goals`, `features`, `constraints`, `raw_content`, `source`, `metadata`
  - [x] Add `to_dict()` methods to all dataclasses for JSON serialization

- [x] Task 2: Create Parser Protocol and Base Infrastructure (AC: 1, 5)
  - [x] Create `src/yolo_developer/seed/parser.py` module
  - [x] Define `SeedParser` Protocol with method: `async def parse(content: str, source: SeedSource) -> SeedParseResult`
  - [x] Implement `detect_source_format(content: str, filename: str | None) -> SeedSource` utility
  - [x] Implement `normalize_content(content: str) -> str` to clean and standardize input
  - [x] Handle encoding detection for file content
  - [x] Use structlog for all logging operations

- [x] Task 3: Implement LLM-Based Seed Parser (AC: 1, 2, 3, 4)
  - [x] Implement `LLMSeedParser` class implementing `SeedParser` protocol
  - [x] Create structured prompt for seed analysis with clear output schema
  - [x] Use LiteLLM for provider-agnostic LLM calls (via `llm/router.py` pattern)
  - [x] Parse LLM response into typed dataclasses
  - [x] Handle parsing failures with retry logic (Tenacity)
  - [x] Support configurable model tier (default: "routine" for cost efficiency)

- [x] Task 4: Implement Goal Extraction Logic (AC: 2)
  - [x] Create `_extract_goals(llm_output: dict) -> list[SeedGoal]` helper
  - [x] Identify project-level objectives from parsed content
  - [x] Extract "what" (capability) and "why" (value) components
  - [x] Assign priority scores based on emphasis in source text
  - [x] Handle implicit goals (e.g., "must be fast" implies performance goal)

- [x] Task 5: Implement Feature Extraction Logic (AC: 3)
  - [x] Create `_extract_features(llm_output: dict, goals: list[SeedGoal]) -> list[SeedFeature]` helper
  - [x] Identify discrete functional capabilities
  - [x] Link features to related goals when relationships are clear
  - [x] Ensure features are atomic and actionable
  - [x] Deduplicate similar features with confidence scoring

- [x] Task 6: Implement Constraint Extraction Logic (AC: 4)
  - [x] Create `_extract_constraints(llm_output: dict) -> list[SeedConstraint]` helper
  - [x] Categorize constraints by type (technical, business, timeline, resource, compliance)
  - [x] Extract constraint descriptions and impact statements
  - [x] Link constraints to affected goals/features
  - [x] Handle implicit constraints (e.g., "Python project" implies Python technical constraint)

- [x] Task 7: Implement Plain Text Parser (AC: 5)
  - [x] Implement `_parse_plain_text(content: str) -> str` preprocessor
  - [x] Split content into logical sections based on whitespace/formatting
  - [x] Identify list items (-, *, numbered) as potential discrete requirements
  - [x] Preserve source line numbers for traceability

- [x] Task 8: Implement Markdown Parser (AC: 5)
  - [x] Implement `_parse_markdown(content: str) -> str` preprocessor
  - [x] Extract structure from headings (h1-h6)
  - [x] Parse lists as requirement candidates
  - [x] Handle code blocks (may contain technical constraints)
  - [x] Extract emphasis (**bold**, *italic*) as potential priority signals

- [x] Task 9: Create High-Level Parse API (AC: all)
  - [x] Create `src/yolo_developer/seed/api.py` module
  - [x] Implement `async def parse_seed(content: str, source: SeedSource | None = None, filename: str | None = None) -> SeedParseResult`
  - [x] Auto-detect source format if not specified
  - [x] Apply appropriate preprocessor based on format
  - [x] Invoke LLM parser for extraction
  - [x] Aggregate and return structured results

- [x] Task 10: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/seed/test_types.py` - Test all dataclass creation and serialization
  - [x] Create `tests/unit/seed/test_parser.py` - Test parser infrastructure and utilities
  - [x] Create `tests/unit/seed/test_extractors.py` - Test extraction helpers with mocked LLM
  - [x] Test edge cases: empty content, malformed input, unicode handling
  - [x] Test format detection for .txt, .md files
  - [x] Aim for >80% coverage

- [x] Task 11: Write Integration Tests (AC: all)
  - [x] Create `tests/integration/test_seed_parsing.py`
  - [x] Create sample seed documents in `tests/fixtures/seeds/`
  - [x] Test end-to-end parsing with real (mocked) LLM responses
  - [x] Test parsing of simple, medium, and complex seed documents
  - [x] Verify structured output matches expected schema

- [x] Task 12: Update Exports and Documentation (AC: 6)
  - [x] Export all types from `seed/__init__.py`
  - [x] Export `parse_seed` API function from `seed/__init__.py`
  - [x] Add module docstring with usage examples
  - [x] Ensure all public functions have comprehensive docstrings

## Dev Notes

### Architecture Compliance

- **ADR-001 (TypedDict + Pydantic):** SeedInput uses Pydantic for boundary validation, internal types use frozen dataclasses
- **ADR-003 (LLM Provider Abstraction):** Use LiteLLM via the `llm/router.py` pattern for provider-agnostic calls
- **FR1:** Users can provide seed requirements as natural language text documents
- **FR3:** System can parse and structure unstructured seed requirements into actionable components
- [Source: architecture.md#Seed Processing] - `seed/` module handles FR1-8

### Technical Requirements

- **Immutable Types:** All dataclasses must be frozen for thread safety
- **Async Operations:** Parser must be fully async for non-blocking operation
- **Structured Logging:** Use structlog for all parsing events and errors
- **Type Safety:** Full type annotations on all functions and return values
- **Error Handling:** Graceful degradation if LLM parsing fails

### Existing Pattern References

**From architecture.md (SeedInput Boundary Type):**
```python
# Boundary validation (Pydantic) - from architecture.md line 290-293
class SeedInput(BaseModel):
    content: str
    source: Literal["file", "text", "url"]
    model_config = ConfigDict(strict=True)
```

**From architecture.md (YoloState TypedDict):**
```python
# Internal graph state uses TypedDict with SeedInput
class YoloState(TypedDict):
    seed: SeedInput
    requirements: list[Requirement]
    stories: list[Story]
    current_agent: str
    messages: Annotated[list[Message], add_messages]
```

**From gates module (Frozen Dataclass Pattern):**
```python
@dataclass(frozen=True)
class GateMetricRecord:
    gate_name: str
    passed: bool
    score: float
    # ... other fields

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            # ... serialize all fields
        }
```

**From decorator.py (Structlog Pattern):**
```python
import structlog
logger = structlog.get_logger(__name__)

logger.info("seed_parsing_started", source=source.value, content_length=len(content))
```

### Project Structure Notes

**Seed Module Location:**
```
src/yolo_developer/seed/
├── __init__.py              # UPDATE: Export types and parse_seed API
├── types.py                 # NEW: SeedComponent, SeedGoal, SeedFeature, SeedConstraint, SeedParseResult
├── parser.py                # NEW: SeedParser protocol, LLMSeedParser implementation
└── api.py                   # NEW: parse_seed() high-level API
```

**Test Structure:**
```
tests/
├── fixtures/seeds/          # NEW: Sample seed documents for testing
│   ├── simple_seed.txt
│   ├── complex_seed.md
│   └── edge_case_seed.txt
├── unit/seed/
│   ├── test_types.py        # NEW: Type creation and serialization tests
│   ├── test_parser.py       # NEW: Parser infrastructure tests
│   └── test_extractors.py   # NEW: Extraction helper tests
└── integration/
    └── test_seed_parsing.py # NEW: End-to-end parsing tests
```

### LLM Prompt Design Notes

The LLM parser should use a structured prompt that:
1. Explains the task clearly (extract goals, features, constraints from natural language)
2. Provides output schema (JSON structure expected)
3. Includes few-shot examples for better extraction quality
4. Requests confidence scores for extracted items

**Example Prompt Structure:**
```
You are a requirements analyst. Parse the following seed document and extract:
1. Goals: High-level project objectives (what and why)
2. Features: Discrete functional capabilities
3. Constraints: Technical, business, or other limitations

Output as JSON with this structure:
{
  "goals": [{"title": "...", "description": "...", "priority": 1-5, "rationale": "..."}],
  "features": [{"name": "...", "description": "...", "user_value": "...", "related_goals": [...]}],
  "constraints": [{"category": "technical|business|...", "description": "...", "impact": "...", "related_items": [...]}]
}

Seed Document:
<content>
```

### Previous Epic Learnings

**From Epic 3 (Quality Gates):**
- Export new types from `__init__.py` immediately
- Use `()` literal instead of `tuple()` for empty tuples in frozen dataclasses
- Use `eval_*` prefix for test helper functions to avoid pytest collection
- Include comprehensive docstrings with examples

**From Epic 2 (Memory Layer):**
- Protocol-based design enables multiple implementations
- Async operations must handle cancellation properly
- JSON serialization should handle datetime with `.isoformat()`

### Git Intelligence (Recent Commits)

Recent commits show patterns for:
- Creating frozen dataclasses with `to_dict()` methods
- Implementing protocols with async methods
- Integration test patterns with fixtures
- Export patterns in `__init__.py`

### Testing Standards

- Use pytest with pytest-asyncio for async tests
- Create fixtures for mock LLM responses
- Test parsing with various input formats
- Test edge cases: empty input, very long input, malformed content
- Verify type safety with frozen dataclasses
- Mock LLM calls to avoid API costs during testing

### Implementation Approach

1. **Types First:** Create all dataclass types in `types.py`
2. **Protocol Definition:** Define `SeedParser` protocol in `parser.py`
3. **Preprocessing:** Implement format-specific preprocessors
4. **LLM Parser:** Implement `LLMSeedParser` with structured prompts
5. **Extractors:** Implement helper functions for each component type
6. **API Layer:** Create high-level `parse_seed()` function
7. **Testing:** Unit tests per module, then integration tests
8. **Exports:** Update `__init__.py` with all public exports

### Dependencies

This story has no dependencies on other stories. It is the first story in Epic 4.

**Downstream Dependencies:**
- Story 4.2 (CLI Seed Command) - Will use `parse_seed()` API
- Story 4.3 (Ambiguity Detection) - Will operate on `SeedParseResult`
- Story 5.1 (Analyst Agent) - Will receive parsed seed components

### References

- [Source: architecture.md#ADR-001] - TypedDict for internal state, Pydantic at boundaries
- [Source: architecture.md#ADR-003] - LiteLLM for LLM provider abstraction
- [Source: architecture.md#Seed Processing] - seed/ module structure (FR1-8)
- [Source: architecture.md#YoloState] - SeedInput type definition
- [Source: prd.md#FR1] - Users can provide seed requirements as natural language text documents
- [Source: prd.md#FR3] - System can parse and structure unstructured seed requirements
- [Source: epics.md#Story-4.1] - Parse Natural Language Seed Documents requirements
- [Story 3.9 Implementation] - Frozen dataclass patterns, to_dict() serialization

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None - clean implementation with no significant debug issues.

### Completion Notes List

1. **Types Module:** Created comprehensive frozen dataclasses with `to_dict()` methods following patterns from Epic 3 (gates module)
2. **Parser Protocol:** Implemented runtime-checkable `SeedParser` protocol with `detect_source_format()` and `normalize_content()` utilities
3. **LLM Parser:** Implemented `LLMSeedParser` using LiteLLM with tenacity retry logic and structured JSON prompt
4. **Extraction Helpers:** Created `_extract_goals()`, `_extract_features()`, `_extract_constraints()` with robust error handling for malformed entries
5. **Preprocessors:** Implemented `_parse_plain_text()` and `_parse_markdown()` with line number annotations and structure markers
6. **High-Level API:** Created `parse_seed()` function with auto-detection, preprocessing, and graceful error handling
7. **Test Coverage:** 108 total tests (98 unit + 10 integration) all passing
8. **Integration Tests:** Used `patch.object(LLMSeedParser, "_call_llm")` pattern for reliable async mocking

### File List

**New Files Created:**
- `src/yolo_developer/seed/types.py` - Seed types and dataclasses
- `src/yolo_developer/seed/parser.py` - Parser protocol and LLM implementation
- `src/yolo_developer/seed/api.py` - High-level parse_seed API
- `tests/unit/seed/test_types.py` - Type unit tests (36 tests)
- `tests/unit/seed/test_parser.py` - Parser unit tests (33 tests)
- `tests/unit/seed/test_extractors.py` - Extractor unit tests (29 tests)
- `tests/integration/test_seed_parsing.py` - Integration tests (10 tests)
- `tests/fixtures/seeds/simple_seed.txt` - Simple test fixture
- `tests/fixtures/seeds/complex_seed.md` - Complex markdown fixture
- `tests/fixtures/seeds/edge_case_seed.txt` - Edge case fixture

**Files Modified:**
- `src/yolo_developer/seed/__init__.py` - Added all exports

---

## Code Review

**Date:** 2026-01-06
**Reviewer:** Adversarial Senior Developer
**Result:** APPROVED (after fixes)

### Issues Found & Fixed

| # | Severity | Issue | Fix Applied |
|---|----------|-------|-------------|
| 1 | HIGH | `raw_content` in `SeedParseResult` contained preprocessed content with `[L1]` annotations instead of original input | Used `dataclasses.replace()` in `api.py` to restore original content after parsing |
| 2 | MEDIUM | Empty string test `assert "" in result.raw_content` in `test_seed_parsing.py` was trivially true | Added actual unicode characters (🚀, 日本語, Ελληνικά) to `edge_case_seed.txt` fixture and updated assertions |
| 3 | MEDIUM | Unrelated `metrics_api.py` change included in story files | Reverted with `git checkout` and applied proper formatting |
| 4 | MEDIUM | No tests for retry logic in `_call_llm` method | Added 4 tests in `TestLLMSeedParserRetryLogic` class |
| 5 | LOW | Dead `TYPE_CHECKING` code block (just `pass`) in `parser.py` | Removed unused import and empty code block |
| 6 | LOW | No tests for `_looks_like_markdown()` helper function | Added 14 tests in new `tests/unit/seed/test_api.py` file |

### Hidden Bug Found & Fixed

**Issue:** `SEED_ANALYSIS_PROMPT` in `parser.py` contained unescaped curly braces in the JSON example schema. When `prompt.format(content=content)` was called, Python's string formatting interpreted `{ "goals":` as a format placeholder, causing `KeyError: '\n  "goals"'`.

**Root Cause:** The prompt template mixed `.format()` placeholder `{content}` with literal JSON curly braces that weren't doubled.

**Fix:** Escaped all JSON curly braces by doubling them (`{` → `{{`, `}` → `}}`).

**Impact:** This bug would have caused 100% failure rate for any real LLM call. It was hidden because all existing tests mocked `_call_llm` entirely, never executing the prompt formatting.

### Files Modified During Code Review

- `src/yolo_developer/seed/api.py` - Added `dataclasses.replace()` import, fixed raw_content preservation
- `src/yolo_developer/seed/parser.py` - Escaped JSON braces in prompt, removed dead TYPE_CHECKING code, added type annotation for json.loads result
- `src/yolo_developer/seed/__init__.py` - Sorted `__all__` exports
- `tests/unit/seed/test_parser.py` - Added `TestLLMSeedParserRetryLogic` class (4 tests)
- `tests/unit/seed/test_api.py` - **NEW FILE** - Added `TestLooksLikeMarkdown` class (14 tests)
- `tests/fixtures/seeds/edge_case_seed.txt` - Added unicode characters for meaningful assertions
- `tests/integration/test_seed_parsing.py` - Updated unicode assertions

### Test Results After Code Review

- **Total Tests:** 126 (up from 108)
- **New Tests Added:** 18 (4 retry + 14 _looks_like_markdown)
- **All Tests Passing:** ✅
- **Code Quality Checks:** ✅ (ruff check, ruff format, mypy)
