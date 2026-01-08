# Story 4.4: Clarification Question Generation

Status: done

## Story

As a developer,
I want the system to generate clarification questions,
So that I know exactly what additional information is needed.

## Acceptance Criteria

1. **AC1: Specific Questions per Ambiguity**
   - **Given** ambiguities are detected via `detect_ambiguities()`
   - **When** clarification questions are generated
   - **Then** each ambiguity has at least one specific question
   - **And** questions directly address the ambiguity's core issue
   - **And** questions avoid vague phrasing like "please clarify"

2. **AC2: Actionable Questions**
   - **Given** generated clarification questions
   - **When** displayed to the user
   - **Then** questions can be answered definitively (yes/no, number, choice)
   - **And** questions don't require additional context to understand
   - **And** questions have clear expected response types

3. **AC3: Answer Format Suggestions**
   - **Given** generated clarification questions
   - **When** the question requires a specific format
   - **Then** suggested answer formats are provided (e.g., "Enter a number 1-100")
   - **And** format hints match the question type (numeric, boolean, choice, free-text)
   - **And** format validation rules can be inferred from suggestions

4. **AC4: Question Prioritization by Impact**
   - **Given** multiple ambiguities with questions
   - **When** questions are presented to the user
   - **Then** questions are prioritized by impact (severity + type)
   - **And** HIGH severity questions appear first
   - **And** BLOCKING ambiguity types (UNDEFINED, SCOPE) rank higher than ADVISORY (PRIORITY)
   - **And** priority score is calculable and deterministic

## Tasks / Subtasks

- [x] Task 1: Enhance ResolutionPrompt Type (AC: 2, 3)
  - [x] Add `answer_format: AnswerFormat` field to `ResolutionPrompt` dataclass
  - [x] Create `AnswerFormat` enum: BOOLEAN, NUMERIC, CHOICE, FREE_TEXT, DATE, LIST
  - [x] Add `format_hint: str | None` field for human-readable format guidance
  - [x] Add `validation_pattern: str | None` field for optional regex validation
  - [x] Update `ResolutionPrompt.to_dict()` to include new fields
  - [x] Maintain backward compatibility with existing code

- [x] Task 2: Implement Question Quality Validation (AC: 1, 2)
  - [x] Create `validate_question_quality(question: str) -> tuple[bool, list[str]]` function
  - [x] Check for vague phrases: "please clarify", "more information", "elaborate"
  - [x] Check for definitive answerability (not open-ended without bounds)
  - [x] Check minimum question length (>10 chars)
  - [x] Return validation result and list of improvement suggestions
  - [x] Add unit tests for validation function

- [x] Task 3: Enhance LLM Question Generation (AC: 1, 2, 3)
  - [x] Update `AMBIGUITY_DETECTION_PROMPT` to request answer_format
  - [x] Add format inference logic: analyze question text to suggest format
  - [x] Add question quality guidance to prompt (actionable, definitive)
  - [x] Parse `answer_format` from LLM response into `AnswerFormat` enum
  - [x] Generate `format_hint` based on `answer_format` type
  - [x] Add fallback for missing format data (default to FREE_TEXT)

- [x] Task 4: Implement Question Prioritization (AC: 4)
  - [x] Create `calculate_question_priority(ambiguity: Ambiguity) -> int` function
  - [x] Define priority scoring: severity weight + type weight
  - [x] Severity weights: HIGH=30, MEDIUM=20, LOW=10
  - [x] Type weights: UNDEFINED=25, SCOPE=20, TECHNICAL=15, DEPENDENCY=10, PRIORITY=5
  - [x] Create `prioritize_questions(ambiguities: list[Ambiguity]) -> list[Ambiguity]` function
  - [x] Return ambiguities sorted by priority (highest first)
  - [x] Add deterministic tie-breaking (by source_text for consistency)

- [x] Task 5: Update Interactive CLI Display (AC: 3, 4)
  - [x] Update `_prompt_for_resolution()` to display format hints
  - [x] Sort ambiguities by priority before prompting
  - [x] Display priority indicator (e.g., "[HIGH PRIORITY]") for top items
  - [x] Add format-specific input validation where applicable
  - [x] Show answer format guidance in prompt text

- [x] Task 6: Update AmbiguityResult to Support Prioritization (AC: 4)
  - [x] Add `prioritized_ambiguities` property to `AmbiguityResult`
  - [x] Cache prioritization result for efficiency
  - [x] Update `AmbiguityResult.to_dict()` to include priority scores
  - [x] Add `get_highest_priority_ambiguity()` convenience method

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Test `AnswerFormat` enum values and serialization
  - [x] Test `validate_question_quality()` with good and bad questions
  - [x] Test `calculate_question_priority()` scoring logic
  - [x] Test `prioritize_questions()` ordering
  - [x] Test format hint generation for each `AnswerFormat` type
  - [x] Test backward compatibility with existing `ResolutionPrompt` usage

- [x] Task 8: Write Integration Tests (AC: all)
  - [x] Test full flow with enhanced question generation
  - [x] Test CLI displays format hints correctly
  - [x] Test prioritization in interactive mode
  - [x] Test JSON output includes new fields
  - [x] Mock LLM calls to return various answer formats

- [x] Task 9: Update Exports and Documentation (AC: all)
  - [x] Export `AnswerFormat` from `seed/__init__.py`
  - [x] Export `validate_question_quality` from `seed/__init__.py`
  - [x] Export `calculate_question_priority` from `seed/__init__.py`
  - [x] Export `prioritize_questions` from `seed/__init__.py`
  - [x] Update docstrings with usage examples

## Dev Notes

### Architecture Compliance

- **ADR-003 (LLM Abstraction):** Continue using LiteLLM for enhanced question generation
- **ADR-005 (CLI Framework):** Use Typer + Rich for format-aware prompts
- **FR5:** System can generate clarification questions for vague requirements
- [Source: architecture.md#Seed Input] - `seed/` module handles FR1-8
- [Source: epics.md#Story-4.4] - Clarification question generation requirements

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses for `AnswerFormat`
- **Backward Compatibility:** Existing code using `ResolutionPrompt` must continue to work
- **Async Pattern:** Any new LLM calls must be async
- **Rich Prompts:** Format hints displayed using Rich formatting

### Previous Story Intelligence (Story 4.3)

**Files Created/Modified in Story 4.3:**
- `src/yolo_developer/seed/ambiguity.py` (586 lines) - Core ambiguity types and detection
- `src/yolo_developer/seed/types.py` - SeedParseResult with ambiguity fields
- `src/yolo_developer/seed/api.py` - parse_seed() with detect_ambiguities parameter
- `src/yolo_developer/cli/commands/seed.py` - Interactive mode and display
- Tests: 51 passing (30 unit + 21 integration)

**Key Patterns from Story 4.3:**
```python
# ResolutionPrompt dataclass pattern (to be extended)
@dataclass(frozen=True)
class ResolutionPrompt:
    question: str
    suggestions: tuple[str, ...]
    default: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "suggestions": list(self.suggestions),
            "default": self.default,
        }
```

**LLM Prompt Pattern (from ambiguity.py):**
```python
AMBIGUITY_DETECTION_PROMPT = """...
For each ambiguity, provide:
- type: One of SCOPE, TECHNICAL, PRIORITY, DEPENDENCY, UNDEFINED
- severity: HIGH (blocks implementation), MEDIUM (causes confusion), LOW (minor clarification)
- source_text: The exact ambiguous phrase from the document
- location: Line number or section where found
- description: Why this is ambiguous
- question: A specific question to clarify this ambiguity
- suggestions: 2-3 possible interpretations or answers
..."""
```

**Critical Learnings from 4.3:**
1. Always use `from __future__ import annotations`
2. Handle LLM JSON responses wrapped in markdown code blocks
3. Mock LLM calls with `patch("yolo_developer.seed.ambiguity.litellm.acompletion")`
4. Use `typer.testing.CliRunner` for CLI tests
5. Test Rich output by checking `result.output` contains expected strings

### Git Intelligence (Recent Commits)

**Story 4.3 Commit (e6bbc98):**
- feat: Implement ambiguity detection and resolution prompts (Story 4.3)
- 11 files changed, 3,111 insertions
- Files: ambiguity.py, types.py, api.py, seed.py, test files

**Commit Message Pattern:**
```
feat: <description> (Story X.Y)

- Bullet point 1
- Bullet point 2
...

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Implementation Approach

1. **Types First:** Add `AnswerFormat` enum and extend `ResolutionPrompt`
2. **Validation Function:** Implement question quality validation
3. **Priority Scoring:** Implement deterministic priority calculation
4. **LLM Enhancement:** Update prompt to request answer formats
5. **CLI Updates:** Display format hints and prioritize questions
6. **Tests:** Unit tests for new functions, integration tests for full flow
7. **Exports:** Update `__init__.py` with new public API

### Priority Scoring Algorithm

```python
def calculate_question_priority(ambiguity: Ambiguity) -> int:
    """Calculate priority score for question ordering.

    Higher score = higher priority (shown first).
    """
    severity_weights = {
        AmbiguitySeverity.HIGH: 30,
        AmbiguitySeverity.MEDIUM: 20,
        AmbiguitySeverity.LOW: 10,
    }
    type_weights = {
        AmbiguityType.UNDEFINED: 25,  # Missing info is most critical
        AmbiguityType.SCOPE: 20,       # Scope unclear blocks planning
        AmbiguityType.TECHNICAL: 15,   # Tech unclear affects design
        AmbiguityType.DEPENDENCY: 10,  # Dependencies can be clarified later
        AmbiguityType.PRIORITY: 5,     # Priority is lowest impact
    }
    return severity_weights[ambiguity.severity] + type_weights[ambiguity.type]
```

### Answer Format Enum Design

```python
class AnswerFormat(str, Enum):
    """Expected format for user responses to clarification questions."""
    BOOLEAN = "boolean"      # Yes/No questions
    NUMERIC = "numeric"      # Number input (possibly with range)
    CHOICE = "choice"        # Pick from suggestions
    FREE_TEXT = "free_text"  # Open-ended text
    DATE = "date"            # Date/time input
    LIST = "list"            # Multiple items (comma-separated)
```

### Format Hint Examples

| AnswerFormat | format_hint Example |
|--------------|---------------------|
| BOOLEAN | "Answer yes or no" |
| NUMERIC | "Enter a number (e.g., 100, 1000)" |
| CHOICE | "Choose from the options above" |
| FREE_TEXT | "Provide a brief description" |
| DATE | "Enter a date (YYYY-MM-DD)" |
| LIST | "Enter items separated by commas" |

### Project Structure Notes

**Files to Modify:**
```
src/yolo_developer/seed/
├── ambiguity.py           # ADD: AnswerFormat, extend ResolutionPrompt, priority functions
└── __init__.py            # UPDATE: Export new types

src/yolo_developer/cli/
└── commands/seed.py       # UPDATE: Display format hints, prioritized ordering
```

**Files to Create:**
```
tests/unit/seed/
└── test_question_generation.py  # NEW: Unit tests for question enhancement
```

### Dependencies

**Depends On:**
- Story 4.3 (Ambiguity Detection) - **COMPLETED**
  - Uses `Ambiguity`, `ResolutionPrompt`, `AmbiguityResult` types
  - Uses `detect_ambiguities()` function
  - Uses CLI interactive mode patterns

**Downstream Dependencies:**
- Story 4.5 (SOP Constraint Validation) - Will use prioritized questions
- Story 4.6 (Semantic Validation Reports) - Will include question metadata
- Story 12.3 (yolo seed Command Full Features) - Will extend question handling

### External Dependencies

- **litellm** (installed) - LLM provider abstraction
- **rich** (installed) - Format-aware prompts
- **typer** (installed) - CLI handling
- No new dependencies required

### References

- [Source: architecture.md#FR5] - Clarification question generation
- [Source: epics.md#Story-4.4] - Story requirements
- [Source: Story 4.3] - Ambiguity detection patterns
- [Source: seed/ambiguity.py] - Existing ResolutionPrompt, Ambiguity types
- [Source: cli/commands/seed.py] - Interactive mode patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 71 tests pass (50 unit + 21 integration) after code review fixes
- mypy strict mode: 0 issues
- ruff check: All checks passed

### Completion Notes List

1. Implemented `AnswerFormat` enum with 6 response types (BOOLEAN, NUMERIC, CHOICE, FREE_TEXT, DATE, LIST)
2. Extended `ResolutionPrompt` dataclass with `answer_format`, `format_hint`, and `validation_pattern` fields
3. Implemented `validate_question_quality()` function that detects vague phrases and checks length
4. Implemented priority scoring algorithm: `calculate_question_priority()` and `prioritize_questions()`
5. Updated CLI to display format hints and priority indicators in interactive mode
6. Enhanced `AmbiguityResult` with `prioritized_ambiguities`, `get_highest_priority_ambiguity()`, `get_priority_score()`, and updated `to_dict()` with priority_scores
7. Updated LLM prompt to request answer_format and added format hint generation
8. Maintained full backward compatibility with existing code
9. Exported new functions from `seed/__init__.py`

### Code Review Fixes (2026-01-08)

**Issues Found (4):**

1. **HIGH - validate_question_quality() not used**: Function was exported but never called
   - Fixed: Added call in `_parse_ambiguity_response()` with warning logging for low-quality questions

2. **MEDIUM - No unit tests for _validate_format_response()**: 60-line function with zero test coverage
   - Fixed: Added 12 new unit tests in `TestValidateFormatResponse` class covering BOOLEAN, NUMERIC, DATE normalization

3. **MEDIUM - Silent skip on mismatched prompts**: Ambiguities without prompts were silently skipped
   - Fixed: Added warning log in interactive mode when prompt is None

4. **LOW - Priority display threshold inconsistency**: Blocking types (UNDEFINED, SCOPE) at MEDIUM severity showed same as advisory types at HIGH
   - Fixed: Added `is_blocking` check to boost priority display for blocking types with score >= 40

### File List

**Modified Files:**
- `src/yolo_developer/seed/ambiguity.py` - Core implementation + code review fix (validate_question_quality call)
- `src/yolo_developer/seed/__init__.py` - Updated exports
- `src/yolo_developer/cli/commands/seed.py` - CLI format hints, priority display + code review fixes (logging, blocking types)

**New Files:**
- `tests/unit/seed/test_question_generation.py` - 50 unit tests (38 original + 12 code review fixes)

**Updated Test Files:**
- `tests/integration/test_cli_seed.py` - Added 6 integration tests for Story 4.4
