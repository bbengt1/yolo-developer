# Story 4.3: Ambiguity Detection & Resolution Prompts

Status: done

## Story

As an Analyst Agent,
I want seeds flagged for ambiguities,
So that I can request clarification before proceeding.

## Acceptance Criteria

1. **AC1: Ambiguity Detection**
   - **Given** a seed with vague requirements
   - **When** analyzed via `detect_ambiguities()`
   - **Then** ambiguous phrases are identified and listed
   - **And** each ambiguity includes the source text and location
   - **And** ambiguity types are categorized (scope, technical, priority, dependency)

2. **AC2: Resolution Prompts**
   - **Given** identified ambiguities
   - **When** flagged via ambiguity detection
   - **Then** specific clarification questions are generated for each
   - **And** questions are actionable and context-aware
   - **And** suggested resolution options are provided where applicable

3. **AC3: User Interaction Flow**
   - **Given** resolution prompts
   - **When** displayed to user via CLI (`yolo seed --interactive`)
   - **Then** user can provide clarifications
   - **And** clarifications update the seed context
   - **And** parsing can be re-run with clarified context

4. **AC4: Multiple Ambiguity Handling**
   - **Given** a seed with multiple ambiguities
   - **When** processed
   - **Then** all ambiguities are detected and listed
   - **And** user can resolve individually or skip
   - **And** unresolved ambiguities are flagged in output

5. **AC5: Confidence Integration**
   - **Given** ambiguity detection results
   - **When** combined with parsing
   - **Then** `SeedParseResult` includes ambiguity metadata
   - **And** overall confidence reflects ambiguity count/severity
   - **And** goals/features with ambiguities are flagged

## Tasks / Subtasks

- [x] Task 1: Create Ambiguity Types (AC: 1, 5)
  - [x] Create `src/yolo_developer/seed/ambiguity.py` module
  - [x] Define `AmbiguityType` enum (SCOPE, TECHNICAL, PRIORITY, DEPENDENCY, UNDEFINED)
  - [x] Define `AmbiguitySeverity` enum (LOW, MEDIUM, HIGH)
  - [x] Define `Ambiguity` dataclass with fields: type, severity, source_text, location, description
  - [x] Define `ResolutionPrompt` dataclass with fields: question, suggestions, default
  - [x] Define `AmbiguityResult` dataclass with fields: ambiguities, overall_confidence, resolution_prompts
  - [x] Use `from __future__ import annotations` and frozen dataclasses

- [x] Task 2: Implement LLM Ambiguity Detection (AC: 1, 2)
  - [x] Create `AMBIGUITY_DETECTION_PROMPT` template in `ambiguity.py`
  - [x] Implement `async def detect_ambiguities(content: str) -> AmbiguityResult`
  - [x] Use LiteLLM with structured JSON output
  - [x] Parse LLM response into `Ambiguity` objects
  - [x] Generate `ResolutionPrompt` for each ambiguity
  - [x] Add tenacity retry decorator (consistent with parser.py)
  - [x] Add structlog logging for all operations

- [x] Task 3: Implement Confidence Scoring (AC: 5)
  - [x] Create `calculate_ambiguity_confidence()` function
  - [x] Score based on: count of ambiguities, severity distribution
  - [x] HIGH severity = -0.15 confidence per item
  - [x] MEDIUM severity = -0.10 confidence per item
  - [x] LOW severity = -0.05 confidence per item
  - [x] Minimum confidence floor at 0.1
  - [x] Return confidence score (0.0-1.0)

- [x] Task 4: Integrate with SeedParseResult (AC: 5)
  - [x] Add `ambiguities: tuple[Ambiguity, ...]` field to `SeedParseResult`
  - [x] Add `ambiguity_confidence: float` field to `SeedParseResult`
  - [x] Update `SeedParseResult.to_dict()` to include ambiguity data
  - [x] Add `has_ambiguities` property to `SeedParseResult`
  - [x] Update `parse_seed()` in `api.py` to optionally run ambiguity detection

- [x] Task 5: Create Resolution Context (AC: 3)
  - [x] Define `Resolution` dataclass: ambiguity_id, user_response, timestamp
  - [x] Define `SeedContext` dataclass: original_content, resolutions, clarified_content
  - [x] Implement `_apply_resolutions_to_content()` in CLI (alternative to standalone function)
  - [x] Track which ambiguities have been resolved

- [x] Task 6: Implement CLI Interactive Mode (AC: 3, 4)
  - [x] Add `--interactive` / `-i` flag to `yolo seed` command
  - [x] Create `_prompt_for_resolution(ambiguity: Ambiguity, prompt: ResolutionPrompt)` helper
  - [x] Use Rich `Prompt.ask()` for user input
  - [x] Display ambiguity context with Rich Panel
  - [x] Allow "skip" option for each ambiguity
  - [x] Re-run parsing after resolutions applied

- [x] Task 7: Update Seed Command Display (AC: 1, 4)
  - [x] Add ambiguity summary section to normal output
  - [x] Create Rich Table for ambiguities: Type, Severity, Description
  - [x] Show resolution prompts in verbose mode
  - [x] Include ambiguity data in JSON output mode
  - [x] Display overall confidence with ambiguity impact

- [x] Task 8: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/seed/test_ambiguity.py`
  - [x] Test `AmbiguityType`, `AmbiguitySeverity` enums
  - [x] Test `detect_ambiguities()` with mock LLM
  - [x] Test `calculate_ambiguity_confidence()` scoring
  - [x] Test CLI ambiguity display functions
  - [x] Test with seeds having 0, 1, and multiple ambiguities
  - [x] Test edge cases: empty content, all ambiguous, no resolutions

- [x] Task 9: Write Integration Tests (AC: all)
  - [x] Create `tests/integration/test_ambiguity_detection.py`
  - [x] Test full flow: parse → detect ambiguities → resolve → re-parse
  - [x] Test CLI interactive mode with mock input
  - [x] Test `parse_seed()` with `detect_ambiguities=True`
  - [x] Test ambiguity data in JSON output
  - [x] Mock LLM calls to avoid API costs

- [x] Task 10: Update Exports and Documentation (AC: all)
  - [x] Export new types from `seed/__init__.py`
  - [x] Export `detect_ambiguities` from `seed/__init__.py`
  - [x] Update CLI command help text
  - [x] Add docstrings with usage examples

## Dev Notes

### Architecture Compliance

- **ADR-003 (LLM Abstraction):** Use LiteLLM for ambiguity detection (consistent with parser.py)
- **ADR-005 (CLI Framework):** Use Typer + Rich for interactive prompts
- **ADR-008 (Configuration):** Ambiguity detection can be configured via `yolo.yaml`
- **FR3:** Semantic validation of seeds (ambiguity is semantic validation)
- **FR4:** Interactive mode for disambiguation prompts
- [Source: architecture.md#Seed Input] - `seed/` module handles FR1-8

### Technical Requirements

- **LiteLLM Pattern:** Follow `LLMSeedParser._call_llm()` pattern exactly
- **Retry Pattern:** Use same tenacity configuration as `parser.py`
- **Structured Output:** Request JSON from LLM, parse with error handling
- **Rich Prompts:** Use `rich.prompt.Prompt` for interactive input
- **Async Pattern:** All LLM calls must be async
- **Immutable Types:** Use frozen dataclasses for all new types

### Existing Pattern References

**From seed/parser.py (LLM Call Pattern):**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, KeyError)),
    reraise=True,
)
async def _call_llm(self, content: str) -> dict[str, Any]:
    prompt = SEED_ANALYSIS_PROMPT.format(content=content)
    response = await litellm.acompletion(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=self.temperature,
    )
    response_text = response.choices[0].message.content
    # JSON extraction logic...
    return json.loads(json_text)
```

**From seed/types.py (Frozen Dataclass Pattern):**
```python
@dataclass(frozen=True)
class SeedGoal:
    title: str
    description: str
    priority: int
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "rationale": self.rationale,
        }
```

**From cli/commands/seed.py (Rich Display Pattern):**
```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

# Table display
table = Table(title="Ambiguities Detected")
table.add_column("Type", style="cyan")
table.add_column("Severity", style="yellow")
table.add_column("Description")
console.print(table)

# Interactive prompt
response = Prompt.ask(
    "[bold]How should we interpret this?[/bold]",
    choices=["option1", "option2", "skip"],
    default="skip",
)
```

### Project Structure Notes

**New Files:**
```
src/yolo_developer/seed/
├── ambiguity.py           # NEW: Ambiguity types and detection
└── __init__.py            # UPDATE: Export new types

tests/
├── unit/seed/
│   └── test_ambiguity.py  # NEW: Ambiguity unit tests
└── integration/
    └── test_ambiguity_detection.py  # NEW: Integration tests
```

**Modified Files:**
```
src/yolo_developer/seed/
├── types.py               # UPDATE: Add ambiguity fields to SeedParseResult
├── api.py                 # UPDATE: Add detect_ambiguities parameter
└── __init__.py            # UPDATE: Export new types

src/yolo_developer/cli/
├── commands/seed.py       # UPDATE: Add --interactive flag, ambiguity display
└── main.py                # UPDATE: Register new options
```

### Ambiguity Detection Prompt Design

```python
AMBIGUITY_DETECTION_PROMPT = """You are a requirements analyst. Analyze the following seed document for ambiguities that could cause implementation confusion.

Identify ambiguities in these categories:
1. **SCOPE**: Unclear boundaries (e.g., "handle all edge cases", "support many users")
2. **TECHNICAL**: Vague technical requirements (e.g., "fast", "scalable", "modern")
3. **PRIORITY**: Unclear importance (e.g., "nice to have", "should", "ideally")
4. **DEPENDENCY**: Unclear relationships (e.g., "integrate with the system", "work with existing")
5. **UNDEFINED**: Missing critical details (e.g., no error handling mentioned, no auth specified)

For each ambiguity, provide:
- type: One of SCOPE, TECHNICAL, PRIORITY, DEPENDENCY, UNDEFINED
- severity: HIGH (blocks implementation), MEDIUM (causes confusion), LOW (minor clarification)
- source_text: The exact ambiguous phrase from the document
- location: Line number or section where found
- description: Why this is ambiguous
- question: A specific question to clarify this ambiguity
- suggestions: 2-3 possible interpretations or answers

Output as JSON:
{{
  "ambiguities": [
    {{
      "type": "SCOPE|TECHNICAL|PRIORITY|DEPENDENCY|UNDEFINED",
      "severity": "HIGH|MEDIUM|LOW",
      "source_text": "exact phrase",
      "location": "line X or section name",
      "description": "why this is ambiguous",
      "question": "clarification question",
      "suggestions": ["option 1", "option 2", "option 3"]
    }}
  ]
}}

Seed Document:
---
{content}
---

Respond ONLY with the JSON object."""
```

### Previous Story Learnings (Story 4.1, 4.2)

**Critical Learnings to Apply:**

1. **Async Handling:** All LLM calls must use `await litellm.acompletion()`
2. **JSON Extraction:** Handle LLM wrapping JSON in markdown code blocks
3. **Mock LLM Calls:** Use `patch("yolo_developer.seed.ambiguity.litellm.acompletion")` pattern
4. **Export Updates:** Immediately update `__init__.py` when adding new public types
5. **Rich Formatting:** Test Rich output by checking `result.output` contains expected strings
6. **Exit Codes:** Use `raise typer.Exit(code=1)` for errors, not exceptions
7. **Filterwarnings:** May need `@pytest.mark.filterwarnings("ignore::RuntimeWarning")` for async mocks

**Code Review Issues to Avoid:**
- Don't forget to test edge cases (empty content, no ambiguities, all high severity)
- Include tests for JSON output mode with ambiguities
- Test interactive mode with both resolution and skip paths
- Ensure confidence calculations don't go below 0.0 or above 1.0

### Git Intelligence (Recent Commits)

**Story 4.2 Commit (5eb9241) Patterns:**
- Comprehensive commit message with bullet points
- Code review fixes listed separately
- Test count mentioned in commit message
- Files organized: types → implementation → tests → exports

### Testing Standards

- Use `pytest-asyncio` for async tests
- Mock `litellm.acompletion` with `AsyncMock`
- Create fixtures for seeds with known ambiguities
- Test confidence scoring with deterministic inputs
- Use `typer.testing.CliRunner` for CLI tests
- Verify exit codes: 0 for success, 1 for errors

### Implementation Approach

1. **Types First:** Create all dataclasses in `ambiguity.py`
2. **LLM Integration:** Implement `detect_ambiguities()` with mocked tests
3. **Confidence Scoring:** Add scoring function with unit tests
4. **SeedParseResult Integration:** Extend types.py carefully
5. **CLI Interactive:** Add flag and prompts with Rich
6. **Display Updates:** Integrate ambiguities into normal/verbose/JSON output
7. **Exports:** Update all `__init__.py` files

### Dependencies

**Depends On:**
- Story 4.1 (Parse Natural Language Seed Documents) - **COMPLETED**
  - Uses `SeedParseResult` type
  - Uses `LLMSeedParser` pattern
  - Uses `parse_seed()` API
- Story 4.2 (CLI Seed Command Implementation) - **COMPLETED**
  - Uses `seed_command()` function
  - Uses Rich display patterns
  - Uses CLI flag patterns

**Downstream Dependencies:**
- Story 4.4 (Seed Validation Rules) - Will use ambiguity results
- Story 12.3 (yolo seed Command Full Features) - Will extend interactive mode

### External Dependencies

- **litellm** (installed) - LLM provider abstraction
- **rich** (installed) - Rich prompts for interactive mode
- **typer** (installed) - CLI flag handling
- **tenacity** (installed) - Retry logic

### References

- [Source: architecture.md#FR3] - Semantic validation of seeds
- [Source: architecture.md#FR4] - Interactive mode for disambiguation
- [Source: architecture.md#ADR-003] - LLM Provider Abstraction (LiteLLM)
- [Source: architecture.md#ADR-005] - CLI Framework (Typer + Rich)
- [Source: epics.md#Story-4.3] - Ambiguity Detection requirements
- [Source: Story 4.1] - parse_seed() API, LLM patterns
- [Source: Story 4.2] - CLI seed command, Rich display patterns
- [Source: seed/parser.py] - LLM call patterns, retry configuration
- [Source: seed/types.py] - Frozen dataclass patterns

## Dev Agent Record

### Agent Model Used

Claude (Anthropic) - Claude Code CLI

### Debug Log References

- Tests verified: 51 passing (30 unit + 21 integration)
- Code review identified story file tracking issues (tasks not marked complete)

### Completion Notes List

- Implemented full ambiguity detection system with LLM integration
- Created `AmbiguityType`, `AmbiguitySeverity` enums and `Ambiguity`, `ResolutionPrompt`, `AmbiguityResult`, `Resolution`, `SeedContext` dataclasses
- Implemented `detect_ambiguities()` async function with tenacity retry
- Implemented `calculate_ambiguity_confidence()` scoring function
- Integrated ambiguity fields into `SeedParseResult` (`ambiguities`, `ambiguity_confidence`, `has_ambiguities`, `ambiguity_count`)
- Added `--interactive` / `-i` flag to CLI seed command
- Implemented Rich-based interactive resolution prompts
- Added ambiguity display tables and JSON output support
- All exports updated in `__init__.py`

### File List

**New Files:**
- `src/yolo_developer/seed/ambiguity.py` - Ambiguity types, detection, and resolution (586 lines)
- `tests/unit/seed/test_ambiguity.py` - Unit tests for ambiguity module (30 tests)
- `tests/integration/test_ambiguity_detection.py` - Integration tests (21 tests)

**Modified Files:**
- `src/yolo_developer/seed/types.py` - Added ambiguity fields to SeedParseResult
- `src/yolo_developer/seed/api.py` - Added detect_ambiguities parameter to parse_seed()
- `src/yolo_developer/seed/__init__.py` - Exported new types and functions
- `src/yolo_developer/cli/commands/seed.py` - Added interactive mode and ambiguity display
- `src/yolo_developer/cli/main.py` - CLI flag registration
- `src/yolo_developer/cli/commands/__init__.py` - Command updates
- `tests/integration/test_seed_command.py` - Updated CLI tests
