# Story 4.2: CLI Seed Command Implementation

Status: done

## Story

As a developer,
I want to provide seeds via CLI command,
So that I can easily feed requirements to the system.

## Acceptance Criteria

1. **AC1: File-Based Seed Input**
   - **Given** I have a seed document file
   - **When** I run `yolo seed requirements.md`
   - **Then** the file is read and processed
   - **And** the seed content is parsed using the `parse_seed()` API
   - **And** the file path is validated before reading

2. **AC2: Parse Results Display**
   - **Given** a seed file has been successfully parsed
   - **When** parsing completes
   - **Then** parsing results are displayed in a formatted table
   - **And** goals are listed with titles and priorities
   - **And** features are listed with names and descriptions
   - **And** constraints are listed with categories and impacts

3. **AC3: Error Display**
   - **Given** a seed file that cannot be read or parsed
   - **When** an error occurs
   - **Then** errors are clearly shown with Rich formatting
   - **And** error messages explain what went wrong
   - **And** suggestions for resolution are provided when possible

4. **AC4: Graceful File Error Handling**
   - **Given** a file path that doesn't exist or can't be read
   - **When** the command is executed
   - **Then** a helpful error message is shown (not a stack trace)
   - **And** the exit code is non-zero
   - **And** the error explains what file was expected

5. **AC5: Verbose Output Option**
   - **Given** the user wants detailed output
   - **When** I run `yolo seed requirements.md --verbose`
   - **Then** additional details are shown (confidence scores, metadata)
   - **And** the raw content length and source format are displayed

6. **AC6: JSON Output Option**
   - **Given** the user wants machine-readable output
   - **When** I run `yolo seed requirements.md --json`
   - **Then** the parse result is output as JSON
   - **And** the JSON structure matches `SeedParseResult.to_dict()`

## Tasks / Subtasks

- [x] Task 1: Create Seed Command Module (AC: 1)
  - [x] Create `src/yolo_developer/cli/commands/seed.py` module
  - [x] Define `seed_command(file_path: Path, verbose: bool, json_output: bool)` function
  - [x] Add structlog logging for command execution
  - [x] Use `from __future__ import annotations` for type hints

- [x] Task 2: Implement File Reading Logic (AC: 1, 4)
  - [x] Validate file path exists with `Path.exists()`
  - [x] Validate file is readable with `Path.is_file()`
  - [x] Read file content with proper encoding detection (UTF-8 default)
  - [x] Handle file reading errors with `typer.Exit(code=1)`
  - [x] Display file size and path confirmation

- [x] Task 3: Integrate parse_seed API (AC: 1, 2)
  - [x] Import `parse_seed` from `yolo_developer.seed`
  - [x] Call `await parse_seed(content, filename=file_path.name)`
  - [x] Handle async call using `asyncio.run()` in CLI context
  - [x] Capture `SeedParseResult` for display

- [x] Task 4: Implement Rich Display Formatting (AC: 2, 5)
  - [x] Create `_display_parse_results(result: SeedParseResult, verbose: bool)` helper
  - [x] Display summary panel with counts (goals, features, constraints)
  - [x] Create Rich Table for goals with columns: Title, Priority, Rationale
  - [x] Create Rich Table for features with columns: Name, Description, User Value
  - [x] Create Rich Table for constraints with columns: Category, Description, Impact
  - [x] Add confidence scores and metadata in verbose mode

- [x] Task 5: Implement JSON Output (AC: 6)
  - [x] Create `_output_json(result: SeedParseResult)` helper
  - [x] Call `result.to_dict()` for serialization
  - [x] Use `rich.print_json()` for formatted JSON output
  - [x] Suppress other output when JSON mode is enabled

- [x] Task 6: Implement Error Handling (AC: 3, 4)
  - [x] Catch `FileNotFoundError` with helpful message
  - [x] Catch `PermissionError` with access denied message
  - [x] Catch `UnicodeDecodeError` with encoding suggestion
  - [x] Catch parsing errors from `parse_seed()` with details
  - [x] Use `console.print("[red]Error:[/red] ...")` for error display

- [x] Task 7: Register Command in CLI App (AC: all)
  - [x] Import `seed_command` in `cli/main.py`
  - [x] Add `@app.command("seed")` with all options
  - [x] Define `--verbose` / `-v` flag (default: False)
  - [x] Define `--json` / `-j` flag (default: False)
  - [x] Add comprehensive help text

- [x] Task 8: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/cli/test_seed_command.py`
  - [x] Test file reading with valid file
  - [x] Test file reading with non-existent file
  - [x] Test file reading with permission error (mock)
  - [x] Test display formatting with mock SeedParseResult
  - [x] Test JSON output mode
  - [x] Test verbose mode

- [x] Task 9: Write Integration Tests (AC: all)
  - [x] Create `tests/integration/test_cli_seed.py`
  - [x] Test full CLI invocation with `typer.testing.CliRunner`
  - [x] Test with sample fixtures from `tests/fixtures/seeds/`
  - [x] Test exit codes for success and failure cases
  - [x] Mock LLM calls to avoid API costs

- [x] Task 10: Update Exports and Documentation (AC: all)
  - [x] Export `seed_command` from `cli/commands/__init__.py`
  - [x] Update `cli/main.py` imports
  - [x] Add docstrings with usage examples
  - [x] Verify `yolo seed --help` displays correctly

## Dev Notes

### Architecture Compliance

- **ADR-005 (CLI Framework):** Use Typer + Rich per architecture specification
- **ADR-008 (Configuration):** Integrate with config loader for any settings
- **FR99:** Users can provide seed documents via `yolo seed` command
- [Source: architecture.md#CLI Interface] - `cli/` module handles FR98-105

### Technical Requirements

- **Typer Patterns:** Use `typer.Argument` for positional args, `typer.Option` for flags
- **Rich Console:** Use shared `Console()` instance from `cli/main.py`
- **Async Handling:** Use `asyncio.run()` to call async `parse_seed()` function
- **Error Exit Codes:** Use `raise typer.Exit(code=1)` for errors
- **Structured Logging:** Use structlog for all command events

### Existing Pattern References

**From cli/main.py (Typer App Pattern):**
```python
app = typer.Typer(
    name="yolo",
    help="YOLO Developer - Autonomous multi-agent AI development system",
    no_args_is_help=True,
)
console = Console()

@app.command("init")
def init(
    path: str | None = typer.Argument(
        None,
        help="Directory to initialize the project in. Defaults to current directory.",
    ),
    # ... options with typer.Option
) -> None:
    """Command docstring becomes help text."""
    init_command(path=path, ...)
```

**From cli/commands/init.py (Command Implementation Pattern):**
```python
from __future__ import annotations

from pathlib import Path
import structlog
from rich.console import Console
from rich.panel import Panel

logger = structlog.get_logger(__name__)
console = Console()

def init_command(
    path: str | None = None,
    name: str | None = None,
    # ... parameters
) -> None:
    """Initialize command implementation."""
    logger.info("init_command_started", path=path, name=name)
    # ... implementation
```

**From seed/__init__.py (parse_seed API):**
```python
from yolo_developer.seed import parse_seed, SeedSource

# parse_seed is async
result = await parse_seed("Build an e-commerce platform with auth")
print(f"Found {result.goal_count} goals, {result.feature_count} features")

# Parse from file with filename hint
with open("requirements.md") as f:
    content = f.read()
result = await parse_seed(content, filename="requirements.md")
```

**From seed/types.py (SeedParseResult to_dict):**
```python
@dataclass(frozen=True)
class SeedParseResult:
    goals: tuple[SeedGoal, ...]
    features: tuple[SeedFeature, ...]
    constraints: tuple[SeedConstraint, ...]
    raw_content: str
    source: SeedSource
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goals": [g.to_dict() for g in self.goals],
            "features": [f.to_dict() for f in self.features],
            "constraints": [c.to_dict() for c in self.constraints],
            "raw_content": self.raw_content,
            "source": self.source.value,
            "metadata": self.metadata,
        }

    @property
    def goal_count(self) -> int:
        return len(self.goals)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def constraint_count(self) -> int:
        return len(self.constraints)
```

### Project Structure Notes

**CLI Module Location:**
```
src/yolo_developer/cli/
├── __init__.py              # Existing
├── main.py                  # UPDATE: Add seed command registration
└── commands/
    ├── __init__.py          # UPDATE: Export seed_command
    ├── init.py              # Existing init command
    └── seed.py              # NEW: Seed command implementation
```

**Test Structure:**
```
tests/
├── fixtures/seeds/          # Existing from Story 4.1
│   ├── simple_seed.txt
│   ├── complex_seed.md
│   └── edge_case_seed.txt
├── unit/cli/
│   ├── __init__.py          # NEW
│   └── test_seed_command.py # NEW: Seed command unit tests
└── integration/
    ├── test_seed_parsing.py # Existing from Story 4.1
    └── test_cli_seed.py     # NEW: CLI integration tests
```

### Previous Story Learnings (Story 4.1)

**Critical Learnings to Apply:**

1. **Async Handling:** `parse_seed()` is async - must use `asyncio.run()` in synchronous CLI context
2. **Mock LLM Calls:** Use `patch.object(LLMSeedParser, "_call_llm")` pattern for tests
3. **Export Updates:** Update `commands/__init__.py` immediately when adding new command
4. **Rich Formatting:** Use Rich Panel, Table, Console for consistent output
5. **Hidden Bug Risk:** Test with real prompt formatting (not just mocked) to catch template issues

**Code Review Issues to Avoid:**
- Don't forget to test error paths (file not found, permission errors)
- Include tests for all output modes (normal, verbose, JSON)
- Use proper exit codes for error conditions

### Git Intelligence (Recent Commits)

**Story 4.1 Commit (f35dd83) Patterns:**
- Comprehensive commit message with bullet points
- Code review fixes listed separately
- Test count mentioned in commit message
- Files organized logically (types, parser, api, tests)

**Testing Patterns from Story 4.1:**
- 126 tests covering unit, integration
- Mock LLM calls with `patch.object`
- Fixtures in `tests/fixtures/seeds/`
- Test edge cases explicitly

### Testing Standards

- Use `typer.testing.CliRunner` for CLI integration tests
- Mock `parse_seed()` in unit tests to isolate CLI logic
- Test with real fixtures from `tests/fixtures/seeds/`
- Verify exit codes: 0 for success, 1 for errors
- Test output formatting with `result.output` assertions
- Mock LLM calls to avoid API costs

### Implementation Approach

1. **Command Module:** Create `seed.py` with command implementation
2. **File Handling:** Implement file reading with error handling
3. **API Integration:** Connect to `parse_seed()` with async handling
4. **Display Formatting:** Create Rich output helpers
5. **JSON Mode:** Implement alternative output format
6. **Error Handling:** Comprehensive error catching and display
7. **CLI Registration:** Add command to main app
8. **Testing:** Unit tests per helper, integration tests for full flow
9. **Exports:** Update all `__init__.py` files

### Dependencies

**Depends On:**
- Story 4.1 (Parse Natural Language Seed Documents) - **COMPLETED**
  - Uses `parse_seed()` API from `yolo_developer.seed`
  - Uses `SeedParseResult`, `SeedGoal`, `SeedFeature`, `SeedConstraint` types
  - Uses existing fixtures in `tests/fixtures/seeds/`

**Downstream Dependencies:**
- Story 4.3 (Ambiguity Detection) - Will extend seed command with validation
- Story 12.3 (yolo seed Command Full Features) - Will add additional options

### External Dependencies

- **typer** (installed) - CLI framework
- **rich** (installed) - Terminal formatting
- **asyncio** (stdlib) - Async handling in sync CLI context

### References

- [Source: architecture.md#ADR-005] - CLI Framework: Typer + Rich
- [Source: architecture.md#CLI Interface] - cli/ module structure (FR98-105)
- [Source: architecture.md#Packaging] - Entry point: `yolo` command
- [Source: prd.md#FR99] - Users can provide seed documents via `yolo seed` command
- [Source: epics.md#Story-4.2] - CLI Seed Command Implementation requirements
- [Source: Story 4.1] - parse_seed() API, SeedParseResult types
- [Source: cli/main.py] - Existing CLI app and command patterns
- [Source: cli/commands/init.py] - Command implementation patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

1. **Seed Command Implementation:** Created comprehensive `seed.py` module with file reading, display formatting, JSON output, and error handling. The command follows existing CLI patterns from `init_command`.

2. **Rich Display Formatting:** Implemented `_display_parse_results()` helper with Rich Panel for summary and Rich Tables for goals, features, and constraints. Verbose mode shows additional metadata like source type and content length.

3. **Integration Test Fix:** Fixed mocking issue in integration tests - `LLMSeedParser._call_llm` returns a parsed dict (not JSON string), and the parser catches exceptions internally returning empty results. Changed test to mock `parse_seed` directly for CLI error handling validation.

4. **Test Coverage:** 37 total tests (22 unit tests + 15 integration tests) covering:
   - File reading with various error conditions
   - Display formatting for normal and verbose modes
   - JSON output structure validation
   - CLI argument parsing and help display
   - Exit codes for success and failure cases
   - Edge cases (unicode, special characters, empty results)

5. **Async Handling:** Used `asyncio.run()` pattern to call async `parse_seed()` from synchronous CLI context, consistent with project patterns.

6. **AC5 Clarification (Confidence Scores):** The AC mentions "confidence scores" in verbose mode, but the data model from Story 4.1 (`SeedGoal`, `SeedFeature`, `SeedConstraint`) does not include confidence scores - only the intermediate `SeedComponent` type has them. Verbose mode shows source type, content length, metadata keys, and detailed goal/feature information. If per-item confidence scores are needed, the data model in Story 4.1 would need to be extended.

7. **Code Review Fix:** Fixed RuntimeWarning in `test_seed_command_parsing_error` by mocking `parse_seed` directly instead of `asyncio.run`, preventing unawaited coroutine warnings.

### File List

**New Files:**
- `src/yolo_developer/cli/commands/seed.py` - Main seed command implementation
- `tests/unit/cli/__init__.py` - CLI unit tests package init
- `tests/unit/cli/test_seed_command.py` - 22 unit tests for seed command
- `tests/integration/test_cli_seed.py` - 15 integration tests for CLI

**Modified Files:**
- `src/yolo_developer/cli/commands/__init__.py` - Added `seed_command` export
- `src/yolo_developer/cli/main.py` - Registered `seed` command with Typer app
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status
- `tests/unit/cli/test_seed_command.py` - Fixed RuntimeWarning in parsing error test (code review)

