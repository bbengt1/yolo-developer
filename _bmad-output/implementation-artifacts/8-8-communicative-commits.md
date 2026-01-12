# Story 8.8: Communicative Commits

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want commit messages that explain the "why",
So that git history is useful for understanding changes.

## Acceptance Criteria

1. **AC1: Generate Purpose-Explaining Commit Messages**
   - **Given** code changes to commit
   - **When** a commit message is generated via `generate_commit_message()`
   - **Then** the message explains the purpose/intent of the change (the "why")
   - **And** the message follows conventional commit format: `<type>(<scope>): <description>`
   - **And** supported types include: feat, fix, refactor, test, docs, chore, style
   - **And** scope is optional but recommended for multi-module changes
   - **And** description is imperative mood, lowercase, max 50 characters

2. **AC2: Story Reference Integration**
   - **Given** code changes associated with a story
   - **When** a commit message is generated
   - **Then** the message body references the story ID being implemented
   - **And** the format follows: `Story: <story_id>` in the message body
   - **And** story context (title, summary) is included when available
   - **And** multiple story references are supported for cross-cutting changes

3. **AC3: Decision Rationale Inclusion**
   - **Given** code changes with associated design decisions
   - **When** a commit message is generated
   - **Then** key decision rationale is included in the commit body
   - **And** architecture pattern choices are documented
   - **And** trade-off explanations are included for non-obvious choices
   - **And** decision references link to ADRs when applicable

4. **AC4: Concise But Informative Messages**
   - **Given** commit message generation
   - **When** `generate_commit_message()` formats the message
   - **Then** subject line is concise (max 50 chars)
   - **And** body provides sufficient context without being verbose
   - **And** bullet points are used for multiple changes
   - **And** total message length is reasonable (< 500 chars recommended)

5. **AC5: LLM-Powered Message Generation**
   - **Given** implementation artifacts and story context
   - **When** LLM generates a commit message
   - **Then** `generate_commit_message_with_llm()` uses "routine" tier model
   - **And** prompt includes code diff summary, story context, and decisions
   - **And** retry logic handles LLM failures (max 2 retries)
   - **And** fallback to template-based generation on failure

6. **AC6: Commit Message Validation**
   - **Given** a generated commit message
   - **When** `validate_commit_message()` checks the message
   - **Then** conventional commit format is verified
   - **And** subject line length is checked (max 50 chars warning, 72 hard limit)
   - **And** body formatting is validated (blank line after subject, line wrap at 72)
   - **And** `CommitMessageValidationResult` with passed, warnings, errors is returned

7. **AC7: Integration with DevOutput**
   - **Given** dev_node generates implementation artifacts
   - **When** implementation is complete
   - **Then** `suggested_commit_message` is included in DevOutput
   - **And** commit message context includes all implementation artifacts
   - **And** commit message references all stories implemented in the batch
   - **And** decision record includes commit message generation rationale

8. **AC8: Exports from Dev Module**
   - **Given** new commit message utilities
   - **When** importing from `yolo_developer.agents.dev`
   - **Then** `generate_commit_message` is exported
   - **And** `generate_commit_message_with_llm` is exported
   - **And** `validate_commit_message` is exported
   - **And** `CommitMessageValidationResult` is exported
   - **And** `CommitMessageContext` is exported

## Tasks / Subtasks

- [x] Task 1: Create Commit Message Types (AC: 6, 8)
  - [x] Create `src/yolo_developer/agents/dev/commit_utils.py`
  - [x] Create `CommitMessageContext` dataclass with fields: story_ids, story_titles, decisions, code_summary, files_changed, change_type
  - [x] Create `CommitMessageValidationResult` dataclass with fields: passed, subject_line, body_lines, warnings, errors
  - [x] Add conventional commit type enum: feat, fix, refactor, test, docs, chore, style
  - [x] Add type annotations for all new types

- [x] Task 2: Implement Template-Based Message Generation (AC: 1, 2, 3, 4)
  - [x] Create `generate_commit_message(context: CommitMessageContext) -> str`
  - [x] Implement subject line generation with conventional commit format
  - [x] Implement body generation with story reference
  - [x] Include decision rationale in body when available
  - [x] Ensure concise output with length limits

- [x] Task 3: Implement Commit Message Validation (AC: 6)
  - [x] Create `validate_commit_message(message: str) -> CommitMessageValidationResult`
  - [x] Validate conventional commit format in subject
  - [x] Check subject line length (50 char warning, 72 char error)
  - [x] Validate body formatting (blank line, line wrap at 72)
  - [x] Return structured validation result

- [x] Task 4: Create LLM Prompt Template (AC: 5)
  - [x] Create `src/yolo_developer/agents/dev/prompts/commit_message.py`
  - [x] Create `build_commit_message_prompt()` with context parameters
  - [x] Include conventional commit format instructions
  - [x] Include examples of good commit messages
  - [x] Add retry prompt for format corrections

- [x] Task 5: Implement LLM-Powered Generation (AC: 5)
  - [x] Create `generate_commit_message_with_llm(context: CommitMessageContext, router: LLMRouter) -> tuple[str, bool]`
  - [x] Use "routine" tier model per ADR-003
  - [x] Implement retry logic (max 2 retries)
  - [x] Extract commit message from LLM response
  - [x] Fallback to template-based generation on failure

- [x] Task 6: Integrate with Dev Node (AC: 7)
  - [x] Update DevOutput in types.py to add `suggested_commit_message: str | None`
  - [x] Update dev_node in node.py to generate commit messages
  - [x] Build CommitMessageContext from implementations
  - [x] Include commit message in dev_node output
  - [x] Add commit message to decision record rationale

- [x] Task 7: Export Functions from Dev Module (AC: 8)
  - [x] Update `src/yolo_developer/agents/dev/__init__.py`
  - [x] Export `generate_commit_message`
  - [x] Export `generate_commit_message_with_llm`
  - [x] Export `validate_commit_message`
  - [x] Export `CommitMessageValidationResult`
  - [x] Export `CommitMessageContext`

- [x] Task 8: Write Unit Tests for Commit Types (AC: 6, 8)
  - [x] Create `tests/unit/agents/dev/test_commit_utils.py`
  - [x] Test `CommitMessageContext` dataclass construction
  - [x] Test `CommitMessageValidationResult` dataclass construction
  - [x] Test conventional commit type enum values
  - [x] Test serialization methods

- [x] Task 9: Write Unit Tests for Template Generation (AC: 1, 2, 3, 4)
  - [x] Test `generate_commit_message()` with minimal context
  - [x] Test `generate_commit_message()` with story reference
  - [x] Test `generate_commit_message()` with decisions
  - [x] Test subject line length enforcement
  - [x] Test body formatting

- [x] Task 10: Write Unit Tests for Validation (AC: 6)
  - [x] Test `validate_commit_message()` with valid message
  - [x] Test `validate_commit_message()` with invalid format
  - [x] Test `validate_commit_message()` with long subject
  - [x] Test `validate_commit_message()` with missing blank line

- [x] Task 11: Write Unit Tests for LLM Generation (AC: 5)
  - [x] Test `generate_commit_message_with_llm()` success path
  - [x] Test `generate_commit_message_with_llm()` retry logic
  - [x] Test `generate_commit_message_with_llm()` fallback
  - [x] Mock LLMRouter for isolated testing

- [x] Task 12: Write Integration Tests for Dev Node (AC: 7)
  - [x] Create `tests/integration/agents/dev/test_commit_generation.py`
  - [x] Test dev_node includes commit message in output
  - [x] Test commit message references story IDs
  - [x] Test commit message reflects implementation artifacts

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for commit message types
- **ADR-003 (LLM Integration):** Use "routine" tier for commit message generation (cost-effective)
- **ADR-005 (LangGraph Communication):** Commit message included in state updates via DevOutput
- **ADR-006 (Quality Gates):** Commit message validation is advisory (warnings, not blocking)
- **ADR-007 (Error Handling):** Retry with fallback to template-based generation
- **ARCH-QUALITY-6:** Use structlog for all logging with structured fields
- **ARCH-QUALITY-7:** Full type annotations on all functions
- **FR64:** Dev Agent can produce communicative commit messages with decision rationale

### Technical Requirements

- Use `from __future__ import annotations` in all new files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/dev/node.py` (Stories 8.1-8.7)
- Use frozen dataclasses for immutable types
- Use AST parsing for code analysis if needed for diff summary
- Follow conventional commit specification: https://www.conventionalcommits.org/

### Library Versions (from architecture.md)

| Library | Version | Purpose |
|---------|---------|---------|
| LangGraph | 1.0.5 | Orchestration framework |
| structlog | latest | Structured logging |
| tenacity | latest | Retry with backoff |
| pytest | latest | Test framework |
| pytest-asyncio | latest | Async test support |

### Project Structure Notes

**New Files to Create:**
- `src/yolo_developer/agents/dev/commit_utils.py` - Commit message utilities
- `src/yolo_developer/agents/dev/prompts/commit_message.py` - LLM prompt template

**Files to Modify:**
- `src/yolo_developer/agents/dev/__init__.py` - Export new functions and types
- `src/yolo_developer/agents/dev/types.py` - Add suggested_commit_message to DevOutput
- `src/yolo_developer/agents/dev/node.py` - Integrate commit message generation

**Test Files:**
- `tests/unit/agents/dev/test_commit_utils.py`
- `tests/integration/agents/dev/test_commit_generation.py`

### Key Type Definitions

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class CommitType(str, Enum):
    """Conventional commit types."""
    FEAT = "feat"
    FIX = "fix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    CHORE = "chore"
    STYLE = "style"


@dataclass(frozen=True)
class CommitMessageContext:
    """Context for generating a commit message.

    Attributes:
        story_ids: List of story IDs being implemented.
        story_titles: Mapping of story ID to title.
        decisions: List of decision rationales to include.
        code_summary: Brief summary of code changes.
        files_changed: List of file paths changed.
        change_type: Primary type of change (feat, fix, etc.).
        scope: Optional scope for the commit.
    """
    story_ids: tuple[str, ...]
    story_titles: dict[str, str] = field(default_factory=dict)
    decisions: tuple[str, ...] = field(default_factory=tuple)
    code_summary: str = ""
    files_changed: tuple[str, ...] = field(default_factory=tuple)
    change_type: CommitType = CommitType.FEAT
    scope: str | None = None


@dataclass
class CommitMessageValidationResult:
    """Result of commit message validation.

    Attributes:
        passed: Whether validation passed (no errors).
        subject_line: Extracted subject line.
        body_lines: Extracted body lines.
        warnings: List of warning messages.
        errors: List of error messages.
    """
    passed: bool = True
    subject_line: str = ""
    body_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "subject_line": self.subject_line,
            "body_line_count": len(self.body_lines),
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "warnings": self.warnings,
            "errors": self.errors,
        }
```

### Key Function Signatures

```python
def generate_commit_message(context: CommitMessageContext) -> str:
    """Generate a commit message from context using templates.

    Creates a conventional commit message with subject line, story reference,
    and decision rationale. Uses template-based generation (no LLM).

    Args:
        context: CommitMessageContext with story and change information.

    Returns:
        Formatted commit message string.

    Example:
        >>> context = CommitMessageContext(
        ...     story_ids=("8-8",),
        ...     story_titles={"8-8": "Communicative Commits"},
        ...     change_type=CommitType.FEAT,
        ... )
        >>> msg = generate_commit_message(context)
        >>> msg.startswith("feat")
        True
    """


async def generate_commit_message_with_llm(
    context: CommitMessageContext,
    router: LLMRouter,
    max_retries: int = 2,
) -> tuple[str, bool]:
    """Generate a commit message using LLM.

    Uses the "routine" tier model for cost-effective generation.
    Falls back to template-based generation on failure.

    Args:
        context: CommitMessageContext with story and change information.
        router: LLMRouter for making LLM calls.
        max_retries: Maximum retry attempts on format issues.

    Returns:
        Tuple of (message, is_valid). Message is the commit message string,
        is_valid indicates if LLM generation succeeded.

    Example:
        >>> message, valid = await generate_commit_message_with_llm(context, router)
        >>> valid
        True
    """


def validate_commit_message(message: str) -> CommitMessageValidationResult:
    """Validate a commit message for conventional commit compliance.

    Checks format, subject line length, body formatting.
    Returns warnings for soft limits, errors for hard limits.

    Args:
        message: Commit message string to validate.

    Returns:
        CommitMessageValidationResult with passed, warnings, errors.

    Example:
        >>> result = validate_commit_message("feat: add login")
        >>> result.passed
        True
    """
```

### Conventional Commit Format Reference

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Subject Line Rules:**
- Type is required: feat, fix, refactor, test, docs, chore, style
- Scope is optional but recommended for multi-module changes
- Description starts lowercase, imperative mood, no period
- Max 50 characters (soft limit), 72 characters (hard limit)

**Body Rules:**
- Blank line between subject and body
- Wrap lines at 72 characters
- Explain "what" and "why", not "how"
- Use bullet points for multiple items

**Example Good Commit Messages:**
```
feat(dev): add commit message generation

Implement communicative commit messages for Dev agent output.
Includes LLM-powered generation with template fallback.

Story: 8-8

- Add CommitMessageContext and validation types
- Implement template-based generation
- Add LLM-powered generation with retry
- Integrate with dev_node output
```

```
fix(gates): handle missing state in DoD validation

Prevents KeyError when memory_context is not present in state.
Uses defensive .get() with default values.

Story: 8-6-dod-validation

Decision: Use defensive programming pattern for optional state
keys to ensure robustness across all agent invocations.
```

### Previous Story Learnings Applied (Stories 8.1-8.7)

From Story 8.3 (Unit Test Generation):
- LLM generation pattern with retry and fallback
- Quality validation of generated content
- Integration with dev_node output

From Story 8.5 (Documentation Generation):
- LLM prompt building with context
- Content extraction from LLM response
- Validation of generated content quality

From Story 8.6 (DoD Validation):
- Dataclass patterns for validation results
- `to_dict()` method for serialization
- Score-based pass/fail with threshold

From Story 8.7 (Pattern Following):
- Pattern cache clearing on node start
- Integration of analysis results in notes
- Decision record rationale augmentation

### Git Commit Pattern (from recent commits)

Recent commits follow pattern:
```
feat: Implement <feature> with code review fixes (Story X.Y)
```

This story should enable the Dev agent to generate similar messages automatically.

### Story Dependencies

This story builds on:
- Story 8.1 (Create Dev Agent Node) - dev_node structure
- Story 8.2 (Maintainable Code Generation) - LLM code generation patterns
- Story 8.3 (Unit Test Generation) - LLM generation with retry/fallback
- Stories 8.5-8.7 - DevOutput enhancements and validation patterns

This story completes Epic 8 (Dev Agent).

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR64 | Dev Agent can produce communicative commit messages with decision rationale | LLM-powered message generation with story context and decision inclusion |

### Integration with DevOutput

The DevOutput dataclass needs a new field:

```python
@dataclass
class DevOutput:
    """Output from Dev agent processing."""
    implementations: tuple[ImplementationArtifact, ...]
    processing_notes: str = ""
    suggested_commit_message: str | None = None  # NEW for Story 8.8

    def to_dict(self) -> dict:
        return {
            "implementations": [impl.to_dict() for impl in self.implementations],
            "processing_notes": self.processing_notes,
            "suggested_commit_message": self.suggested_commit_message,  # NEW
        }
```

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-8] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-8.8] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-003] - LLM tier routing
- [Source: _bmad-output/planning-artifacts/architecture.md#Implementation-Patterns] - Naming and logging conventions
- [Source: src/yolo_developer/agents/dev/node.py] - Existing dev_node implementation
- [Source: src/yolo_developer/agents/dev/test_utils.py] - LLM generation pattern reference
- [Source: src/yolo_developer/agents/dev/doc_utils.py] - Documentation generation pattern reference
- [Source: _bmad-output/implementation-artifacts/8-7-pattern-following.md] - Previous story learnings
- [Conventional Commits Specification: https://www.conventionalcommits.org/]
- [FR64: Dev Agent can produce communicative commit messages with decision rationale]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None - all tests pass.

### Completion Notes List

- Implemented all 12 tasks following red-green-refactor TDD cycle
- All 60 commit-related tests pass (49 unit + 11 integration)
- Linting (ruff) and type checking (mypy) pass
- Fixed 2 existing integration tests that had stale return type expectations

### Code Review Fixes Applied

1. Fixed docstring example for CommitType to use `.value` correctly
2. Made code block regex more permissive (handles any language hint)
3. Added 6 tests for module exports (AC8 verification)
4. Skip body line length warning for URLs
5. Added test for empty stories edge case
6. Fixed scope prompt wording from "none" to "(omit scope)"

### File List

**New Files:**
- `src/yolo_developer/agents/dev/commit_utils.py` - Core commit message utilities (types, generation, validation)
- `src/yolo_developer/agents/dev/prompts/commit_message.py` - LLM prompt templates for commit generation
- `tests/unit/agents/dev/test_commit_utils.py` - Unit tests for commit utilities (43 tests)
- `tests/integration/agents/dev/test_commit_generation.py` - Integration tests for dev node commit generation (10 tests)

**Modified Files:**
- `src/yolo_developer/agents/dev/__init__.py` - Added exports for commit utilities
- `src/yolo_developer/agents/dev/types.py` - Added `suggested_commit_message` field to DevOutput
- `src/yolo_developer/agents/dev/node.py` - Integrated commit message generation with dev_node
- `tests/integration/agents/dev/test_code_generation_integration.py` - Fixed return type unpacking for 3-tuple

