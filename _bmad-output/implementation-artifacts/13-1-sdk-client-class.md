# Story 13.1: SDK Client Class

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a YoloClient class for programmatic access,
so that I can integrate YOLO Developer into my scripts.

## Acceptance Criteria

### AC1: YoloClient Instantiation with Configuration
**Given** I import yolo_developer
**When** I instantiate YoloClient
**Then** I can configure it with settings
**And** default configuration is loaded if none provided
**And** configuration can be overridden via constructor parameters

### AC2: Full Functionality Access
**Given** I have a YoloClient instance
**When** I access the client's methods
**Then** all core functionality is accessible:
- Project initialization (`init()`)
- Seed document processing (`seed()`)
- Workflow execution (`run()`)
- Status retrieval (`status()`)
- Audit trail access (`get_audit()`)

### AC3: Complete Type Hints
**Given** I use YoloClient in a typed Python project
**When** I run mypy type checking
**Then** all public methods have complete type annotations
**And** return types are properly specified
**And** IDE autocompletion works correctly

### AC4: Comprehensive Documentation
**Given** I explore YoloClient in an IDE
**When** I view docstrings
**Then** all public methods have comprehensive docstrings
**And** docstrings include Args, Returns, Raises, and Examples
**And** module-level docstring explains overall usage

### AC5: SDK-Specific Exceptions
**Given** an error occurs during SDK operations
**When** the SDK raises an exception
**Then** SDK-specific exception types are used
**And** exceptions include helpful error messages
**And** original exceptions are preserved for debugging

## Tasks / Subtasks

- [x] Task 1: Create SDK Module Structure (AC: #1, #4)
  - [x] Create `src/yolo_developer/sdk/` directory
  - [x] Create `src/yolo_developer/sdk/__init__.py` with public API exports
  - [x] Create `src/yolo_developer/sdk/types.py` for SDK-specific type definitions
  - [x] Create `src/yolo_developer/sdk/exceptions.py` for SDK exception hierarchy

- [x] Task 2: Implement YoloClient Class Foundation (AC: #1, #3, #4)
  - [x] Create `src/yolo_developer/sdk/client.py` with YoloClient class
  - [x] Implement `__init__` with optional config parameter
  - [x] Add configuration loading with defaults fallback
  - [x] Add `from_config_file()` class method for file-based config
  - [x] Add complete type hints for all attributes

- [x] Task 3: Implement Core Client Methods (AC: #2, #3, #4)
  - [x] Implement `init()` method for project initialization (FR106)
  - [x] Implement `seed()` method for seed document processing (FR107)
  - [x] Implement `run()` method for workflow execution (FR107)
  - [x] Implement `run_async()` for async workflow execution
  - [x] Implement `status()` method for status retrieval
  - [x] Implement `get_audit()` method for audit trail access (FR108)
  - [x] Add comprehensive docstrings with examples

- [x] Task 4: Implement SDK Exceptions (AC: #5)
  - [x] Create `SDKError` base exception class
  - [x] Create `ClientNotInitializedError` for uninitialized operations
  - [x] Create `WorkflowExecutionError` for run failures
  - [x] Create `SeedValidationError` for seed issues
  - [x] Create `ProjectNotFoundError` for missing project paths
  - [x] Ensure exception chaining preserves original errors

- [x] Task 5: Add SDK Type Definitions (AC: #3)
  - [x] Define `SeedResult` type for seed operation results
  - [x] Define `RunResult` type for workflow execution results
  - [x] Define `StatusResult` type for status query results
  - [x] Define `AuditEntry` type for audit trail entries
  - [x] Define `InitResult` type for initialization results
  - [x] Export all types from `sdk/__init__.py`

- [x] Task 6: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/sdk/` directory structure
  - [x] Test YoloClient initialization with default config
  - [x] Test YoloClient initialization with custom config
  - [x] Test `from_config_file()` class method
  - [x] Test each core method with mocked dependencies
  - [x] Test exception handling and error messages
  - [x] Test type annotations with mypy
  - [x] 48 tests passing, comprehensive coverage for SDK module

- [x] Task 7: Update Package Exports (AC: #1, #4)
  - [x] Update `src/yolo_developer/__init__.py` with SDK exports
  - [x] Ensure `YoloClient` is importable from `yolo_developer`
  - [x] Verify imports work correctly

## Dev Notes

### Architecture Patterns

Per ADR-009 (Packaging & Distribution) and Architecture document:

1. **SDK Layer Position**: SDK sits between external consumers and the orchestrator layer
2. **Direct Import Pattern**: SDK imports directly from orchestrator module
   ```python
   from yolo_developer.orchestrator import run_workflow, stream_workflow, create_initial_state
   from yolo_developer.config import load_config, YoloConfig
   ```

3. **Configuration Integration**: SDK should leverage existing config system
   ```python
   from yolo_developer.config import load_config, YoloConfig, ConfigurationError
   ```

4. **Async Pattern**: Support both sync and async operations
   - Sync methods wrap async operations for convenience
   - Async methods expose full async API for advanced users

### Implementation Approach

**YoloClient Class Design:**
```python
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from yolo_developer.config import load_config, YoloConfig
from yolo_developer.sdk.exceptions import SDKError, ClientNotInitializedError
from yolo_developer.sdk.types import SeedResult, RunResult, StatusResult

if TYPE_CHECKING:
    from yolo_developer.orchestrator import YoloState

class YoloClient:
    """Python SDK client for YOLO Developer.

    Provides programmatic access to all YOLO Developer functionality including
    project initialization, seed processing, workflow execution, and audit access.

    Example:
        >>> from yolo_developer import YoloClient
        >>>
        >>> # Initialize with default config
        >>> client = YoloClient()
        >>>
        >>> # Or with custom config
        >>> client = YoloClient(config=my_config)
        >>>
        >>> # Run a workflow
        >>> result = await client.run_async(seed_content="Build a REST API...")
    """

    def __init__(
        self,
        config: YoloConfig | None = None,
        *,
        project_path: Path | str | None = None,
    ) -> None:
        """Initialize YoloClient.

        Args:
            config: Optional YoloConfig instance. If not provided, loads from
                default location (./yolo.yaml) or uses defaults.
            project_path: Optional project directory path. Defaults to current directory.
        """
        ...
```

### Module Structure

Per architecture document, SDK module structure:
```
src/yolo_developer/sdk/
├── __init__.py         # Public SDK API exports
├── client.py           # YoloClient class
├── types.py            # SDK-specific types (SeedResult, RunResult, etc.)
└── exceptions.py       # SDK exception hierarchy
```

### Testing Standards

Follow project patterns from existing tests:
- Use `pytest` with `pytest-asyncio` for async tests
- Mock orchestrator and config dependencies
- Test file naming: `test_<module>.py`
- Test function naming: `test_<behavior>_<scenario>`
- Run `ruff check` and `mypy` before committing

### Dependencies

**Internal Dependencies:**
- `yolo_developer.config` - Configuration loading and validation
- `yolo_developer.orchestrator` - Workflow execution
- `yolo_developer.audit` - Audit trail access (FR108)
- `yolo_developer.seed` - Seed document processing

**No External Dependencies Required** - SDK uses existing project dependencies.

### Project Structure Notes

- **Alignment**: SDK module follows architecture.md structure exactly
- **Entry Point**: SDK will be importable as `from yolo_developer import YoloClient`
- **API Boundary**: SDK is one of three external entry points (CLI, SDK, MCP)

### Previous Story Learnings

From Epic 12 stories:
1. Run `ruff check` and `mypy` before committing
2. Follow existing code patterns from similar modules
3. Use `from __future__ import annotations` in all files
4. Export public API from `__init__.py`
5. Include comprehensive docstrings with examples
6. Test both success and error paths

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Python SDK] - SDK structure and design
- [Source: _bmad-output/planning-artifacts/prd.md#Python SDK] - FR106-FR111 requirements
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 13] - Epic 13 overview and stories
- [Source: src/yolo_developer/orchestrator/__init__.py] - Orchestrator public API
- [Source: src/yolo_developer/config/__init__.py] - Config public API
- [Related: Story 12.1 (Typer CLI Setup)] - Pattern reference for module setup
- [Related: ADR-009] - PyPI package with entry points

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Test run: All 48 SDK tests pass
- mypy: Success, no issues found in 4 SDK source files
- ruff check: All checks passed
- ruff format: Applied formatting to SDK files

### Completion Notes List

1. Created complete SDK module with YoloClient class as main entry point
2. Implemented 5 exception classes with chaining support (SDKError, ClientNotInitializedError, WorkflowExecutionError, SeedValidationError, ProjectNotFoundError)
3. Created 5 frozen dataclass types for SDK results (SeedResult, RunResult, StatusResult, AuditEntry, InitResult)
4. YoloClient supports both sync and async APIs (run() wraps run_async(), etc.)
5. All methods have comprehensive docstrings with Args, Returns, Raises, and Examples
6. Used timezone-aware datetime (datetime.now(timezone.utc)) per ruff DTZ005 rule
7. SDK exports added to main yolo_developer/__init__.py for top-level import
8. 48 unit tests covering initialization, configuration, core methods, exceptions, and type properties
9. Test file mock paths fixed to match actual import locations (yolo_developer.seed.parse_seed, etc.)
10. workflow execution extracts agents from Decision objects' agent attribute

### File List

**New Files:**
- src/yolo_developer/sdk/__init__.py
- src/yolo_developer/sdk/client.py
- src/yolo_developer/sdk/types.py
- src/yolo_developer/sdk/exceptions.py
- tests/unit/sdk/__init__.py
- tests/unit/sdk/test_client.py
- tests/unit/sdk/test_exceptions.py

**Modified Files:**
- src/yolo_developer/__init__.py (add SDK exports)
- _bmad-output/implementation-artifacts/sprint-status.yaml

### Change Log

- 2026-01-19: Story file created for Story 13.1 - SDK Client Class
- 2026-01-19: Code review completed - Fixed 6 issues:
  1. Non-UTC aware timestamps in types.py (HIGH)
  2. Deprecated asyncio.get_event_loop() usage (HIGH)
  3. Missing test for sync run() method (MEDIUM)
  4. Type: ignore comment with proper typing (MEDIUM)
  5. Documented hardcoded quality_score as limitation (MEDIUM)
  6. Added test for from_config_file with string path (LOW)
  - All 50 tests passing, mypy clean, ruff clean
