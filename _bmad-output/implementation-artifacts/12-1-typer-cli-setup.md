# Story 12.1: Typer CLI Setup

## Story

**As a** developer,
**I want** a CLI built with Typer and Rich,
**So that** I have a modern, beautiful command-line interface.

## Status

- **Epic:** 12 - CLI Interface
- **Status:** done
- **Priority:** P1
- **Story Points:** 3

## Acceptance Criteria

### AC1: Help Information Display
**Given** I install yolo-developer
**When** I run `yolo`
**Then** help information is displayed
**And** all available commands are listed with descriptions

### AC2: Beautiful Rich Formatting
**Given** I run any `yolo` command
**When** output is displayed
**Then** output is beautifully formatted with Rich
**And** colors are used consistently (green=success, red=error, yellow=warning)
**And** tables are used for structured data

### AC3: System-Wide Entry Point
**Given** I install yolo-developer via `pip install` or `uv sync`
**When** I open a new terminal
**Then** the `yolo` entry point is available system-wide
**And** `yolo --help` works from any directory

### AC4: Command Structure Foundation
**Given** the CLI is set up
**When** I examine the command structure
**Then** placeholder commands exist for all Epic 12 commands (run, status, logs, tune, config)
**And** each placeholder displays "Coming soon" message with Rich formatting
**And** existing commands (init, seed, version) continue to work

### AC5: Display Utilities Module
**Given** the CLI needs to display formatted output
**When** commands need to show panels, tables, or status indicators
**Then** a `display.py` module provides reusable Rich formatting utilities
**And** all commands use consistent styling patterns

## Technical Requirements

### Functional Requirements Mapping
- **FR98:** Users can initialize new projects via `yolo init` command (EXISTS)
- **FR99:** Users can provide seed documents via `yolo seed` command (EXISTS)
- **FR100:** Users can execute autonomous sprints via `yolo run` command (PLACEHOLDER)
- **FR101:** Users can view sprint status via `yolo status` command (PLACEHOLDER)
- **FR102:** Users can view decision logs via `yolo logs` command (PLACEHOLDER)
- **FR103:** Users can modify agent templates via `yolo tune` command (PLACEHOLDER)
- **FR104:** Users can manage configuration via `yolo config` command (PLACEHOLDER)
- **FR105:** CLI can display real-time agent activity (FUTURE - Story 12.9)

### Architecture References
- **ADR-009:** PyPI package with CLI entry point (Typer + Rich)
- **ARCH-DEP-5:** Typer + Rich framework selection
- **ARCH-STRUCT-2:** cli/ module structure in src/yolo_developer/

### Technology Stack
- **Typer:** CLI framework (already installed)
- **Rich:** Terminal formatting (already installed)
- **Python 3.10+:** Runtime requirement

## Tasks

### Task 1: Create Display Utilities Module (AC: #5)
**File:** `src/yolo_developer/cli/display.py`

Create reusable Rich formatting utilities:

```python
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def success_panel(message: str, title: str = "Success") -> None:
    """Display a success panel."""
    console.print(Panel(f"[green]{message}[/green]", title=title, border_style="green"))

def error_panel(message: str, title: str = "Error") -> None:
    """Display an error panel."""
    console.print(Panel(f"[red]{message}[/red]", title=title, border_style="red"))

def info_panel(message: str, title: str = "Info") -> None:
    """Display an info panel."""
    console.print(Panel(f"[blue]{message}[/blue]", title=title, border_style="blue"))

def warning_panel(message: str, title: str = "Warning") -> None:
    """Display a warning panel."""
    console.print(Panel(f"[yellow]{message}[/yellow]", title=title, border_style="yellow"))

def coming_soon(command: str) -> None:
    """Display a 'coming soon' message for unimplemented commands."""
    console.print(Panel(
        f"[yellow]The '{command}' command is not yet implemented.[/yellow]\n\n"
        f"[dim]This command will be available in a future release.[/dim]",
        title="Coming Soon",
        border_style="yellow",
    ))

def create_table(title: str, columns: list[tuple[str, str]]) -> Table:
    """Create a styled table with given columns."""
    table = Table(title=title, show_header=True, header_style="bold")
    for name, style in columns:
        table.add_column(name, style=style)
    return table
```

**Subtasks:**
1. Create `display.py` with Rich utilities
2. Add `success_panel`, `error_panel`, `info_panel`, `warning_panel`
3. Add `coming_soon` function for placeholder commands
4. Add `create_table` factory function
5. Export from `cli/__init__.py`

### Task 2: Create Placeholder Command Files (AC: #4)
**Files:**
- `src/yolo_developer/cli/commands/run.py`
- `src/yolo_developer/cli/commands/status.py`
- `src/yolo_developer/cli/commands/logs.py`
- `src/yolo_developer/cli/commands/tune.py`
- `src/yolo_developer/cli/commands/config.py`

Each file follows this pattern:

```python
"""YOLO {command} command implementation (Story 12.X)."""

from __future__ import annotations

import structlog

from yolo_developer.cli.display import coming_soon

logger = structlog.get_logger(__name__)


def {command}_command() -> None:
    """Execute the {command} command.

    This command will be implemented in Story 12.X.
    """
    logger.debug("{command}_command_invoked")
    coming_soon("{command}")
```

**Subtasks:**
1. Create `run.py` with placeholder
2. Create `status.py` with placeholder
3. Create `logs.py` with placeholder
4. Create `tune.py` with placeholder
5. Create `config.py` with placeholder
6. Export all from `commands/__init__.py`

### Task 3: Update Main CLI with All Commands (AC: #1, #4)
**File:** `src/yolo_developer/cli/main.py`

Add all placeholder commands to main app:

```python
@app.command("run")
def run() -> None:
    """Execute an autonomous sprint.

    Triggers the multi-agent orchestration to execute a sprint
    based on the seed requirements and project configuration.
    """
    from yolo_developer.cli.commands.run import run_command
    run_command()


@app.command("status")
def status() -> None:
    """View current sprint status.

    Shows the progress of the current sprint including completed stories,
    in-progress work, and any blocked items.
    """
    from yolo_developer.cli.commands.status import status_command
    status_command()


@app.command("logs")
def logs() -> None:
    """View decision logs and audit trail.

    Browse the audit trail of agent decisions with filtering options.
    """
    from yolo_developer.cli.commands.logs import logs_command
    logs_command()


@app.command("tune")
def tune() -> None:
    """Modify agent templates and behavior.

    Customize how agents make decisions by modifying their templates.
    """
    from yolo_developer.cli.commands.tune import tune_command
    tune_command()


@app.command("config")
def config() -> None:
    """Manage project configuration.

    View, set, import, or export project configuration values.
    """
    from yolo_developer.cli.commands.config import config_command
    config_command()
```

**Subtasks:**
1. Add `run` command with docstring
2. Add `status` command with docstring
3. Add `logs` command with docstring
4. Add `tune` command with docstring
5. Add `config` command with docstring
6. Verify `no_args_is_help=True` shows all commands

### Task 4: Update Module Exports (AC: #3)
**File:** `src/yolo_developer/cli/__init__.py`

Update exports to include new modules:

```python
from yolo_developer.cli.main import app
from yolo_developer.cli.display import (
    console,
    success_panel,
    error_panel,
    info_panel,
    warning_panel,
    coming_soon,
    create_table,
)

__all__ = [
    "app",
    "console",
    "success_panel",
    "error_panel",
    "info_panel",
    "warning_panel",
    "coming_soon",
    "create_table",
]
```

**Subtasks:**
1. Add display module exports
2. Update `__all__` list
3. Verify entry point works (`yolo --help`)

### Task 5: Write Comprehensive Tests (AC: #1-5)
**Files:**
- `tests/unit/cli/test_display.py`
- `tests/unit/cli/test_main.py` (update)
- `tests/unit/cli/test_placeholder_commands.py`

Test all new functionality:

**test_display.py:**
- Test `success_panel` outputs green panel
- Test `error_panel` outputs red panel
- Test `coming_soon` outputs warning panel
- Test `create_table` returns configured Table

**test_main.py (additions):**
- Test `yolo --help` lists all commands
- Test each placeholder command invokes coming_soon

**test_placeholder_commands.py:**
- Test each placeholder command logs debug
- Test each placeholder command displays coming_soon

**Subtasks:**
1. Create `test_display.py` with display utility tests
2. Update `test_main.py` with command listing tests
3. Create `test_placeholder_commands.py`
4. Ensure >90% coverage on new code

### Task 6: Verify Entry Point (AC: #3)
**File:** `pyproject.toml` (verify)

Confirm entry point configuration:

```toml
[project.scripts]
yolo = "yolo_developer.cli.main:app"
```

**Subtasks:**
1. Verify entry point is correctly configured
2. Test `uv run yolo --help` works
3. Test all commands appear in help output

## Dev Notes

### Current CLI State Analysis

The CLI already has significant implementation from earlier stories:

**Existing Commands (Working):**
- `yolo init` - Full implementation in `commands/init.py` (Story 1.x)
- `yolo seed` - Comprehensive implementation in `commands/seed.py` (Story 4.x)
- `yolo version` - Simple version display

**Missing Commands (This Story):**
- `yolo run` - Placeholder needed
- `yolo status` - Placeholder needed
- `yolo logs` - Placeholder needed
- `yolo tune` - Placeholder needed
- `yolo config` - Placeholder needed

**Missing Infrastructure:**
- `display.py` - Reusable Rich formatting utilities

### Existing Code Patterns to Follow

**From `commands/seed.py` (comprehensive reference):**
```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from yolo_developer.seed.sop import SOPStore

logger = structlog.get_logger(__name__)
console = Console()

def seed_command(
    file_path: Path,
    verbose: bool = False,
    json_output: bool = False,
) -> None:
    """Main command function."""
    logger.info("seed_command_started", file_path=str(file_path))
    try:
        # Implementation
        result = asyncio.run(_parse_seed_async(content))
        _display_results(result, verbose)
    except Exception as e:
        logger.error("seed_command_failed", error=str(e))
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e
```

**Key Patterns:**
1. All files start with `from __future__ import annotations`
2. Use structlog for logging with context
3. Use Rich Console for all output
4. Use `typer.Exit(code=1)` for errors
5. Async operations via `asyncio.run()`
6. Type hints on all function signatures

### File Structure After Implementation

```
src/yolo_developer/cli/
├── __init__.py           # Export app and display utilities
├── main.py               # Typer app with all commands registered
├── display.py            # NEW: Rich formatting utilities
└── commands/
    ├── __init__.py       # Export all command functions
    ├── init.py           # EXISTS: yolo init
    ├── seed.py           # EXISTS: yolo seed
    ├── run.py            # NEW: placeholder
    ├── status.py         # NEW: placeholder
    ├── logs.py           # NEW: placeholder
    ├── tune.py           # NEW: placeholder
    └── config.py         # NEW: placeholder

tests/unit/cli/
├── __init__.py
├── test_seed_command.py  # EXISTS: comprehensive tests
├── test_display.py       # NEW: display utility tests
├── test_main.py          # NEW/UPDATE: main app tests
└── test_placeholder_commands.py  # NEW: placeholder tests
```

### NFR Considerations

- **NFR-PERF-4:** CLI command response <2 seconds (achieved - no blocking calls)
- **NFR-MAINT-1:** YAML configuration (config integration in place)

### Dependencies

All dependencies already installed:
- `typer` - CLI framework
- `rich` - Terminal formatting
- `structlog` - Structured logging
- `pydantic` - Validation

### Testing Approach

Follow existing patterns from `test_seed_command.py`:
- Use `MagicMock(spec=Console)` for mocking console
- Use `@patch("yolo_developer.cli.commands.X.console")` for output verification
- Test both success and error paths
- Verify exit codes with `pytest.raises(typer.Exit)`

## Definition of Done

- [x] All acceptance criteria implemented and verified
- [x] Display utilities module created with all functions
- [x] All placeholder commands created and registered
- [x] Unit tests for display module with >90% coverage
- [x] Unit tests for placeholder commands
- [x] `yolo --help` displays all commands
- [x] Type hints on all public functions (mypy passes)
- [x] Code formatted with ruff
- [x] Docstrings following Google style on all public APIs
- [x] No breaking changes to existing commands (init, seed, version)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No debugging issues encountered

### Completion Notes List

1. **Task 1 - Display Utilities Module**: Created `src/yolo_developer/cli/display.py` with Rich formatting utilities including `success_panel`, `error_panel`, `info_panel`, `warning_panel`, `coming_soon`, and `create_table`. All 16 unit tests pass.

2. **Task 2 - Placeholder Command Files**: Created 5 placeholder command files (`run.py`, `status.py`, `logs.py`, `tune.py`, `config.py`) in `src/yolo_developer/cli/commands/`. Each follows the established pattern with structlog debug logging and coming_soon display.

3. **Task 3 - Update Main CLI**: Updated `src/yolo_developer/cli/main.py` to register all 5 new placeholder commands with Typer decorators and descriptive docstrings.

4. **Task 4 - Module Exports**: Updated `src/yolo_developer/cli/__init__.py` to export display utilities from the CLI module public API.

5. **Task 5 - Comprehensive Tests**: Created `test_main_commands.py` (14 tests) and `test_placeholder_commands.py` (11 tests). Combined with `test_display.py` (16 tests), all 41 Story 12.1 tests pass.

6. **Task 6 - Verify Entry Point**: Confirmed `yolo --help` displays all 8 commands (init, version, seed, run, status, logs, tune, config). All placeholder commands display "Coming Soon" panel correctly.

### File List

**Created:**
- `src/yolo_developer/cli/display.py` - Rich formatting utilities module
- `src/yolo_developer/cli/commands/run.py` - Run command placeholder
- `src/yolo_developer/cli/commands/status.py` - Status command placeholder
- `src/yolo_developer/cli/commands/logs.py` - Logs command placeholder
- `src/yolo_developer/cli/commands/tune.py` - Tune command placeholder
- `src/yolo_developer/cli/commands/config.py` - Config command placeholder
- `tests/unit/cli/test_display.py` - Display utilities tests (16 tests)
- `tests/unit/cli/test_main_commands.py` - Main CLI command tests (14 tests)
- `tests/unit/cli/test_placeholder_commands.py` - Placeholder command tests (11 tests)

**Modified:**
- `src/yolo_developer/cli/main.py` - Added 5 new commands
- `src/yolo_developer/cli/__init__.py` - Added display exports
- `src/yolo_developer/cli/commands/__init__.py` - Added new command exports
- `src/yolo_developer/cli/commands/seed.py` - Ruff formatting fixes (code review)
- `tests/unit/cli/test_seed_command.py` - Ruff formatting fixes (code review)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status

## Senior Developer Review (AI)

**Reviewer:** Claude Opus 4.5
**Date:** 2026-01-18
**Outcome:** ✅ APPROVED (with fixes applied)

### Issues Found and Fixed

| ID | Severity | Description | Resolution |
|----|----------|-------------|------------|
| H1 | HIGH | Incomplete File List - seed.py, test_seed_command.py, sprint-status.yaml not documented | Added to File List |
| M1 | MEDIUM | Top-level imports instead of lazy imports in main.py | Converted to lazy imports per spec |
| M2 | MEDIUM | Minor docstring wording differences | Accepted as-is - functionally equivalent |
| M3 | MEDIUM | Story status "done" but sprint-status "review" | Synced to "done" after fixes |
| M4 | MEDIUM | No regression tests for existing commands (init, version, seed) | Added 3 regression tests |

### Test Results After Fixes
- **44 tests pass** (was 41, added 3 regression tests)
- mypy: Success, no issues
- ruff: All checks passed

### Acceptance Criteria Verification
- ✅ AC1: Help Information Display - `yolo --help` shows all 8 commands
- ✅ AC2: Beautiful Rich Formatting - Consistent colors via display.py
- ✅ AC3: System-Wide Entry Point - `yolo` works from any directory
- ✅ AC4: Command Structure Foundation - All 5 placeholders + existing commands work
- ✅ AC5: Display Utilities Module - display.py provides reusable utilities

## References

- Epic 12: CLI Interface requirements
- FR98-FR105: CLI functional requirements
- ADR-009: PyPI package with CLI entry point
- Story 4.2: Existing seed command patterns
- Architecture: cli/ module structure (lines 1044-1051)
