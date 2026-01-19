# Story 12.7: yolo tune Command

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to modify agent templates via `yolo tune`,
so that I can customize agent behavior for my project's specific needs.

## Acceptance Criteria

### AC1: View Current Agent Template
**Given** agents with templates exist
**When** I run `yolo tune analyst`
**Then** the current template/prompt for that agent is displayed
**And** the template shows system prompt and configurable sections
**And** output is formatted with Rich styling for readability

### AC2: Modify Agent Template
**Given** I have identified a template to modify
**When** I run `yolo tune analyst --edit`
**Then** I can modify the template content
**And** changes are validated for syntax/structure
**And** changes take effect immediately for subsequent runs

### AC3: List Available Agents
**Given** multiple agents with templates exist
**When** I run `yolo tune --list`
**Then** all configurable agents are listed
**And** each agent shows name and brief description
**And** customization status (default/customized) is shown

### AC4: Reset to Default Template
**Given** an agent has a customized template
**When** I run `yolo tune analyst --reset`
**Then** the template is reset to its default value
**And** a confirmation message is displayed
**And** the change takes effect immediately

## Tasks / Subtasks

- [x] Task 1: Add CLI Flags to main.py (AC: #1, #2, #3, #4)
  - [x] Add positional `agent_name` argument (optional)
  - [x] Add `--list/-l` flag to show all configurable agents
  - [x] Add `--edit/-e` flag to open template for editing
  - [x] Add `--reset/-r` flag to reset to default
  - [x] Add `--export` flag to export template to file
  - [x] Add `--import` flag to import template from file
  - [x] Add `--json/-j` flag for machine-readable output
  - [x] Update tune command help text with examples

- [x] Task 2: Create Agent Template Registry (AC: #1, #3)
  - [x] Create `src/yolo_developer/cli/commands/tune.py` with template registry
  - [x] Define AGENT_TEMPLATES dict mapping agent names to prompt module paths
  - [x] Include all 6 agents: analyst, pm, architect, dev, sm, tea
  - [x] Add agent descriptions for --list output
  - [x] Add function to detect custom vs default templates

- [x] Task 3: Implement Template Viewing (AC: #1)
  - [x] Load template content from agent prompt modules
  - [x] Format system prompt with Rich Panel
  - [x] Format user prompt template with syntax highlighting
  - [x] Show configurable sections (guidelines, conventions)
  - [x] Display template metadata (agent name, version)

- [x] Task 4: Implement Template Customization Storage (AC: #2, #4)
  - [x] Create `.yolo/templates/` directory for custom templates
  - [x] Define template override file format (YAML with prompt sections)
  - [x] Implement save_custom_template() function
  - [x] Implement load_custom_template() function
  - [x] Implement has_custom_template() function
  - [x] Implement reset_template() to remove custom overrides

- [x] Task 5: Implement Template Editing (AC: #2)
  - [x] Create interactive editing workflow
  - [x] Option 1: Open in $EDITOR (environment variable)
  - [ ] Option 2: Use Rich prompt for inline editing (deferred - $EDITOR sufficient for MVP)
  - [x] Validate template syntax after editing
  - [ ] Show diff of changes before saving (deferred - future enhancement)
  - [ ] Confirm save with user (deferred - editor exit is implicit confirmation)

- [x] Task 6: Implement Template Export/Import (AC: #2)
  - [x] Export current template to YAML file
  - [x] Import template from YAML file
  - [x] Validate imported template structure
  - [x] Handle file not found gracefully

- [x] Task 7: Implement Agent Listing (AC: #3)
  - [x] Build Rich table with agent information
  - [x] Show columns: Agent, Description, Status (default/customized)
  - [x] Color-code customized agents
  - [x] Add helpful footer with usage tips

- [x] Task 8: Write Unit Tests (AC: all)
  - [x] Test CLI flag parsing for all options
  - [x] Test agent template registry completeness
  - [x] Test template viewing output
  - [x] Test custom template save/load
  - [x] Test template reset functionality
  - [x] Test export/import round-trip
  - [x] Test validation of invalid templates
  - [x] Test --list output structure
  - [x] Test JSON output format

## Dev Notes

### Existing Implementation

The tune command exists as a placeholder at `src/yolo_developer/cli/commands/tune.py` (28 lines) that currently just displays "coming soon".

**CLI wiring already exists in main.py:**
```python
@app.command("tune")
def tune() -> None:
    """Customize agent behavior.

    Modify agent templates and decision-making parameters
    to tune how agents approach development tasks.

    This command will be fully implemented in Story 12.7.
    """
    from yolo_developer.cli.commands.tune import tune_command

    tune_command()
```

### Agent Prompts Structure (CRITICAL CONTEXT)

**Prompts are stored in two locations:**

1. **Centralized Prompts:**
   - `src/yolo_developer/agents/prompts/analyst.py` - Analyst agent prompts

2. **Agent-Specific Prompts:**
   - `src/yolo_developer/agents/dev/prompts/` - Dev agent (code_generation.py, test_generation.py, etc.)
   - `src/yolo_developer/agents/pm/` - PM agent (llm.py, breakdown.py, dependencies.py)

**Standard Template Pattern:**
```python
# Constants for guidelines
MAINTAINABILITY_GUIDELINES = """..."""
PROJECT_CONVENTIONS = """..."""

# System prompt (defines agent persona/role)
ANALYST_SYSTEM_PROMPT = """You are a Requirements Analyst AI agent...
CORE RESPONSIBILITIES:
1. CRYSTALLIZE: Transform vague requirements
2. CATEGORIZE: Classify requirements
...
OUTPUT FORMAT:
You MUST respond with valid JSON matching this schema:
..."""

# User prompt template (with placeholders)
ANALYST_USER_PROMPT_TEMPLATE = """Analyze the following seed content...
SEED CONTENT:
---
{seed_content}
---
"""

# Builder functions for complex prompts
def build_code_generation_prompt(
    story_title: str,
    requirements: str,
    acceptance_criteria: list[str] | None = None,
    ...
) -> str:
    """Build complete prompt with templating logic."""
```

### Template Override File Format

Custom templates should be stored in `.yolo/templates/<agent_name>.yaml`:

```yaml
# .yolo/templates/analyst.yaml
agent: analyst
version: "1.0"
customized_at: "2026-01-19T10:00:00Z"

system_prompt: |
  You are a Requirements Analyst AI agent...
  (full system prompt content)

user_prompt_template: |
  Analyze the following seed content...
  {seed_content}

# Optional: Override specific guidelines
guidelines:
  maintainability: |
    Custom maintainability guidelines...
  conventions: |
    Custom project conventions...
```

### Key Dependencies

**Configuration (config/):**
- `load_config()` - Load project configuration
- Project root: Determined by presence of `yolo.yaml` or `.yolo/` directory

**Display Utilities (cli/display.py):**
- `console` - Rich console instance
- `create_table()` - Create styled table
- `info_panel()`, `success_panel()`, `warning_panel()` - Styled panels

### Display Patterns

Follow existing CLI patterns from logs/status commands:

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from yolo_developer.cli.display import create_table, info_panel, success_panel

# Agent colors (match status/logs patterns)
AGENT_COLORS = {
    "analyst": "cyan",
    "pm": "blue",
    "architect": "magenta",
    "dev": "green",
    "tea": "yellow",
    "sm": "white",
}

# Display template with syntax highlighting
syntax = Syntax(template_content, "python", theme="monokai", line_numbers=True)
console.print(Panel(syntax, title=f"{agent_name} System Prompt"))
```

### Agent Registry Definition

```python
CONFIGURABLE_AGENTS: dict[str, dict[str, str]] = {
    "analyst": {
        "description": "Crystallizes vague requirements into specific, implementable statements",
        "prompt_module": "yolo_developer.agents.prompts.analyst",
        "system_prompt_var": "ANALYST_SYSTEM_PROMPT",
        "user_prompt_var": "ANALYST_USER_PROMPT_TEMPLATE",
    },
    "pm": {
        "description": "Transforms requirements into testable user stories with acceptance criteria",
        "prompt_module": "yolo_developer.agents.pm.llm",
        "system_prompt_var": "PM_SYSTEM_PROMPT",
        "user_prompt_var": "PM_USER_PROMPT_TEMPLATE",
    },
    "architect": {
        "description": "Designs system architecture following 12-Factor principles",
        "prompt_module": "yolo_developer.agents.architect.prompts",
        "system_prompt_var": "ARCHITECT_SYSTEM_PROMPT",
        "user_prompt_var": "ARCHITECT_USER_PROMPT_TEMPLATE",
    },
    "dev": {
        "description": "Implements maintainable code with tests and documentation",
        "prompt_module": "yolo_developer.agents.dev.prompts.code_generation",
        "system_prompt_var": "CODE_GENERATION_TEMPLATE",
        "user_prompt_var": None,  # Uses builder function
    },
    "sm": {
        "description": "Orchestrates sprint execution and monitors system health",
        "prompt_module": "yolo_developer.agents.sm.prompts",
        "system_prompt_var": "SM_SYSTEM_PROMPT",
        "user_prompt_var": "SM_USER_PROMPT_TEMPLATE",
    },
    "tea": {
        "description": "Validates test coverage and calculates deployment confidence",
        "prompt_module": "yolo_developer.agents.tea.prompts",
        "system_prompt_var": "TEA_SYSTEM_PROMPT",
        "user_prompt_var": "TEA_USER_PROMPT_TEMPLATE",
    },
}
```

### Architecture Patterns

Per ADR-005 and existing CLI patterns:
- Use structlog for logging
- Use typer.Exit() for error exits
- Follow same flag patterns as logs/status commands (--json)
- Handle missing templates gracefully
- Use Path objects for file operations
- Ensure .yolo/ directory exists before writing

### Project Structure Notes

- CLI command: `src/yolo_developer/cli/commands/tune.py`
- CLI wiring: `src/yolo_developer/cli/main.py:400-411`
- Tests: `tests/unit/cli/test_tune_command.py` (new)
- Display utilities: `src/yolo_developer/cli/display.py`
- Template storage: `.yolo/templates/` (created on first customization)

### JSON Output Structure

```json
{
  "agent": "analyst",
  "status": "customized",
  "customized_at": "2026-01-19T10:00:00+00:00",
  "template": {
    "system_prompt": "You are a Requirements Analyst AI agent...",
    "user_prompt_template": "Analyze the following seed content...",
    "guidelines": {}
  }
}
```

For --list:
```json
{
  "agents": [
    {
      "name": "analyst",
      "description": "Crystallizes vague requirements into specific statements",
      "status": "default"
    },
    {
      "name": "pm",
      "description": "Transforms requirements into testable user stories",
      "status": "customized"
    }
  ]
}
```

### Previous Story Intelligence (Story 12.6)

**Learnings from yolo logs implementation:**
- Added validation at entry point (decision_type, limit) - do same for agent_name
- Consolidated case-sensitivity handling (normalize agent names to lowercase)
- Used constants for magic numbers (TABLE_SUMMARY_MAX_LENGTH, etc.)
- Removed dead code (unused verbose parameter)
- 43 tests provided comprehensive coverage

**Patterns to follow:**
- Validate input early, return with warning_panel on invalid input
- Use normalized variables (normalized_agent) for consistent handling
- Use constants for display values
- Keep functions focused (single responsibility)

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-12.7]
- [Source: src/yolo_developer/cli/commands/tune.py]
- [Source: src/yolo_developer/cli/main.py:400-411]
- [Source: src/yolo_developer/agents/prompts/analyst.py]
- [Source: src/yolo_developer/agents/dev/prompts/code_generation.py]
- [Source: src/yolo_developer/agents/pm/llm.py]
- [Related: FR103 - Users can modify agent templates via yolo tune command]
- [Related: FR91 - Users can customize agent templates and rules]
- [Related: Story 12.6 (yolo logs command) - display patterns]

## Dev Agent Record

### Agent Model Used
Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References
N/A

### Completion Notes List
- Implemented complete `yolo tune` command replacing placeholder
- Created CONFIGURABLE_AGENTS registry mapping all 6 agents to their prompt modules
- Template storage uses `.yolo/templates/<agent>.yaml` format with system_prompt, user_prompt_template, and guidelines sections
- Template viewing displays Rich-formatted panels with syntax highlighting
- Edit workflow opens template in $EDITOR, validates YAML syntax
- Export/import supports YAML format with validation
- Agent listing shows Rich table with customization status (default/customized)
- Tests use `strip_ansi()` helper for Rich output assertions
- Removed tune tests from placeholder tests file

**Code Review Fixes Applied:**
- Added YAML error handling in `_edit_template()` for invalid syntax after editing
- Added `typer.Exit(code=1)` for invalid agent name (per project pattern)
- Removed empty `TYPE_CHECKING` block (dead code)
- Added 6 unit tests for `_edit_template()` function
- Updated story subtasks: deferred Rich prompt inline editing, diff display, and save confirmation (MVP uses $EDITOR workflow)
- Added sprint-status.yaml to File List

### Change Log
| File | Change |
|------|--------|
| src/yolo_developer/cli/main.py | Added CLI flags: agent_name, --list, --edit, --reset, --export, --import, --json |
| src/yolo_developer/cli/commands/tune.py | Complete rewrite - full tune command implementation with YAML error handling, typer.Exit() |
| tests/unit/cli/test_tune_command.py | New file - 57 unit tests (including 6 for _edit_template) |
| tests/unit/cli/test_placeholder_commands.py | Removed TestTuneCommand class, added note about moved tests |
| _bmad-output/implementation-artifacts/sprint-status.yaml | Updated story status |

### File List
- `src/yolo_developer/cli/main.py:400-468` - CLI command definition with all flags
- `src/yolo_developer/cli/commands/tune.py` - Full implementation
- `tests/unit/cli/test_tune_command.py` - Unit tests (including edit template tests)
- `tests/unit/cli/test_placeholder_commands.py` - Updated to remove tune tests
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Sprint status tracking update
