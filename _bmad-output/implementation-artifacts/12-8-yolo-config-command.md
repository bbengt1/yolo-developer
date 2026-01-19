# Story 12.8: yolo config Command

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to manage configuration via `yolo config`,
so that I can adjust settings easily without manually editing YAML files.

## Acceptance Criteria

### AC1: View Current Configuration
**Given** project configuration exists (yolo.yaml file)
**When** I run `yolo config`
**Then** the current configuration is displayed
**And** nested settings (llm, quality, memory) are shown hierarchically
**And** sensitive values (API keys) are masked in the output
**And** output is formatted with Rich styling for readability

### AC2: Set Configuration Value
**Given** a valid configuration key
**When** I run `yolo config set key value`
**Then** the value is updated in the configuration file
**And** nested keys are supported (e.g., `yolo config set llm.cheap_model gpt-4o`)
**And** invalid keys or values produce clear error messages
**And** a confirmation message shows what was changed

### AC3: Export Configuration
**Given** project configuration exists
**When** I run `yolo config export`
**Then** configuration is exported to a portable YAML file
**And** sensitive values (API keys) are excluded with placeholder comments
**And** the file can be specified with --output/-o flag
**And** a success message shows the export path

### AC4: Import Configuration
**Given** an exported configuration file exists
**When** I run `yolo config import config-file.yaml`
**Then** the configuration is imported to the project
**And** validation is performed before applying changes
**And** conflicts with existing config are handled gracefully
**And** a success message confirms the import

## Tasks / Subtasks

- [x] Task 1: Add CLI Flags to main.py (AC: #1, #2, #3, #4)
  - [x] Add subcommand approach with `yolo config` (show), `yolo config set`, `yolo config export`, `yolo config import`
  - [x] Add `key` argument for set subcommand
  - [x] Add `value` argument for set subcommand
  - [x] Add `file` argument for import subcommand
  - [x] Add `--output/-o` option for export subcommand
  - [x] Add `--json/-j` flag for machine-readable output
  - [x] Add `--no-mask` flag to show API keys (for debugging)
  - [x] Update config command help text with examples

- [x] Task 2: Implement Configuration Viewing (AC: #1)
  - [x] Create `show_config()` function in config command module
  - [x] Load configuration using `load_config()` from config module
  - [x] Format nested configuration with Rich Tree or Panel
  - [x] Mask sensitive values (API keys) by default
  - [x] Support JSON output format with `--json` flag
  - [x] Handle missing config file gracefully (suggest `yolo init`)

- [x] Task 3: Implement Configuration Setting (AC: #2)
  - [x] Create `set_config_value()` function
  - [x] Parse dotted key paths (e.g., `llm.cheap_model`)
  - [x] Validate key exists in schema before setting
  - [x] Validate value type matches schema (string, float, bool)
  - [x] Load existing yolo.yaml, update value, write back
  - [x] Write back YAML (note: comments not preserved - uses yaml.safe_dump)
  - [x] Show before/after values in confirmation

- [x] Task 4: Implement Configuration Export (AC: #3)
  - [x] Wire existing `export_config()` to CLI command
  - [x] Default output path to `yolo-config-export.yaml`
  - [x] Support custom output path with `--output/-o`
  - [x] Show success panel with exported file path
  - [x] JSON output mode for scripting

- [x] Task 5: Implement Configuration Import (AC: #4)
  - [x] Wire existing `import_config()` to CLI command
  - [x] Validate source file exists
  - [x] Load and validate configuration before applying
  - [x] Handle validation errors with clear messages
  - [x] Show success panel with applied changes summary
  - [x] JSON output mode for scripting

- [x] Task 6: Implement JSON Output Mode (AC: #1, #2, #3, #4)
  - [x] Structure JSON output consistently across all subcommands
  - [x] For show: return full config as JSON (with masked API keys by default)
  - [x] For set: return {"key": key, "old_value": old, "new_value": new}
  - [x] For export: return {"path": export_path, "status": "success"}
  - [x] For import: return {"source": source_path, "status": "success"}

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Test CLI flag parsing for all subcommands
  - [x] Test config viewing with and without config file
  - [x] Test config setting with valid/invalid keys
  - [x] Test config setting with nested keys
  - [x] Test config export to default and custom paths
  - [x] Test config import with valid/invalid files
  - [x] Test API key masking in output
  - [x] Test JSON output format for all subcommands
  - [x] Test error handling for missing config file

## Dev Notes

### Existing Implementation

The config command exists as a placeholder at `src/yolo_developer/cli/commands/config.py` (28 lines) that currently displays "coming soon" via `coming_soon("config")`.

**CLI wiring already exists in main.py (lines 471-481):**
```python
@app.command("config")
def config() -> None:
    """Manage project configuration.

    View, set, import, or export project configuration values.

    This command will be fully implemented in Story 12.8.
    """
    from yolo_developer.cli.commands.config import config_command

    config_command()
```

### Configuration System (CRITICAL CONTEXT)

**Config module is fully implemented:**
- `src/yolo_developer/config/__init__.py` - Public API exports
- `src/yolo_developer/config/schema.py` - Pydantic Settings models (YoloConfig, LLMConfig, QualityConfig, MemoryConfig)
- `src/yolo_developer/config/loader.py` - Configuration loading with YAML + env var support
- `src/yolo_developer/config/export.py` - Export/import functionality (already implemented!)
- `src/yolo_developer/config/validators.py` - Configuration validation

**Key imports available:**
```python
from yolo_developer.config import (
    load_config,
    export_config,
    import_config,
    YoloConfig,
    ConfigurationError,
    validate_config,
)
```

**Export/Import already work (Story 1.8):**
```python
# Export excludes API keys automatically
export_config(config, Path("exported.yaml"))

# Import validates before applying
import_config(Path("exported.yaml"), Path("yolo.yaml"))
```

### Configuration Schema Structure

```python
class YoloConfig(BaseSettings):
    project_name: str  # Required
    llm: LLMConfig = Field(default_factory=LLMConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

class LLMConfig(BaseModel):
    cheap_model: str = "gpt-4o-mini"
    premium_model: str = "claude-sonnet-4-20250514"
    best_model: str = "claude-opus-4-5-20251101"
    openai_api_key: SecretStr | None = None  # Env var only
    anthropic_api_key: SecretStr | None = None  # Env var only

class QualityConfig(BaseModel):
    test_coverage_threshold: float = 0.80
    confidence_threshold: float = 0.90
    gate_thresholds: dict[str, GateThreshold] = {}
    seed_thresholds: SeedThresholdConfig = ...
    critical_paths: list[str] = ["orchestrator/", "gates/", "agents/"]

class MemoryConfig(BaseModel):
    persist_path: str = ".yolo/memory"
    vector_store_type: Literal["chromadb"] = "chromadb"
    graph_store_type: Literal["json", "neo4j"] = "json"
```

### CLI Pattern Options

**Option 1: Subcommands (Recommended)**
```bash
yolo config                    # Show current config
yolo config set key value      # Set a value
yolo config export             # Export to file
yolo config import file.yaml   # Import from file
```

Using Typer's app.command() with sub-apps or callback pattern:

```python
config_app = typer.Typer(help="Manage project configuration")
app.add_typer(config_app, name="config")

@config_app.callback(invoke_without_command=True)
def config_show(ctx: typer.Context, json_output: bool = False):
    """Show current configuration."""
    if ctx.invoked_subcommand is None:
        show_config(json_output)

@config_app.command("set")
def config_set(key: str, value: str, json_output: bool = False):
    """Set a configuration value."""
    ...

@config_app.command("export")
def config_export(output: Path = Path("yolo-config-export.yaml")):
    """Export configuration to file."""
    ...

@config_app.command("import")
def config_import(file: Path):
    """Import configuration from file."""
    ...
```

### Display Patterns

Follow existing CLI patterns from logs/status/tune commands:

```python
from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel
from yolo_developer.cli.display import (
    console,
    create_table,
    info_panel,
    success_panel,
    warning_panel,
    error_panel,
)

# Display config as tree
def display_config_tree(config: YoloConfig) -> None:
    tree = Tree("[bold]yolo.yaml[/bold]")

    # Project name
    tree.add(f"[cyan]project_name:[/cyan] {config.project_name}")

    # LLM section
    llm_branch = tree.add("[cyan]llm:[/cyan]")
    llm_branch.add(f"cheap_model: {config.llm.cheap_model}")
    llm_branch.add(f"premium_model: {config.llm.premium_model}")
    llm_branch.add(f"best_model: {config.llm.best_model}")
    llm_branch.add(f"openai_api_key: {'[dim]****[/dim]' if config.llm.openai_api_key else '[dim]not set[/dim]'}")
    llm_branch.add(f"anthropic_api_key: {'[dim]****[/dim]' if config.llm.anthropic_api_key else '[dim]not set[/dim]'}")

    # Quality section
    quality_branch = tree.add("[cyan]quality:[/cyan]")
    quality_branch.add(f"test_coverage_threshold: {config.quality.test_coverage_threshold}")
    quality_branch.add(f"confidence_threshold: {config.quality.confidence_threshold}")

    # Memory section
    memory_branch = tree.add("[cyan]memory:[/cyan]")
    memory_branch.add(f"persist_path: {config.memory.persist_path}")
    memory_branch.add(f"vector_store_type: {config.memory.vector_store_type}")
    memory_branch.add(f"graph_store_type: {config.memory.graph_store_type}")

    console.print(tree)
```

### Setting Nested Values

Support dotted key paths for nested config updates:

```python
def set_nested_value(data: dict, key_path: str, value: Any) -> None:
    """Set a value in nested dict using dotted key path.

    Examples:
        set_nested_value(data, "llm.cheap_model", "gpt-4o")
        set_nested_value(data, "quality.test_coverage_threshold", 0.85)
    """
    keys = key_path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            raise KeyError(f"Unknown configuration key: {key}")
        current = current[key]

    final_key = keys[-1]
    if final_key not in current:
        raise KeyError(f"Unknown configuration key: {final_key}")

    current[final_key] = value
```

### Key Dependencies

**Configuration (config/):**
- `load_config()` - Load project configuration
- `export_config()` - Export config to YAML (excludes secrets)
- `import_config()` - Import config from YAML
- `ConfigurationError` - Error class for config issues
- `YoloConfig` - Main configuration class

**Display Utilities (cli/display.py):**
- `console` - Rich console instance
- `create_table()` - Create styled table
- `info_panel()`, `success_panel()`, `warning_panel()`, `error_panel()` - Styled panels
- `coming_soon()` - Placeholder display (to be replaced)

### Architecture Patterns

Per ADR-008 and existing CLI patterns:
- Use structlog for logging
- Use typer.Exit() for error exits with code=1
- Follow same flag patterns as tune/logs commands (--json)
- Handle missing yolo.yaml gracefully
- Use Path objects for file operations
- Validate inputs early, return with warning_panel on invalid input

### Previous Story Intelligence (Story 12.7)

**Learnings from yolo tune implementation:**
- Added YAML error handling for invalid syntax
- Used `typer.Exit(code=1)` for invalid inputs (per project pattern)
- Removed empty `TYPE_CHECKING` blocks (dead code)
- Tests use `strip_ansi()` helper for Rich output assertions
- Consolidated case-sensitivity handling (normalize agent names to lowercase)

**Patterns to follow:**
- Validate input early, return with warning_panel on invalid input
- Use normalized variables for consistent handling
- Use constants for display values and magic strings
- Keep functions focused (single responsibility)
- Test coverage should include error cases

### JSON Output Structure

For `yolo config` (show):
```json
{
  "project_name": "yolo-developer",
  "llm": {
    "cheap_model": "gpt-4o-mini",
    "premium_model": "claude-sonnet-4-20250514",
    "best_model": "claude-opus-4-5-20251101",
    "openai_api_key": "****",
    "anthropic_api_key": null
  },
  "quality": {
    "test_coverage_threshold": 0.80,
    "confidence_threshold": 0.90,
    "gate_thresholds": {},
    "seed_thresholds": {
      "overall": 0.70,
      "ambiguity": 0.60,
      "sop": 0.80
    },
    "critical_paths": ["orchestrator/", "gates/", "agents/"]
  },
  "memory": {
    "persist_path": ".yolo/memory",
    "vector_store_type": "chromadb",
    "graph_store_type": "json"
  }
}
```

For `yolo config set`:
```json
{
  "key": "llm.cheap_model",
  "old_value": "gpt-4o-mini",
  "new_value": "gpt-4o",
  "status": "success"
}
```

For `yolo config export`:
```json
{
  "path": "/absolute/path/to/exported.yaml",
  "status": "success"
}
```

For `yolo config import`:
```json
{
  "source": "/absolute/path/to/source.yaml",
  "target": "/absolute/path/to/yolo.yaml",
  "status": "success"
}
```

### Project Structure Notes

- CLI command: `src/yolo_developer/cli/commands/config.py` (replace placeholder)
- CLI wiring: `src/yolo_developer/cli/main.py:471-481` (needs update for subcommands)
- Tests: `tests/unit/cli/test_config_command.py` (new)
- Display utilities: `src/yolo_developer/cli/display.py`
- Config module: `src/yolo_developer/config/` (fully implemented)

### Test File Location

Tests should mirror the pattern from tune:
- `tests/unit/cli/test_config_command.py` - New test file

### Valid Configuration Keys

For reference when implementing validation:
```python
VALID_CONFIG_KEYS = {
    "project_name",
    "llm.cheap_model",
    "llm.premium_model",
    "llm.best_model",
    "quality.test_coverage_threshold",
    "quality.confidence_threshold",
    "quality.seed_thresholds.overall",
    "quality.seed_thresholds.ambiguity",
    "quality.seed_thresholds.sop",
    "memory.persist_path",
    "memory.vector_store_type",
    "memory.graph_store_type",
}

# Keys that cannot be set via CLI (security)
PROTECTED_KEYS = {
    "llm.openai_api_key",
    "llm.anthropic_api_key",
}
```

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-12.8]
- [Source: src/yolo_developer/cli/commands/config.py] - Placeholder to replace
- [Source: src/yolo_developer/cli/main.py:471-481] - CLI wiring
- [Source: src/yolo_developer/config/schema.py] - Configuration schema
- [Source: src/yolo_developer/config/export.py] - Export/import already implemented
- [Source: src/yolo_developer/config/__init__.py] - Public API
- [Related: FR104 - Users can manage configuration via yolo config command]
- [Related: FR96 - Users can export and import project configurations]
- [Related: ADR-008 - Pydantic Settings with YAML override]
- [Related: Story 12.7 (yolo tune command) - CLI patterns]
- [Related: Story 1.8 (export/import) - Export functionality already implemented]

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

N/A

### Completion Notes List

1. Implemented using Typer sub-app pattern (`add_typer`) for `yolo config` subcommands
2. Leveraged existing `export_config()` and `import_config()` functions from config module (Story 1.8)
3. Added `VALID_CONFIG_KEYS` and `PROTECTED_KEYS` constants for key validation and security
4. Used `print()` instead of `console.print()` for ALL JSON output to avoid Rich line-wrapping issues
5. All tests pass (39 config tests + 18 other CLI tests = 57 CLI tests total)
6. mypy and ruff checks pass with strict mode
7. API keys are masked by default in both Rich tree and JSON output
8. Protected keys (API keys) cannot be set via CLI - environment variables required
9. YAML comments are NOT preserved when using `yolo config set` (uses yaml.safe_dump)

### Code Review Fixes Applied

- Fixed inconsistent JSON error output (changed `console.print()` to `print()` for all JSON)
- Added tests for `--no-mask` flag
- Added tests for invalid numeric value conversion
- Added tests for YAML syntax error handling
- Clarified `_set_nested_value` docstring (intermediate key creation is safe after validation)

### File List

**Created:**
- `tests/unit/cli/test_config_command.py` (~490 lines) - 39 comprehensive tests

**Modified:**
- `src/yolo_developer/cli/commands/config.py` - Full implementation (~520 lines)
- `src/yolo_developer/cli/main.py` - Added config sub-app with subcommands
- `tests/unit/cli/test_main_commands.py` - Updated to test show_config and handle groups
- `tests/unit/cli/test_placeholder_commands.py` - Removed config placeholder tests
