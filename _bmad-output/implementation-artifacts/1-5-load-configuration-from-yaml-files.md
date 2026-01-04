# Story 1.5: Load Configuration from YAML Files

Status: done

## Story

As a developer,
I want to configure my project using YAML files,
So that I can version control and share configuration easily.

## Acceptance Criteria

1. **AC1: YAML File Loading**
   - **Given** a `yolo.yaml` file exists in the project root
   - **When** YOLO Developer loads configuration
   - **Then** all settings from yolo.yaml are applied
   - **And** the configuration priority is: defaults → YAML → environment variables

2. **AC2: Nested Configuration Support**
   - **Given** a YAML file with nested structure (llm, quality, memory sections)
   - **When** configuration is loaded
   - **Then** nested values are correctly mapped to nested Pydantic models
   - **And** `config.llm.cheap_model` returns the YAML-specified value

3. **AC3: Default Value Fallback**
   - **Given** a YAML file with only some settings defined
   - **When** configuration is loaded
   - **Then** missing optional values use defaults from YoloConfig
   - **And** the configuration is valid without all settings specified

4. **AC4: Helpful Parse Error Messages**
   - **Given** a YAML file with syntax errors
   - **When** loading is attempted
   - **Then** a clear error message is raised
   - **And** the error includes the line number where the problem occurred
   - **And** the error explains what is wrong

5. **AC5: Missing File Handling**
   - **Given** no `yolo.yaml` file exists
   - **When** configuration is loaded
   - **Then** defaults are used (no error)
   - **And** environment variables can still override defaults

## Tasks / Subtasks

- [x] Task 1: Add pydantic-settings-yaml Dependency (AC: 1)
  - [x] ~~Add `pydantic-settings-yaml>=2.0.0` to pyproject.toml dependencies~~ (Not needed - used built-in pydantic-settings YamlConfigSettingsSource)
  - [x] ~~Run `uv sync` to install the dependency~~ (pydantic-settings already has YAML support)
  - [x] ~~Verify import works~~ (Verified: `from pydantic_settings import YamlConfigSettingsSource` works)
  - [x] Added `types-PyYAML` to dev dependencies for mypy type checking

- [x] Task 2: Create Config Loader Module (AC: 1, 2, 5)
  - [x] Create `src/yolo_developer/config/loader.py`
  - [x] Implement `load_config(config_path: Path | None = None) -> YoloConfig` function
  - [x] Support default path of `yolo.yaml` in current directory
  - [x] Handle missing file gracefully (use defaults only)
  - [x] Ensure YAML → env variable priority order with custom `_merge_with_env_vars()` function

- [x] Task 3: Update YoloConfig for YAML Support (AC: 1, 2)
  - [x] ~~Modify `YoloConfig.model_config`~~ (Not needed - loader handles YAML loading externally)
  - [x] Loader parses YAML and merges with env vars before passing to YoloConfig
  - [x] Ensure nested models (LLMConfig, QualityConfig, MemoryConfig) load from YAML
  - [x] Maintain backward compatibility with environment-only loading

- [x] Task 4: Implement Parse Error Handling (AC: 4)
  - [x] Wrap YAML parsing in try/except for yaml.YAMLError
  - [x] Create custom `ConfigurationError` exception class
  - [x] Include line number in error message using yaml.YAMLError.problem_mark
  - [x] Provide actionable error messages explaining what's wrong

- [x] Task 5: Update Config Module Exports (AC: all)
  - [x] Export `load_config` from `config/__init__.py`
  - [x] Export `ConfigurationError` exception
  - [x] Ensure public API is clean and documented

- [x] Task 6: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/config/test_loader.py`
  - [x] Test: load_config with valid YAML file
  - [x] Test: load_config with nested configuration
  - [x] Test: load_config with partial YAML (some defaults used)
  - [x] Test: load_config with missing file (uses defaults)
  - [x] Test: load_config with invalid YAML syntax (error with line number)
  - [x] Test: load_config with invalid config values (ValidationError)
  - [x] Test: environment variables override YAML values
  - [x] Test: YAML overrides defaults

## Dev Notes

### Critical Architecture Requirements

**From ADR-008 (Configuration Management):**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class YoloConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YOLO_",
        yaml_file="yolo.yaml",  # <-- ADD THIS
    )
```

The architecture specifies a layered configuration approach:
- Schema validation catches config errors early
- **YAML is human-readable for project config**
- Environment variables for secrets
- **Layered priority: defaults → YAML → env → CLI**

### Current State (from Story 1.4)

The `YoloConfig` class already exists with:
```python
model_config = SettingsConfigDict(
    env_prefix="YOLO_",
    env_nested_delimiter="__",
    extra="forbid",
)
```

**What's missing:** YAML file loading capability

### Implementation Options

**Option 1: pydantic-settings-yaml (Recommended)**

```python
# pyproject.toml
dependencies = [
    "pydantic-settings-yaml>=2.0.0",
    # ... existing deps
]

# schema.py or loader.py
from pydantic_settings_yaml import YamlBaseSettings

class YoloConfig(YamlBaseSettings):
    yaml_file = "yolo.yaml"
    # ... rest stays the same
```

**Option 2: Custom YAML Loader**

```python
import yaml
from pathlib import Path
from pydantic import ValidationError

def load_config(config_path: Path | None = None) -> YoloConfig:
    """Load configuration from YAML file with env override."""
    yaml_data = {}

    path = config_path or Path("yolo.yaml")
    if path.exists():
        try:
            with open(path) as f:
                yaml_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            if hasattr(e, 'problem_mark'):
                mark = e.problem_mark
                raise ConfigurationError(
                    f"YAML parse error at line {mark.line + 1}, column {mark.column + 1}: {e.problem}"
                ) from e
            raise ConfigurationError(f"YAML parse error: {e}") from e

    # YoloConfig will merge YAML with env vars
    return YoloConfig(**yaml_data)
```

### Expected YAML File Format

```yaml
# yolo.yaml - YOLO Developer Configuration
project_name: my-awesome-project

llm:
  cheap_model: gpt-4o-mini
  premium_model: claude-sonnet-4-20250514
  best_model: claude-opus-4-5-20251101

quality:
  test_coverage_threshold: 0.85
  confidence_threshold: 0.92

memory:
  persist_path: .yolo/memory
  vector_store_type: chromadb
  graph_store_type: json
```

### Error Message Examples

**Good error message (AC4):**
```
ConfigurationError: Invalid YAML at line 5, column 3:
  mapping values are not allowed here

  Did you mean to use proper YAML indentation?

  Problematic section:
    llm:
      cheap_model:gpt-4o-mini  # Missing space after colon
```

**Validation error:**
```
ConfigurationError: Invalid configuration value:
  quality.test_coverage_threshold: Value 1.5 is not valid.
  Must be between 0.0 and 1.0.
```

### Testing Approach

```python
import pytest
from pathlib import Path
from yolo_developer.config import load_config, ConfigurationError

class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Load configuration from valid YAML file."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("""
project_name: test-project
llm:
  cheap_model: custom-model
""")
        config = load_config(yaml_file)
        assert config.project_name == "test-project"
        assert config.llm.cheap_model == "custom-model"
        # Defaults still apply
        assert config.quality.test_coverage_threshold == 0.80

    def test_load_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        """Missing YAML file should use defaults without error."""
        # No file created - should still work with env var
        import os
        os.environ["YOLO_PROJECT_NAME"] = "env-project"
        try:
            config = load_config(tmp_path / "nonexistent.yaml")
            assert config.project_name == "env-project"
        finally:
            del os.environ["YOLO_PROJECT_NAME"]

    def test_yaml_syntax_error_includes_line_number(self, tmp_path: Path) -> None:
        """YAML syntax errors should include line numbers."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("""
project_name: test
llm:
  cheap_model:missing-space
""")
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        assert "line" in str(exc_info.value).lower()

    def test_env_overrides_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables should override YAML values."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("""
project_name: yaml-project
llm:
  cheap_model: yaml-model
""")
        monkeypatch.setenv("YOLO_LLM__CHEAP_MODEL", "env-model")
        config = load_config(yaml_file)
        assert config.project_name == "yaml-project"
        assert config.llm.cheap_model == "env-model"  # Env wins
```

### Project Structure Notes

**File location:** `src/yolo_developer/config/loader.py`

**Architecture reference:**
```
└── config/                 # Configuration (FR89-97)
    ├── __init__.py        # Exports load_config, YoloConfig, etc.
    ├── schema.py           # Pydantic Settings models (Story 1.4) ✅
    ├── loader.py           # Config loading logic (THIS STORY)
    └── defaults.py         # Default configurations (future)
```

### Dependencies

**Already in pyproject.toml:**
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `pyyaml` (for YAML parsing)

**May need to add:**
- `pydantic-settings-yaml>=2.0.0` (simplifies YAML integration)

### Previous Story Learnings (Story 1.4)

1. Use `from __future__ import annotations` in all Python files
2. Export public API from `__init__.py` (e.g., `YoloConfig`, `LLMConfig`)
3. Field constraints with `ge=0.0, le=1.0` for thresholds work well
4. Comprehensive tests for all edge cases (43 tests for schema)
5. Run `ruff check`, `ruff format`, and `mypy` before marking complete

### Git Commit Patterns

Recent commits follow format:
```
feat: Description (Story X.X)
```

Example for this story:
```
feat: Add YAML configuration file loading (Story 1.5)
```

### References

- [Source: architecture.md#ADR-008] - Configuration Management decision
- [Source: epics.md#Story 1.5] - Story requirements
- [Source: Story 1.4 completion notes] - Previous implementation patterns
- [pydantic-settings-yaml PyPI](https://pypi.org/project/pydantic-settings-yaml/)
- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Pydantic Settings Sources](https://docs.pydantic.dev/latest/concepts/pydantic_settings/#settings-sources)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 156 tests pass (129 from Stories 1.1-1.4 + 27 new loader tests)
- mypy: Success, no issues found in config module
- ruff check: All checks passed
- ruff format: All files formatted correctly

### Completion Notes List

1. **Design Decision:** Used custom YAML loader instead of `pydantic-settings-yaml` package because:
   - `pydantic-settings` already includes `YamlConfigSettingsSource` (built-in)
   - Custom loader gives more control over priority order (defaults → YAML → env vars)
   - No new dependency needed beyond existing `pyyaml`

2. Created `src/yolo_developer/config/loader.py` with:
   - `load_config(config_path: Path | None = None) -> YoloConfig` function
   - `ConfigurationError` exception class for all config errors
   - `_load_yaml_file()` with YAML parsing and error handling
   - `_merge_with_env_vars()` to properly layer env vars on top of YAML
   - `_create_yaml_error()` for helpful error messages with line/column numbers
   - `_create_validation_error()` for Pydantic validation error wrapping

3. Updated `src/yolo_developer/config/__init__.py` to export:
   - `load_config` function
   - `ConfigurationError` exception

4. Added `types-PyYAML` to dev dependencies for mypy type checking

5. Created comprehensive test suite with 29 tests covering:
   - Valid YAML loading with project_name, nested configs
   - Partial YAML with default fallbacks
   - Empty YAML file handling
   - Missing file handling with env var support
   - YAML syntax errors with line/column numbers
   - Validation errors with descriptive messages
   - Environment variable overrides (priority order verified)
   - Module exports and code quality

6. **Code Review Fixes (7 issues addressed):**
   - Added logging warning for env vars with >2 levels of nesting
   - Improved `_convert_value()` with automatic type inference (bool, int, float)
   - Fixed hardcoded test path to use `inspect.getfile()`
   - Updated test to properly validate hint text in error messages
   - Added tests for empty YAML file and comments-only YAML file
   - Added `@pytest.mark.slow` marker for mypy subprocess test
   - Registered slow marker in pyproject.toml

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Implemented all tasks, 156 tests pass | Dev Agent |
| 2026-01-04 | Code review completed, fixed 7 issues, 158 tests pass | Code Review |

### File List

- `src/yolo_developer/config/loader.py` - New configuration loader module with load_config function and ConfigurationError exception
- `src/yolo_developer/config/__init__.py` - Updated to export load_config and ConfigurationError
- `tests/unit/config/test_loader.py` - Comprehensive test suite for loader module (29 tests after code review fixes)
- `pyproject.toml` - Added types-PyYAML to dev dependencies
- `uv.lock` - Auto-updated by uv sync (dependency lockfile)
