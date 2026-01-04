# Story 1.8: Export and Import Project Configurations

Status: done

## Story

As a developer,
I want to export my configuration and import it to other projects,
So that I can standardize settings across my work.

## Acceptance Criteria

1. **AC1: Export Configuration to YAML**
   - **Given** I have a configured YOLO Developer project
   - **When** I call `export_config(config, output_path)`
   - **Then** configuration is exported to a portable YAML file
   - **And** the output file is valid YAML that can be loaded

2. **AC2: Exclude Sensitive Values from Export**
   - **Given** a configuration with API keys set
   - **When** I export the configuration
   - **Then** sensitive values (API keys) are excluded from export
   - **And** placeholders indicate where secrets should be set via env vars
   - **And** export comments explain how to set secrets

3. **AC3: Import Configuration from YAML**
   - **Given** a previously exported YAML configuration file
   - **When** I call `import_config(source_path, target_path)`
   - **Then** the configuration is imported to the target path
   - **And** the imported configuration is validated before writing

4. **AC4: Preserve Configuration Structure**
   - **Given** a configuration with nested settings
   - **When** I export and import the configuration
   - **Then** all nested sections (llm, quality, memory) are preserved
   - **And** field order is consistent and readable

5. **AC5: Handle Missing Source Files**
   - **Given** an import operation with a non-existent source file
   - **When** I call `import_config()`
   - **Then** a ConfigurationError is raised with helpful message
   - **And** the message includes the full path that was not found

## Tasks / Subtasks

- [x] Task 1: Create Export Module Structure (AC: 1, 4)
  - [x] Create `src/yolo_developer/config/export.py` module
  - [x] Define `export_config(config: YoloConfig, output_path: Path) -> None` function
  - [x] Add export module to `__init__.py` exports
  - [x] Use `yaml.safe_dump()` with appropriate options for readable output

- [x] Task 2: Implement Secret Exclusion (AC: 2)
  - [x] Create helper function to convert config to dict excluding secrets
  - [x] Replace `SecretStr` values with placeholder comment/value
  - [x] Add header comment explaining environment variable usage
  - [x] Use `# Set via YOLO_LLM__OPENAI_API_KEY` style comments

- [x] Task 3: Implement Configuration Import (AC: 3, 5)
  - [x] Create `import_config(source_path: Path, target_path: Path | None = None) -> None`
  - [x] Validate source file exists before attempting import
  - [x] Load and validate source YAML before writing to target
  - [x] Default target_path to `yolo.yaml` in current directory

- [x] Task 4: Add Input Validation (AC: 5)
  - [x] Raise `ConfigurationError` for missing source files
  - [x] Include full absolute path in error message
  - [x] Validate imported YAML parses correctly before writing

- [x] Task 5: Write Unit Tests (AC: all)
  - [x] Test: Export creates valid YAML file
  - [x] Test: Export excludes API keys from output
  - [x] Test: Export includes all non-secret configuration
  - [x] Test: Import reads from source and writes to target
  - [x] Test: Import validates configuration before writing
  - [x] Test: Import raises error for missing source file
  - [x] Test: Round-trip export/import preserves configuration values

## Dev Notes

### Critical Architecture Requirements

**From ADR-008 (Configuration Management):**
- Configuration export/import for portability
- Secrets excluded from exports (env vars only)
- YAML format for human readability

**From NFR-SEC-1:**
- API keys via environment variables only
- Never persist secrets to config files

### Current State (from Stories 1.4-1.7)

The existing implementation provides:

1. **`YoloConfig` schema** in `schema.py`:
   - Pydantic model with nested `LLMConfig`, `QualityConfig`, `MemoryConfig`
   - API keys stored as `SecretStr` (masked in repr/logs)
   - `model_dump()` method available for dict conversion

2. **`load_config()` in `loader.py`**:
   - Loads from YAML with env var overrides
   - Full validation pipeline with `validate_config()`

3. **`ConfigurationError`** for error handling

### Implementation Approach

**Export Implementation:**

```python
# src/yolo_developer/config/export.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from yolo_developer.config.schema import YoloConfig

# Fields that contain secrets and should be excluded from export
SECRET_FIELDS = {"openai_api_key", "anthropic_api_key"}


def export_config(config: YoloConfig, output_path: Path) -> None:
    """Export configuration to YAML file, excluding secrets.

    Args:
        config: The YoloConfig instance to export.
        output_path: Path where YAML file will be written.
    """
    # Convert to dict, excluding secrets
    data = _config_to_exportable_dict(config)

    # Write with header comment
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# YOLO Developer Configuration\n")
        f.write("# API keys must be set via environment variables:\n")
        f.write("#   YOLO_LLM__OPENAI_API_KEY=your-key-here\n")
        f.write("#   YOLO_LLM__ANTHROPIC_API_KEY=your-key-here\n")
        f.write("\n")
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _config_to_exportable_dict(config: YoloConfig) -> dict[str, Any]:
    """Convert config to dict, excluding secret fields."""
    data = config.model_dump(mode="json")

    # Remove secrets from llm section
    if "llm" in data:
        for field in SECRET_FIELDS:
            data["llm"].pop(field, None)

    return data
```

**Import Implementation:**

```python
def import_config(
    source_path: Path,
    target_path: Path | None = None,
) -> None:
    """Import configuration from YAML file.

    Args:
        source_path: Path to source YAML configuration.
        target_path: Path to write imported config. Defaults to ./yolo.yaml.

    Raises:
        ConfigurationError: If source file doesn't exist or is invalid.
    """
    from yolo_developer.config.loader import ConfigurationError, _load_yaml_file

    if not source_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {source_path.absolute()}"
        )

    # Load and validate source
    yaml_data = _load_yaml_file(source_path)

    # Validate by attempting to create config
    # This catches validation errors before writing
    from yolo_developer.config.schema import YoloConfig
    try:
        YoloConfig(**yaml_data)
    except Exception as e:
        raise ConfigurationError(
            f"Invalid configuration in {source_path}: {e}"
        ) from e

    # Write to target
    target = target_path or Path("yolo.yaml")
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, sort_keys=False)
```

### Testing Approach

```python
import pytest
from pathlib import Path
from yolo_developer.config import load_config
from yolo_developer.config.export import export_config, import_config
from yolo_developer.config.loader import ConfigurationError


class TestExportConfig:
    def test_export_creates_valid_yaml(self, tmp_path: Path) -> None:
        """Exported file should be valid YAML."""
        config = load_config_with_defaults(tmp_path)
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        assert output.exists()
        # Should be loadable
        loaded = load_config(output)
        assert loaded.project_name == config.project_name

    def test_export_excludes_api_keys(self, tmp_path: Path) -> None:
        """API keys should not appear in export."""
        config = load_config_with_api_keys(tmp_path)
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        content = output.read_text()
        assert "openai_api_key" not in content
        assert "anthropic_api_key" not in content
        assert "secret" not in content.lower()


class TestImportConfig:
    def test_import_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Missing source file should raise ConfigurationError."""
        missing = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            import_config(missing)

        assert "not found" in str(exc_info.value)
        assert str(missing.absolute()) in str(exc_info.value)
```

### Project Structure Notes

- New module: `src/yolo_developer/config/export.py`
- Updates to: `src/yolo_developer/config/__init__.py` (export public API)
- New tests: `tests/unit/config/test_export.py`

### Previous Story Learnings (Stories 1.5-1.7)

1. Use `from __future__ import annotations` in all Python files
2. Export public API from `__init__.py`
3. Run `ruff check`, `ruff format`, and `mypy` before marking complete
4. Use `ConfigurationError` for all config-related errors
5. Include full paths in error messages for better debugging
6. Test both success paths and error conditions

### References

- [Source: architecture.md#ADR-008] - Configuration Management
- [Source: prd.md#NFR-SEC-1] - API key security
- [Source: epics.md#Story 1.8] - Story requirements
- [Source: schema.py] - YoloConfig schema with SecretStr API keys
- [Source: loader.py] - Existing load_config, _load_yaml_file functions

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 229 tests pass (16 new export tests + 213 existing tests)
- mypy: Success, no issues found in 5 source files (config module)
- ruff check: All checks passed
- ruff format: All files formatted correctly

### Completion Notes List

1. **Task 1 - Export Module Structure**: Created `src/yolo_developer/config/export.py` with `export_config()` function that uses `yaml.safe_dump()` with `default_flow_style=False` and `sort_keys=False` for readable output. Added exports to `__init__.py`.

2. **Task 2 - Secret Exclusion**: Implemented `_config_to_exportable_dict()` helper that removes `openai_api_key` and `anthropic_api_key` from the llm section. Added `EXPORT_HEADER` constant with comments explaining how to set API keys via environment variables.

3. **Task 3 - Configuration Import**: Implemented `import_config(source_path, target_path)` that validates source exists, loads and validates YAML, then writes to target. Defaults target to `yolo.yaml` in current directory.

4. **Task 4 - Input Validation**: `import_config()` raises `ConfigurationError` for missing files with full absolute path in message. Validates YAML by attempting to create `YoloConfig` before writing.

5. **Task 5 - Unit Tests**: Created 16 comprehensive tests in `tests/unit/config/test_export.py` covering:
   - Export creates valid YAML file (6 tests for different config sections)
   - Export excludes API keys (3 tests for secrets and header comments)
   - Import reads and writes correctly (3 tests)
   - Import error handling (3 tests for missing files and invalid YAML)
   - Round-trip preservation (1 test verifying all values survive export/import)

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Implemented all tasks, 229 tests pass | Dev Agent |
| 2026-01-04 | Code review: 4 issues fixed, 232 tests pass | Code Reviewer |

### File List

- `src/yolo_developer/config/export.py` - New module with export_config(), import_config(), _config_to_exportable_dict()
- `src/yolo_developer/config/__init__.py` - Added exports for export_config, import_config
- `tests/unit/config/test_export.py` - New test file with 19 tests covering all AC

## Senior Developer Review (AI)

**Reviewer:** Claude Opus 4.5 (claude-opus-4-5-20251101)
**Review Date:** 2026-01-04
**Verdict:** PASS with fixes applied

### Issues Found and Resolved

| # | Severity | Issue | Resolution |
|---|----------|-------|------------|
| 1 | HIGH | No directory creation for target path - both export and import fail with FileNotFoundError if parent dir doesn't exist | Added `output.parent.mkdir(parents=True, exist_ok=True)` to both functions |
| 2 | MEDIUM | Import loses header comments from exported files | Added EXPORT_HEADER to import_config output |
| 3 | MEDIUM | Missing tests for directory creation edge case | Added 3 new tests: export creates dirs, import creates dirs, import adds header |
| 4 | MEDIUM | Silent overwrite behavior undocumented | Added documentation noting standard config tool behavior |

### Test Results After Fixes

- 232 tests pass (19 export tests + 213 existing)
- mypy: Success, no issues found
- ruff check: All checks passed
- ruff format: All files formatted

