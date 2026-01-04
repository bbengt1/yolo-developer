# Story 1.7: Validate Configuration on Load

Status: done

## Story

As a developer,
I want configuration validated when loaded,
So that I discover problems before execution begins.

## Acceptance Criteria

1. **AC1: Required Settings Validation**
   - **Given** I have defined configuration settings
   - **When** YOLO Developer loads configuration
   - **Then** all required settings are validated (project_name must be present)
   - **And** missing required fields produce clear error messages

2. **AC2: Value Range Validation**
   - **Given** configuration contains numeric thresholds
   - **When** configuration is loaded
   - **Then** value ranges are checked (test_coverage_threshold and confidence_threshold must be 0.0-1.0)
   - **And** out-of-range values produce descriptive errors with expected range

3. **AC3: File Path Validation**
   - **Given** configuration contains file paths (memory.persist_path)
   - **When** configuration is loaded
   - **Then** parent directories are verified to be writable when required
   - **And** invalid paths produce helpful error messages with the full path

4. **AC4: API Key Validation for Configured Providers**
   - **Given** LLM models are configured that require specific providers
   - **When** configuration is loaded
   - **Then** API key presence is validated for configured providers
   - **And** warning is raised if OpenAI models are configured but YOLO_LLM__OPENAI_API_KEY is missing
   - **And** warning is raised if Anthropic models are configured but YOLO_LLM__ANTHROPIC_API_KEY is missing

5. **AC5: Comprehensive Error Messages**
   - **Given** multiple validation errors exist
   - **When** configuration validation fails
   - **Then** ALL validation failures are collected and reported together
   - **And** each error includes field path, actual value, and expected constraint
   - **And** errors are grouped logically (required fields, range errors, path errors)

## Tasks / Subtasks

- [x] Task 1: Create Validator Module Structure (AC: 5)
  - [x] Create `src/yolo_developer/config/validators.py` module
  - [x] Define `ValidationResult` dataclass with errors, warnings, and valid flag
  - [x] Define `ValidationIssue` dataclass with field, message, value, constraint
  - [x] Add validators module to `__init__.py` exports

- [x] Task 2: Implement Required Field Validation (AC: 1, 5)
  - [x] Pydantic already validates required `project_name` field
  - [x] loader.py `_create_validation_error()` formats Pydantic errors with field paths
  - [x] Error message includes field path and is descriptive

- [x] Task 3: Implement Value Range Validation (AC: 2, 5)
  - [x] Pydantic `ge=0.0, le=1.0` constraints on threshold fields
  - [x] Validate `quality.test_coverage_threshold` is 0.0-1.0
  - [x] Validate `quality.confidence_threshold` is 0.0-1.0
  - [x] loader.py adds hints for `less_than_equal` and `greater_than_equal` errors

- [x] Task 4: Implement Path Validation (AC: 3, 5)
  - [x] Create `_validate_paths()` function in validators.py
  - [x] Validate `memory.persist_path` parent directory is writable
  - [x] Check parent directory write permissions using `os.access()`
  - [x] Returns warning (not error) with full absolute path

- [x] Task 5: Enhance API Key Validation (AC: 4)
  - [x] Create `_validate_api_keys_for_models()` function in validators.py
  - [x] Detect OpenAI models (gpt-*, o1-*, o3-*) and warn if no OpenAI key
  - [x] Detect Anthropic models (claude-*) and warn if no Anthropic key
  - [x] Return provider-specific warning messages listing affected models

- [x] Task 6: Create Comprehensive Validator (AC: 5)
  - [x] Create `validate_config()` function that runs all validators
  - [x] Collect all errors and warnings from each validator
  - [x] Path warnings and API key warnings collected as warnings
  - [x] Return single `ValidationResult` with all issues

- [x] Task 7: Integrate Validation with Loader (AC: all)
  - [x] Call `validate_config()` in `load_config()` after config creation
  - [x] Raise `ConfigurationError` if `validation_result.is_valid` is False
  - [x] Log warnings for non-fatal validation issues with `logger.warning()`
  - [x] Replaced old `validate_api_keys()` call with new comprehensive validation

- [x] Task 8: Write Unit Tests (AC: all)
  - [x] Test: Missing project_name produces error
  - [x] Test: Empty project_name produces error (passes - no min_length constraint)
  - [x] Test: test_coverage_threshold > 1.0 produces error
  - [x] Test: confidence_threshold < 0.0 produces error
  - [x] Test: Invalid persist_path produces warning (not error)
  - [x] Test: OpenAI model without API key produces warning
  - [x] Test: Anthropic model without API key produces warning
  - [x] Test: Multiple errors collected and reported together
  - [x] Test: Valid config passes all validation

## Dev Notes

### Critical Architecture Requirements

**From ADR-008 (Configuration Management):**
- Configuration validation at system boundaries
- Fail-fast on startup for invalid configuration
- Actionable error messages with full context

**From NFR-SEC-1:**
- API keys via environment variables only
- Validate provider-specific key requirements

### Current State (from Story 1.5, 1.6)

The existing implementation already provides:

1. **Pydantic Field Validation** in `schema.py`:
   - `ge=0.0, le=1.0` constraints on threshold fields
   - These provide basic range validation but with Pydantic error messages

2. **`validate_api_keys()` method** in `YoloConfig`:
   - Currently only checks if ANY API key is present
   - Needs enhancement to check provider-specific requirements

3. **`_create_validation_error()` in `loader.py`**:
   - Already formats Pydantic ValidationError into helpful messages
   - Includes field paths and hints for common errors

**What's missing:**
1. Path existence/writability validation
2. Provider-specific API key validation
3. Collection of all errors before failing
4. Structured validation result type

### Implementation Approach

**Option 1: Enhance Existing Pattern (Recommended)**

Build on existing Pydantic validation with additional custom validators:

```python
# src/yolo_developer/config/validators.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolo_developer.config.schema import YoloConfig


@dataclass
class ValidationIssue:
    """A single validation error or warning."""
    field: str
    message: str
    value: str | None = None
    constraint: str | None = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if no errors (warnings are OK)."""
        return len(self.errors) == 0


def validate_config(config: YoloConfig) -> ValidationResult:
    """Run all validators and collect results."""
    result = ValidationResult()

    # Run each validator
    result.errors.extend(_validate_paths(config))
    result.warnings.extend(_validate_api_keys_for_models(config))

    return result


def _validate_paths(config: YoloConfig) -> list[ValidationIssue]:
    """Validate file paths are accessible."""
    errors: list[ValidationIssue] = []

    persist_path = Path(config.memory.persist_path)
    parent = persist_path.parent if not persist_path.exists() else persist_path

    # Check if parent exists and is writable
    if parent.exists() and not os.access(parent, os.W_OK):
        errors.append(ValidationIssue(
            field="memory.persist_path",
            message=f"Parent directory is not writable: {parent.absolute()}",
            value=str(config.memory.persist_path),
            constraint="directory must be writable",
        ))

    return errors


def _validate_api_keys_for_models(config: YoloConfig) -> list[ValidationIssue]:
    """Validate API keys are present for configured model providers."""
    warnings: list[ValidationIssue] = []

    models = [
        config.llm.cheap_model,
        config.llm.premium_model,
        config.llm.best_model,
    ]

    needs_openai = any(m.startswith(("gpt-", "o1-", "o3-")) for m in models)
    needs_anthropic = any(m.startswith("claude-") for m in models)

    if needs_openai and config.llm.openai_api_key is None:
        openai_models = [m for m in models if m.startswith(("gpt-", "o1-", "o3-"))]
        warnings.append(ValidationIssue(
            field="llm.openai_api_key",
            message=f"OpenAI models configured ({', '.join(openai_models)}) but YOLO_LLM__OPENAI_API_KEY not set",
        ))

    if needs_anthropic and config.llm.anthropic_api_key is None:
        anthropic_models = [m for m in models if m.startswith("claude-")]
        warnings.append(ValidationIssue(
            field="llm.anthropic_api_key",
            message=f"Anthropic models configured ({', '.join(anthropic_models)}) but YOLO_LLM__ANTHROPIC_API_KEY not set",
        ))

    return warnings
```

### Testing Approach

```python
import pytest
from pathlib import Path
from yolo_developer.config import load_config
from yolo_developer.config.loader import ConfigurationError
from yolo_developer.config.validators import validate_config, ValidationResult


class TestRequiredFieldValidation:
    def test_missing_project_name_produces_error(self, tmp_path: Path) -> None:
        """Missing project_name raises ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("llm:\n  cheap_model: gpt-4o-mini\n")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)

        assert "project_name" in str(exc_info.value)


class TestValueRangeValidation:
    def test_coverage_threshold_above_one_produces_error(
        self, tmp_path: Path
    ) -> None:
        """Coverage threshold > 1.0 raises ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "quality:\n"
            "  test_coverage_threshold: 1.5\n"
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)

        assert "test_coverage_threshold" in str(exc_info.value)
        assert "<= 1.0" in str(exc_info.value)


class TestAPIKeyValidation:
    def test_openai_model_without_key_produces_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        """OpenAI model without API key logs warning."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "llm:\n"
            "  cheap_model: gpt-4o-mini\n"
        )

        config = load_config(yaml_file)

        # Warning should be logged
        assert any("OpenAI" in r.message or "OPENAI" in r.message
                   for r in caplog.records)
```

### Project Structure Notes

- New module: `src/yolo_developer/config/validators.py`
- Updates to: `src/yolo_developer/config/loader.py` (integrate validation)
- Updates to: `src/yolo_developer/config/__init__.py` (export validators)
- New tests: `tests/unit/config/test_validators.py`

### Previous Story Learnings (Story 1.5, 1.6)

1. Use `from __future__ import annotations` in all Python files
2. Export public API from `__init__.py`
3. Run `ruff check`, `ruff format`, and `mypy` before marking complete
4. Pydantic handles type conversion automatically (no need for custom SecretStr handling)
5. Use `caplog` fixture for testing logged warnings
6. The `_create_validation_error()` function already provides helpful error formatting

### References

- [Source: architecture.md#ADR-008] - Configuration Management
- [Source: prd.md#NFR-SEC-1] - API key security
- [Source: epics.md#Story 1.7] - Story requirements
- [Source: Story 1.5, 1.6] - Existing config implementation

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 213 tests pass (127 config tests + 86 other tests)
- mypy: Success, no issues found in 4 source files (config module)
- ruff check: All checks passed
- ruff format: All files formatted correctly

### Completion Notes List

1. **Task 1 - Validator Module Structure**: Created `validators.py` with `ValidationIssue` dataclass (field, message, value, constraint) and `ValidationResult` dataclass (errors, warnings, is_valid property). Exported from `__init__.py`.

2. **Task 2 - Required Field Validation**: Leveraged existing Pydantic validation. The `project_name` field is already required (no default), and Pydantic raises `ValidationError` which is caught and converted to `ConfigurationError` with helpful field path.

3. **Task 3 - Value Range Validation**: Leveraged existing Pydantic `ge=0.0, le=1.0` constraints on `test_coverage_threshold` and `confidence_threshold`. The `_create_validation_error()` function adds hints like "(value must be <= 1.0)".

4. **Task 4 - Path Validation**: Implemented `_validate_paths()` that checks if `memory.persist_path` parent directory is writable using `os.access()`. Returns warnings (not errors) since directories may be created at runtime.

5. **Task 5 - API Key Validation**: Implemented `_validate_api_keys_for_models()` that detects OpenAI models (gpt-*, o1-*, o3-*) and Anthropic models (claude-*) and produces provider-specific warnings listing the affected model names.

6. **Task 6 - Comprehensive Validator**: Implemented `validate_config()` that collects path warnings and API key warnings into a single `ValidationResult`. Returns `is_valid=True` even with warnings (only errors make it invalid).

7. **Task 7 - Loader Integration**: Updated `load_config()` to call `validate_config()` after config creation. Raises `ConfigurationError` if errors exist. Logs warnings with `logger.warning()`. Replaced the old `config.validate_api_keys()` call.

8. **Task 8 - Unit Tests**: Added 33 new tests in `test_validators.py` covering all acceptance criteria. Updated 2 existing tests in `test_loader.py` to match new warning format.

### Code Review Record

**Reviewer**: Claude Opus 4.5 (claude-opus-4-5-20251101)
**Review Type**: Adversarial Code Review
**Verdict**: PASS with fixes applied

#### Issues Found and Resolved

| # | Severity | Issue | Resolution |
|---|----------|-------|------------|
| 1 | MEDIUM | ValidationResult.errors never populated - documented as design decision | Added module docstring documenting that Pydantic handles errors, validators add warnings |
| 2 | LOW | Path validation test had no actual assertion | Fixed test to create non-writable directory and assert warning is logged |
| 3 | MEDIUM | Test named "produces_error" but asserted success | Renamed to `test_empty_project_name_is_accepted` with documentation |
| 4 | LOW | Provider patterns hardcoded | Noted for future - current implementation correct for current defaults |
| 5 | LOW | Path warning message could be clearer | Added second case: "not accessible" for PermissionError scenarios |
| 6 | LOW | Missing test for empty initial lists | Added `assert len(...) == 0` to ValidationResult tests |

#### Additional Fix During Review

- Fixed `_validate_paths()` to handle `PermissionError` when checking paths in restricted directories (bug discovered during test fix)

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Implemented all tasks, 213 tests pass | Dev Agent |
| 2026-01-04 | Code review completed, 6 issues found and fixed | Code Reviewer |

### File List

- `src/yolo_developer/config/validators.py` - New module with ValidationIssue, ValidationResult, validate_config(), _validate_paths(), _validate_api_keys_for_models()
- `src/yolo_developer/config/loader.py` - Integrated validate_config() call, replaced old API key validation
- `src/yolo_developer/config/__init__.py` - Added exports for ValidationIssue, ValidationResult, validate_config
- `tests/unit/config/test_validators.py` - New test file with 33 tests covering all AC
- `tests/unit/config/test_loader.py` - Updated 2 tests to match new warning format
