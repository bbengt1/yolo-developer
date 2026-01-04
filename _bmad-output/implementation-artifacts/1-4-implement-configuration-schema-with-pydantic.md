# Story 1.4: Implement Configuration Schema with Pydantic

Status: done

## Story

As a developer,
I want a strongly-typed configuration schema,
So that configuration errors are caught early with helpful messages.

## Acceptance Criteria

1. **AC1: YoloConfig Class Uses Pydantic Settings**
   - **Given** I need to configure YOLO Developer settings
   - **When** I import YoloConfig from config/schema.py
   - **Then** YoloConfig inherits from pydantic_settings.BaseSettings
   - **And** the class uses SettingsConfigDict with env_prefix="YOLO_"

2. **AC2: All Configuration Options Have Type Hints**
   - **Given** the YoloConfig class is defined
   - **When** I inspect the configuration fields
   - **Then** every field has explicit type annotations
   - **And** mypy validates the schema with no errors

3. **AC3: Default Values for Optional Settings**
   - **Given** no configuration is provided
   - **When** I instantiate YoloConfig()
   - **Then** all optional settings have sensible default values
   - **And** the instance is valid without any external configuration

4. **AC4: ValidationError with Clear Messages**
   - **Given** an invalid configuration value is provided
   - **When** I attempt to instantiate YoloConfig with invalid data
   - **Then** a pydantic.ValidationError is raised
   - **And** the error message clearly describes what is wrong and how to fix it

5. **AC5: Nested Configuration Models**
   - **Given** configuration is logically grouped (LLM, Quality, Memory)
   - **When** I define the schema
   - **Then** nested Pydantic models group related settings
   - **And** the structure follows ADR-008 design

## Tasks / Subtasks

- [x] Task 1: Create LLM Configuration Model (AC: 2, 3, 5)
  - [x] Create `src/yolo_developer/config/schema.py` file
  - [x] Define `LLMConfig` nested model with cheap_model, premium_model, best_model fields
  - [x] Add type hints for all fields (str)
  - [x] Add default values matching ADR-008 (gpt-4o-mini, claude-sonnet-4-20250514, claude-opus-4-5-20251101)
  - [x] Add Field descriptions for documentation

- [x] Task 2: Create Quality Configuration Model (AC: 2, 3, 5)
  - [x] Define `QualityConfig` nested model
  - [x] Add test_coverage_threshold: float field with 0.80 default
  - [x] Add confidence_threshold: float field with 0.90 default
  - [x] Add Field validators for range constraints (0.0 to 1.0)

- [x] Task 3: Create Memory Configuration Model (AC: 2, 3, 5)
  - [x] Define `MemoryConfig` nested model
  - [x] Add persist_path: str field with ".yolo/memory" default
  - [x] Add vector_store_type: Literal["chromadb"] field
  - [x] Add graph_store_type: Literal["json", "neo4j"] field with "json" default

- [x] Task 4: Create Main YoloConfig Class (AC: 1, 2, 3, 5)
  - [x] Define `YoloConfig` class inheriting from BaseSettings
  - [x] Add SettingsConfigDict with env_prefix="YOLO_", env_nested_delimiter="__"
  - [x] Compose nested models: llm: LLMConfig, quality: QualityConfig, memory: MemoryConfig
  - [x] Add project_name: str field (required, no default)
  - [x] Ensure all fields have `from __future__ import annotations`

- [x] Task 5: Implement Custom Validation (AC: 4)
  - [x] Add @field_validator for threshold fields to ensure 0.0-1.0 range
  - [x] Add @model_validator for cross-field validation if needed
  - [x] Create custom error messages that are user-friendly
  - [x] Ensure ValidationError messages include field path and expected format

- [x] Task 6: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/config/test_schema.py`
  - [x] Test YoloConfig instantiation with defaults
  - [x] Test YoloConfig with environment variable overrides (monkeypatch)
  - [x] Test ValidationError for invalid threshold values
  - [x] Test ValidationError messages are clear and actionable
  - [x] Test nested model access (config.llm.cheap_model)
  - [x] Test that all fields have type hints (use typing.get_type_hints)
  - [x] Test mypy passes on schema.py

## Dev Notes

### Critical Architecture Requirements

**From ADR-008 (Configuration Management):**

The architecture specifies a layered configuration approach:
- Schema validation catches config errors early
- YAML is human-readable for project config
- Environment variables for secrets
- Layered priority: defaults → YAML → env → CLI

**Implementation Pattern from architecture.md:**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class YoloConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YOLO_",
        yaml_file="yolo.yaml",
    )

    # LLM settings
    cheap_model: str = "gpt-4o-mini"
    premium_model: str = "claude-sonnet-4-20250514"

    # Quality thresholds
    test_coverage_threshold: float = 0.80
    confidence_threshold: float = 0.90

    # Memory settings
    memory_persist_path: str = ".yolo/memory"
```

**From ADR-001 (State Management Pattern):**

> Pydantic validates at system boundaries (API inputs, user configs)

Configuration is a **system boundary** - this is where Pydantic validation is critical.

**From Project Structure (architecture.md):**

```
└── config/                 # Configuration (FR89-97)
    ├── __init__.py
    ├── schema.py           # Pydantic Settings models  <-- THIS STORY
    ├── loader.py           # Config loading logic      <-- Story 1.5
    └── defaults.py         # Default configurations
```

### Pydantic v2 Best Practices (2025)

Based on current Pydantic documentation:

1. **Use `pydantic-settings` package** - BaseSettings moved to separate package in v2
2. **Use `SettingsConfigDict`** - replaces inner `Config` class from v1
3. **Use `env_nested_delimiter`** - for nested environment variables like `YOLO_LLM__CHEAP_MODEL`
4. **Use `Field(description=...)`** - for auto-documentation
5. **Use `@field_validator`** - replaces `@validator` from v1
6. **Use `model_config`** - class attribute, not `Config` inner class

### Environment Variable Mapping

With `env_prefix="YOLO_"` and `env_nested_delimiter="__"`:

| Config Field | Environment Variable |
|--------------|---------------------|
| `llm.cheap_model` | `YOLO_LLM__CHEAP_MODEL` |
| `llm.premium_model` | `YOLO_LLM__PREMIUM_MODEL` |
| `quality.test_coverage_threshold` | `YOLO_QUALITY__TEST_COVERAGE_THRESHOLD` |
| `quality.confidence_threshold` | `YOLO_QUALITY__CONFIDENCE_THRESHOLD` |
| `memory.persist_path` | `YOLO_MEMORY__PERSIST_PATH` |
| `project_name` | `YOLO_PROJECT_NAME` |

### Required Dependencies

Already in pyproject.toml from Story 1.1:
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`

### Testing Approach

```python
import pytest
from pydantic import ValidationError

class TestYoloConfigDefaults:
    def test_instantiation_with_defaults(self) -> None:
        """YoloConfig should instantiate with sensible defaults."""
        config = YoloConfig(project_name="test-project")
        assert config.llm.cheap_model == "gpt-4o-mini"
        assert config.quality.test_coverage_threshold == 0.80

class TestYoloConfigValidation:
    def test_invalid_threshold_raises_error(self) -> None:
        """Threshold outside 0-1 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YoloConfig(
                project_name="test",
                quality={"test_coverage_threshold": 1.5}
            )
        assert "test_coverage_threshold" in str(exc_info.value)

    def test_error_message_is_clear(self) -> None:
        """Error messages should explain what's wrong."""
        with pytest.raises(ValidationError) as exc_info:
            YoloConfig(
                project_name="test",
                quality={"test_coverage_threshold": -0.5}
            )
        error = exc_info.value.errors()[0]
        assert error["loc"] == ("quality", "test_coverage_threshold")
```

### Project Structure Notes

- File location: `src/yolo_developer/config/schema.py`
- All Python files must use `from __future__ import annotations`
- Follow existing patterns from `cli/main.py`
- Use ClassVar for any mutable class attributes (per RUF012)

### References

- [Source: architecture.md#ADR-008] - Configuration Management decision
- [Source: architecture.md#ADR-001] - Pydantic at boundaries pattern
- [Source: architecture.md#Project Structure] - config/ module location
- [Source: epics.md#Story 1.4] - Story requirements
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [pydantic-settings-yaml PyPI](https://pypi.org/project/pydantic-settings-yaml/) - For YAML support in Story 1.5

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 129 tests pass (86 from Stories 1.1-1.3 + 43 config schema tests)
- mypy: Success, no issues found in schema.py
- ruff check: All checks passed
- ruff format: All files formatted correctly

### Completion Notes List

1. Created `src/yolo_developer/config/schema.py` with complete configuration schema per ADR-008
2. Implemented `LLMConfig` nested model with cheap_model, premium_model, best_model fields and Field descriptions
3. Implemented `QualityConfig` nested model with test_coverage_threshold and confidence_threshold fields (0.0-1.0 range validation)
4. Implemented `MemoryConfig` nested model with persist_path, vector_store_type, and graph_store_type fields
5. Implemented `YoloConfig` main class inheriting from BaseSettings with env_prefix="YOLO_" and env_nested_delimiter="__"
6. Added @field_validator for threshold range validation with clear error messages
7. Created comprehensive test suite with 33 tests covering all acceptance criteria
8. Tests verify: default values, type hints, Field descriptions, validation errors, environment variable overrides, mypy compliance

**Code Review Fixes (2026-01-04):**
9. Added `sprint-status.yaml` to File List (was modified but not documented)
10. Added exports to `config/__init__.py` for public API (`YoloConfig`, `LLMConfig`, `QualityConfig`, `MemoryConfig`)
11. Removed redundant `@field_validator` from `QualityConfig` (duplicated `Field(ge=0.0, le=1.0)` constraints)
12. Added 5 more env variable override tests (premium_model, best_model, memory.persist_path, confidence_threshold)
13. Added type hint tests for `QualityConfig` and `MemoryConfig`
14. Added 4 module export tests to verify public API imports work

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Story implemented - all tasks complete, 119 tests pass | Dev Agent |
| 2026-01-04 | Code review - found 5 issues, all fixed, 129 tests pass | Code Review |

### File List

- `src/yolo_developer/config/schema.py` - New configuration schema with LLMConfig, QualityConfig, MemoryConfig, and YoloConfig classes
- `src/yolo_developer/config/__init__.py` - Updated to export schema classes
- `tests/unit/config/test_schema.py` - Test file for configuration schema
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status to review

