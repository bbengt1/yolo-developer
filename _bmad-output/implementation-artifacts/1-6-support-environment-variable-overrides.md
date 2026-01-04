# Story 1.6: Support Environment Variable Overrides

Status: review

## Story

As a developer,
I want environment variables to override config file settings,
So that I can customize behavior in different environments without changing files.

## Acceptance Criteria

1. **AC1: Environment Variable Override**
   - **Given** configuration is defined in yolo.yaml
   - **When** I set an environment variable with YOLO_ prefix
   - **Then** the environment variable value overrides the file value
   - **And** the priority order is: defaults → YAML → environment variables

2. **AC2: Nested Settings Support**
   - **Given** nested configuration (llm, quality, memory sections)
   - **When** I set YOLO_LLM__CHEAP_MODEL environment variable
   - **Then** the nested setting is correctly overridden
   - **And** the `__` delimiter maps to nested config paths

3. **AC3: API Keys via Environment Variables Only**
   - **Given** I need to configure LLM API keys
   - **When** I set YOLO_LLM__OPENAI_API_KEY or YOLO_LLM__ANTHROPIC_API_KEY
   - **Then** the API key is available in configuration
   - **And** API keys are NEVER written to YAML files during export
   - **And** missing required API keys produce clear error messages

4. **AC4: Secure API Key Handling**
   - **Given** an API key is set via environment variable
   - **When** configuration is logged or displayed
   - **Then** API keys are masked/redacted in output
   - **And** API keys do not appear in error messages
   - **And** API keys are excluded from `repr()` output

5. **AC5: API Key Validation**
   - **Given** LLM operations require API keys
   - **When** configuration is validated on load
   - **Then** a warning is logged if no API keys are configured
   - **And** validation does not fail (API keys may be set later via SDK)

## Tasks / Subtasks

- [x] Task 1: Add API Key Fields to LLMConfig (AC: 3, 4)
  - [x] Add `openai_api_key: str | None` field with SecretStr type
  - [x] Add `anthropic_api_key: str | None` field with SecretStr type
  - [x] Configure fields to read from environment (YOLO_LLM__OPENAI_API_KEY, etc.)
  - [x] Use Pydantic SecretStr for automatic masking in repr/logs

- [x] Task 2: Implement API Key Masking (AC: 4)
  - [x] Ensure SecretStr masks values in __repr__ and __str__
  - [x] Add `get_secret_value()` method for accessing actual key
  - [x] Verify masked output in configuration display

- [x] Task 3: Add API Key Validation Warning (AC: 5)
  - [x] Add `validate_api_keys()` method to YoloConfig
  - [x] Log warning (not error) if no API keys are configured
  - [x] Return list of missing API key warnings for display

- [x] Task 4: Document Environment Variable Pattern (AC: 1, 2)
  - [x] Update docstrings in schema.py with full env var mapping
  - [x] Document all supported environment variables
  - [x] Include examples in module docstring

- [x] Task 5: Write Unit Tests (AC: all)
  - [x] Test: API key loaded from environment variable
  - [x] Test: API key masked in repr output
  - [x] Test: API key accessible via get_secret_value()
  - [x] Test: Missing API key produces warning (not error)
  - [x] Test: Nested env var override (verify Story 1.5 coverage)
  - [x] Test: Multiple env vars override multiple settings

## Dev Notes

### Critical Architecture Requirements

**From ADR-008 (Configuration Management):**
- Environment variables for secrets
- Layered priority: defaults → YAML → env → CLI
- API keys NEVER in config files

**From NFR-SEC-1:**
- API keys stored via environment variables or encrypted secrets manager
- No credentials in generated code

### Current State (from Story 1.5)

The `load_config()` function in `loader.py` already implements:
```python
def _merge_with_env_vars(yaml_data: dict[str, Any]) -> dict[str, Any]:
    """Merge YAML data with environment variables.

    Environment variables with YOLO_ prefix override YAML values.
    Uses __ as nested delimiter (e.g., YOLO_LLM__CHEAP_MODEL).
    """
```

**What's missing:**
1. API key fields in LLMConfig schema
2. SecretStr type for automatic masking
3. Validation warnings for missing API keys

### Implementation Approach

**Option 1: Pydantic SecretStr (Recommended)**

```python
from pydantic import SecretStr

class LLMConfig(BaseModel):
    """Configuration for LLM provider settings."""

    cheap_model: str = Field(default="gpt-4o-mini")
    premium_model: str = Field(default="claude-sonnet-4-20250514")
    best_model: str = Field(default="claude-opus-4-5-20251101")

    # API keys - read from env only, masked in output
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (set via YOLO_LLM__OPENAI_API_KEY env var)",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (set via YOLO_LLM__ANTHROPIC_API_KEY env var)",
    )
```

**SecretStr Benefits:**
- `str(api_key)` returns `'**********'` (masked)
- `api_key.get_secret_value()` returns actual value
- Prevents accidental logging of secrets
- Native Pydantic support

### Environment Variable Mapping

| Environment Variable | Config Path | Description |
|---------------------|-------------|-------------|
| YOLO_PROJECT_NAME | project_name | Project name (required) |
| YOLO_LLM__CHEAP_MODEL | llm.cheap_model | Cheap LLM model |
| YOLO_LLM__PREMIUM_MODEL | llm.premium_model | Premium LLM model |
| YOLO_LLM__BEST_MODEL | llm.best_model | Best LLM model |
| YOLO_LLM__OPENAI_API_KEY | llm.openai_api_key | OpenAI API key |
| YOLO_LLM__ANTHROPIC_API_KEY | llm.anthropic_api_key | Anthropic API key |
| YOLO_QUALITY__TEST_COVERAGE_THRESHOLD | quality.test_coverage_threshold | Coverage threshold |
| YOLO_QUALITY__CONFIDENCE_THRESHOLD | quality.confidence_threshold | Confidence threshold |
| YOLO_MEMORY__PERSIST_PATH | memory.persist_path | Memory storage path |
| YOLO_MEMORY__VECTOR_STORE_TYPE | memory.vector_store_type | Vector store type |
| YOLO_MEMORY__GRAPH_STORE_TYPE | memory.graph_store_type | Graph store type |

### Testing Approach

```python
import pytest
from yolo_developer.config import load_config, YoloConfig
from pydantic import SecretStr

class TestAPIKeyHandling:
    def test_api_key_loaded_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API key is loaded from environment variable."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("project_name: test-project\n")

        monkeypatch.setenv("YOLO_LLM__OPENAI_API_KEY", "sk-test-key-12345")
        config = load_config(yaml_file)

        assert config.llm.openai_api_key is not None
        assert config.llm.openai_api_key.get_secret_value() == "sk-test-key-12345"

    def test_api_key_masked_in_repr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API key is masked in repr output."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("project_name: test-project\n")

        monkeypatch.setenv("YOLO_LLM__OPENAI_API_KEY", "sk-secret-key")
        config = load_config(yaml_file)

        repr_output = repr(config.llm)
        assert "sk-secret-key" not in repr_output
        assert "**********" in repr_output or "SecretStr" in repr_output

    def test_missing_api_key_logs_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        """Missing API key logs a warning, not an error."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("project_name: test-project\n")

        config = load_config(yaml_file)
        warnings = config.validate_api_keys()

        assert len(warnings) > 0
        assert "api key" in warnings[0].lower()
```

### Previous Story Learnings (Story 1.5)

1. Use `from __future__ import annotations` in all Python files
2. Export public API from `__init__.py`
3. Run `ruff check`, `ruff format`, and `mypy` before marking complete
4. The `_merge_with_env_vars()` function already handles env var loading
5. Type conversion via `_convert_value()` handles floats, bools, ints

### File Changes Required

1. **`src/yolo_developer/config/schema.py`**
   - Add `SecretStr` import from pydantic
   - Add `openai_api_key` and `anthropic_api_key` fields to LLMConfig
   - Add `validate_api_keys()` method to YoloConfig

2. **`src/yolo_developer/config/loader.py`**
   - ~~Update `_convert_value()` to handle SecretStr fields~~ (Not needed - Pydantic handles SecretStr conversion automatically)
   - Added API key warning logging in `load_config()` to satisfy AC5

3. **`tests/unit/config/test_schema.py`**
   - Add tests for API key fields
   - Add tests for masking behavior

4. **`tests/unit/config/test_loader.py`**
   - Add tests for API key env var loading

### Git Commit Pattern

```
feat: Add API key support with SecretStr masking (Story 1.6)
```

### References

- [Source: architecture.md#ADR-008] - Configuration Management
- [Source: prd.md#NFR-SEC-1] - API key security
- [Source: epics.md#Story 1.6] - Story requirements
- [Source: Story 1.5] - Existing env var implementation
- [Pydantic SecretStr](https://docs.pydantic.dev/latest/concepts/types/#secret-types)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 180 tests pass (94 config tests + 86 other tests) - increased from 177 after code review fixes
- mypy: Success, no issues found in 3 source files
- ruff check: All checks passed
- ruff format: All 3 config files already formatted correctly

### Completion Notes List

1. **Task 1 - API Key Fields**: Added `openai_api_key` and `anthropic_api_key` fields to LLMConfig using Pydantic SecretStr type. Fields default to None and are configured via YOLO_LLM__OPENAI_API_KEY and YOLO_LLM__ANTHROPIC_API_KEY environment variables.

2. **Task 2 - API Key Masking**: SecretStr provides automatic masking in __repr__ and __str__ output. Values are accessible via `.get_secret_value()` method. Verified with 9 dedicated tests.

3. **Task 3 - Validation Warning**: Added `validate_api_keys()` method to YoloConfig that returns a list of warning strings. Returns warning if no API keys are configured (neither OpenAI nor Anthropic). Does NOT raise an error - just returns warnings for display.

4. **Task 4 - Documentation**: Updated module docstring in schema.py with comprehensive documentation including:
   - Full environment variable mapping table
   - Configuration priority order (defaults → YAML → env)
   - API key security notes
   - Example usage code

5. **Task 5 - Tests**: Added 22 new tests across test_schema.py and test_loader.py:
   - TestLLMConfigAPIKeys (9 tests): Field existence, SecretStr type, masking, get_secret_value()
   - TestYoloConfigAPIKeyValidation (5 tests): validate_api_keys() method behavior
   - TestAPIKeyLoading (8 tests): Loading from env vars via loader.py + warning logging tests

6. **Code Review Fix - AC5 Warning Logging**: Updated load_config() to call validate_api_keys() and log warnings using logger.warning(). This ensures AC5 requirement "a warning is logged if no API keys are configured" is fully satisfied.

7. **AC3 Export Test Note**: The AC3 requirement "API keys are NEVER written to YAML files during export" will be tested in Story 1.8 (Export and Import Project Configurations) when export functionality is implemented. SecretStr type already prevents accidental exposure.

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Implemented all tasks, 177 tests pass | Dev Agent |
| 2026-01-04 | Code review fixes: AC5 warning logging, File List update, Dev Notes clarification | Code Review |

### File List

- `src/yolo_developer/config/schema.py` - Added SecretStr import, openai_api_key and anthropic_api_key fields to LLMConfig, validate_api_keys() method to YoloConfig, updated docstrings with full env var documentation
- `src/yolo_developer/config/loader.py` - Added API key warning logging in load_config() to satisfy AC5 (calls validate_api_keys() and logs warnings)
- `tests/unit/config/test_schema.py` - Added TestLLMConfigAPIKeys class (9 tests) and TestYoloConfigAPIKeyValidation class (5 tests)
- `tests/unit/config/test_loader.py` - Added TestAPIKeyLoading class (8 tests: 6 original + 2 for warning logging)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status tracking
