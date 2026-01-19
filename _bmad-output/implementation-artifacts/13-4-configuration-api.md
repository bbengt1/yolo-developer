# Story 13.4: Configuration API

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to configure all settings via SDK,
so that I have full programmatic control.

## Acceptance Criteria

### AC1: Read All Configuration Settings
**Given** a YoloClient instance
**When** I access `client.config`
**Then** I receive the complete YoloConfig object
**And** all nested settings are accessible (llm, quality, memory)
**And** SecretStr fields (API keys) are masked when accessing `.get_secret_value()`
**And** the returned config reflects current in-memory state

### AC2: Update Configuration Settings
**Given** a YoloClient instance
**When** I call `update_config()` with partial settings
**Then** only the specified settings are updated
**And** unspecified settings retain their current values
**And** nested settings can be updated (e.g., `quality.test_coverage_threshold`)
**And** updates are validated before applying
**And** invalid updates raise ConfigurationAPIError

### AC3: Configuration Validation
**Given** configuration changes to apply
**When** I call `validate_config()` or update with validation enabled
**Then** all validation rules are checked
**And** errors prevent changes from being applied
**And** warnings are returned but don't block
**And** validation result includes specific issue details

### AC4: Configuration Persistence
**Given** configuration changes have been made via SDK
**When** I call `save_config()` or update with `persist=True`
**Then** changes are written to yolo.yaml
**And** API keys are excluded from the saved file
**And** the file includes helpful comments
**And** subsequent client instances see the changes

### AC5: Async Versions Available
**Given** a YoloClient instance
**When** I call `update_config_async()`, `validate_config_async()`, `save_config_async()`
**Then** they return awaitable coroutines
**And** sync methods correctly wrap async versions using `_run_sync()`
**And** both work correctly in sync and async contexts

## Tasks / Subtasks

- [x] Task 1: Review Existing Configuration Implementation (AC: #1, #3)
  - [x] Subtask 1.1: Analyze config module's public API (schema.py, loader.py)
  - [x] Subtask 1.2: Review existing client.config property (client.py line 220-227)
  - [x] Subtask 1.3: Review validators.py for validation patterns
  - [x] Subtask 1.4: Review export.py for persistence patterns

- [x] Task 2: Implement Configuration Read Access (AC: #1)
  - [x] Subtask 2.1: Ensure config property returns full YoloConfig
  - [x] Subtask 2.2: Add helper methods for nested config access if needed (not needed - direct attribute access works)
  - [x] Subtask 2.3: Document SecretStr masking behavior

- [x] Task 3: Implement Configuration Updates (AC: #2)
  - [x] Subtask 3.1: Create update_config_async() with partial update support
  - [x] Subtask 3.2: Create update_config() sync wrapper
  - [x] Subtask 3.3: Support nested updates using dot notation or dict merge
  - [x] Subtask 3.4: Validate before applying updates
  - [x] Subtask 3.5: Add ConfigurationAPIError exception type

- [x] Task 4: Implement Validation API (AC: #3)
  - [x] Subtask 4.1: Create validate_config_async() method
  - [x] Subtask 4.2: Create validate_config() sync wrapper
  - [x] Subtask 4.3: Return structured ValidationResult with errors/warnings
  - [x] Subtask 4.4: Integrate with existing validators.py logic

- [x] Task 5: Implement Configuration Persistence (AC: #4)
  - [x] Subtask 5.1: Create save_config_async() method
  - [x] Subtask 5.2: Create save_config() sync wrapper
  - [x] Subtask 5.3: Exclude API keys from saved file
  - [x] Subtask 5.4: Add persist parameter to update_config() for auto-save

- [x] Task 6: Add Result Types (AC: all)
  - [x] Subtask 6.1: Create ConfigUpdateResult dataclass
  - [x] Subtask 6.2: Create ConfigValidationResult dataclass (or reuse from validators)
  - [x] Subtask 6.3: Create ConfigSaveResult dataclass
  - [x] Subtask 6.4: Export from sdk/__init__.py

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Subtask 7.1: Test config read access and nested properties
  - [x] Subtask 7.2: Test update_config() with various scenarios
  - [x] Subtask 7.3: Test validation with valid/invalid configs
  - [x] Subtask 7.4: Test persistence to yolo.yaml
  - [x] Subtask 7.5: Test async/sync parity

- [x] Task 8: Update Documentation (AC: all)
  - [x] Subtask 8.1: Update client.py docstrings
  - [x] Subtask 8.2: Add usage examples for common config operations

## Dev Notes

### Architecture Patterns

Per Story 13.1/13.2/13.3 implementation and architecture.md:

1. **SDK Layer Position**: SDK sits between external consumers and the config layer
2. **Direct Import Pattern**: SDK imports from config module:
   ```python
   from yolo_developer.config import (
       YoloConfig,
       LLMConfig,
       QualityConfig,
       MemoryConfig,
       load_config,
       validate_config,
       export_config,
   )
   ```

3. **Async/Sync Pattern**: Sync methods wrap async versions using `_run_sync()` helper
4. **Result Types**: Use `@dataclass(frozen=True)` for immutable results with timestamp

### Existing Configuration Structure (schema.py)

```python
class YoloConfig(BaseSettings):
    project_name: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

class LLMConfig(BaseModel):
    cheap_model: str = "gpt-4o-mini"
    premium_model: str = "claude-sonnet-4-20250514"
    best_model: str = "claude-opus-4-5-20251101"
    openai_api_key: SecretStr | None = None  # env vars only
    anthropic_api_key: SecretStr | None = None  # env vars only

class QualityConfig(BaseModel):
    test_coverage_threshold: float = 0.80  # 0.0-1.0
    confidence_threshold: float = 0.90  # 0.0-1.0
    gate_thresholds: dict[str, GateThreshold] = {}
    seed_thresholds: SeedThresholdConfig = Field(default_factory=SeedThresholdConfig)
    critical_paths: list[str] = []

class MemoryConfig(BaseModel):
    persist_path: str = ".yolo/memory"
    vector_store_type: Literal["chromadb"] = "chromadb"
    graph_store_type: Literal["json", "neo4j"] = "json"
```

### Existing Client Config Property (client.py lines 220-227)

```python
@property
def config(self) -> YoloConfig:
    """Get the current configuration."""
    return self._config
```

Currently read-only. Need to add update methods.

### Proposed API Design

```python
# Read access (existing)
config = client.config
coverage = client.config.quality.test_coverage_threshold

# Update with partial settings
result = client.update_config(
    quality={"test_coverage_threshold": 0.85},
    persist=True,  # Optional: save to yolo.yaml
)

# Validation only
validation = client.validate_config(config_changes)

# Save current config to file
save_result = client.save_config()

# Async versions
result = await client.update_config_async(quality={"test_coverage_threshold": 0.85})
validation = await client.validate_config_async(config_changes)
save_result = await client.save_config_async()
```

### Key Files to Touch

**Modify:**
- `src/yolo_developer/sdk/client.py` - Add config API methods
- `src/yolo_developer/sdk/types.py` - Add result dataclasses
- `src/yolo_developer/sdk/__init__.py` - Export new types
- `src/yolo_developer/sdk/exceptions.py` - Add ConfigurationAPIError
- `tests/unit/sdk/test_client.py` - Add config API tests

**Reference:**
- `src/yolo_developer/config/schema.py` - YoloConfig structure
- `src/yolo_developer/config/loader.py` - load_config() implementation
- `src/yolo_developer/config/export.py` - export_config() / import_config()
- `src/yolo_developer/config/validators.py` - Validation logic

### Previous Story Learnings (Stories 13.1, 13.2, 13.3)

1. Run `ruff check` and `mypy` before committing
2. Use `from __future__ import annotations` in all files
3. Use timezone-aware datetime: `datetime.now(timezone.utc)` per ruff DTZ005 rule
4. Use `_run_sync()` helper instead of deprecated `asyncio.get_event_loop()`
5. Frozen dataclasses for immutable results
6. Exception chaining with `raise ... from e`
7. Test both success and error paths
8. 55 tests currently passing for SDK module (13 JsonDecisionStore + 42 SDK)

### Error Handling Pattern

```python
try:
    # operation
except ConfigurationError as e:
    raise ConfigurationAPIError(
        f"Failed to update configuration: {e}",
        original_error=e,
        details={"field": field_name, "value": value},
    ) from e
```

### Project Structure Notes

- Alignment: SDK module follows architecture.md structure
- Entry Point: `from yolo_developer import YoloClient`
- API Boundary: SDK is one of three external entry points (CLI, SDK, MCP)
- Config stored in: `./yolo.yaml` (project root)
- Secrets: API keys via environment variables only, never in config file

### Testing Standards

Follow patterns from `tests/unit/sdk/test_client.py`:
- Use `pytest` with `pytest-asyncio` for async tests
- Mock file operations for unit tests
- Test file naming: `test_<module>.py`
- Test function naming: `test_<behavior>_<scenario>`
- Mark async tests with `@pytest.mark.asyncio`

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Python SDK] - SDK structure and design
- [Source: _bmad-output/planning-artifacts/prd.md#Python SDK] - FR109 requirement
- [Source: _bmad-output/planning-artifacts/epics.md#Story 13.4] - Story definition
- [Source: src/yolo_developer/config/schema.py] - YoloConfig implementation
- [Source: src/yolo_developer/config/loader.py] - Configuration loading
- [Source: src/yolo_developer/config/export.py] - Configuration export/import
- [Source: src/yolo_developer/config/validators.py] - Validation logic
- [Source: src/yolo_developer/sdk/client.py:220-227] - Existing config property
- [Related: Story 13.1 (SDK Client Class)] - Foundation implementation
- [Related: Story 13.2 (Programmatic Init/Seed/Run)] - SDK method patterns
- [Related: Story 13.3 (Audit Trail Access)] - Async/sync patterns, result types

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 66 tests passing (42 existing + 24 new configuration API tests)
- ruff check: All checks passed
- mypy: Success, no issues found in 4 source files

### Code Review Fixes Applied

1. **M1**: Added sprint-status.yaml to File List documentation
2. **M2**: Updated sdk/__init__.py module docstring to mention Stories 13.1-13.4
3. **M3**: Clarified Subtask 2.2 that helper methods weren't needed
4. **M4**: Added test for empty update_config() call (test_update_config_empty_update)

### Completion Notes List

1. **Task 1**: Reviewed existing configuration implementation
   - Analyzed config module's public API (schema.py with YoloConfig, LLMConfig, QualityConfig, MemoryConfig)
   - Reviewed existing client.config property (read-only)
   - Reviewed validators.py for validation patterns (ValidationResult with errors/warnings)
   - Reviewed export.py for persistence patterns (export_config excludes secrets)

2. **Task 2**: Config read access already complete via existing `client.config` property
   - Returns full YoloConfig with all nested settings accessible
   - SecretStr fields properly masked

3. **Task 3**: Implemented update_config() and update_config_async()
   - Partial update support using dict merge pattern
   - Validates before applying changes
   - Raises ConfigurationAPIError on validation failure
   - Added persist parameter for auto-save

4. **Task 4**: Implemented validate_config() and validate_config_async()
   - Integrates with config module's validate_config()
   - Returns ConfigValidationResult with errors/warnings separation
   - is_valid=True only when no errors (warnings allowed)

5. **Task 5**: Implemented save_config() and save_config_async()
   - Uses config module's export_config() for persistence
   - API keys excluded from saved file
   - Returns ConfigSaveResult with secrets_excluded list

6. **Task 6**: Added result types to types.py
   - ConfigValidationIssue (field, message, severity)
   - ConfigValidationResult (is_valid, issues, errors/warnings properties)
   - ConfigUpdateResult (success, previous_values, new_values, persisted, validation)
   - ConfigSaveResult (success, config_path, secrets_excluded)
   - All exported from sdk/__init__.py

7. **Task 7**: Wrote 24 unit tests covering all 5 acceptance criteria
   - TestYoloClientConfigRead (3 tests): AC1
   - TestYoloClientConfigUpdate (7 tests): AC2 (includes empty update edge case)
   - TestYoloClientConfigValidation (4 tests): AC3
   - TestYoloClientConfigPersistence (5 tests): AC4
   - TestYoloClientConfigAsyncSync (5 tests): AC5

8. **Task 8**: Documentation complete
   - All methods have comprehensive docstrings with Args, Returns, Raises, Examples
   - Module docstring updated to reference FR109
   - types.py dataclasses have full docstrings with examples

### File List

**Modified:**
- `src/yolo_developer/sdk/client.py` - Added ~290 lines with config API methods
- `src/yolo_developer/sdk/types.py` - Added ConfigValidationIssue, ConfigValidationResult, ConfigUpdateResult, ConfigSaveResult
- `src/yolo_developer/sdk/exceptions.py` - Added ConfigurationAPIError
- `src/yolo_developer/sdk/__init__.py` - Exported new types and exception, updated docstring for Story 13.4
- `tests/unit/sdk/test_client.py` - Added 24 new tests in 5 test classes
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status
