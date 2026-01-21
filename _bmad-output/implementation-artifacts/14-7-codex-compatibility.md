# Story 14.7: Codex Compatibility

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want YOLO Developer to support OpenAI's ChatGPT Codex models with hybrid routing,
so that I can use code-optimized models for Dev/TEA while keeping reasoning tasks on other providers.

## Acceptance Criteria

1. Given `yolo.yaml` or env config, when `llm.provider` is set to `openai` or `hybrid`, then the schema validates `llm.openai.*` fields, defaults are applied, and missing API keys produce clear, path-specific `ConfigurationError` messages (e.g., `llm.openai.api_key`).
2. Given hybrid routing enabled, when a task is `code_generation` or `code_review`, then the router selects the OpenAI/Codex provider (or configured routing override) and uses `llm.openai.code_model` when present.
3. Given hybrid routing enabled, when a task is `architecture` or `analysis`, then the router selects the configured non-code provider (default: `anthropic`) and uses the reasoning tier (`llm.openai.reasoning_model` or equivalent configured tier).
4. Given Dev/TEA agents run, when they call the LLM via routing, then provider/model are recorded in audit metadata fields and surfaced in state/logs for traceability.
5. Given docs are updated, when users read setup guidance, then OpenAI/Codex config keys and env var names (e.g., `YOLO_LLM__OPENAI__API_KEY`) plus hybrid routing examples are documented.

## Tasks / Subtasks

- [x] Task 1: Extend LLM config schema for Codex/OpenAI + hybrid routing (AC: 1, 2, 3)
  - [x] Add OpenAI/Codex config fields and routing config with defaults
  - [x] Add validation for provider selection and missing API keys
- [x] Task 2: Implement router updates for task-based provider selection (AC: 2, 3)
  - [x] Add task-type mapping for code vs reasoning tasks
  - [x] Ensure code tasks use `code_model` when configured
- [x] Task 3: Wire Dev to routing + audit metadata (AC: 4)
  - [x] Update Dev LLM calls to use task-type routing
  - [x] Record provider/model in audit metadata
  - [x] TEA has no LLM calls yet (no routing changes required)
- [x] Task 4: Documentation updates (AC: 5)
  - [x] Update README + docs for OpenAI/Codex + hybrid routing
- [x] Task 5: Tests for schema, routing, and agent wiring (AC: 1-4)
  - [x] Unit tests for config validation and routing behavior
  - [x] Integration coverage for Dev/TEA routing path

## Dev Notes

- Use existing LiteLLM abstraction (ADR-003) and extend `LLMConfig` rather than introducing a new client abstraction. Prefer config-driven routing over new SDK clients.
- Prefer explicit error messaging with full config paths and `ConfigurationError` for missing API keys or invalid provider selection.
- Keep backward compatibility: default behavior should remain current when hybrid routing is disabled.
- Maintain async patterns for all LLM calls and avoid introducing sync I/O.

## Developer Context

- The Codex compatibility scope should remain within the existing LiteLLM routing layer and configuration schema updates; do not add a separate OpenAI SDK client unless required by existing patterns.
- Hybrid routing must be task-based (code vs reasoning) and should not change default provider behavior when hybrid routing is disabled.
- Web research was not performed in this pass; verify current OpenAI model names and availability during implementation.
- Prior story learnings: no 14-6 story file exists yet; run `dev-story` for 14-6 before implementation if it is created later.

## Technical Requirements

- Add OpenAI/Codex configuration fields and hybrid routing configuration with sane defaults.
- Validate provider selection and API key presence at load time with explicit `ConfigurationError` messages.
- Ensure task routing supports at least: `code_generation`, `code_review`, `architecture`, `analysis`, `documentation`, `testing`.
- Routing map (minimum):\n  - code_generation → openai (code_model)\n  - code_review → openai (code_model)\n  - architecture → anthropic (premium/reasoning)\n  - analysis → anthropic (premium/reasoning)\n  - documentation → openai (cheap)\n  - testing → openai (code_model or premium)
- Config keys (minimum): `llm.provider`, `llm.openai.api_key`, `llm.openai.code_model`, `llm.openai.cheap_model`, `llm.openai.premium_model`, `llm.openai.reasoning_model`, `llm.hybrid.enabled`, `llm.hybrid.routing.*`.
- Env vars (minimum): `YOLO_LLM__PROVIDER`, `YOLO_LLM__OPENAI__API_KEY`, `YOLO_LLM__OPENAI__CODE_MODEL`, `YOLO_LLM__HYBRID__ENABLED`, `YOLO_LLM__HYBRID__ROUTING__CODE_GENERATION`.

## Architecture Compliance

- Follow ADR-003 (LiteLLM abstraction) and ADR-008 (Pydantic config + YAML/env overrides).
- Keep async-first I/O and type annotations on all functions touched.
- Maintain existing module boundaries and avoid cross-module circular imports.

## Library/Framework Requirements

- LiteLLM remains the LLM provider abstraction; OpenAI and Anthropic should be configured through LiteLLM-compatible model IDs.
- Pydantic v2 schema updates should continue using existing validators and SecretStr handling.

## File Structure Requirements

- Update only within `src/yolo_developer/llm/`, `src/yolo_developer/config/`, and `src/yolo_developer/agents/` unless new CLI/docs changes are required by ACs.
- Keep new tests mirrored under `tests/unit/` and `tests/integration/` following existing structure.
- If CLI commands are added, limit to existing `src/yolo_developer/cli/commands/` patterns.

## Testing Requirements

- Add unit tests for config validation (provider selection, missing keys, model selection).
- Add unit tests for routing behavior by task type.
- Add integration coverage for Dev/TEA agent routing and audit metadata.
- Add regression checks to ensure default (non-hybrid) routing remains unchanged.
- Definition of Done: schema validation tests green, routing tests green, Dev/TEA routing metadata asserted, docs updated.

### Project Structure Notes

- LLM routing: `src/yolo_developer/llm/router.py`
- LLM config schema: `src/yolo_developer/config/schema.py`
- Config loading/validation: `src/yolo_developer/config/loader.py`
- Dev agent: `src/yolo_developer/agents/dev/node.py`
- TEA agent: `src/yolo_developer/agents/tea/node.py`
- Audit metadata: `src/yolo_developer/audit/` (log entry fields)
- Tests: `tests/unit/llm/`, `tests/unit/config/`, `tests/integration/agents/`

### References

- LiteLLM provider abstraction (ADR-003): `_bmad-output/planning-artifacts/architecture.md#ADR-003-LLM-Provider-Abstraction`
- Configuration management (ADR-008): `_bmad-output/planning-artifacts/architecture.md#ADR-008-Configuration-Management`
- Epic 14 integration goals: `_bmad-output/planning-artifacts/epics.md#Epic-14-MCP-Integration`
- Configurability requirement (FR92): `_bmad-output/planning-artifacts/epics.md#FR89-97`

## Project Context Reference

- No `project-context.md` found in the repository at story creation time.

## Scope Boundaries

- Do: Extend existing `LLMConfig` + LiteLLM router; add hybrid routing logic.
- Do: Update Dev/TEA to use task routing and emit provider/model metadata.
- Do: Update README + docs/mcp/index.md with config examples.
- Do not: Add a separate OpenAI SDK client or new CLI commands beyond config/docs.
- Do not: Add Azure OpenAI support unless it already exists in config patterns.

## Dev Agent Record

### Agent Model Used

GPT-5 (Codex CLI)

### Debug Log References

- 2026-01-21: `uv run pytest tests/unit/llm/test_router.py::TestLLMRouterTaskRouting::test_task_routing_uses_openai_code_model_in_hybrid tests/unit/config/test_loader.py::TestAPIKeyLoading::test_api_key_loaded_from_nested_env_openai tests/unit/config/test_loader.py::TestAPIKeyLoading::test_missing_openai_key_raises_when_provider_openai tests/unit/agents/dev/test_code_utils.py::TestGenerateCodeWithValidation::test_uses_correct_tier tests/unit/agents/dev/test_doc_utils.py::TestGenerateDocumentationWithLLM::test_uses_complex_tier tests/unit/agents/dev/test_commit_utils.py::TestGenerateCommitMessageWithLLM::test_llm_generation_uses_routine_tier tests/unit/agents/dev/test_test_utils.py::TestGenerateUnitTestsWithLLM::test_includes_implementation_code_in_prompt tests/integration/agents/dev/test_commit_generation.py::TestDevNodeCommitMessageIntegration::test_dev_node_records_llm_usage_metadata -q` (pass: 8 tests)
- 2026-01-21: `uv run pytest tests/unit/agents/dev/test_test_utils.py::TestGenerateUnitTestsWithLLM::test_calls_router_with_testing_task_type -q` (pass: 1 test)

### Completion Notes List

1. Added OpenAI/Codex + hybrid routing config with provider validation and nested env var support.
2. Implemented task-based routing in LLMRouter and updated Dev LLM usage to record provider/model metadata.
3. Updated README + docs/mcp/index.md with Codex/hybrid examples and env vars; added unit/integration coverage for routing and metadata.
4. Dev router now honors project config with fallback defaults, clears per-run usage logs, and tests align with task-based routing.

### File List

- _bmad-output/implementation-artifacts/14-7-codex-compatibility.md
- _bmad-output/implementation-artifacts/sprint-status.yaml
- README.md
- docs/mcp/index.md
- src/yolo_developer/agents/dev/commit_utils.py
- src/yolo_developer/agents/dev/code_utils.py
- src/yolo_developer/agents/dev/doc_utils.py
- src/yolo_developer/agents/dev/integration_utils.py
- src/yolo_developer/agents/dev/node.py
- src/yolo_developer/agents/dev/prompts/integration_test_generation.py
- src/yolo_developer/agents/dev/test_utils.py
- src/yolo_developer/cli/commands/config.py
- src/yolo_developer/config/__init__.py
- src/yolo_developer/config/export.py
- src/yolo_developer/config/loader.py
- src/yolo_developer/config/schema.py
- src/yolo_developer/config/validators.py
- src/yolo_developer/llm/__init__.py
- src/yolo_developer/llm/router.py
- src/yolo_developer/mcp/__init__.py
- src/yolo_developer/mcp/server.py
- src/yolo_developer/mcp/tools.py
- tests/integration/agents/dev/test_code_generation_integration.py
- tests/integration/agents/dev/test_commit_generation.py
- tests/integration/agents/dev/test_documentation_generation.py
- tests/integration/agents/dev/test_integration_test_generation.py
- tests/integration/agents/dev/test_test_generation_integration.py
- tests/integration/mcp/test_mcp_server.py
- tests/unit/agents/dev/test_code_utils.py
- tests/unit/agents/dev/test_commit_utils.py
- tests/unit/agents/dev/test_doc_utils.py
- tests/unit/agents/dev/test_test_utils.py
- tests/unit/cli/test_config_command.py
- tests/unit/config/test_export.py
- tests/unit/config/test_loader.py
- tests/unit/config/test_schema.py
- tests/unit/llm/test_router.py
- tests/unit/mcp/test_tools.py

## Change Log

- 2026-01-21: Added Codex/hybrid config support, task routing, Dev metadata logging, docs updates, and unit/integration tests.
- 2026-01-21: Aligned Dev router config loading and usage logging, updated routing tests, and synced file list with repo changes.

## Story Completion Status

Status: done

Completion Note: Implemented Codex/hybrid routing, provider validation, Dev LLM metadata logging, docs updates, and tests; router now honors project config and usage logging per run; review fixes complete.
