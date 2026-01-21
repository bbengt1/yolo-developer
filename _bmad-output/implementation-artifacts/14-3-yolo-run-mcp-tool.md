# Story 14.3: yolo_run MCP Tool

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an MCP client (e.g., Claude Code),
I want to invoke a `yolo_run` tool via MCP,
so that I can trigger a sprint and receive a sprint_id for status queries.

## Acceptance Criteria

### AC1: yolo_run Tool Registration
**Given** the YOLO Developer MCP server is running
**When** a client queries available tools
**Then** `yolo_run` is listed as an available tool
**And** tool description explains its purpose for LLM understanding
**And** tool parameters are properly documented with types

### AC2: Trigger Sprint Execution
**Given** a validated seed
**When** `yolo_run` is invoked
**Then** sprint execution begins
**And** execution runs without blocking the MCP response

### AC3: Return Sprint ID
**Given** `yolo_run` starts a sprint
**When** the tool completes its response
**Then** a unique sprint_id is returned
**And** the sprint_id can be used to query status later

### AC4: Long-Running Execution Handling
**Given** a sprint may take minutes or hours
**When** `yolo_run` is invoked
**Then** the tool responds quickly with a sprint_id
**And** the sprint continues in the background

### AC5: Error Handling
**Given** `yolo_run` is invoked without a valid seed
**When** the tool validates inputs
**Then** it returns a structured error response
**And** no sprint is started

## Tasks / Subtasks

- [x] Task 1: Implement yolo_run MCP Tool (AC: #1, #2, #3, #4)
  - [x] Subtask 1.1: Add `yolo_run` tool in `src/yolo_developer/mcp/tools.py`
  - [x] Subtask 1.2: Define input schema (seed_id and/or seed_content)
  - [x] Subtask 1.3: Start sprint execution in background task
  - [x] Subtask 1.4: Return structured response with sprint_id
- [x] Task 2: Sprint Registry for Status Queries (AC: #3, #4)
  - [x] Subtask 2.1: Store sprint metadata by sprint_id (status, start time, thread_id)
  - [x] Subtask 2.2: Ensure registry is thread-safe for MCP concurrency
- [x] Task 3: Input Validation + Error Responses (AC: #5)
  - [x] Subtask 3.1: Validate seed_id exists in MCP seed store
  - [x] Subtask 3.2: Validate seed_content is non-empty when provided
  - [x] Subtask 3.3: Return MCP-compatible error payloads
- [x] Task 4: Tests (AC: all)
  - [x] Subtask 4.1: Unit tests for successful run + error cases
  - [x] Subtask 4.2: Integration tests via FastMCP Client

## Dev Notes

### Developer Context
- Extend the existing FastMCP tool module in `src/yolo_developer/mcp/tools.py`; follow the `yolo_seed` async + structured response pattern.
- Use MCP seed storage (`get_seed`) for `seed_id` lookups; if the seed is missing or invalid, return an MCP error and do not start a sprint.
- Trigger orchestration via `yolo_developer.orchestrator` APIs (e.g., `run_workflow` / `stream_workflow`) instead of invoking CLI commands.

### Technical Requirements
- Inputs: accept `seed_id` (primary) and optional `seed_content`; document precedence if both supplied.
- Outputs: return a JSON-serializable response with `status`, `sprint_id`, and timing metadata; include `seed_id` when provided.
- Long-running execution: start sprint execution in a background task and return immediately.
- Persist sessions via `SessionManager` under `.yolo/sessions` with checkpointing enabled.

### Architecture Compliance
- ADR-001: Keep TypedDict state intact; avoid mutating shared state directly.
- ADR-004: Use FastMCP decorator-based tools; keep `SERVER_INSTRUCTIONS` updated.
- ADR-007: Use resilient error handling paths (retry or explicit error capture) around orchestrator calls.

### Library/Framework Requirements
- FastMCP 2.x tool registration (current: 2.14.3).
- Stdlib `asyncio`, `uuid`, `datetime`, and `threading` if adding a sprint registry with locks.
- Structured logging via `structlog` consistent with existing modules.

### Testing Requirements
- Unit tests in `tests/unit/mcp/test_tools.py` for successful run, missing seed, and invalid inputs.
- Integration tests in `tests/integration/mcp/test_mcp_server.py` using FastMCP Client; stub orchestrator to avoid real long runs.
- Add test fixtures to clear seed and sprint registries between tests (match Story 14.2 isolation pattern).

### Previous Story Intelligence
- Story 14.2 added thread-safe seed storage with `_seeds_lock` and `clear_seeds()`; mirror this for sprint registry state.
- Prior tests emphasized structured errors for invalid inputs and isolation via autouse fixtures.

### Git Intelligence Summary
- Recent commits are repo setup only; no MCP-specific changes to leverage.

### Latest Tech Information
- FastMCP latest stable on PyPI: `2.14.3`.
- MCP SDK latest stable on PyPI: `1.25.0`.

### Project Structure Notes

- Primary implementation: `src/yolo_developer/mcp/tools.py` (add `yolo_run` and sprint registry helpers).
- Update `src/yolo_developer/mcp/server.py` `SERVER_INSTRUCTIONS` to document `yolo_run`.
- Tests: extend `tests/unit/mcp/test_tools.py` and `tests/integration/mcp/test_mcp_server.py`.
- Prefer keeping registry state co-located with MCP tools for MVP consistency; extract to a dedicated module only if it grows beyond tool scope.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 14.3] MCP yolo_run story definition and ACs
- [Source: _bmad-output/planning-artifacts/prd.md#FR112-FR117] MCP protocol requirements
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-004] FastMCP 2.x server pattern
- [Source: _bmad-output/planning-artifacts/architecture.md#ARCH-PATTERN-4] Decorator-based MCP tools
- [Source: _bmad-output/implementation-artifacts/14-2-yolo-seed-mcp-tool.md] Existing MCP tool patterns and test strategy
- Project context: no `project-context.md` found under `docs/` (none loaded)

## Dev Agent Record

### Agent Model Used

GPT-5 (Codex CLI)

### Debug Log References

- `uv run pytest tests/unit/mcp/test_tools.py tests/integration/mcp/test_mcp_server.py` (pass)
- `uv run pytest` (timeout; failures in `tests/integration/agents/architect/test_adr_integration.py` due to missing `project_name` config and missing `OPENAI_API_KEY`)
- `uv run mypy src/yolo_developer` (pass)
- `uv run ruff check src tests` (pass)
- `YOLO_PROJECT_NAME=yolo-developer uv run pytest tests/integration/agents/architect/test_adr_integration.py -x` (pass)
- `YOLO_PROJECT_NAME=yolo-developer uv run pytest` (pass)
- `uv run ruff format src tests` (reformatted `src/yolo_developer/mcp/tools.py`)
- `uv run ruff check src tests` (pass)
- `uv run pytest tests/unit/mcp/test_tools.py tests/integration/mcp/test_mcp_server.py` (pass)
- `uv run pytest` (timeout after 120s; `tests/integration/agents/architect/test_adr_integration.py` passed)
- `YOLO_PROJECT_NAME=yolo-developer OPENAI_API_KEY=$OPENAI_API_KEY uv run pytest` (pass; 6508 tests, 44 warnings)
- `uv run ruff check src tests` (pass)
- `uv run pytest tests/unit/mcp/test_tools.py tests/integration/mcp/test_mcp_server.py` (pass)

### Completion Notes List

1. Story context assembled from epics, PRD, and architecture sources.
2. Guardrails added for FastMCP tool patterns, background execution, and registry handling.
3. Status set to ready-for-dev with implementation-ready tasks and tests.
4. Implemented `yolo_run` MCP tool with sprint registry and background execution.
5. Added unit and integration tests for yolo_run tool behavior and registration.
6. Added unit test fixture to clear `YOLO_PROJECT_NAME` for unit tests to avoid config override conflicts.
7. Full test suite passes with `YOLO_PROJECT_NAME=yolo-developer`.
8. Applied `ruff format` after formatting drift from earlier full-suite run.
9. Validated `seed_content` against seed quality thresholds in `yolo_run`.
10. Added thread-safe sprint task registry updates with cleanup on completion.
11. Updated MCP module docs to reflect `yolo_run` availability.

### File List

- _bmad-output/implementation-artifacts/14-3-yolo-run-mcp-tool.md
- _bmad-output/implementation-artifacts/sprint-status.yaml
- src/yolo_developer/mcp/__init__.py
- src/yolo_developer/mcp/server.py
- src/yolo_developer/mcp/tools.py
- tests/unit/conftest.py
- tests/integration/mcp/test_mcp_server.py
- tests/unit/mcp/test_tools.py

### Change Log

- 2026-01-21: Implemented yolo_run MCP tool, added sprint registry, tests, and unit env isolation fixture.
- 2026-01-21: Code review fixes for seed validation, sprint task registry safety/cleanup, and doc updates.
