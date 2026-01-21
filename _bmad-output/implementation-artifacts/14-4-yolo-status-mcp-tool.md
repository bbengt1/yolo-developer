# Story 14.4: yolo_status MCP Tool

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an MCP client (e.g., Claude Code),
I want to invoke a `yolo_status` tool via MCP,
so that I can query sprint progress by sprint_id.

## Acceptance Criteria

### AC1: yolo_status Tool Registration
**Given** the YOLO Developer MCP server is running
**When** a client queries available tools
**Then** `yolo_status` is listed as an available tool
**And** tool description explains its purpose for LLM understanding
**And** tool parameters are properly documented with types

### AC2: Return Sprint Status
**Given** a sprint is in progress or complete
**When** `yolo_status` is invoked with a valid sprint_id
**Then** current status is returned
**And** progress details are included (timestamps, status, error if failed)
**And** response format matches the tool schema

### AC3: Unknown Sprint Handling
**Given** `yolo_status` is invoked with an unknown sprint_id
**When** the tool validates inputs
**Then** it returns a structured error response
**And** no status fields are returned

## Tasks / Subtasks

- [x] Task 1: Implement yolo_status MCP Tool (AC: #1, #2, #3)
  - [x] Subtask 1.1: Add `yolo_status` tool in `src/yolo_developer/mcp/tools.py`
  - [x] Subtask 1.2: Define input schema (sprint_id)
  - [x] Subtask 1.3: Return structured response using sprint registry data
  - [x] Subtask 1.4: Return MCP-compatible error payloads for unknown sprint
- [x] Task 2: Documentation Updates (AC: #1)
  - [x] Subtask 2.1: Update `src/yolo_developer/mcp/server.py` instructions
  - [x] Subtask 2.2: Update `src/yolo_developer/mcp/__init__.py` tool list
- [x] Task 3: Tests (AC: all)
  - [x] Subtask 3.1: Unit tests for success + unknown sprint_id
  - [x] Subtask 3.2: Integration tests via FastMCP Client

## Dev Notes

### Developer Context
- Mirror the FastMCP tool pattern used in `yolo_seed` and `yolo_run` in `src/yolo_developer/mcp/tools.py`.
- Use the in-memory sprint registry (`get_sprint`, `StoredSprint`) to serve status without blocking or spawning new tasks.
- Return MCP-compatible structured responses with stable fields and explicit errors for unknown sprint IDs.

### Technical Requirements
- Inputs: `sprint_id` (required, string). Validate non-empty before lookup.
- Outputs (success): JSON-serializable dict with `status`, `sprint_id`, `seed_id`, `thread_id`, `started_at`, and `completed_at` (nullable), plus `error` when status is `failed`.
- Outputs (error): `{"status": "error", "error": "Sprint not found"}` with no sprint metadata.
- Concurrency: read sprint data via `get_sprint()` (locked); do not mutate sprint state inside `yolo_status`.

### Architecture Compliance
- ADR-004: Use FastMCP 2.x decorator-based tools; keep `SERVER_INSTRUCTIONS` updated for tool discovery.
- ADR-001: Keep TypedDict state intact; `yolo_status` should return new dicts and avoid mutating shared state directly.
- ADR-007: Use resilient error handling paths; return structured errors rather than raising.

### Library/Framework Requirements
- FastMCP 2.x tool registration (current: 2.14.3).
- Stdlib `datetime` for ISO-8601 timestamps and `typing` for literal types.
- Structured logging via `structlog` consistent with existing MCP tools.

### Project Structure Notes
- Primary implementation: `src/yolo_developer/mcp/tools.py` (add `yolo_status` tool and response shaping).
- Update `src/yolo_developer/mcp/server.py` `SERVER_INSTRUCTIONS` to document `yolo_status` input/output.
- Update `src/yolo_developer/mcp/__init__.py` tool list to remove "coming" wording once implemented.

### Testing Requirements
- Unit tests in `tests/unit/mcp/test_tools.py` for:
  - Valid sprint_id returns metadata (running/completed/failed).
  - Unknown sprint_id returns structured error.
- Integration tests in `tests/integration/mcp/test_mcp_server.py` using FastMCP client:
  - Ensure tool registration.
  - Validate responses for success and unknown sprint.
- Clear sprint registry between tests using existing `clear_sprints()` fixture pattern (see Story 14.3).

### Previous Story Intelligence
- Story 14.3 introduced `StoredSprint`, `_sprints_lock`, and `get_sprint()` in `src/yolo_developer/mcp/tools.py` to support status queries.
- `_run_sprint` updates sprint status to `completed` or `failed` and sets timestamps; `yolo_status` should surface these fields without mutation.
- Sprint task registry is now thread-safe and cleaned up; avoid adding new shared mutable state without locks.
- Tests already use `clear_sprints()` between runs; align new tests to the same pattern.

### Git Intelligence Summary
- Recent work added `yolo_run` and sprint registry behavior in `src/yolo_developer/mcp/tools.py`; follow those patterns for tool registration and response shape.

### Latest Tech Information
- No external web research performed in this environment; using architecture-provided versions.
- FastMCP latest stable on PyPI per architecture: `2.14.3`.
- MCP SDK latest stable on PyPI per epics context: `1.25.0`.

### Project Context Reference
- No `project-context.md` found under `docs/` at story creation time.

## References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 14.4] MCP yolo_status story definition and ACs
- [Source: _bmad-output/planning-artifacts/prd.md#FR112-FR117] MCP protocol requirements
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-004] FastMCP 2.x server pattern
- [Source: _bmad-output/implementation-artifacts/14-3-yolo-run-mcp-tool.md] Sprint registry and MCP tool patterns

## Dev Agent Record

### Agent Model Used

GPT-5 (Codex CLI)

### Implementation Plan

- Add `yolo_status` MCP tool that reads sprint registry and returns structured status data.
- Update MCP server instructions and public API exports.
- Extend unit and integration tests for registration and status/error responses.
- Run full test suite to confirm no regressions.

### Debug Log References

- `uv run pytest tests/unit/mcp/test_tools.py -k yolo_status` (fail; missing yolo_status)
- `uv run pytest tests/unit/mcp/test_tools.py -k yolo_status` (pass)
- `uv run pytest tests/unit/mcp/test_tools.py tests/integration/mcp/test_mcp_server.py` (pass)
- `YOLO_PROJECT_NAME=yolo-developer OPENAI_API_KEY=$OPENAI_API_KEY uv run pytest` (fail; ruff import/order errors)
- `uv run ruff check --fix src/yolo_developer/mcp/__init__.py src/yolo_developer/mcp/tools.py` (fix imports/__all__)
- `YOLO_PROJECT_NAME=yolo-developer OPENAI_API_KEY=$OPENAI_API_KEY uv run pytest` (pass; 6516 tests, 44 warnings)
- `uv run ruff check src tests` (pass)
- `uv run pytest tests/unit/mcp/test_tools.py -k yolo_status` (pass)
- `uv run pytest tests/integration/mcp/test_mcp_server.py -k yolo_status` (pass)

### Completion Notes List

1. Implemented `yolo_status` MCP tool with input validation and structured status/error responses.
2. Updated MCP server instructions and public exports to include `yolo_status`.
3. Added unit and integration coverage for tool registration and status/error responses.
4. Full test suite passes with `YOLO_PROJECT_NAME=yolo-developer`.
5. Generated definition-of-done validation report for this story.
6. Documented the yolo_status walkthrough in the README and MCP docs.
7. Added a CLI-based tool listing example to MCP documentation.
8. Clarified when the in-process MCP client is appropriate vs. a running server.
9. Updated docs to align CLI seed usage and clarified MCP tool availability wording.
10. Added thread-safe sprint snapshot reads for yolo_status.

### File List

- _bmad-output/implementation-artifacts/14-4-yolo-status-mcp-tool.md
- _bmad-output/implementation-artifacts/sprint-status.yaml
- _bmad-output/implementation-artifacts/validation-report-2026-01-21-085015.md
- _bmad-output/implementation-artifacts/validation-report-2026-01-21-091955.md
- README.md
- docs/mcp/index.md
- src/yolo_developer/mcp/__init__.py
- src/yolo_developer/mcp/server.py
- src/yolo_developer/mcp/tools.py
- tests/integration/mcp/test_mcp_server.py
- tests/unit/mcp/test_tools.py

### Change Log

- 2026-01-21: Implemented yolo_status MCP tool, updated MCP docs/exports, and added unit/integration tests.
- 2026-01-21: Documented yolo_status walkthrough and tool verification example in README and MCP docs, including usage note on in-process vs server clients.
- 2026-01-21: Code review fixes for CLI doc accuracy, MCP server doc drift, and thread-safe sprint snapshots.
