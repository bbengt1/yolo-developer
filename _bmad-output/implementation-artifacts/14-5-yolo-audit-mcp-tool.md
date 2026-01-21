# Story 14.5: yolo_audit MCP Tool

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a `yolo_audit` MCP tool,
so that external systems can access audit data.

## Acceptance Criteria

### AC1: yolo_audit Tool Registration
**Given** the YOLO Developer MCP server is running
**When** a client queries available tools
**Then** `yolo_audit` is listed as an available tool
**And** tool description explains its purpose for LLM understanding
**And** tool parameters are properly documented with types

### AC2: Audit Data Retrieval
**Given** audit data exists
**When** `yolo_audit` is invoked
**Then** audit data is returned
**And** data format is MCP-compliant JSON
**And** results are ordered chronologically

### AC3: Filtering Support
**Given** audit data exists
**When** `yolo_audit` is invoked with filters
**Then** filtering parameters are supported
**And** only matching audit entries are returned

### AC4: Pagination Support
**Given** large audit datasets
**When** `yolo_audit` is invoked with limit/offset
**Then** pagination is supported
**And** the response includes total counts or paging metadata

## Tasks / Subtasks

- [x] Task 1: Implement yolo_audit MCP Tool (AC: #1, #2, #3, #4)
  - [x] Subtask 1.1: Add `yolo_audit` tool in `src/yolo_developer/mcp/tools.py`
  - [x] Subtask 1.2: Define input schema (filters, limit, offset)
  - [x] Subtask 1.3: Return MCP-compatible audit payloads
  - [x] Subtask 1.4: Support filtering + pagination in responses
- [x] Task 2: Documentation Updates (AC: #1)
  - [x] Subtask 2.1: Update `src/yolo_developer/mcp/server.py` instructions
  - [x] Subtask 2.2: Update `src/yolo_developer/mcp/__init__.py` tool list
- [x] Task 3: Tests (AC: all)
  - [x] Subtask 3.1: Unit tests for filtering, pagination, and empty results
  - [x] Subtask 3.2: Integration tests via FastMCP Client

## Dev Notes

### Developer Context
- Mirror the FastMCP tool pattern used in `yolo_seed`, `yolo_run`, and `yolo_status` in `src/yolo_developer/mcp/tools.py`.
- Audit data can be retrieved via the SDK audit utilities in `src/yolo_developer/sdk/client.py` or direct audit store queries.
- Return MCP-compatible structured responses with explicit errors for invalid filters or missing audit store.

### Technical Requirements
- Inputs: optional filters (`agent`, `decision_type`, `artifact_type`, `start_time`, `end_time`) plus `limit` and `offset`.
- Outputs (success): JSON-serializable dict with `status`, `entries`, `limit`, `offset`, and `total`.
- Outputs (error): `{"status": "error", "error": "..."}` with no data payload.
- Ordering: results must be chronological (oldest first).
- Pagination: apply `offset` then `limit` in that order.

### Architecture Compliance
- ADR-004: Use FastMCP 2.x decorator-based tools; keep `SERVER_INSTRUCTIONS` updated for tool discovery.
- ADR-001: Keep TypedDict state intact; avoid mutating shared state directly.
- ADR-007: Use resilient error handling paths; return structured errors rather than raising.

### Library/Framework Requirements
- FastMCP 2.x tool registration (current: 2.14.3).
- Stdlib `datetime` for ISO-8601 timestamps and `typing` for filter literals.
- Structured logging via `structlog` consistent with existing MCP tools.

### Project Structure Notes
- Primary implementation: `src/yolo_developer/mcp/tools.py` (add `yolo_audit` tool and response shaping).
- Audit retrieval helpers exist in `src/yolo_developer/sdk/client.py` (`get_audit` / `get_audit_async`).
- Filtering types available in `src/yolo_developer/audit/filter_types.py` and stores in `src/yolo_developer/audit/json_decision_store.py`.
- Update `src/yolo_developer/mcp/server.py` `SERVER_INSTRUCTIONS` to document `yolo_audit` input/output.
- Update `src/yolo_developer/mcp/__init__.py` tool list to remove "coming" wording once implemented.

### Testing Requirements
- Unit tests in `tests/unit/mcp/test_tools.py` for:
  - Empty audit store returns empty list with correct metadata.
  - Filtering by agent or decision_type returns expected subset.
  - Pagination applies `offset` then `limit`.
- Integration tests in `tests/integration/mcp/test_mcp_server.py` using FastMCP client:
  - Ensure tool registration.
  - Validate responses for success and filter parameters.

### Previous Story Intelligence
- Story 14.4 added thread-safe sprint status reporting and MCP tool registration tests; follow the same tool registration patterns.
- SDK audit access (`YoloClient.get_audit_async`) uses `JsonDecisionStore` under `.yolo/audit/decisions.json` and applies filters via `AuditFilters`.
- Audit store returns decisions in chronological order; reuse ordering to match AC2.

### Git Intelligence Summary
- Recent work added MCP tool patterns and tests in `src/yolo_developer/mcp/tools.py` and `tests/*/mcp`.

### Latest Tech Information
- No external web research performed in this environment; using architecture-provided versions.
- FastMCP latest stable on PyPI per architecture: `2.14.3`.
- MCP SDK latest stable on PyPI per epics context: `1.25.0`.

### Project Context Reference
- No `project-context.md` found under `docs/` at story creation time.

## References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 14.5] MCP yolo_audit story definition and ACs
- [Source: _bmad-output/planning-artifacts/prd.md#FR112-FR117] MCP protocol requirements
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-004] FastMCP 2.x server pattern
- [Source: src/yolo_developer/sdk/client.py#get_audit_async] Existing audit retrieval logic
- [Source: src/yolo_developer/audit/filter_types.py] Audit filter definitions
- [Source: src/yolo_developer/audit/json_decision_store.py] Audit storage and ordering

## Dev Agent Record

### Agent Model Used

GPT-5 (Codex CLI)

### Debug Log References

- 2026-01-21: `uv run pytest` (fails: missing OPENAI_API_KEY for architect integration tests)
- 2026-01-21: `uv run pytest` (still fails: OPENAI_API_KEY not visible to uv run)
- 2026-01-21: `uv run pytest` (178 failed, 6303 passed, 44 warnings, 41 errors; ConfigurationError + chromadb file errors + "Too many open files")
- 2026-01-21: `uv run pytest tests/integration/agents/architect/test_adr_integration.py::TestArchitectNodeAdrIntegration::test_architect_node_generates_adrs -vv` (fails: missing required config `project_name`)
- 2026-01-21: `YOLO_PROJECT_NAME=test-project YOLO_MEMORY__PERSIST_PATH=.yolo/memory uv run pytest tests/integration/agents/architect -q` (pass: 38 passed)
- 2026-01-21: `YOLO_PROJECT_NAME=test-project YOLO_MEMORY__PERSIST_PATH=.yolo/memory uv run pytest tests/integration/test_memory_persistence.py -q` (pass: 15 passed)
- 2026-01-21: `YOLO_PROJECT_NAME=test-project YOLO_MEMORY__PERSIST_PATH=.yolo/memory uv run pytest tests/integration/test_pattern_learning.py -q` (pass: 11 passed)
- 2026-01-21: `YOLO_PROJECT_NAME=test-project YOLO_MEMORY__PERSIST_PATH=.yolo/memory uv run pytest tests/integration/test_project_isolation.py -q` (pass: 12 passed)
- 2026-01-21: `YOLO_PROJECT_NAME=test-project YOLO_MEMORY__PERSIST_PATH=.yolo/memory uv run pytest tests/unit/memory -q` (pass: 267 passed)
- 2026-01-21: `YOLO_PROJECT_NAME=test-project YOLO_MEMORY__PERSIST_PATH=.yolo/memory uv run pytest tests/unit/config -q` (fails: env override forced persist_path)
- 2026-01-21: `uv run pytest tests/unit/config -q` (pass: 176 passed)
- 2026-01-21: `uv run pytest tests/unit/mcp/test_tools.py::TestYoloAuditTool tests/integration/mcp/test_mcp_server.py::TestYoloAuditIntegration::test_yolo_audit_via_client_returns_entries` (pass: 4 passed)

### Completion Notes List

1. Ultimate context engine analysis completed - comprehensive developer guide created.
2. Story status set to ready-for-dev.
3. Web research unavailable; referenced architecture and epics for version constraints.
4. Implemented `yolo_audit` tool with ISO timestamp validation, filtering, and pagination responses.
5. Updated MCP tool docs and public exports for `yolo_audit`.
6. Tests added for unit/integration coverage of audit tool behavior.
7. Tests: `uv run pytest` (fails due to missing OPENAI_API_KEY in architect integration tests); `uv run pytest tests/unit/mcp/test_tools.py::TestYoloAuditTool tests/integration/mcp/test_mcp_server.py::TestYoloAuditIntegration::test_yolo_audit_via_client_returns_entries` (pass).
8. Full regression suite still failing: OPENAI_API_KEY not visible to `uv run pytest` (architect integration tests).
9. Full regression suite run completed with failures unrelated to this story (config validation errors, chromadb file errors, and too many open files).
10. Targeted suites pass when env configured; full regression still pending.
11. Code review fixes: added missing audit store error handling, structured exceptions, and adjusted limit=0 pagination behavior with updated tests.

### File List

- _bmad-output/implementation-artifacts/14-5-yolo-audit-mcp-tool.md
- _bmad-output/implementation-artifacts/sprint-status.yaml
- _bmad-output/implementation-artifacts/validation-report-2026-01-21-100006.md
- src/yolo_developer/mcp/tools.py
- src/yolo_developer/mcp/server.py
- src/yolo_developer/mcp/__init__.py
- tests/unit/mcp/test_tools.py
- tests/integration/mcp/test_mcp_server.py

## Change Log

- 2026-01-21: Implemented yolo_audit MCP tool with filtering/pagination, updated docs, and added tests.
- 2026-01-21: Code review fixes for audit error handling and pagination semantics.
