# Story 14.6: Claude Code Compatibility

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want YOLO Developer to work with Claude Code and other MCP-compatible AI assistants (including Codex CLI),
so that I can use it from my preferred AI assistant regardless of LLM provider.

## Acceptance Criteria

### AC1: Tool Discovery (Multi-Provider)
**Given** any MCP-compatible client (Claude Code, Codex CLI, or other)
**When** YOLO Developer server is added
**Then** all tools are discoverable via MCP protocol
**And** tool descriptions are clear and LLM-agnostic (no provider-specific language)
**And** tool parameters are properly typed and documented per MCP spec

### AC2: Tool Invocation (Multi-Provider)
**Given** any MCP-compatible client connected to YOLO Developer server
**When** tools are invoked
**Then** `yolo_seed`, `yolo_run`, `yolo_status`, and `yolo_audit` work correctly
**And** input parameters are correctly parsed regardless of client
**And** responses are properly formatted for any MCP client display

### AC3: Response Rendering (Provider-Agnostic)
**Given** any MCP tool invocation from any client
**When** the tool returns a response
**Then** responses render correctly in all MCP clients
**And** success responses display meaningful data
**And** error responses display clear error messages
**And** structured content uses standard JSON (no provider-specific formatting)

### AC4: Error Handling (Provider-Agnostic)
**Given** any MCP tool invocation from any client
**When** an error occurs (validation, not found, internal)
**Then** error handling works properly across all clients
**And** errors are returned in MCP-compliant format
**And** any MCP client can display and understand the error
**And** no unhandled exceptions are exposed to any client

### AC5: Transport Compatibility
**Given** different MCP clients with varying transport preferences
**When** connecting to YOLO Developer MCP server
**Then** STDIO transport works for Claude Code and Codex CLI
**And** HTTP/SSE transport remains available for other integrations
**And** no transport-specific assumptions break compatibility

## Tasks / Subtasks

- [x] Task 1: Verify and Fix MCP Tool Registration (AC: #1)
  - [x] Subtask 1.1: Audit all tool docstrings for LLM-agnostic language (no Claude/Codex-specific terms)
  - [x] Subtask 1.2: Verify parameter annotations use proper MCP-compatible types
  - [x] Subtask 1.3: Test tool discovery via generic MCP client list_tools()
  - [x] Subtask 1.4: Ensure tool descriptions work for both Claude and GPT-based LLMs

- [x] Task 2: Test and Fix Tool Invocations (AC: #2)
  - [x] Subtask 2.1: Test yolo_seed with content and file_path parameters
  - [x] Subtask 2.2: Test yolo_run with seed_id and seed_content parameters
  - [x] Subtask 2.3: Test yolo_status with valid and invalid sprint_ids
  - [x] Subtask 2.4: Test yolo_audit with various filter combinations

- [x] Task 3: Validate Response Formatting (AC: #3)
  - [x] Subtask 3.1: Verify JSON responses are properly serializable (no custom objects)
  - [x] Subtask 3.2: Ensure datetime fields are ISO-8601 formatted strings
  - [x] Subtask 3.3: Test structured content rendering in generic MCP clients
  - [x] Subtask 3.4: Verify all response fields have consistent naming (snake_case)
  - [x] Subtask 3.5: Ensure no provider-specific response formatting

- [x] Task 4: Error Handling Hardening (AC: #4)
  - [x] Subtask 4.1: Verify all error paths return structured error responses
  - [x] Subtask 4.2: Ensure no raw exceptions escape to MCP clients
  - [x] Subtask 4.3: Test error responses with FastMCP mask_error_details enabled
  - [x] Subtask 4.4: Add comprehensive error handling tests

- [x] Task 5: Transport Compatibility (AC: #5)
  - [x] Subtask 5.1: Verify STDIO transport works (primary for Claude Code + Codex)
  - [x] Subtask 5.2: Verify HTTP transport works (for remote/web integrations)
  - [x] Subtask 5.3: Test run_server() with both TransportType.STDIO and TransportType.HTTP
  - [x] Subtask 5.4: Ensure no transport-specific assumptions in tool implementations

- [x] Task 6: Documentation Updates (AC: #1, #5)
  - [x] Subtask 6.1: Update SERVER_INSTRUCTIONS for multi-provider usage
  - [x] Subtask 6.2: Verify pyproject.toml MCP entry point is correct
  - [x] Subtask 6.3: Create Claude Desktop config example in README/docs
  - [x] Subtask 6.4: Create Codex CLI config example in README/docs
  - [x] Subtask 6.5: Document generic MCP client integration pattern

- [x] Task 7: Integration Testing (AC: all)
  - [x] Subtask 7.1: Create end-to-end integration test with FastMCP client
  - [x] Subtask 7.2: Test full workflow: seed -> run -> status -> audit
  - [x] Subtask 7.3: Test concurrent MCP requests (thread safety)
  - [x] Subtask 7.4: Verify existing Codex compatibility tests still pass (from 14-7)

## Dev Notes

### Developer Context
- The MCP server is implemented using FastMCP 2.x at `src/yolo_developer/mcp/server.py`
- Tools are registered in `src/yolo_developer/mcp/tools.py` via `@mcp.tool` decorator
- The server already has `mask_error_details = True` configured for production safety
- Previous stories (14.2-14.5) implemented the core tools; this story validates compatibility
- **Story 14-7 already added Codex compatibility fixes** - must preserve these patterns

### Multi-Provider Compatibility Requirements
- **CRITICAL**: Must work with both Claude Code (Anthropic) AND Codex CLI (OpenAI)
- Tool descriptions must be LLM-agnostic (work equally well for Claude and GPT models)
- No provider-specific language in docstrings or responses
- Standard MCP protocol compliance ensures cross-provider compatibility

### Technical Requirements
- **Transport**: STDIO transport is primary for both Claude Code and Codex CLI
- **Tool Registration**: FastMCP decorator pattern with provider-agnostic docstrings
- **Response Format**: JSON-serializable dicts with `status` field and consistent naming
- **Error Format**: `{"status": "error", "error": "message"}` pattern (MCP standard)
- **Threading**: Thread-safe via locks for concurrent MCP requests
- **No Provider Lock-in**: Avoid Claude-specific or OpenAI-specific formatting

### Architecture Compliance
- ADR-004: FastMCP 2.x server pattern (already implemented)
- ADR-007: Resilient error handling with structured error responses
- ADR-008: Configuration via Pydantic Settings (secrets via env vars)

### Library/Framework Requirements
- **FastMCP**: 2.14.3+ (current: 2.0.0+ in pyproject.toml)
- **MCP SDK**: Compatible with MCP protocol 1.0+
- **Claude Code**: Requires proper STDIO transport support
- **pytest-asyncio**: For async integration tests

### Project Structure Notes
- Primary files: `src/yolo_developer/mcp/server.py`, `src/yolo_developer/mcp/tools.py`
- Tests: `tests/unit/mcp/test_tools.py`, `tests/integration/mcp/test_mcp_server.py`
- No structural changes expected - this story validates and hardens existing implementation

### Testing Requirements
- Unit tests for each tool's edge cases and error paths
- Integration tests using FastMCP client `create_connected_server_and_client_session`
- Test tool discovery (`list_tools()`) returns all expected tools
- Test tool invocation with valid and invalid parameters
- Test concurrent requests for thread safety validation

### Previous Story Intelligence
- Story 14.5 (yolo_audit) added comprehensive filtering and pagination
- Story 14.4 (yolo_status) established sprint status reporting patterns
- Story 14.3 (yolo_run) implemented async sprint execution
- Story 14.2 (yolo_seed) established seed storage patterns
- All tools follow consistent `{"status": "...", ...}` response pattern

### Git Intelligence Summary
- Recent commits added MCP tools and fixed documentation
- **`56c43de fix: codex compatibility review fixes (14-7)`** - CRITICAL: Story 14-7 already addressed Codex compatibility
  - This story must preserve and extend those fixes
  - Review 14-7 changes to ensure no regression
- MCP implementation is mature with 4 tools already registered
- Must verify 14-7 patterns are maintained while adding Claude Code validation

### Latest Tech Information
- FastMCP 2.14.3+ supports both STDIO and HTTP transports
- MCP SDK 1.25.0+ is current stable version
- Claude Code uses STDIO transport for local MCP servers
- `mcp install` command can register servers in Claude Desktop config
- Tool docstrings become the LLM-readable descriptions
- Parameter type hints determine MCP input schema

### Multi-Provider Integration Notes

#### Claude Code Configuration
- Config location: `~/.claude/claude_desktop_config.json` (macOS)
- Server config format:
  ```json
  {
    "mcpServers": {
      "yolo-developer": {
        "command": "uv",
        "args": ["run", "yolo-mcp"]
      }
    }
  }
  ```

#### Codex CLI Configuration
- Codex CLI uses MCP servers via similar configuration
- Ensure `yolo-mcp` entry point works with both `uv run` and direct Python invocation
- Codex may use different environment variable handling - test both

#### Generic MCP Client Integration
- STDIO transport is default and preferred for local AI assistants
- HTTP transport available for remote/web-based integrations
- The `yolo-mcp` entry point should be defined in pyproject.toml
- All tools use standard MCP types - no provider-specific extensions

### Project Context Reference
- CLAUDE.md specifies: `uv run yolo --help` for CLI entry point
- MCP server can be started via Python module: `python -m yolo_developer.mcp`
- Entry point pattern from architecture: `yolo_developer.mcp:run_server`

## References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 14.6] Claude Code compatibility story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-004] FastMCP 2.x server pattern
- [Source: src/yolo_developer/mcp/server.py] MCP server implementation
- [Source: src/yolo_developer/mcp/tools.py] MCP tool implementations
- [Source: _bmad-output/implementation-artifacts/14-5-yolo-audit-mcp-tool.md] Previous story patterns
- [Source: MCP Python SDK docs] Transport and integration patterns
- [Source: FastMCP GitHub] Tool registration and server configuration

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (Claude Code)

### Debug Log References

### Completion Notes List

1. Story created from epics definition with exhaustive context analysis.
2. Analyzed existing MCP implementation in server.py and tools.py.
3. Researched latest MCP SDK patterns for Claude Code integration.
4. Previous stories (14.2-14.5) established solid foundation.
5. **CRITICAL**: Story 14-7 already added Codex compatibility - must preserve these patterns.
6. Updated story to ensure multi-LLM provider compatibility (Claude Code + Codex CLI + others).
7. This story focuses on validation, hardening, multi-provider testing, and documentation.
8. **Implementation Complete**: All 7 tasks and 30 subtasks completed successfully.
9. Fixed provider-specific language in server.py module docstring and TransportType docstring.
10. Added `yolo-mcp` entry point to pyproject.toml for direct MCP server invocation.
11. Created `__main__.py` for `python -m yolo_developer.mcp` execution.
12. Updated docs/mcp/index.md with multi-provider configuration examples (Claude Code, Codex CLI, generic clients).
13. Added 8 new test classes with 53 new tests covering all acceptance criteria.
14. All 122 MCP tests pass, linting and type checking pass.

### File List

- _bmad-output/implementation-artifacts/14-6-claude-code-compatibility.md
- src/yolo_developer/mcp/server.py (updated docstrings for provider-agnostic language)
- src/yolo_developer/mcp/__init__.py (updated docstring for provider-agnostic language)
- src/yolo_developer/mcp/__main__.py (new - module entry point)
- pyproject.toml (added yolo-mcp entry point)
- tests/unit/mcp/test_tools.py (added TestMcpProviderAgnosticLanguage, TestMcpResponseFormatting, TestMcpErrorHandling)
- tests/unit/mcp/test_server.py (added TestTransportCompatibility)
- tests/integration/mcp/test_mcp_server.py (added TestMcpToolDiscovery, TestMcpFullWorkflow, TestMcpProviderCompatibility)
- docs/mcp/index.md (updated with multi-provider configuration examples)

## Change Log

- 2026-01-22: Story created with full context analysis from create-story workflow.
- 2026-01-22: Updated to ensure multi-LLM provider compatibility (Claude Code + Codex CLI).
- 2026-01-22: Implementation complete - all tasks/subtasks done. Status changed to review.
