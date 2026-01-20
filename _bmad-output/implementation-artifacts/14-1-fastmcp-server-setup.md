# Story 14.1: FastMCP Server Setup

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want YOLO Developer exposed as an MCP server,
so that MCP clients can invoke it.

## Acceptance Criteria

### AC1: FastMCP Server Module
**Given** YOLO Developer is installed
**When** I import `from yolo_developer.mcp import mcp`
**Then** a FastMCP 2.x server instance is available
**And** server name is "YOLO Developer"
**And** server metadata includes version from package

### AC2: STDIO Transport Support
**Given** the MCP server is started
**When** using STDIO transport
**Then** the server listens on stdin/stdout
**And** MCP messages are properly serialized
**And** the server can be used with Claude Desktop configuration

### AC3: HTTP Transport Support
**Given** the MCP server is started with HTTP transport
**When** running `yolo mcp --transport http --port 8000`
**Then** the server listens on the specified port
**And** HTTP endpoints respond to MCP requests
**And** server can be accessed by remote MCP clients

### AC4: Server Metadata Configuration
**Given** the MCP server is running
**When** a client queries server capabilities
**Then** server returns name "YOLO Developer"
**And** server returns version from `yolo_developer.__version__`
**And** server returns instructions describing available tools
**And** error details are masked in production (mask_error_details=True)

### AC5: CLI Integration
**Given** the YOLO CLI is installed
**When** I run `yolo mcp`
**Then** the MCP server starts with STDIO transport (default)
**And** `yolo mcp --transport http` starts HTTP transport
**And** `yolo mcp --port 8080` configures the HTTP port
**And** `yolo mcp --help` shows available options

### AC6: Test Coverage
**Given** the MCP server implementation
**When** running tests
**Then** server initialization is tested
**And** transport configuration is tested
**And** CLI command integration is tested
**And** tests use FastMCP's testing patterns with mocked clients

## Tasks / Subtasks

- [x] Task 1: Create FastMCP Server Module (AC: #1, #4)
  - [x] Subtask 1.1: Create `src/yolo_developer/mcp/server.py` with FastMCP instance
  - [x] Subtask 1.2: Configure server name "YOLO Developer" with version from package
  - [x] Subtask 1.3: Set `mask_error_details=True` for production safety
  - [x] Subtask 1.4: Add server instructions describing available tools (placeholder for future tools)
  - [x] Subtask 1.5: Export `mcp` instance from `src/yolo_developer/mcp/__init__.py`

- [x] Task 2: Implement Transport Configuration (AC: #2, #3)
  - [x] Subtask 2.1: Add `run()` method wrapper for transport selection
  - [x] Subtask 2.2: Implement STDIO transport as default (for Claude Desktop)
  - [x] Subtask 2.3: Implement HTTP transport with configurable port
  - [x] Subtask 2.4: Add transport type enum or literal for type safety

- [x] Task 3: Add CLI Command (AC: #5)
  - [x] Subtask 3.1: Create `src/yolo_developer/cli/commands/mcp.py` with mcp command
  - [x] Subtask 3.2: Add `--transport` option with choices [stdio, http]
  - [x] Subtask 3.3: Add `--port` option for HTTP transport (default 8000)
  - [x] Subtask 3.4: Register command in `src/yolo_developer/cli/main.py`

- [x] Task 4: Write Unit Tests (AC: #6)
  - [x] Subtask 4.1: Test server initialization and metadata
  - [x] Subtask 4.2: Test transport configuration methods
  - [x] Subtask 4.3: Test CLI command parsing and options
  - [x] Subtask 4.4: Test server can be instantiated without errors

- [x] Task 5: Update Documentation (AC: all)
  - [x] Subtask 5.1: Update module docstrings with usage examples
  - [x] Subtask 5.2: Export server from package `__init__.py` for programmatic access

## Dev Notes

### Architecture Patterns (MANDATORY)

Per architecture.md ADR-004 and ARCH-PATTERN-4:

1. **FastMCP 2.x** is the required framework for MCP server implementation
2. **Decorator-based tool definition** will be used for tool registration (future stories)
3. **STDIO and HTTP transports** must be supported
4. **Error masking** must be enabled for production safety

### FastMCP 2.x API Reference (Current as of 2026-01)

Based on latest FastMCP documentation research:

```python
from fastmcp import FastMCP

# Server initialization
mcp = FastMCP(
    name="YOLO Developer",          # Server name visible to clients
    instructions="...",              # Server instructions for LLMs
    version="1.0.0",                 # Optional: defaults to FastMCP version
    mask_error_details=True,         # Hide internal errors from clients
)

# Tool registration (for future stories 14.2-14.5)
@mcp.tool
def tool_name(param: str) -> dict:
    """Tool description for LLM."""
    return {"result": "value"}

# Resource registration (optional)
@mcp.resource("resource://path")
def resource_name() -> str:
    """Resource description."""
    return "resource content"

# Running the server
if __name__ == "__main__":
    mcp.run()  # Default STDIO transport

# Or with explicit transport
mcp.run(transport="stdio")  # STDIO for Claude Desktop
mcp.run(transport="http", port=8000)  # HTTP for remote access
```

### Server Configuration

```python
# src/yolo_developer/mcp/server.py
from __future__ import annotations

from fastmcp import FastMCP

from yolo_developer import __version__

# Server instructions for MCP clients
SERVER_INSTRUCTIONS = """
YOLO Developer MCP Server

Available tools will include:
- yolo_seed: Provide seed requirements for development
- yolo_run: Execute autonomous sprint
- yolo_status: Query sprint status
- yolo_audit: Access audit trail

Use these tools to integrate YOLO Developer into your AI workflow.
"""

mcp = FastMCP(
    name="YOLO Developer",
    instructions=SERVER_INSTRUCTIONS,
    version=__version__,
    mask_error_details=True,  # Production safety
)
```

### CLI Command Design

```python
# src/yolo_developer/cli/commands/mcp.py
from __future__ import annotations

from typing import Literal

import typer

from yolo_developer.mcp import mcp

app = typer.Typer(help="MCP server commands")

@app.callback(invoke_without_command=True)
def serve(
    transport: Literal["stdio", "http"] = typer.Option(
        "stdio",
        "--transport", "-t",
        help="Transport protocol to use",
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port for HTTP transport",
    ),
) -> None:
    """Start the YOLO Developer MCP server."""
    if transport == "http":
        mcp.run(transport="http", port=port)
    else:
        mcp.run(transport="stdio")
```

### File Structure (Alignment with Architecture)

Per architecture.md project structure:

```
src/yolo_developer/
├── mcp/                    # MCP Server (FR112-117)
│   ├── __init__.py         # Export mcp instance
│   ├── server.py           # FastMCP server definition
│   └── tools.py            # MCP tool implementations (future stories)
├── cli/
│   └── commands/
│       └── mcp.py          # CLI command for MCP server
```

### Testing Patterns (from Epic 13)

Per test patterns from Story 13.6 and architecture.md:

```python
# tests/unit/mcp/test_server.py
from __future__ import annotations

import pytest

from yolo_developer.mcp import mcp
from yolo_developer.mcp.server import SERVER_INSTRUCTIONS


def test_server_initialization():
    """Test MCP server is properly initialized."""
    assert mcp.name == "YOLO Developer"


def test_server_has_version():
    """Test server version is set from package."""
    from yolo_developer import __version__
    assert mcp.version == __version__


def test_server_has_instructions():
    """Test server has instructions configured."""
    assert mcp.instructions == SERVER_INSTRUCTIONS
    assert "yolo_seed" in SERVER_INSTRUCTIONS


def test_server_masks_errors():
    """Test error masking is enabled."""
    assert mcp.mask_error_details is True
```

### FastMCP Testing with Client (Integration Tests)

```python
# tests/integration/test_mcp_server.py
from __future__ import annotations

import pytest
from fastmcp import Client

from yolo_developer.mcp import mcp


@pytest.mark.asyncio
async def test_server_responds_to_client():
    """Test server can handle client requests."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        # No tools registered yet in this story
        assert isinstance(tools, list)
```

### Key Files to Touch

**Create:**
- `src/yolo_developer/mcp/server.py` - FastMCP server definition
- `src/yolo_developer/cli/commands/mcp.py` - CLI command
- `tests/unit/mcp/test_server.py` - Unit tests
- `tests/unit/mcp/__init__.py` - Test package marker

**Modify:**
- `src/yolo_developer/mcp/__init__.py` - Export `mcp` instance
- `src/yolo_developer/cli/main.py` - Register mcp command

### Dependencies

Per architecture.md ARCH-DEP-4, FastMCP 2.x should already be in dependencies. Verify in `pyproject.toml`:

```toml
dependencies = [
    # ... other dependencies
    "fastmcp>=2.0.0",
]
```

If not present, add it:
```bash
uv add fastmcp
```

### Previous Story Learnings (Epic 13)

From Story 13.6 completion notes:
1. Run `ruff check` and `mypy` before committing
2. Use `from __future__ import annotations` in all files
3. All SDK tests (153) currently passing - don't break them
4. Exception chaining with `raise ... from e` pattern
5. Test both success and error paths
6. Export new types from package `__init__.py`

### Claude Desktop Configuration

For testing with Claude Desktop, the server can be configured in `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yolo-developer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yolo-developer", "yolo", "mcp"]
    }
  }
}
```

Or using fastmcp install command:
```bash
fastmcp install claude-desktop src/yolo_developer/mcp/server.py:mcp --server-name "YOLO Developer"
```

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-004] - MCP Implementation decision
- [Source: _bmad-output/planning-artifacts/architecture.md#ARCH-DEP-4] - FastMCP 2.x dependency
- [Source: _bmad-output/planning-artifacts/architecture.md#ARCH-PATTERN-4] - FastMCP decorator pattern
- [Source: _bmad-output/planning-artifacts/epics.md#Story 14.1] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR112-FR117] - MCP functional requirements
- [FastMCP Docs: https://gofastmcp.com] - FastMCP 2.x documentation
- [Related: Story 13.6 (Event Emission)] - SDK patterns and testing approach
- [Related: Story 14.2-14.5] - Future tool implementations that will use this server

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

1. **FastMCP 2.x Global Settings**: FastMCP 2.x uses `fastmcp.settings` global module for configuration like `mask_error_details` rather than instance attributes. The server.py imports and configures this at module load time.

2. **Transport Enum Pattern**: Added `TransportType` enum for type-safe transport selection. The `run_server()` function accepts both string and enum values for flexibility.

3. **CLI Registration**: Used Typer's `app.callback(invoke_without_command=True)` pattern to register the mcp command as both a subcommand and allow it to run directly.

4. **Async Methods**: FastMCP's `get_tools()` method is async. Tests check for method existence rather than calling async methods directly.

5. **All 28 tests pass**: 18 server tests + 7 CLI command tests + 3 integration tests covering AC1-AC6.

### File List

**Created:**
- `src/yolo_developer/mcp/server.py` - FastMCP server implementation with TransportType, run_server()
- `src/yolo_developer/cli/commands/mcp.py` - CLI command for `yolo mcp`
- `tests/unit/mcp/__init__.py` - Test package marker
- `tests/unit/mcp/test_server.py` - 18 unit tests for server module
- `tests/unit/cli/test_mcp_command.py` - 7 tests for CLI command
- `tests/integration/mcp/__init__.py` - Integration test package marker
- `tests/integration/mcp/test_mcp_server.py` - 3 integration tests with FastMCP Client

**Modified:**
- `src/yolo_developer/mcp/__init__.py` - Added exports: TransportType, run_server
- `src/yolo_developer/cli/main.py` - Registered mcp subcommand
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status

## Senior Developer Review (AI)

**Reviewer:** Claude Opus 4.5 (code-review workflow)
**Date:** 2026-01-19
**Outcome:** ✅ APPROVED (after fixes)

### Issues Found and Fixed

| ID | Severity | Issue | Resolution |
|----|----------|-------|------------|
| M1 | Medium | `sprint-status.yaml` modified but not in File List | Added to File List |
| M2 | Medium | Redundant `mask_error_details` in FastMCP constructor | Removed, kept global setting only |
| M3 | Medium | Missing integration tests with FastMCP Client | Created 3 integration tests |
| L1 | Low | Incorrect type annotation `patch` → `MagicMock` | Fixed in CLI tests |
| L2 | Low | Missing `__all__` documentation | Added comment explaining exports |
| L3 | Low | Unclear docstring about `run_server()` | Clarified type-safe wrapper purpose |

### Final Test Results
- **28 tests pass** (18 unit server + 7 unit CLI + 3 integration)
- **ruff check:** All checks passed
- **mypy:** Success, no issues found

### Verification
All ACs validated as fully implemented. No critical issues found.
