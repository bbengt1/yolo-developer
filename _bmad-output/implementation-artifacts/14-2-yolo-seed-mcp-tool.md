# Story 14.2: yolo_seed MCP Tool

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an MCP client (e.g., Claude Code),
I want to invoke a `yolo_seed` tool via MCP protocol,
so that I can provide seed requirements for autonomous development.

## Acceptance Criteria

### AC1: yolo_seed Tool Registration
**Given** the YOLO Developer MCP server is running
**When** a client queries available tools
**Then** `yolo_seed` is listed as an available tool
**And** tool description explains its purpose for LLM understanding
**And** tool parameters are properly documented with types

### AC2: Text-Based Seed Input
**Given** a client calls `yolo_seed` with content parameter
**When** content is provided as plain text
**Then** the tool accepts the seed content
**And** returns a unique seed_id for tracking
**And** stores the seed for later processing

### AC3: File-Based Seed Input
**Given** a client calls `yolo_seed` with file_path parameter
**When** file_path points to an existing file
**Then** the tool reads the file content
**And** processes it as seed requirements
**And** returns a seed_id with file source noted

### AC4: Input Validation
**Given** a client provides seed content
**When** the content is empty or invalid
**Then** the tool returns an error response
**And** error includes descriptive message for client
**And** does not create a seed_id for invalid input

### AC5: Seed Storage
**Given** a valid seed is provided
**When** yolo_seed processes the input
**Then** seed is stored in memory/cache
**And** seed can be retrieved by seed_id
**And** seed includes metadata (timestamp, source type)

### AC6: Tool Response Format
**Given** a successful seed operation
**When** the tool completes
**Then** response includes status ("accepted")
**And** response includes seed_id (UUID)
**And** response includes content_length and source type
**And** response is JSON-serializable for MCP protocol

### AC7: Test Coverage
**Given** the yolo_seed MCP tool implementation
**When** running tests
**Then** tool registration is tested
**And** text and file inputs are tested
**And** validation error cases are tested
**And** response format is validated
**And** integration with FastMCP Client is tested

## Tasks / Subtasks

- [x] Task 1: Implement yolo_seed MCP Tool (AC: #1, #2, #3, #6)
  - [x] Subtask 1.1: Create `src/yolo_developer/mcp/tools.py` with tool implementations
  - [x] Subtask 1.2: Add `@mcp.tool` decorator for `yolo_seed` function
  - [x] Subtask 1.3: Implement text content parameter handling
  - [x] Subtask 1.4: Implement file_path parameter handling with file reading
  - [x] Subtask 1.5: Return structured response with seed_id, status, metadata

- [x] Task 2: Implement Seed Storage (AC: #5)
  - [x] Subtask 2.1: Create seed storage mechanism (in-memory dict or simple cache)
  - [x] Subtask 2.2: Generate UUID for each seed
  - [x] Subtask 2.3: Store seed with metadata (timestamp, source, content_length)
  - [x] Subtask 2.4: Implement seed retrieval by seed_id

- [x] Task 3: Implement Input Validation (AC: #4)
  - [x] Subtask 3.1: Validate content is non-empty
  - [x] Subtask 3.2: Validate file_path exists when provided
  - [x] Subtask 3.3: Return MCP-compatible error responses
  - [x] Subtask 3.4: Handle edge cases (both content and file_path, neither provided)

- [x] Task 4: Write Unit Tests (AC: #7)
  - [x] Subtask 4.1: Test tool is registered with correct metadata
  - [x] Subtask 4.2: Test text content input
  - [x] Subtask 4.3: Test file_path input
  - [x] Subtask 4.4: Test validation error cases
  - [x] Subtask 4.5: Test response format and structure

- [x] Task 5: Write Integration Tests (AC: #7)
  - [x] Subtask 5.1: Test yolo_seed via FastMCP Client
  - [x] Subtask 5.2: Test full flow: seed → retrieve
  - [x] Subtask 5.3: Verify tool appears in list_tools()

- [x] Task 6: Update Exports and Documentation (AC: all)
  - [x] Subtask 6.1: Import tools.py in mcp/server.py to register tools
  - [x] Subtask 6.2: Update SERVER_INSTRUCTIONS with yolo_seed details
  - [x] Subtask 6.3: Add docstrings with usage examples

## Dev Notes

### Architecture Patterns (MANDATORY)

Per architecture.md ADR-004 and ARCH-PATTERN-4:

1. **FastMCP 2.x** decorator-based tool definition
2. **Error masking** is already configured in server.py
3. **Structured responses** for MCP protocol compatibility

### FastMCP Tool Definition Pattern

Based on Story 14.1 foundation and FastMCP 2.x:

```python
from fastmcp import FastMCP

from yolo_developer.mcp.server import mcp

@mcp.tool
async def yolo_seed(
    content: str | None = None,
    file_path: str | None = None,
) -> dict:
    """Provide seed requirements for autonomous development.

    Provide EITHER content (as text) OR file_path (to read from).

    Args:
        content: Seed requirements as plain text
        file_path: Path to file containing seed requirements

    Returns:
        dict with seed_id, status, content_length, source
    """
    # Implementation here
    ...
```

### Seed Storage Pattern

Simple in-memory storage for MVP (can be replaced with persistent storage later):

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class StoredSeed:
    """Stored seed with metadata."""
    seed_id: str
    content: str
    source: Literal["text", "file"]
    created_at: datetime
    content_length: int
    file_path: str | None = None

# In-memory storage
_seeds: dict[str, StoredSeed] = {}

def store_seed(content: str, source: Literal["text", "file"], file_path: str | None = None) -> StoredSeed:
    """Store a seed and return the stored seed."""
    seed_id = str(uuid.uuid4())
    seed = StoredSeed(
        seed_id=seed_id,
        content=content,
        source=source,
        created_at=datetime.utcnow(),
        content_length=len(content),
        file_path=file_path,
    )
    _seeds[seed_id] = seed
    return seed

def get_seed(seed_id: str) -> StoredSeed | None:
    """Retrieve a seed by ID."""
    return _seeds.get(seed_id)
```

### Response Format

```python
# Success response
{
    "status": "accepted",
    "seed_id": "550e8400-e29b-41d4-a716-446655440000",
    "content_length": 1234,
    "source": "text",  # or "file"
}

# Error response
{
    "status": "error",
    "error": "Content cannot be empty",
}
```

### File Structure (Per Architecture)

```
src/yolo_developer/
├── mcp/
│   ├── __init__.py         # Exports mcp, run_server, TransportType
│   ├── server.py           # FastMCP server definition (from 14.1)
│   └── tools.py            # MCP tool implementations (NEW)
```

### Testing Patterns (from Story 14.1)

Per established patterns from Story 14.1:

```python
# tests/unit/mcp/test_tools.py
from __future__ import annotations

import pytest

from yolo_developer.mcp.tools import yolo_seed, store_seed, get_seed


@pytest.mark.asyncio
async def test_yolo_seed_with_text_content():
    """Test yolo_seed with text content."""
    result = await yolo_seed(content="Build a REST API for user management")

    assert result["status"] == "accepted"
    assert "seed_id" in result
    assert result["content_length"] == len("Build a REST API for user management")
    assert result["source"] == "text"


@pytest.mark.asyncio
async def test_yolo_seed_with_empty_content():
    """Test yolo_seed validation for empty content."""
    result = await yolo_seed(content="")

    assert result["status"] == "error"
    assert "error" in result


# tests/integration/mcp/test_tools_integration.py
@pytest.mark.asyncio
async def test_yolo_seed_via_mcp_client():
    """Test yolo_seed through FastMCP Client."""
    from fastmcp import Client
    from yolo_developer.mcp import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("yolo_seed", {"content": "Test seed"})
        assert result["status"] == "accepted"
```

### Key Files to Touch

**Create:**
- `src/yolo_developer/mcp/tools.py` - MCP tool implementations

**Modify:**
- `src/yolo_developer/mcp/server.py` - Import tools.py to register tools
- `src/yolo_developer/mcp/__init__.py` - Export tool-related functions if needed

**Tests Create:**
- `tests/unit/mcp/test_tools.py` - Unit tests for tools
- Update `tests/integration/mcp/test_mcp_server.py` - Integration tests for yolo_seed

### Dependencies

Uses existing FastMCP from Story 14.1. Standard library only:
- `uuid` for seed ID generation
- `datetime` for timestamps
- `pathlib` for file operations

### Previous Story Learnings (Story 14.1)

1. FastMCP 2.x uses `fastmcp.settings` for global configuration
2. Use `from __future__ import annotations` in all files
3. Run `ruff check` and `mypy` before committing
4. Tests should use `pytest.mark.asyncio` for async functions
5. Use FastMCP Client for integration testing

### Claude Desktop Testing

After implementation, the yolo_seed tool can be tested with Claude Desktop:

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

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-004] - MCP Implementation decision
- [Source: _bmad-output/planning-artifacts/architecture.md#ARCH-PATTERN-4] - FastMCP decorator pattern
- [Source: _bmad-output/planning-artifacts/epics.md#Story 14.2] - Story definition
- [Source: _bmad-output/planning-artifacts/prd.md#FR112-FR117] - MCP functional requirements
- [FastMCP Docs: https://gofastmcp.com] - FastMCP 2.x documentation
- [Related: Story 14.1 (FastMCP Server Setup)] - Foundation story
- [Related: Story 14.3-14.5] - Future MCP tools that will follow same pattern

## Story Progress Notes

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - Implementation completed without issues.

### Completion Notes List

1. Created `src/yolo_developer/mcp/tools.py` with yolo_seed MCP tool using @mcp.tool decorator
2. Implemented StoredSeed dataclass for seed storage with UUID, content, source, timestamp metadata
3. Implemented in-memory seed storage with store_seed(), get_seed(), clear_seeds() functions
4. Added comprehensive input validation (empty content, missing file, edge cases)
5. Created 17 unit tests in `tests/unit/mcp/test_tools.py` covering all ACs
6. Created 8 integration tests using FastMCP Client in `tests/integration/mcp/test_mcp_server.py`
7. Updated SERVER_INSTRUCTIONS with yolo_seed documentation
8. Updated mcp/__init__.py with new exports (yolo_seed, StoredSeed, store_seed, get_seed, clear_seeds)
9. All 2766 unit tests pass, 43 MCP-specific tests pass
10. Ruff check and mypy pass with no issues

### Code Review Fixes (2026-01-19)

7 issues identified and fixed during adversarial code review:

1. **Issue 1 (MEDIUM)**: Added thread safety with `threading.Lock` to `_seeds` storage for concurrent MCP requests
2. **Issue 2 (MEDIUM)**: Added autouse fixture to unit tests for test isolation (clear_seeds before each test)
3. **Issue 3 (LOW)**: Added test for directory path validation error
4. **Issue 4 (LOW)**: Added test for empty file validation error
5. **Issue 5 (LOW)**: Added test for file read error handling (OSError)
6. **Issue 6 (LOW)**: Enhanced docstring for `clear_seeds()` clarifying it's for testing
7. **Issue 7 (LOW)**: Added seed retrieval verification to file input integration test

Post-review: 22 unit tests + 8 integration tests (29 total MCP tests), all passing.

### File List

**Created:**
- src/yolo_developer/mcp/tools.py

**Modified:**
- src/yolo_developer/mcp/server.py (added _register_tools() function, updated SERVER_INSTRUCTIONS)
- src/yolo_developer/mcp/__init__.py (added exports for tools module)
- tests/unit/mcp/test_tools.py (created new test file, 22 tests)
- tests/integration/mcp/test_mcp_server.py (added yolo_seed integration tests, 8 tests)

### Change Log

- 2026-01-19: Implemented Story 14.2 - yolo_seed MCP tool with full test coverage
- 2026-01-19: Applied 7 code review fixes (thread safety, test isolation, edge case tests)
