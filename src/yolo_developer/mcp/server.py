"""FastMCP server implementation for YOLO Developer.

This module provides the MCP (Model Context Protocol) server that exposes
YOLO Developer functionality to MCP-compatible AI assistants and tools.

Example:
    >>> from yolo_developer.mcp import mcp
    >>> # Server is ready to be used with MCP clients
    >>> mcp.run()  # Start with default STDIO transport

    >>> # Or with HTTP transport
    >>> mcp.run(transport="http", port=8000)

The server exposes the following tools:
    - yolo_seed: Provide seed requirements for development
    - yolo_run: Execute autonomous sprint
    - yolo_status: Query sprint status
    - yolo_audit: Access audit trail
    - yolo_git_commit: Commit changes to git
    - yolo_pr_create: Create pull requests
    - yolo_pr_respond: Respond to PR review comments
    - yolo_issue_create: Create issues
    - yolo_release_create: Create releases

References:
    - ADR-004: FastMCP 2.x for MCP server implementation
    - ARCH-PATTERN-4: FastMCP decorator-based MCP server
    - FR112-FR117: MCP protocol integration requirements
"""

from __future__ import annotations

from enum import Enum

from fastmcp import FastMCP
from fastmcp import settings as fastmcp_settings

from yolo_developer.version import __version__


class TransportType(Enum):
    """Transport protocol types for the MCP server.

    Attributes:
        STDIO: Standard input/output transport for local AI assistant integration.
        HTTP: HTTP transport for remote MCP client access.
    """

    STDIO = "stdio"
    HTTP = "http"


# Configure FastMCP global settings for production safety
# Error masking hides internal error details from MCP clients
fastmcp_settings.mask_error_details = True

# Server instructions for MCP clients
# These help LLMs understand what tools are available and how to use them
SERVER_INSTRUCTIONS = """
YOLO Developer MCP Server

Available tools:

## yolo_seed
Provide seed requirements for autonomous development. You can provide requirements
either as text content or by specifying a file path to read from.

Parameters:
- content (optional): Seed requirements as plain text
- file_path (optional): Path to file containing seed requirements

Returns seed_id for tracking, content_length, and source type.

Example:
  yolo_seed(content="Build a REST API for user management")
  yolo_seed(file_path="/path/to/requirements.txt")

## yolo_run
Execute a sprint from a previously seeded requirements document.

Parameters:
- seed_id (optional): Seed identifier returned by yolo_seed (preferred)
- seed_content (optional): Raw seed content (used if seed_id not provided)

Returns sprint_id for status queries and thread_id for checkpointing.

Example:
  yolo_run(seed_id="550e8400-e29b-41d4-a716-446655440000")

## yolo_status
Query the current status of a sprint by sprint_id.

Parameters:
- sprint_id (required): Sprint identifier returned by yolo_run

Returns sprint status, timestamps, and error details if the sprint failed.

Example:
  yolo_status(sprint_id="sprint-abcdef12")

## yolo_audit
Access audit trail entries with optional filtering and pagination.

Parameters:
- agent (optional): Filter by agent name (e.g., "analyst")
- decision_type (optional): Filter by decision type (e.g., "requirement_analysis")
- artifact_type (optional): Filter by artifact type (e.g., "requirement")
- start_time (optional): ISO-8601 start timestamp (inclusive)
- end_time (optional): ISO-8601 end timestamp (inclusive)
- limit (optional): Max entries to return (default 100)
- offset (optional): Number of entries to skip (default 0)

Returns audit entries, pagination metadata, and total count.

Example:
  yolo_audit(agent="analyst", limit=25, offset=0)

## yolo_git_commit
Commit staged changes and optionally push.

## yolo_pr_create
Create a pull request from the current branch.

## yolo_pr_respond
Reply to a pull request review comment.

## yolo_issue_create
Create a GitHub issue.

## yolo_release_create
Create a GitHub release with generated notes.

## yolo_import_issue
Import a GitHub issue and convert it into a user story.

## yolo_import_issues
Import multiple GitHub issues into user stories.

## yolo_preview_import
Preview the issue import output without updating the issue.

Use these tools to integrate YOLO Developer into your AI workflow.
"""

# Create the FastMCP server instance
# This is the main entry point for MCP clients
# Note: mask_error_details is configured via fastmcp_settings (line 49), not constructor
mcp = FastMCP(
    name="YOLO Developer",
    instructions=SERVER_INSTRUCTIONS,
    version=__version__,
)


def run_server(
    transport: TransportType | str = TransportType.STDIO,
    port: int = 8000,
) -> None:
    """Run the MCP server with the specified transport.

    Args:
        transport: Transport protocol to use. Defaults to STDIO for local AI assistants.
        port: Port number for HTTP transport. Defaults to 8000.

    Example:
        >>> from yolo_developer.mcp.server import run_server, TransportType
        >>> run_server()  # STDIO transport (default)
        >>> run_server(TransportType.HTTP, port=8080)  # HTTP transport
        >>> run_server("http", port=8080)  # String transport type also works
    """
    # Normalize transport to enum
    if isinstance(transport, str):
        transport = TransportType(transport.lower())

    _register_tools()

    if transport == TransportType.HTTP:
        mcp.run(transport="http", port=port)
    else:
        mcp.run(transport="stdio")


__all__ = ["SERVER_INSTRUCTIONS", "TransportType", "mcp", "run_server"]


_TOOLS_REGISTERED = False


def _register_tools() -> None:
    """Register MCP tools with the server.

    This function imports the tools module to trigger tool registration
    via the @mcp.tool decorator. It's called lazily to avoid importing
    LLM dependencies during CLI help output.
    """
    global _TOOLS_REGISTERED
    if _TOOLS_REGISTERED:
        return
    from yolo_developer.mcp import tools as _tools  # noqa: F401

    _TOOLS_REGISTERED = True
