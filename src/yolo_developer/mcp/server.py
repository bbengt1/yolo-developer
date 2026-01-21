"""FastMCP server implementation for YOLO Developer.

This module provides the MCP (Model Context Protocol) server that exposes
YOLO Developer functionality to external tools like Claude Code.

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

References:
    - ADR-004: FastMCP 2.x for MCP server implementation
    - ARCH-PATTERN-4: FastMCP decorator-based MCP server
    - FR112-FR117: MCP protocol integration requirements
"""

from __future__ import annotations

from enum import Enum

from fastmcp import FastMCP
from fastmcp import settings as fastmcp_settings

from yolo_developer import __version__


class TransportType(Enum):
    """Transport protocol types for the MCP server.

    Attributes:
        STDIO: Standard input/output transport for Claude Desktop integration.
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

## Coming Soon
- yolo_audit: Access audit trail

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
        transport: Transport protocol to use. Defaults to STDIO for Claude Desktop.
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

    if transport == TransportType.HTTP:
        mcp.run(transport="http", port=port)
    else:
        mcp.run(transport="stdio")


__all__ = ["SERVER_INSTRUCTIONS", "TransportType", "mcp", "run_server"]


def _register_tools() -> None:
    """Register MCP tools with the server.

    This function imports the tools module to trigger tool registration
    via the @mcp.tool decorator. It's called lazily to avoid circular imports.
    """
    from yolo_developer.mcp import tools as _tools  # noqa: F401


# Register tools when module is fully loaded
_register_tools()
