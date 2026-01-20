"""FastMCP server implementation for YOLO Developer.

This module provides the MCP (Model Context Protocol) server that exposes
YOLO Developer functionality to external tools like Claude Code.

Example:
    >>> from yolo_developer.mcp import mcp
    >>> # Server is ready to be used with MCP clients
    >>> mcp.run()  # Start with default STDIO transport

    >>> # Or with HTTP transport
    >>> mcp.run(transport="http", port=8000)

The server exposes the following tools (implemented in future stories):
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

Available tools will include:
- yolo_seed: Provide seed requirements for development
- yolo_run: Execute autonomous sprint
- yolo_status: Query sprint status
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
