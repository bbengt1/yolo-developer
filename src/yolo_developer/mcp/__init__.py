"""MCP (Model Context Protocol) server for YOLO Developer.

This module exposes YOLO Developer functionality to external tools
via the MCP protocol, enabling integration with Claude Code and
other MCP-compatible clients.

Example:
    >>> from yolo_developer.mcp import mcp, run_server, TransportType
    >>> mcp.run()  # Start with STDIO transport (default)
    >>> run_server()  # Start with STDIO transport (type-safe wrapper)
    >>> run_server(TransportType.HTTP, port=8080)  # HTTP transport

Note:
    Use `run_server()` for type-safe transport selection via the TransportType enum.
    Use `mcp.run()` directly for quick testing or when passing transport as string.

The server is configured with:
    - Name: "YOLO Developer"
    - Version: Package version from yolo_developer.__version__
    - Instructions: Description of available tools
    - Error masking: Enabled for production safety

Tools available (implemented in future stories):
    - yolo_seed: Provide seed requirements
    - yolo_run: Execute autonomous sprint
    - yolo_status: Query sprint status
    - yolo_audit: Access audit trail
"""

from __future__ import annotations

from yolo_developer.mcp.server import (
    SERVER_INSTRUCTIONS,
    TransportType,
    mcp,
    run_server,
)

# Public API exports:
# - mcp: FastMCP server instance for direct use
# - run_server: Type-safe wrapper for starting the server
# - TransportType: Enum for transport protocol selection
# - SERVER_INSTRUCTIONS: Server instructions string for reference
__all__ = ["SERVER_INSTRUCTIONS", "TransportType", "mcp", "run_server"]
